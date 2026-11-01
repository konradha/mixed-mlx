#include "metal_spd.hpp"
#include "../common/buffers.hpp"
#include "../common/clock.hpp"
#include "../common/order.hpp"
#include "mps_matmul.hpp"

#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace msqrt {
namespace {

using Clock = common::Clock;

struct SquareParams {
  std::uint32_t order;
};

struct NormPair {
  float residual_squared;
  float matrix_squared;
};

struct WorkingHost {
  float high;
  float low;
};

double elapsed(Clock::time_point start) {
  return common::elapsed(start);
}

std::uint32_t checked_order(std::size_t order) {
  if (order == 0) {
    throw std::runtime_error("Matrix order exceeds Metal range.");
  }
  return common::to_u32(order);
}

Matrix read_working_matrix(MTL::Buffer *buffer, std::size_t count) {
  const WorkingHost *source =
      static_cast<const WorkingHost *>(buffer->contents());
  Matrix result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = static_cast<double>(source[index].high) +
                    static_cast<double>(source[index].low);
  }
  return result;
}

void encode_residual_and_norm(Metal &metal, MTL::ComputeCommandEncoder *encoder,
                              MTL::Buffer *matrix_work, MTL::Buffer *root_work,
                              MTL::Buffer *residual_work,
                              MTL::Buffer *residual_f32, MTL::Buffer *norm,
                              const SquareParams &square, std::uint32_t count) {
  MTL::ComputePipelineState *residual_pipeline =
      metal.pipeline("working_residual");
  encoder->setBuffer(matrix_work, 0, 0);
  encoder->setBuffer(root_work, 0, 1);
  encoder->setBuffer(residual_work, 0, 2);
  encoder->setBuffer(residual_f32, 0, 3);
  encoder->setBytes(&square, sizeof(square), 4);
  Metal::dispatch2d(encoder, residual_pipeline, square.order, square.order);

  MTL::ComputePipelineState *norm_pipeline = metal.pipeline("working_norm");
  encoder->setBuffer(matrix_work, 0, 0);
  encoder->setBuffer(residual_work, 0, 1);
  encoder->setBuffer(norm, 0, 2);
  encoder->setBytes(&count, sizeof(count), 3);
  Metal::dispatch1d(encoder, norm_pipeline, 256, 256);
}

double measured_residual(MTL::Buffer *norm) {
  const NormPair value = *static_cast<const NormPair *>(norm->contents());
  if (!(value.matrix_squared > 0.0f) ||
      !std::isfinite(value.residual_squared)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(static_cast<double>(value.residual_squared) /
                   static_cast<double>(value.matrix_squared));
}

}  // namespace

GpuFactor shared_eigen(const Problem &problem) {
  const std::size_t count = problem.order * problem.order;
  GpuFactor result;
  result.basis.assign(problem.matrix.begin(), problem.matrix.end());
  result.root_spectrum.resize(problem.order);
  __LAPACK_int order = static_cast<__LAPACK_int>(checked_order(problem.order));
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_size = -1;
  __LAPACK_int info = 0;
  float work_query = 0.0f;
  __LAPACK_int integer_query = 0;
  const auto start = Clock::now();
  ssyevd_(&job, &triangle, &order, result.basis.data(), &order,
          result.root_spectrum.data(), &work_query, &work_size, &integer_query,
          &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate ssyevd workspace query failed.");
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  integer_size = integer_query;
  std::vector<float> work(static_cast<std::size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<std::size_t>(integer_size));
  ssyevd_(&job, &triangle, &order, result.basis.data(), &order,
          result.root_spectrum.data(), work.data(), &work_size,
          integer_work.data(), &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate ssyevd failed.");
  }
  result.seconds = elapsed(start);
  for (float &value : result.root_spectrum) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
      throw std::runtime_error(
          "The shared FP32 factor has an invalid spectrum.");
    }
    value = std::sqrt(value);
  }
  if (result.basis.size() != count) {
    throw std::runtime_error("The shared FP32 factor has an invalid size.");
  }
  return result;
}

Run metal_fp32(Metal &metal, const Problem &problem,
               std::size_t max_corrections, double tolerance) {
  const std::uint32_t order = checked_order(problem.order);
  const std::size_t count = problem.order * problem.order;
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Matrix exceeds Metal element range.");
  }
  const std::uint32_t count32 = static_cast<std::uint32_t>(count);
  const std::size_t bytes = count * sizeof(float);
  const std::size_t working_bytes = count * sizeof(WorkingHost);
  Run run;
  run.stage = "baseline";
  run.backend = "metal";
  run.format = "fp32_compensated";
  run.case_name = problem.name;
  run.order = problem.order;
  run.condition = problem.condition;
  const auto total_start = Clock::now();
  GpuFactor factor = shared_eigen(problem);
  run.times.factor_seconds = factor.seconds;

  common::BufferGuard<Metal> owned(metal);
  const auto make = [&](std::size_t size) { return owned.make(size); };
  MTL::Buffer *basis = make(bytes);
  MTL::Buffer *basis_t = make(bytes);
  MTL::Buffer *spectrum = make(problem.order * sizeof(float));
  MTL::Buffer *inverse = make(bytes);
  MTL::Buffer *scaled_basis = make(bytes);
  MTL::Buffer *root_f32 = make(bytes);
  MTL::Buffer *residual_f32 = make(bytes);
  MTL::Buffer *workspace = make(bytes);
  MTL::Buffer *coordinates = make(bytes);
  MTL::Buffer *correction = make(bytes);
  MTL::Buffer *matrix_work = make(working_bytes);
  MTL::Buffer *root_work = make(working_bytes);
  MTL::Buffer *residual_work = make(working_bytes);
  MTL::Buffer *norm = make(sizeof(NormPair));

  WorkingHost *matrix_values =
      static_cast<WorkingHost *>(matrix_work->contents());
  for (std::size_t index = 0; index < count; ++index) {
    const float high = static_cast<float>(problem.matrix[index]);
    matrix_values[index] = {
        high, static_cast<float>(problem.matrix[index] - double{high})};
  }
  std::memcpy(basis->contents(), factor.basis.data(), bytes);
  std::memcpy(spectrum->contents(), factor.root_spectrum.data(),
              problem.order * sizeof(float));

  MpsMatmul mps_matmul(basis, problem.order);
  const SquareParams square{order};
  auto start = Clock::now();
  metal.submit_commands([&](MTL::CommandBuffer *command) {
    MTL::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
    MTL::ComputePipelineState *transpose = metal.pipeline("transpose_f32");
    encoder->setBuffer(basis, 0, 0);
    encoder->setBuffer(basis_t, 0, 1);
    encoder->setBytes(&square, sizeof(square), 2);
    Metal::dispatch2d(encoder, transpose, order, order);

    MTL::ComputePipelineState *inverse_pipeline =
        metal.pipeline("sylvester_inverse_f32");
    encoder->setBuffer(spectrum, 0, 0);
    encoder->setBuffer(inverse, 0, 1);
    encoder->setBytes(&square, sizeof(square), 2);
    Metal::dispatch2d(encoder, inverse_pipeline, order, order);

    MTL::ComputePipelineState *scale = metal.pipeline("scale_columns_f32");
    encoder->setBuffer(basis, 0, 0);
    encoder->setBuffer(spectrum, 0, 1);
    encoder->setBuffer(scaled_basis, 0, 2);
    encoder->setBytes(&square, sizeof(square), 3);
    Metal::dispatch2d(encoder, scale, order, order);
    encoder->endEncoding();

    mps_matmul.encode(command, scaled_basis, basis_t, root_f32);

    encoder = command->computeCommandEncoder();
    MTL::ComputePipelineState *convert = metal.pipeline("working_from_f32");
    encoder->setBuffer(root_f32, 0, 0);
    encoder->setBuffer(root_work, 0, 1);
    encoder->setBytes(&square, sizeof(square), 2);
    Metal::dispatch2d(encoder, convert, order, order);
    encoder->endEncoding();
  });
  run.times.initial_seconds = elapsed(start);

  start = Clock::now();
  metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
    encode_residual_and_norm(metal, encoder, matrix_work, root_work,
                             residual_work, residual_f32, norm, square,
                             count32);
  });
  run.times.residual_seconds += elapsed(start);
  double residual_norm = measured_residual(norm);
  run.steps.push_back({0, residual_norm,
                       relative_error(read_working_matrix(root_work, count),
                                      problem.reference_root)});

  for (std::size_t iteration = 0;
       residual_norm > tolerance && iteration < max_corrections; ++iteration) {
    start = Clock::now();
    metal.submit_commands([&](MTL::CommandBuffer *command) {
      mps_matmul.encode(command, basis_t, residual_f32, workspace);
      mps_matmul.encode(command, workspace, basis, coordinates);

      MTL::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
      MTL::ComputePipelineState *scale = metal.pipeline("apply_scale_f32");
      encoder->setBuffer(coordinates, 0, 0);
      encoder->setBuffer(inverse, 0, 1);
      encoder->setBytes(&count32, sizeof(count32), 2);
      Metal::dispatch1d(encoder, scale, count);
      encoder->endEncoding();

      mps_matmul.encode(command, basis, coordinates, workspace);
      mps_matmul.encode(command, workspace, basis_t, correction);

      encoder = command->computeCommandEncoder();
      MTL::ComputePipelineState *add = metal.pipeline("working_add_symmetric");
      encoder->setBuffer(root_work, 0, 0);
      encoder->setBuffer(correction, 0, 1);
      encoder->setBytes(&square, sizeof(square), 2);
      Metal::dispatch2d(encoder, add, order, order);
      encode_residual_and_norm(metal, encoder, matrix_work, root_work,
                               residual_work, residual_f32, norm, square,
                               count32);
      encoder->endEncoding();
    });
    run.times.correction_seconds += elapsed(start);
    residual_norm = measured_residual(norm);
    run.steps.push_back({iteration + 1, residual_norm,
                         relative_error(read_working_matrix(root_work, count),
                                        problem.reference_root)});
  }
  run.converged = residual_norm <= tolerance;
  run.times.total_seconds = elapsed(total_start);
  return run;
}

}  // namespace msqrt
