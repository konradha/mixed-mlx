#include "spectral.hpp"
#include "../common/buffers.hpp"
#include "../common/clock.hpp"
#include "../common/order.hpp"
#include "lowbit_matmul.hpp"
#include "mps_matmul.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace msqrt {
namespace {

using Clock = common::Clock;
struct SquareParams {
  std::uint32_t order;
};

struct CodecParams {
  std::uint32_t elements;
  std::uint32_t blocks;
};

struct NormPair {
  float residual_squared;
  float matrix_squared;
};

struct WorkingHost {
  float high;
  float low;
};

struct Factor {
  std::vector<float> basis;
  std::vector<float> root_spectrum;
  double seconds = 0.0;
};

struct BufferReleaser {
  void operator()(MTL::Buffer *buffer) const {
    if (buffer != nullptr) {
      buffer->release();
    }
  }
};

using BufferPtr = std::unique_ptr<MTL::Buffer, BufferReleaser>;
struct ResidencyReleaser {
  void operator()(MTL::ResidencySet *set) const {
    if (set != nullptr) {
      set->release();
    }
  }
};

using ResidencyPtr = std::unique_ptr<MTL::ResidencySet, ResidencyReleaser>;

struct Quantized {
  std::vector<float> values;
  BufferPtr encoded;
  double seconds = 0.0;
  double bits_per_value = 0.0;
};

std::size_t elements(std::size_t order) { return common::elements(order); }

__LAPACK_int lapack_order(std::size_t order) {
  if (order >
      static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max())) {
    throw std::runtime_error("The matrix order exceeds the LAPACK range.");
  }
  return static_cast<__LAPACK_int>(order);
}

std::uint32_t metal_count(std::size_t count) { return common::to_u32(count); }

double elapsed(Clock::time_point start) { return common::elapsed(start); }

void dgemm(char left_transpose, char right_transpose, std::size_t order,
           const double *left, const double *right, double *output) {
  const __LAPACK_int n = lapack_order(order);
  const double alpha = 1.0;
  const double beta = 0.0;
  dgemm_(&left_transpose, &right_transpose, &n, &n, &n, &alpha, left, &n, right,
         &n, &beta, output, &n);
}

void eigen_double(Matrix &vectors, std::vector<double> &values) {
  __LAPACK_int order = lapack_order(values.size());
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_size = -1;
  __LAPACK_int info = 0;
  double work_query = 0.0;
  __LAPACK_int integer_query = 0;
  dsyevd_(&job, &triangle, &order, vectors.data(), &order, values.data(),
          &work_query, &work_size, &integer_query, &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("The FP64 eigenvalue workspace query failed.");
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  integer_size = integer_query;
  std::vector<double> work(static_cast<std::size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<std::size_t>(integer_size));
  dsyevd_(&job, &triangle, &order, vectors.data(), &order, values.data(),
          work.data(), &work_size, integer_work.data(), &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("The FP64 eigenvalue calculation failed.");
  }
}

Factor factor_float(const Problem &problem) {
  Factor factor;
  factor.basis.assign(problem.matrix.begin(), problem.matrix.end());
  factor.root_spectrum.resize(problem.order);
  __LAPACK_int order = lapack_order(problem.order);
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_size = -1;
  __LAPACK_int info = 0;
  float work_query = 0.0f;
  __LAPACK_int integer_query = 0;
  const auto start = Clock::now();
  ssyevd_(&job, &triangle, &order, factor.basis.data(), &order,
          factor.root_spectrum.data(), &work_query, &work_size, &integer_query,
          &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("The FP32 eigenvalue workspace query failed.");
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  integer_size = integer_query;
  std::vector<float> work(static_cast<std::size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<std::size_t>(integer_size));
  ssyevd_(&job, &triangle, &order, factor.basis.data(), &order,
          factor.root_spectrum.data(), work.data(), &work_size,
          integer_work.data(), &integer_size, &info);
  factor.seconds = elapsed(start);
  if (info != 0) {
    throw std::runtime_error("The FP32 eigenvalue calculation failed.");
  }
  for (float &value : factor.root_spectrum) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
      throw std::runtime_error("The FP32 factor has an invalid spectrum.");
    }
    value = std::sqrt(value);
  }
  return factor;
}

Matrix reference_root(const Matrix &matrix, std::size_t order) {
  Matrix vectors = matrix;
  std::vector<double> values(order);
  eigen_double(vectors, values);
  Matrix weighted = vectors;
  for (std::size_t column = 0; column < order; ++column) {
    if (!(values[column] > 0.0)) {
      throw std::runtime_error(
          "The reference matrix is not positive definite.");
    }
    const double scale = std::sqrt(values[column]);
    for (std::size_t row = 0; row < order; ++row) {
      weighted[row + column * order] *= scale;
    }
  }
  Matrix root(elements(order));
  dgemm('N', 'T', order, weighted.data(), vectors.data(), root.data());
  return root;
}

Matrix deterministic_basis(std::size_t order) {
  Matrix basis(elements(order));
  for (std::size_t column = 0; column < order; ++column) {
    for (std::size_t row = 0; row < order; ++row) {
      basis[row + column * order] =
          std::sin(0.17 * static_cast<double>((row + 1) * (column + 2))) +
          std::cos(0.11 * static_cast<double>((row + 3) * (column + 1)));
    }
    for (std::size_t prior = 0; prior < column; ++prior) {
      double projection = 0.0;
      for (std::size_t row = 0; row < order; ++row) {
        projection += basis[row + prior * order] * basis[row + column * order];
      }
      for (std::size_t row = 0; row < order; ++row) {
        basis[row + column * order] -= projection * basis[row + prior * order];
      }
    }
    double norm = 0.0;
    for (std::size_t row = 0; row < order; ++row) {
      norm = std::hypot(norm, basis[row + column * order]);
    }
    if (!(norm > std::numeric_limits<double>::epsilon())) {
      throw std::runtime_error("The deterministic basis calculation failed.");
    }
    for (std::size_t row = 0; row < order; ++row) {
      basis[row + column * order] /= norm;
    }
  }
  return basis;
}

class Generator {
public:
  explicit Generator(std::uint64_t seed) : state_(seed) {}

  double uniform() {
    std::uint64_t value = state_;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state_ = value;
    const std::uint64_t bits = (value * UINT64_C(2685821657736338717)) >> 11;
    return static_cast<double>(bits) * 0x1.0p-53;
  }

  double normal() {
    const double first =
        std::max(uniform(), std::numeric_limits<double>::min());
    return std::sqrt(-2.0 * std::log(first)) *
           std::cos(6.283185307179586 * uniform());
  }

private:
  std::uint64_t state_;
};

double infinity_norm(const Matrix &matrix, std::size_t order) {
  double result = 0.0;
  for (std::size_t row = 0; row < order; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < order; ++column) {
      sum += std::abs(matrix[row + column * order]);
    }
    result = std::max(result, sum);
  }
  return result;
}

double relative_error(const Matrix &value, const Matrix &reference) {
  double difference = 0.0;
  double norm = 0.0;
  for (std::size_t index = 0; index < value.size(); ++index) {
    difference = std::hypot(difference, value[index] - reference[index]);
    norm = std::hypot(norm, reference[index]);
  }
  return difference / norm;
}

double relative_error(const std::vector<float> &value,
                      const std::vector<float> &reference) {
  double difference = 0.0;
  double norm = 0.0;
  for (std::size_t index = 0; index < value.size(); ++index) {
    difference =
        std::hypot(difference, double{value[index]} - double{reference[index]});
    norm = std::hypot(norm, double{reference[index]});
  }
  return difference / norm;
}
float binary16_to_float(std::uint16_t bits) {
  const bool negative = (bits & UINT16_C(0x8000)) != 0;
  const std::uint16_t exponent = (bits >> 10) & UINT16_C(0x1f);
  const std::uint16_t fraction = bits & UINT16_C(0x03ff);
  float magnitude = 0.0f;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(fraction), -24);
  } else if (exponent == 31) {
    magnitude = fraction == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
  } else {
    magnitude = std::ldexp(static_cast<float>(UINT16_C(0x0400) + fraction),
                           static_cast<int>(exponent) - 25);
  }
  return negative ? -magnitude : magnitude;
}

double fp16_relative_error(MTL::Buffer *encoded,
                           const std::vector<float> &reference) {
  const auto *values = static_cast<const std::uint16_t *>(encoded->contents());
  double difference = 0.0;
  double norm = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double value = binary16_to_float(values[index]);
    difference =
        std::hypot(difference, value - static_cast<double>(reference[index]));
    norm = std::hypot(norm, static_cast<double>(reference[index]));
  }
  return difference / norm;
}

bool fp16_all_finite(MTL::Buffer *encoded, std::size_t count) {
  const auto *values = static_cast<const std::uint16_t *>(encoded->contents());
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(binary16_to_float(values[index])))
      return false;
  }
  return true;
}

const char *format_name(FactorFormat format) {
  switch (format) {
  case FactorFormat::Fp32:
    return "fp32";
  case FactorFormat::Fp16:
    return "fp16";
  case FactorFormat::Bf16:
    return "bf16";
  case FactorFormat::Fp8E4M3:
    return "fp8_e4m3";
  case FactorFormat::Fp8E5M2:
    return "fp8_e5m2";
  case FactorFormat::MxFp8E4M3:
    return "mxfp8_e4m3";
  case FactorFormat::MxFp4E2M1:
    return "mxfp4_e2m1";
  }
  throw std::runtime_error("The factor format is unknown.");
}

std::size_t value_bytes(FactorFormat format, std::size_t count) {
  switch (format) {
  case FactorFormat::Fp32:
    return 4 * count;
  case FactorFormat::Fp16:
  case FactorFormat::Bf16:
    return 2 * count;
  case FactorFormat::Fp8E4M3:
  case FactorFormat::Fp8E5M2:
  case FactorFormat::MxFp8E4M3:
    return count;
  case FactorFormat::MxFp4E2M1:
    return (count + 1) / 2;
  }
  throw std::runtime_error("The factor format is unknown.");
}

// Vector kernels load 4 bytes at once. Pad 1-byte and 4-bit values so the
// last vector load stays in bounds. Bits math still uses exact value_bytes.
std::size_t encoded_alloc_bytes(FactorFormat format, std::size_t count) {
  switch (format) {
  case FactorFormat::Fp8E4M3:
  case FactorFormat::Fp8E5M2:
  case FactorFormat::MxFp8E4M3:
    return ((count + 3) / 4) * 4;
  case FactorFormat::MxFp4E2M1:
    return ((count + 7) / 8) * 4;
  default:
    return value_bytes(format, count);
  }
}

std::size_t scale_bytes(FactorFormat format, std::size_t blocks) {
  return format == FactorFormat::MxFp8E4M3 || format == FactorFormat::MxFp4E2M1
             ? blocks
             : 0;
}

void encode_codec(Metal &metal, MTL::ComputeCommandEncoder *encoder,
                  FactorFormat format, MTL::Buffer *source,
                  MTL::Buffer *encoded, MTL::Buffer *scales,
                  const CodecParams &params) {
  const bool mx =
      format == FactorFormat::MxFp8E4M3 || format == FactorFormat::MxFp4E2M1;
  const char *name = nullptr;
  switch (format) {
  case FactorFormat::Fp32:
    name = "encode_fp32";
    break;
  case FactorFormat::Fp16:
    name = "encode_fp16";
    break;
  case FactorFormat::Bf16:
    name = "encode_bf16";
    break;
  case FactorFormat::Fp8E4M3:
    name = "encode_fp8_e4m3";
    break;
  case FactorFormat::Fp8E5M2:
    name = "encode_fp8_e5m2";
    break;
  case FactorFormat::MxFp8E4M3:
    name = "encode_mxfp8_e4m3";
    break;
  case FactorFormat::MxFp4E2M1:
    name = "encode_mxfp4_e2m1";
    break;
  }
  MTL::ComputePipelineState *pipeline = metal.pipeline(name);
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(encoded, 0, 1);
  if (mx) {
    encoder->setBuffer(scales, 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->setComputePipelineState(pipeline);
    encoder->dispatchThreadgroups(MTL::Size::Make(params.blocks, 1, 1),
                                  MTL::Size::Make(32, 1, 1));
    return;
  }
  encoder->setBytes(&params, sizeof(params), 2);
  const std::size_t width =
      format == FactorFormat::Fp8E4M3 || format == FactorFormat::Fp8E5M2 ? 4
                                                                         : 1;
  Metal::dispatch1d(encoder, pipeline, (params.elements + width - 1) / width);
}

void decode_codec(Metal &metal, MTL::ComputeCommandEncoder *encoder,
                  FactorFormat format, MTL::Buffer *encoded,
                  MTL::Buffer *scales, MTL::Buffer *decoded,
                  const CodecParams &params) {
  const bool mx =
      format == FactorFormat::MxFp8E4M3 || format == FactorFormat::MxFp4E2M1;
  const char *name = nullptr;
  switch (format) {
  case FactorFormat::Fp32:
    name = "decode_fp32";
    break;
  case FactorFormat::Fp16:
    name = "decode_fp16";
    break;
  case FactorFormat::Bf16:
    name = "decode_bf16";
    break;
  case FactorFormat::Fp8E4M3:
    name = "decode_fp8_e4m3";
    break;
  case FactorFormat::Fp8E5M2:
    name = "decode_fp8_e5m2";
    break;
  case FactorFormat::MxFp8E4M3:
    name = "decode_mxfp8_e4m3";
    break;
  case FactorFormat::MxFp4E2M1:
    name = "decode_mxfp4_e2m1";
    break;
  }
  MTL::ComputePipelineState *pipeline = metal.pipeline(name);
  encoder->setBuffer(encoded, 0, 0);
  if (mx) {
    encoder->setBuffer(scales, 0, 1);
    encoder->setBuffer(decoded, 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
  } else {
    encoder->setBuffer(decoded, 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
  }
  std::size_t width = 1;
  if (format == FactorFormat::Fp8E4M3 || format == FactorFormat::Fp8E5M2 ||
      format == FactorFormat::MxFp8E4M3) {
    width = 4;
  } else if (format == FactorFormat::MxFp4E2M1) {
    width = 8;
  }
  Metal::dispatch1d(encoder, pipeline, (params.elements + width - 1) / width);
}
// Native low-bit products use the physical layouts reported by LowbitMatmul.
bool is_low_mpp_basis(FactorFormat format) {
  return format == FactorFormat::Fp8E4M3 || format == FactorFormat::Fp8E5M2 ||
         format == FactorFormat::MxFp8E4M3 || format == FactorFormat::MxFp4E2M1;
}

LowbitDataType low_datatype(FactorFormat format) {
  switch (format) {
  case FactorFormat::Fp8E4M3:
    return LowbitDataType::Float8E4M3;
  case FactorFormat::Fp8E5M2:
    return LowbitDataType::Float8E5M2;
  case FactorFormat::MxFp8E4M3:
    return LowbitDataType::MxFloat8E4M3;
  case FactorFormat::MxFp4E2M1:
    return LowbitDataType::MxFloat4E2M1;
  default:
    throw std::runtime_error("No homogeneous MPP type for this basis.");
  }
}

const char *low_backend_name(FactorFormat format) {
  switch (format) {
  case FactorFormat::Fp8E4M3:
    return "MPP-e4m3-32x32";
  case FactorFormat::Fp8E5M2:
    return "MPP-e5m2-32x32";
  case FactorFormat::MxFp8E4M3:
    return "MPP-mxfp8-32x32";
  case FactorFormat::MxFp4E2M1:
    return "MPP-mxfp4-32x32";
  default:
    return "MPP-low-32x32";
  }
}

void encode_matrix_codec(Metal &metal, MTL4::ComputeCommandEncoder *encoder,
                         MTL::ResidencySet *residency,
                         MTL4EncoderResources &resources, FactorFormat format,
                         MTL::Buffer *source, MTL::Buffer *encoded,
                         MTL::Buffer *scales, MTL::Buffer *constants) {
  const char *name = nullptr;
  switch (format) {
  case FactorFormat::Fp8E4M3:
    name = "encode_matrix_fp8_e4m3";
    break;
  case FactorFormat::Fp8E5M2:
    name = "encode_matrix_fp8_e5m2";
    break;
  case FactorFormat::MxFp8E4M3:
    name = "encode_matrix_mxfp8_e4m3";
    break;
  case FactorFormat::MxFp4E2M1:
    name = "encode_matrix_mxfp4_e2m1";
    break;
  default:
    throw std::runtime_error("No native matrix codec for this basis.");
  }
  const auto &params =
      *static_cast<const MatrixCodecParams *>(constants->contents());
  const bool mx =
      format == FactorFormat::MxFp8E4M3 || format == FactorFormat::MxFp4E2M1;
  MTL::ComputePipelineState *pipeline = metal.pipeline(name);
  if (mx) {
    const std::size_t groups = std::size_t{params.order} * (params.order / 32);
    Metal::dispatch1d(encoder, pipeline, residency,
                      {source, encoded, scales, constants}, resources,
                      groups * 32, 32);
  } else {
    const std::size_t physical_elements =
        std::size_t{params.order} * params.leading_dimension;
    Metal::dispatch1d(encoder, pipeline, residency,
                      {source, encoded, constants}, resources,
                      physical_elements / 4);
  }
}

Quantized quantize(Metal &metal, FactorFormat format,
                   const std::vector<float> &source, bool keep_encoded = false,
                   bool decode_values = true) {
  Quantized result;
  if (decode_values || format == FactorFormat::Fp32) {
    result.values = source;
  }
  if (format == FactorFormat::Fp32) {
    result.bits_per_value = 32.0;
    return result;
  }
  const std::size_t count = source.size();
  const std::size_t blocks = (count + 31) / 32;
  const CodecParams params{metal_count(count), metal_count(blocks)};
  const std::size_t encoded_bytes = value_bytes(format, count);
  const std::size_t metadata_bytes = scale_bytes(format, blocks);
  common::BufferGuard<Metal> owned(metal);
  const auto make = [&](std::size_t bytes) {
    return owned.make(std::max<std::size_t>(bytes, 1));
  };
  MTL::Buffer *input = make(count * sizeof(float));
  result.encoded.reset(metal.buffer(encoded_alloc_bytes(format, count)));
  MTL::Buffer *scales = make(metadata_bytes);
  MTL::Buffer *decoded = decode_values ? make(count * sizeof(float)) : nullptr;
  std::memcpy(input->contents(), source.data(), count * sizeof(float));
  const auto start = Clock::now();
  metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
    encode_codec(metal, encoder, format, input, result.encoded.get(), scales,
                 params);
    if (decode_values) {
      decode_codec(metal, encoder, format, result.encoded.get(), scales,
                   decoded, params);
    }
  });
  result.seconds = elapsed(start);
  if (decode_values) {
    std::memcpy(result.values.data(), decoded->contents(),
                count * sizeof(float));
  }
  result.bits_per_value = 8.0 *
                          static_cast<double>(encoded_bytes + metadata_bytes) /
                          static_cast<double>(count);
  if (!keep_encoded) {
    result.encoded.reset();
  }
  return result;
}

bool invert_basis(const std::vector<float> &basis, std::size_t order,
                  std::vector<float> &inverse, double &condition) {
  __LAPACK_int n = lapack_order(order);
  const char norm_type = '1';
  std::vector<float> norm_work(order);
  const float norm =
      slange_(&norm_type, &n, &n, basis.data(), &n, norm_work.data());
  inverse = basis;
  std::vector<__LAPACK_int> pivots(order);
  __LAPACK_int info = 0;
  sgetrf_(&n, &n, inverse.data(), &n, pivots.data(), &info);
  if (info != 0) {
    condition = std::numeric_limits<double>::infinity();
    return false;
  }
  float reciprocal_condition = 0.0f;
  std::vector<float> condition_work(4 * order);
  std::vector<__LAPACK_int> integer_work(order);
  sgecon_(&norm_type, &n, inverse.data(), &n, &norm, &reciprocal_condition,
          condition_work.data(), integer_work.data(), &info);
  if (info != 0 || !(reciprocal_condition > 0.0f)) {
    condition = std::numeric_limits<double>::infinity();
    return false;
  }
  condition = 1.0 / static_cast<double>(reciprocal_condition);
  __LAPACK_int work_size = -1;
  float work_query = 0.0f;
  sgetri_(&n, inverse.data(), &n, pivots.data(), &work_query, &work_size,
          &info);
  if (info != 0) {
    return false;
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  std::vector<float> work(static_cast<std::size_t>(work_size));
  sgetri_(&n, inverse.data(), &n, pivots.data(), work.data(), &work_size,
          &info);
  return info == 0;
}

Matrix read_working(MTL::Buffer *buffer, std::size_t count) {
  const WorkingHost *source =
      static_cast<const WorkingHost *>(buffer->contents());
  Matrix result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = double{source[index].high} + double{source[index].low};
  }
  return result;
}

void residual_and_norm(Metal &metal, MTL::ComputeCommandEncoder *encoder,
                       MTL::Buffer *matrix, MTL::Buffer *root,
                       MTL::Buffer *residual, MTL::Buffer *residual_f32,
                       MTL::Buffer *norm, const SquareParams &square,
                       std::uint32_t count) {
  MTL::ComputePipelineState *residual_pipeline =
      metal.pipeline("working_residual");
  encoder->setBuffer(matrix, 0, 0);
  encoder->setBuffer(root, 0, 1);
  encoder->setBuffer(residual, 0, 2);
  encoder->setBuffer(residual_f32, 0, 3);
  encoder->setBytes(&square, sizeof(square), 4);
  Metal::dispatch2d(encoder, residual_pipeline, square.order, square.order);
  MTL::ComputePipelineState *norm_pipeline = metal.pipeline("working_norm");
  encoder->setBuffer(matrix, 0, 0);
  encoder->setBuffer(residual, 0, 1);
  encoder->setBuffer(norm, 0, 2);
  encoder->setBytes(&count, sizeof(count), 3);
  Metal::dispatch1d(encoder, norm_pipeline, 256, 256);
}

void residual_and_norm(Metal &metal, MTL4::ComputeCommandEncoder *encoder,
                       MTL::ResidencySet *residency,
                       MTL4EncoderResources &dispatch_resources,
                       MTL::Buffer *matrix, MTL::Buffer *root,
                       MTL::Buffer *residual, MTL::Buffer *residual_f32,
                       MTL::Buffer *norm, MTL::Buffer *square,
                       MTL::Buffer *count) {
  Metal::dispatch2d(
      encoder, metal.pipeline("working_residual"), residency,
      {matrix, root, residual, residual_f32, square}, dispatch_resources,
      static_cast<const SquareParams *>(square->contents())->order,
      static_cast<const SquareParams *>(square->contents())->order);
  encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
  Metal::dispatch1d(encoder, metal.pipeline("working_norm"), residency,
                    {matrix, residual, norm, count}, dispatch_resources, 256,
                    256);
}

double measured_residual(MTL::Buffer *norm) {
  const NormPair measured = *static_cast<const NormPair *>(norm->contents());
  if (!(measured.matrix_squared > 0.0f) ||
      !std::isfinite(measured.residual_squared)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::sqrt(double{measured.residual_squared} /
                   double{measured.matrix_squared});
}

} // namespace

Problem controlled_problem(std::size_t order, double condition) {
  if (!(condition >= 1.0) || !std::isfinite(condition)) {
    throw std::runtime_error("The condition number is invalid.");
  }
  const Matrix basis = deterministic_basis(order);
  Matrix weighted = basis;
  for (std::size_t column = 0; column < order; ++column) {
    const double fraction = order == 1 ? 0.0
                                       : static_cast<double>(column) /
                                             static_cast<double>(order - 1);
    const double eigenvalue = std::exp(fraction * std::log(condition));
    for (std::size_t row = 0; row < order; ++row) {
      weighted[row + column * order] *= eigenvalue;
    }
  }
  Matrix matrix(elements(order));
  dgemm('N', 'T', order, weighted.data(), basis.data(), matrix.data());
  Problem problem{"controlled", order, condition, std::move(matrix), {}};
  problem.reference_root = reference_root(problem.matrix, order);
  return problem;
}

Problem paper_problem(char set, std::size_t order) {
  if (set != 'V' && set != 'W' && set != 'H') {
    throw std::runtime_error("The paper set must be V, W, or H.");
  }
  Matrix matrix(elements(order), 0.0);
  std::string name;
  if (set == 'H') {
    name = "paper_VII";
    for (std::size_t column = 0; column < order; ++column) {
      for (std::size_t row = 0; row < order; ++row) {
        matrix[row + column * order] =
            1.0 / static_cast<double>(row + column + 1);
      }
      matrix[column + column * order] += 1.0e-6;
    }
  } else {
    name = set == 'V' ? "paper_V" : "paper_VI";
    Generator generator(UINT64_C(0x243f6a8885a308d3) ^
                        (set == 'V' ? UINT64_C(5) : UINT64_C(6)));
    for (std::size_t column = 0; column < order; ++column) {
      for (std::size_t row = column; row < order; ++row) {
        const double value =
            set == 'V' ? generator.uniform() : generator.normal();
        matrix[row + column * order] = value;
        matrix[column + row * order] = value;
      }
    }
    Matrix discovery = matrix;
    std::vector<double> values(order);
    eigen_double(discovery, values);
    const double separation =
        std::max(1.0, 0.01 * infinity_norm(matrix, order));
    const double shift = std::max(0.0, separation - values.front());
    for (std::size_t index = 0; index < order; ++index) {
      matrix[index + index * order] += shift;
    }
  }
  Matrix discovery = matrix;
  std::vector<double> values(order);
  eigen_double(discovery, values);
  Problem problem{
      name, order, values.back() / values.front(), std::move(matrix), {}};
  problem.reference_root = reference_root(problem.matrix, order);
  return problem;
}

std::vector<PrecisionPlan> precision_plans() {
  return {{"fp32_baseline", FactorFormat::Fp32, FactorFormat::Fp32},
          {"spectrum_fp16", FactorFormat::Fp32, FactorFormat::Fp16},
          {"spectrum_bf16", FactorFormat::Fp32, FactorFormat::Bf16},
          {"spectrum_e4m3", FactorFormat::Fp32, FactorFormat::Fp8E4M3},
          {"spectrum_e5m2", FactorFormat::Fp32, FactorFormat::Fp8E5M2},
          {"spectrum_mxfp8", FactorFormat::Fp32, FactorFormat::MxFp8E4M3},
          {"spectrum_mxfp4", FactorFormat::Fp32, FactorFormat::MxFp4E2M1},
          {"basis_fp16", FactorFormat::Fp16, FactorFormat::Fp32},
          {"basis_bf16", FactorFormat::Bf16, FactorFormat::Fp32},
          {"basis_e4m3", FactorFormat::Fp8E4M3, FactorFormat::Fp32},
          {"basis_e5m2", FactorFormat::Fp8E5M2, FactorFormat::Fp32},
          {"basis_mxfp8", FactorFormat::MxFp8E4M3, FactorFormat::Fp32},
          {"basis_mxfp4", FactorFormat::MxFp4E2M1, FactorFormat::Fp32},
          {"fp16_mxfp8", FactorFormat::Fp16, FactorFormat::MxFp8E4M3},
          {"bf16_mxfp8", FactorFormat::Bf16, FactorFormat::MxFp8E4M3},
          {"e4m3_mxfp4", FactorFormat::Fp8E4M3, FactorFormat::MxFp4E2M1}};
}
SpectralResult run_spectral(Metal &metal, const Problem &problem,
                            const PrecisionPlan &plan,
                            std::size_t max_corrections, double tolerance) {
  SpectralResult result;
  result.plan = plan.name;
  result.basis_format = format_name(plan.basis);
  result.spectrum_format = format_name(plan.spectrum);
  result.case_name = problem.name;
  result.order = problem.order;
  result.condition = problem.condition;

  const double not_a_number = std::numeric_limits<double>::quiet_NaN();
  result.basis_relative_error = not_a_number;
  result.spectrum_relative_error = not_a_number;
  result.basis_condition = not_a_number;
  result.initial_error = not_a_number;
  result.final_error = not_a_number;
  result.final_residual = not_a_number;
  result.factor_seconds = not_a_number;
  result.codec_seconds = not_a_number;
  result.initial_seconds = not_a_number;
  result.correction_seconds = not_a_number;
  result.total_seconds = not_a_number;
  result.product_kernel_seconds = not_a_number;
  result.conversion_seconds = not_a_number;
  result.complete_solve_seconds = not_a_number;
  result.forward_error = not_a_number;
  result.residual = not_a_number;

  const std::size_t count = elements(problem.order);
  const auto storage_bits = [](FactorFormat format, std::size_t values) {
    const std::size_t blocks = (values + 31) / 32;
    return 8.0 *
           static_cast<double>(value_bytes(format, values) +
                               scale_bytes(format, blocks)) /
           static_cast<double>(values);
  };
  result.basis_bits_per_value = storage_bits(plan.basis, count);
  result.spectrum_bits_per_value = storage_bits(plan.spectrum, problem.order);
  result.weighted_factor_storage_bits =
      (result.basis_bits_per_value * static_cast<double>(count) +
       result.spectrum_bits_per_value * static_cast<double>(problem.order)) /
      static_cast<double>(count + problem.order);

  const bool fp16_basis = plan.basis == FactorFormat::Fp16;
  const auto total_start = Clock::now();
  constexpr LowbitMatmulTile tile = LowbitMatmulTile::Rows32Columns32;
  std::unique_ptr<LowbitMatmul> fp16_fp16;
  std::unique_ptr<LowbitMatmul> fp16_fp32;
  std::unique_ptr<LowbitMatmul> fp32_fp16;
  const bool low_mpp =
      is_low_mpp_basis(plan.basis) && problem.condition < 1000.0;
  std::unique_ptr<LowbitMatmul> low_product;
  if (fp16_basis) {
    result.product_backend = "MPP-32x32";
    result.arithmetic_format = "fp16_fp32_mixed_to_fp32";
    fp16_fp16 = std::make_unique<LowbitMatmul>(
        metal.device(), metal.library(), problem.order,
        LowbitMatmulSignature::Float16Float16ToFloat32, tile);
    fp16_fp32 = std::make_unique<LowbitMatmul>(
        metal.device(), metal.library(), problem.order,
        LowbitMatmulSignature::Float16Float32ToFloat32, tile);
    fp32_fp16 = std::make_unique<LowbitMatmul>(
        metal.device(), metal.library(), problem.order,
        LowbitMatmulSignature::Float32Float16ToFloat32, tile);
    for (const LowbitMatmul *operation :
         {fp16_fp16.get(), fp16_fp32.get(), fp32_fp16.get()}) {
      if (operation->availability().status != LowbitMatmulStatus::Available) {
        result.reason = "native_product_unavailable";
        return result;
      }
    }
  } else if (low_mpp) {
    low_product = std::make_unique<LowbitMatmul>(
        metal.device(), metal.library(), problem.order,
        low_datatype(plan.basis), tile);
    if (low_product->availability().status != LowbitMatmulStatus::Available) {
      result.reason = "native_product_unavailable";
      return result;
    }
    result.product_backend = low_backend_name(plan.basis);
    result.arithmetic_format = "adaptive_low_mpp_then_mps";
  } else {
    result.product_backend = "MPS";
    result.arithmetic_format = "fp32";
  }

  Factor factor = factor_float(problem);
  result.factor_seconds = factor.seconds;
  std::vector<float> basis_inverse(factor.basis.size());
  if (!fp16_basis) {
    for (std::size_t column = 0; column < problem.order; ++column) {
      for (std::size_t row = 0; row < problem.order; ++row) {
        basis_inverse[row + column * problem.order] =
            factor.basis[column + row * problem.order];
      }
    }
    result.basis_condition = 1.0;
  } else if (!invert_basis(factor.basis, problem.order, basis_inverse,
                           result.basis_condition) ||
             !(result.basis_condition < 1.0e4)) {
    result.reason = "ill_conditioned_basis";
    result.total_seconds = elapsed(total_start);
    result.complete_solve_seconds = result.total_seconds;
    return result;
  }

  Quantized basis =
      quantize(metal, plan.basis, factor.basis, fp16_basis, !fp16_basis);
  const Quantized spectrum =
      quantize(metal, plan.spectrum, factor.root_spectrum);
  result.codec_seconds = basis.seconds + spectrum.seconds;
  result.conversion_seconds = result.codec_seconds;
  result.basis_bits_per_value = basis.bits_per_value;
  result.spectrum_bits_per_value = spectrum.bits_per_value;
  result.weighted_factor_storage_bits =
      (result.basis_bits_per_value * static_cast<double>(count) +
       result.spectrum_bits_per_value * static_cast<double>(problem.order)) /
      static_cast<double>(count + problem.order);
  result.basis_relative_error =
      fp16_basis ? fp16_relative_error(basis.encoded.get(), factor.basis)
                 : relative_error(basis.values, factor.basis);
  result.spectrum_relative_error =
      relative_error(spectrum.values, factor.root_spectrum);
  for (float value : spectrum.values) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
      result.reason = "invalid_spectrum";
      result.total_seconds = elapsed(total_start);
      result.complete_solve_seconds = result.total_seconds;
      return result;
    }
  }
  // Use the quantized basis transpose for both native and decoded products.
  if (!fp16_basis && plan.basis != FactorFormat::Fp32) {
    for (std::size_t column = 0; column < problem.order; ++column) {
      for (std::size_t row = 0; row < problem.order; ++row) {
        basis_inverse[row + column * problem.order] =
            basis.values[column + row * problem.order];
      }
    }
  }

  Quantized fp16_inverse;
  if (fp16_basis) {
    fp16_inverse =
        quantize(metal, FactorFormat::Fp16, basis_inverse, true, false);
    result.codec_seconds += fp16_inverse.seconds;
    result.conversion_seconds = result.codec_seconds;
    if (!fp16_all_finite(fp16_inverse.encoded.get(), count)) {
      result.reason = "ill_conditioned_basis";
      result.total_seconds = elapsed(total_start);
      result.complete_solve_seconds = result.total_seconds;
      return result;
    }
  }
  result.accepted = true;
  result.product_kernel_seconds = 0.0;
  result.correction_seconds = 0.0;

  const std::uint32_t count32 = metal_count(count);
  const std::uint32_t order = metal_count(problem.order);
  const std::size_t bytes = count * sizeof(float);
  const std::size_t factor_bytes =
      fp16_basis ? count * sizeof(std::uint16_t) : bytes;
  const std::size_t working_bytes = count * sizeof(WorkingHost);

  std::vector<float> sylvester_inverse(count);
  for (std::size_t column = 0; column < problem.order; ++column) {
    for (std::size_t row = 0; row < problem.order; ++row) {
      sylvester_inverse[row + column * problem.order] =
          1.0f / (spectrum.values[row] + spectrum.values[column]);
    }
  }

  const SquareParams square{order};

  common::BufferGuard<Metal> owned(metal);
  const auto make = [&](std::size_t bytes_to_allocate) {
    return owned.make(bytes_to_allocate);
  };
  MTL::Buffer *basis_buffer =
      fp16_basis ? basis.encoded.get() : make(factor_bytes);
  MTL::Buffer *basis_inverse_buffer =
      fp16_basis ? fp16_inverse.encoded.get() : make(factor_bytes);
  MTL::Buffer *spectrum_buffer = make(problem.order * sizeof(float));
  MTL::Buffer *sylvester_buffer = make(bytes);
  MTL::Buffer *scaled_basis = make(factor_bytes);
  MTL::Buffer *root_product = make(bytes);
  MTL::Buffer *residual_f32 = make(bytes);
  MTL::Buffer *workspace = make(bytes);
  MTL::Buffer *coordinates = make(bytes);
  MTL::Buffer *correction = make(bytes);
  MTL::Buffer *matrix_work = make(working_bytes);
  MTL::Buffer *root_work = make(working_bytes);
  MTL::Buffer *residual_work = make(working_bytes);
  MTL::Buffer *norm = make(sizeof(NormPair));
  MTL::Buffer *square_constants =
      fp16_basis || low_mpp ? make(sizeof(square)) : nullptr;
  MTL::Buffer *count_constants =
      fp16_basis || low_mpp ? make(sizeof(count32)) : nullptr;

  if (!fp16_basis) {
    std::memcpy(basis_buffer->contents(), basis.values.data(), bytes);
    std::memcpy(basis_inverse_buffer->contents(), basis_inverse.data(), bytes);
  }
  std::memcpy(spectrum_buffer->contents(), spectrum.values.data(),
              problem.order * sizeof(float));
  std::memcpy(sylvester_buffer->contents(), sylvester_inverse.data(), bytes);
  if (fp16_basis || low_mpp) {
    std::memcpy(square_constants->contents(), &square, sizeof(square));
    std::memcpy(count_constants->contents(), &count32, sizeof(count32));
  }
  WorkingHost *matrix_values =
      static_cast<WorkingHost *>(matrix_work->contents());
  for (std::size_t index = 0; index < count; ++index) {
    const float high = static_cast<float>(problem.matrix[index]);
    matrix_values[index] =
        WorkingHost{high, static_cast<float>(problem.matrix[index] - high)};
  }

  // Encode static low-bit factors once; correction operands stay on the GPU.
  MTL::Buffer *low_basis_L = nullptr, *low_basis_R = nullptr;
  MTL::Buffer *low_basis_Ls = nullptr, *low_basis_Rs = nullptr;
  MTL::Buffer *low_inverse_L = nullptr, *low_inverse_R = nullptr;
  MTL::Buffer *low_inverse_Ls = nullptr, *low_inverse_Rs = nullptr;
  MTL::Buffer *low_scaled_L = nullptr, *low_scaled_Ls = nullptr;
  MTL::Buffer *low_left = nullptr, *low_right = nullptr;
  MTL::Buffer *low_left_s = nullptr, *low_right_s = nullptr;
  MTL::Buffer *low_left_constants = nullptr, *low_right_constants = nullptr;
  if (low_mpp) {
    const auto &left_layout = low_product->left_layout();
    const auto &right_layout = low_product->right_layout();
    const bool mx = plan.basis == FactorFormat::MxFp8E4M3 ||
                    plan.basis == FactorFormat::MxFp4E2M1;
    low_basis_L = make(left_layout.data_bytes);
    low_basis_R = make(right_layout.data_bytes);
    low_inverse_L = make(left_layout.data_bytes);
    low_inverse_R = make(right_layout.data_bytes);
    low_scaled_L = make(left_layout.data_bytes);
    low_left = make(left_layout.data_bytes);
    low_right = make(right_layout.data_bytes);
    if (mx) {
      low_basis_Ls = make(left_layout.scale_bytes);
      low_basis_Rs = make(right_layout.scale_bytes);
      low_inverse_Ls = make(left_layout.scale_bytes);
      low_inverse_Rs = make(right_layout.scale_bytes);
      low_scaled_Ls = make(left_layout.scale_bytes);
      low_left_s = make(left_layout.scale_bytes);
      low_right_s = make(right_layout.scale_bytes);
    }
    const auto make_codec_constants =
        [&](const LowbitMatmulOperandLayout &layout) {
          const MatrixCodecParams params{
              order, metal_count(layout.data_leading_dimension_elements),
              metal_count(layout.scale_leading_dimension_bytes),
              layout.storage == LowbitMatrixStorage::TransposedColumnMajor
                  ? 1u
                  : 0u};
          MTL::Buffer *buffer = make(sizeof(params));
          std::memcpy(buffer->contents(), &params, sizeof(params));
          return buffer;
        };
    low_left_constants = make_codec_constants(left_layout);
    low_right_constants = make_codec_constants(right_layout);
  }

  std::vector<float>().swap(factor.basis);
  std::vector<float>().swap(basis.values);
  std::vector<float>().swap(basis_inverse);
  std::vector<float>().swap(fp16_inverse.values);

  ResidencyPtr residency;
  if (fp16_basis || low_mpp) {
    std::vector<MTL::Buffer *> resident_buffers = owned.buffers();
    resident_buffers.push_back(basis_buffer);
    resident_buffers.push_back(basis_inverse_buffer);
    residency.reset(metal.residency_set(resident_buffers));
  }
  std::unique_ptr<MpsMatmul> mps;
  if (!fp16_basis) {
    mps = std::make_unique<MpsMatmul>(root_product, problem.order);
  }

  const auto encode_mpp_product = [&](MTL4::ComputeCommandEncoder *encoder,
                                      LowbitMatmulEncoderResources &resources,
                                      LowbitMatmul &operation,
                                      MTL::Buffer *left, MTL::Buffer *right,
                                      MTL::Buffer *output) {
    const LowbitMatmulBuffers product{left,    right,  nullptr,
                                      nullptr, output, residency.get()};
    const LowbitMatmulResult encoded =
        operation.encode(encoder, product, resources);
    if (!encoded.encoded()) {
      throw std::runtime_error("Native MPP product failed: " + encoded.message);
    }
  };
  const auto encode_low_product = [&](MTL4::ComputeCommandEncoder *encoder,
                                      LowbitMatmulEncoderResources &resources,
                                      MTL::Buffer *left, MTL::Buffer *left_sc,
                                      MTL::Buffer *right, MTL::Buffer *right_sc,
                                      MTL::Buffer *output) {
    const LowbitMatmulBuffers product{left,     right,  left_sc,
                                      right_sc, output, residency.get()};
    const LowbitMatmulResult encoded =
        low_product->encode(encoder, product, resources);
    if (!encoded.encoded()) {
      throw std::runtime_error("Native MPP product failed: " + encoded.message);
    }
  };
  const auto encode_legacy = [](MTL::CommandBuffer *command,
                                const auto &encode) {
    MTL::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
    if (encoder == nullptr) {
      throw std::runtime_error("Cannot create a Metal compute encoder.");
    }
    encode(encoder);
    encoder->endEncoding();
  };
  const auto barrier = [](MTL4::ComputeCommandEncoder *encoder) {
    encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
  };

  if (low_mpp) {
    const auto encode_start = Clock::now();
    metal.submit_mtl4([&](MTL4::CommandBuffer *command) {
      command->useResidencySet(residency.get());
      MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
      if (encoder == nullptr) {
        throw std::runtime_error("Cannot create an MTL4 compute encoder.");
      }
      MTL4EncoderResources dispatch_resources;
      encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                          plan.basis, basis_buffer, low_basis_L, low_basis_Ls,
                          low_left_constants);
      encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                          plan.basis, basis_buffer, low_basis_R, low_basis_Rs,
                          low_right_constants);
      encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                          plan.basis, basis_inverse_buffer, low_inverse_L,
                          low_inverse_Ls, low_left_constants);
      encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                          plan.basis, basis_inverse_buffer, low_inverse_R,
                          low_inverse_Rs, low_right_constants);
      Metal::dispatch2d(
          encoder, metal.pipeline("scale_columns_f32"), residency.get(),
          {basis_buffer, spectrum_buffer, scaled_basis, square_constants},
          dispatch_resources, order, order);
      barrier(encoder);
      encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                          plan.basis, scaled_basis, low_scaled_L, low_scaled_Ls,
                          low_left_constants);
      encoder->endEncoding();
    });
    result.codec_seconds += elapsed(encode_start);
    result.conversion_seconds = result.codec_seconds;
  }

  auto start = Clock::now();
  double command_seconds = 0.0;
  if (fp16_basis) {
    command_seconds = metal.submit_mtl4([&](MTL4::CommandBuffer *command) {
      command->useResidencySet(residency.get());
      MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
      if (encoder == nullptr) {
        throw std::runtime_error("Cannot create an MTL4 compute encoder.");
      }
      LowbitMatmulEncoderResources resources;
      MTL4EncoderResources dispatch_resources;
      Metal::dispatch2d(
          encoder, metal.pipeline("scale_columns_f16"), residency.get(),
          {basis_buffer, spectrum_buffer, scaled_basis, square_constants},
          dispatch_resources, order, order);
      barrier(encoder);
      encode_mpp_product(encoder, resources, *fp16_fp16, scaled_basis,
                         basis_inverse_buffer, root_product);
      barrier(encoder);
      Metal::dispatch2d(encoder, metal.pipeline("working_from_f32"),
                        residency.get(),
                        {root_product, root_work, square_constants},
                        dispatch_resources, order, order);
      barrier(encoder);
      residual_and_norm(metal, encoder, residency.get(), dispatch_resources,
                        matrix_work, root_work, residual_work, residual_f32,
                        norm, square_constants, count_constants);
      encoder->endEncoding();
    });
  } else if (low_mpp) {
    command_seconds = metal.submit_mtl4([&](MTL4::CommandBuffer *command) {
      command->useResidencySet(residency.get());
      MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
      if (encoder == nullptr) {
        throw std::runtime_error("Cannot create an MTL4 compute encoder.");
      }
      LowbitMatmulEncoderResources resources;
      MTL4EncoderResources dispatch_resources;
      encode_low_product(encoder, resources, low_scaled_L, low_scaled_Ls,
                         low_inverse_R, low_inverse_Rs, root_product);
      barrier(encoder);
      Metal::dispatch2d(encoder, metal.pipeline("working_from_f32"),
                        residency.get(),
                        {root_product, root_work, square_constants},
                        dispatch_resources, order, order);
      barrier(encoder);
      residual_and_norm(metal, encoder, residency.get(), dispatch_resources,
                        matrix_work, root_work, residual_work, residual_f32,
                        norm, square_constants, count_constants);
      encoder->endEncoding();
    });
  } else {
    command_seconds = metal.submit_commands([&](MTL::CommandBuffer *command) {
      encode_legacy(command, [&](MTL::ComputeCommandEncoder *encoder) {
        MTL::ComputePipelineState *scale = metal.pipeline("scale_columns_f32");
        encoder->setBuffer(basis_buffer, 0, 0);
        encoder->setBuffer(spectrum_buffer, 0, 1);
        encoder->setBuffer(scaled_basis, 0, 2);
        encoder->setBytes(&square, sizeof(square), 3);
        Metal::dispatch2d(encoder, scale, order, order);
      });
      mps->encode(command, scaled_basis, basis_inverse_buffer, root_product);
      encode_legacy(command, [&](MTL::ComputeCommandEncoder *encoder) {
        MTL::ComputePipelineState *convert = metal.pipeline("working_from_f32");
        encoder->setBuffer(root_product, 0, 0);
        encoder->setBuffer(root_work, 0, 1);
        encoder->setBytes(&square, sizeof(square), 2);
        Metal::dispatch2d(encoder, convert, order, order);
        residual_and_norm(metal, encoder, matrix_work, root_work, residual_work,
                          residual_f32, norm, square, count32);
      });
    });
  }
  result.product_kernel_seconds += command_seconds;
  result.initial_seconds = elapsed(start);

  double measured = measured_residual(norm);
  Matrix root = read_working(root_work, count);
  result.history.push_back(IterationRecord{
      0, measured, relative_error(root, problem.reference_root)});
  result.initial_error = result.history.front().forward_error;
  const double divergence_limit = std::max(1.0, 1.0e4 * measured);
  // Stall fallback: leave low products after two iterations without 2x
  // progress, even above the residual threshold.
  bool low_on = true;
  double low_prev = measured;
  int low_stalls = 0;
  for (std::size_t iteration = 0;
       measured > tolerance && iteration < max_corrections &&
       measured < divergence_limit;
       ++iteration) {
    start = Clock::now();
    if (fp16_basis) {
      command_seconds = metal.submit_mtl4([&](MTL4::CommandBuffer *command) {
        command->useResidencySet(residency.get());
        MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
        if (encoder == nullptr) {
          throw std::runtime_error("Cannot create an MTL4 compute encoder.");
        }
        LowbitMatmulEncoderResources resources;
        MTL4EncoderResources dispatch_resources;
        encode_mpp_product(encoder, resources, *fp16_fp32, basis_inverse_buffer,
                           residual_f32, workspace);
        barrier(encoder);
        encode_mpp_product(encoder, resources, *fp32_fp16, workspace,
                           basis_buffer, coordinates);
        barrier(encoder);
        Metal::dispatch1d(
            encoder, metal.pipeline("apply_scale_f32"), residency.get(),
            {coordinates, sylvester_buffer, coordinates, count_constants},
            dispatch_resources, count32);
        barrier(encoder);
        encode_mpp_product(encoder, resources, *fp16_fp32, basis_buffer,
                           coordinates, workspace);
        barrier(encoder);
        encode_mpp_product(encoder, resources, *fp32_fp16, workspace,
                           basis_inverse_buffer, correction);
        barrier(encoder);
        Metal::dispatch2d(encoder, metal.pipeline("working_add"),
                          residency.get(),
                          {root_work, correction, square_constants},
                          dispatch_resources, order, order);
        barrier(encoder);
        residual_and_norm(metal, encoder, residency.get(), dispatch_resources,
                          matrix_work, root_work, residual_work, residual_f32,
                          norm, square_constants, count_constants);
        encoder->endEncoding();
      });
    } else if (low_mpp && low_on && measured > 1.0e-3) {
      command_seconds = metal.submit_mtl4([&](MTL4::CommandBuffer *command) {
        command->useResidencySet(residency.get());
        MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
        if (encoder == nullptr) {
          throw std::runtime_error("Cannot create an MTL4 compute encoder.");
        }
        LowbitMatmulEncoderResources resources;
        MTL4EncoderResources dispatch_resources;
        encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                            plan.basis, residual_f32, low_right, low_right_s,
                            low_right_constants);
        barrier(encoder);
        encode_low_product(encoder, resources, low_inverse_L, low_inverse_Ls,
                           low_right, low_right_s, workspace);
        barrier(encoder);
        encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                            plan.basis, workspace, low_left, low_left_s,
                            low_left_constants);
        barrier(encoder);
        encode_low_product(encoder, resources, low_left, low_left_s,
                           low_basis_R, low_basis_Rs, coordinates);
        barrier(encoder);
        Metal::dispatch1d(
            encoder, metal.pipeline("apply_scale_f32"), residency.get(),
            {coordinates, sylvester_buffer, coordinates, count_constants},
            dispatch_resources, count32);
        barrier(encoder);
        encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                            plan.basis, coordinates, low_right, low_right_s,
                            low_right_constants);
        barrier(encoder);
        encode_low_product(encoder, resources, low_basis_L, low_basis_Ls,
                           low_right, low_right_s, workspace);
        barrier(encoder);
        encode_matrix_codec(metal, encoder, residency.get(), dispatch_resources,
                            plan.basis, workspace, low_left, low_left_s,
                            low_left_constants);
        barrier(encoder);
        encode_low_product(encoder, resources, low_left, low_left_s,
                           low_inverse_R, low_inverse_Rs, correction);
        barrier(encoder);
        Metal::dispatch2d(encoder, metal.pipeline("working_add"),
                          residency.get(),
                          {root_work, correction, square_constants},
                          dispatch_resources, order, order);
        barrier(encoder);
        residual_and_norm(metal, encoder, residency.get(), dispatch_resources,
                          matrix_work, root_work, residual_work, residual_f32,
                          norm, square_constants, count_constants);
        encoder->endEncoding();
      });
    } else {
      command_seconds = metal.submit_commands([&](MTL::CommandBuffer *command) {
        mps->encode(command, basis_inverse_buffer, residual_f32, workspace);
        mps->encode(command, workspace, basis_buffer, coordinates);
        encode_legacy(command, [&](MTL::ComputeCommandEncoder *encoder) {
          MTL::ComputePipelineState *scale = metal.pipeline("apply_scale_f32");
          encoder->setBuffer(coordinates, 0, 0);
          encoder->setBuffer(sylvester_buffer, 0, 1);
          encoder->setBuffer(coordinates, 0, 2);
          encoder->setBytes(&count32, sizeof(count32), 3);
          Metal::dispatch1d(encoder, scale, count32);
        });
        mps->encode(command, basis_buffer, coordinates, workspace);
        mps->encode(command, workspace, basis_inverse_buffer, correction);
        encode_legacy(command, [&](MTL::ComputeCommandEncoder *encoder) {
          MTL::ComputePipelineState *add = metal.pipeline("working_add");
          encoder->setBuffer(root_work, 0, 0);
          encoder->setBuffer(correction, 0, 1);
          encoder->setBytes(&square, sizeof(square), 2);
          Metal::dispatch2d(encoder, add, order, order);
          residual_and_norm(metal, encoder, matrix_work, root_work,
                            residual_work, residual_f32, norm, square, count32);
        });
      });
    }
    result.product_kernel_seconds += command_seconds;
    result.correction_seconds += elapsed(start);
    measured = measured_residual(norm);
    root = read_working(root_work, count);
    result.history.push_back(IterationRecord{
        iteration + 1, measured, relative_error(root, problem.reference_root)});
    if (low_mpp && low_on && measured > tolerance) {
      if (measured > low_prev * 0.5) {
        ++low_stalls;
      } else {
        low_stalls = 0;
      }
      low_prev = measured;
      low_on = low_stalls < 2;
    }
  }

  result.iterations = result.history.size() - 1;
  result.correction_count = result.iterations;
  result.final_residual = result.history.back().residual;
  result.final_error = result.history.back().forward_error;
  result.residual = result.final_residual;
  result.forward_error = result.final_error;
  result.converged = result.final_residual <= tolerance;
  if (result.converged) {
    result.reason = "converged";
  } else if (!std::isfinite(result.final_residual)) {
    result.reason = "nonfinite_residual";
  } else if (result.final_residual >= divergence_limit) {
    result.reason = "divergent";
  } else {
    result.reason = "correction_limit";
  }
  result.total_seconds = elapsed(total_start);
  result.complete_solve_seconds = result.total_seconds;
  return result;
}

void write_spectral_results(const std::string &directory,
                            const std::vector<SpectralResult> &results) {
  std::filesystem::create_directories(directory);
  std::ofstream summary(directory + "/summary.csv");
  std::ofstream iterations(directory + "/iterations.csv");
  if (!summary || !iterations) {
    throw std::runtime_error("Cannot create the spectral result files.");
  }
  summary
      << "plan,basis_format,spectrum_format,product_backend,"
         "arithmetic_format,case,order,evaluation,condition,accepted,"
         "converged,reason,iterations,basis_bits,spectrum_bits,"
         "weighted_factor_storage_bits,basis_error,spectrum_error,"
         "basis_condition,initial_error,final_error,final_residual,factor_ms,"
         "codec_ms,initial_ms,correction_ms,total_ms,product_kernel_ms,"
         "conversion_ms,correction_count,complete_solve_ms,forward_error,"
         "residual\n";
  iterations << "plan,case,order,evaluation,condition,iteration,residual,"
                "forward_error\n";
  summary << std::setprecision(17);
  iterations << std::setprecision(17);
  for (const SpectralResult &result : results) {
    const auto number = [&](double value) {
      if (std::isfinite(value))
        summary << value;
    };
    const bool unsupported =
        result.reason == "native_product_signature_unsupported" ||
        result.reason == "native_product_unavailable";
    summary << result.plan << ',' << result.basis_format << ','
            << result.spectrum_format << ',' << result.product_backend << ','
            << result.arithmetic_format << ',' << result.case_name << ','
            << result.order << ',' << result.evaluation << ','
            << result.condition << ',' << result.accepted << ','
            << result.converged << ',' << result.reason << ',';
    if (!unsupported)
      summary << result.iterations;
    summary << ',';
    number(result.basis_bits_per_value);
    summary << ',';
    number(result.spectrum_bits_per_value);
    summary << ',';
    number(result.weighted_factor_storage_bits);
    summary << ',';
    number(result.basis_relative_error);
    summary << ',';
    number(result.spectrum_relative_error);
    summary << ',';
    number(result.basis_condition);
    summary << ',';
    number(result.initial_error);
    summary << ',';
    number(result.final_error);
    summary << ',';
    number(result.final_residual);
    summary << ',';
    number(1.0e3 * result.factor_seconds);
    summary << ',';
    number(1.0e3 * result.codec_seconds);
    summary << ',';
    number(1.0e3 * result.initial_seconds);
    summary << ',';
    number(1.0e3 * result.correction_seconds);
    summary << ',';
    number(1.0e3 * result.total_seconds);
    summary << ',';
    number(1.0e3 * result.product_kernel_seconds);
    summary << ',';
    number(1.0e3 * result.conversion_seconds);
    summary << ',';
    if (!unsupported)
      summary << result.correction_count;
    summary << ',';
    number(1.0e3 * result.complete_solve_seconds);
    summary << ',';
    number(result.forward_error);
    summary << ',';
    number(result.residual);
    summary << '\n';
    for (const IterationRecord &step : result.history) {
      iterations << result.plan << ',' << result.case_name << ','
                 << result.order << ',' << result.evaluation << ','
                 << result.condition << ',' << step.iteration << ','
                 << step.residual << ',' << step.forward_error << '\n';
    }
  }
}

} // namespace msqrt
