#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal_runtime.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mixed_sqrt {

namespace {

std::string readTextFile(const char *path) {
  std::ifstream input(path);
  if (!input) {
    fail(std::string("Cannot read shader source: ") + path);
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string errorText(NS::Error *error) {
  return error == nullptr ? "Unknown Metal error."
                          : error->localizedDescription()->utf8String();
}

}  // namespace

class MetalRuntime::Impl {
 public:
  Impl(const char *shader_path, RoundingMode rounding_mode,
       uint32_t rounding_seed)
      : rounding_mode_(rounding_mode), rounding_seed_(rounding_seed) {
    device_ = MTL::CreateSystemDefaultDevice();
    if (device_ == nullptr) {
      fail("No Metal device is available.");
    }
    queue_ = device_->newCommandQueue();
    if (queue_ == nullptr) {
      fail("Cannot create a Metal command queue.");
    }

    const std::string source = readTextFile(shader_path);
    NS::Error *error = nullptr;
    NS::String *source_string =
        NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    MTL::CompileOptions *compile_options = MTL::CompileOptions::alloc()->init();
    compile_options->setFastMathEnabled(false);
    library_ = device_->newLibrary(source_string, compile_options, &error);
    compile_options->release();
    if (library_ == nullptr) {
      fail("Metal shader compilation failed: " + errorText(error));
    }

    for (const PrecisionSpec &precision : kPrecisionSpecs) {
      if (!precision.kernels_enabled) {
        continue;
      }
      KernelPipelines pipelines;
      pipelines.precision = precision.precision;
      pipelines.scale_columns = makePipeline(
          kernelName(precision, KernelOperation::ScaleColumns).c_str());
      pipelines.matmul =
          makePipeline(kernelName(precision, KernelOperation::Matmul).c_str());
      pipelines.diagonal_sylvester = makePipeline(
          kernelName(precision, KernelOperation::DiagonalSylvester).c_str());
      if (precision.precision == Precision::Fp32 ||
          precision.precision == Precision::Bf16) {
        pipelines.apply_sylvester_inverse = makePipeline(
            kernelName(precision, KernelOperation::ApplySylvesterInverse)
                .c_str());
        pipelines.matmul_tiled = makePipeline(
            kernelName(precision, KernelOperation::MatmulTiled).c_str());
        pipelines.matmul_tiled_scaled = makePipeline(
            kernelName(precision, KernelOperation::MatmulTiledScaled).c_str());
        pipelines.diagonal_sylvester_coalesced = makePipeline(
            kernelName(precision, KernelOperation::DiagonalSylvesterCoalesced)
                .c_str());
      }
      if (precision.precision != Precision::Fp32) {
        const std::string format_name = precision.usesLogSpectrum()
                                            ? precision.cli_name
                                            : precision.kernel_suffix;
        pipelines.decode_to_f32 =
            makePipeline(("decode_" + format_name + "_to_f32").c_str());
        pipelines.quantize = makePipeline(("quantize_" + format_name).c_str());
        if (rounding_mode_ == RoundingMode::Stochastic) {
          pipelines.quantize_stochastic =
              makePipeline(("quantize_" + format_name + "_stochastic").c_str());
        }
        const std::string basis_format = precision.kernel_suffix;
        pipelines.matmul_low_left_f32 =
            makePipeline(("matmul_low_left_f32_" + basis_format).c_str());
        pipelines.matmul_f32_low_right =
            makePipeline(("matmul_f32_low_right_" + basis_format).c_str());
        pipelines.scale_low_columns_f32 =
            makePipeline(("scale_low_columns_f32_" + basis_format).c_str());
      }
      pipelines_.push_back(pipelines);
    }
    resident_.jacobi_identity = makePipeline("jacobi_identity_f32");
    resident_.jacobi_rotations = makePipeline("jacobi_rotations_f32");
    resident_.jacobi_similarity = makePipeline("jacobi_similarity_vectors_f32");
    resident_.jacobi_root_eigenvalues =
        makePipeline("jacobi_root_eigenvalues_f32");
    resident_.transpose_square = makePipeline("transpose_square_f32");
    resident_.sylvester_inverse =
        makePipeline("sylvester_inverse_from_spectrum_f32");
    resident_.log_spectrum_metadata = makePipeline("log_spectrum_metadata");
    resident_.working_from_f32 = makePipeline("working_from_symmetric_f32");
    resident_.working_residual = makePipeline("working_residual");
    resident_.working_residual_to_f32 = makePipeline("working_residual_to_f32");
    resident_.working_add_symmetric_correction =
        makePipeline("working_add_symmetric_correction");
    resident_.working_norm = makePipeline("working_norm");
  }

  ~Impl() {
    for (const KernelPipelines &pipelines : pipelines_) {
      if (pipelines.decode_to_f32 != nullptr) {
        pipelines.decode_to_f32->release();
      }
      if (pipelines.scale_low_columns_f32 != nullptr) {
        pipelines.scale_low_columns_f32->release();
      }
      if (pipelines.matmul_f32_low_right != nullptr) {
        pipelines.matmul_f32_low_right->release();
      }
      if (pipelines.matmul_low_left_f32 != nullptr) {
        pipelines.matmul_low_left_f32->release();
      }
      if (pipelines.diagonal_sylvester_coalesced != nullptr) {
        pipelines.diagonal_sylvester_coalesced->release();
      }
      if (pipelines.matmul_tiled_scaled != nullptr) {
        pipelines.matmul_tiled_scaled->release();
      }
      if (pipelines.matmul_tiled != nullptr) {
        pipelines.matmul_tiled->release();
      }
      if (pipelines.apply_sylvester_inverse != nullptr) {
        pipelines.apply_sylvester_inverse->release();
      }
      if (pipelines.quantize != nullptr) {
        pipelines.quantize->release();
      }
      if (pipelines.quantize_stochastic != nullptr) {
        pipelines.quantize_stochastic->release();
      }
      if (pipelines.diagonal_sylvester != nullptr) {
        pipelines.diagonal_sylvester->release();
      }
      if (pipelines.matmul != nullptr) {
        pipelines.matmul->release();
      }
      if (pipelines.scale_columns != nullptr) {
        pipelines.scale_columns->release();
      }
    }
    resident_.release();
    if (library_ != nullptr)
      library_->release();
    if (queue_ != nullptr)
      queue_->release();
    if (device_ != nullptr)
      device_->release();
  }

  void requireEnabledKernels(const PrecisionSpec &precision) const {
    ensureKernelsEnabled(precision);
  }

  RoundingMode roundingMode() const { return rounding_mode_; }

  uint32_t roundingSeed() const { return rounding_seed_; }

  MTL::Buffer *makeSharedBuffer(size_t bytes) const {
    MTL::Buffer *buffer =
        device_->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (buffer == nullptr) {
      fail("Cannot allocate a shared Metal buffer.");
    }
    return buffer;
  }

  void probeF32Edges(MTL::Buffer *source, MTL::Buffer *output,
                     size_t elements) {
    MTL::ComputePipelineState *pipeline = makePipeline("probe_f32_edges");
    dispatch1d(pipeline, elements, [&](MTL::ComputeCommandEncoder *encoder) {
      encoder->setBuffer(source, 0, 0);
      encoder->setBuffer(output, 0, 1);
    });
    pipeline->release();
  }

  void scaleColumns(const PrecisionSpec &precision,
                    const StorageBuffers &eigenvectors,
                    const StorageBuffers &root_eigenvalues,
                    const StorageBuffers &output, MTL::Buffer *temporary_output,
                    uint32_t order) const {
    const size_t elements = checkedElementCount(order, order);
    validateRawOutput(precision, output, temporary_output, elements);
    const SquareParams params{order};
    MTL::Buffer *raw_output =
        precision.requiresQuantizer() ? temporary_output : output.values;
    dispatch2d(pipelineFor(precision, KernelOperation::ScaleColumns), order,
               order, [&](MTL::ComputeCommandEncoder *encoder) {
                 if (precision.requiresScalePlane()) {
                   encoder->setBuffer(eigenvectors.values, 0, 0);
                   encoder->setBuffer(eigenvectors.scales, 0, 1);
                   encoder->setBuffer(root_eigenvalues.values, 0, 2);
                   encoder->setBuffer(root_eigenvalues.scales, 0, 3);
                   encoder->setBuffer(raw_output, 0, 4);
                   encoder->setBytes(&params, sizeof(params), 5);
                 } else {
                   encoder->setBuffer(eigenvectors.values, 0, 0);
                   encoder->setBuffer(root_eigenvalues.values, 0, 1);
                   encoder->setBuffer(raw_output, 0, 2);
                   encoder->setBytes(&params, sizeof(params), 3);
                 }
               });
    quantizeOutput(precision, temporary_output, output, elements);
  }

  void matmul(const PrecisionSpec &precision, const StorageBuffers &left,
              const StorageBuffers &right, const StorageBuffers &output,
              MTL::Buffer *temporary_output, uint32_t rows, uint32_t columns,
              uint32_t inner) const {
    const size_t elements = checkedElementCount(rows, columns);
    validateRawOutput(precision, output, temporary_output, elements);
    const MatmulParams params{rows, columns, inner};
    MTL::Buffer *raw_output =
        precision.requiresQuantizer() ? temporary_output : output.values;
    dispatch2d(pipelineFor(precision, KernelOperation::Matmul), columns, rows,
               [&](MTL::ComputeCommandEncoder *encoder) {
                 if (precision.requiresScalePlane()) {
                   encoder->setBuffer(left.values, 0, 0);
                   encoder->setBuffer(left.scales, 0, 1);
                   encoder->setBuffer(right.values, 0, 2);
                   encoder->setBuffer(right.scales, 0, 3);
                   encoder->setBuffer(raw_output, 0, 4);
                   encoder->setBytes(&params, sizeof(params), 5);
                 } else {
                   encoder->setBuffer(left.values, 0, 0);
                   encoder->setBuffer(right.values, 0, 1);
                   encoder->setBuffer(raw_output, 0, 2);
                   encoder->setBytes(&params, sizeof(params), 3);
                 }
               });
    quantizeOutput(precision, temporary_output, output, elements);
  }

  void diagonalSylvester(const PrecisionSpec &precision,
                         const StorageBuffers &right_hand_side,
                         const StorageBuffers &root_eigenvalues,
                         const StorageBuffers &correction,
                         MTL::Buffer *temporary_output, uint32_t order) const {
    const size_t elements = checkedElementCount(order, order);
    validateRawOutput(precision, correction, temporary_output, elements);
    const SquareParams params{order};
    MTL::Buffer *raw_output =
        precision.requiresQuantizer() ? temporary_output : correction.values;
    dispatch2d(pipelineFor(precision, KernelOperation::DiagonalSylvester),
               order, order, [&](MTL::ComputeCommandEncoder *encoder) {
                 if (precision.requiresScalePlane()) {
                   encoder->setBuffer(right_hand_side.values, 0, 0);
                   encoder->setBuffer(right_hand_side.scales, 0, 1);
                   encoder->setBuffer(root_eigenvalues.values, 0, 2);
                   encoder->setBuffer(root_eigenvalues.scales, 0, 3);
                   encoder->setBuffer(raw_output, 0, 4);
                   encoder->setBytes(&params, sizeof(params), 5);
                 } else {
                   encoder->setBuffer(right_hand_side.values, 0, 0);
                   encoder->setBuffer(root_eigenvalues.values, 0, 1);
                   encoder->setBuffer(raw_output, 0, 2);
                   encoder->setBytes(&params, sizeof(params), 3);
                 }
               });
    quantizeOutput(precision, temporary_output, correction, elements);
  }

  void refineNativePerStage(
      const PrecisionSpec &precision, const StorageBuffers &eigenvectors,
      const StorageBuffers &eigenvectors_transposed,
      const StorageBuffers &root_eigenvalues, const StorageBuffers &residual,
      const StorageBuffers &workspace_one, const StorageBuffers &workspace_two,
      const StorageBuffers &correction_coordinates,
      const StorageBuffers &correction, MTL::Buffer *temporary_output,
      uint32_t order) const {
    matmul(precision, eigenvectors_transposed, residual, workspace_one,
           temporary_output, order, order, order);
    matmul(precision, workspace_one, eigenvectors, workspace_two,
           temporary_output, order, order, order);
    diagonalSylvester(precision, workspace_two, root_eigenvalues,
                      correction_coordinates, temporary_output, order);
    matmul(precision, eigenvectors, correction_coordinates, workspace_one,
           temporary_output, order, order, order);
    matmul(precision, workspace_one, eigenvectors_transposed, correction,
           temporary_output, order, order, order);
  }

  void refineNativeStaged(const PrecisionSpec &precision,
                          const StorageBuffers &eigenvectors,
                          const StorageBuffers &eigenvectors_transposed,
                          const StorageBuffers &root_eigenvalues,
                          const StorageBuffers &residual,
                          const StorageBuffers &workspace_one,
                          const StorageBuffers &workspace_two,
                          const StorageBuffers &correction_coordinates,
                          const StorageBuffers &correction,
                          MTL::Buffer *temporary_output, uint32_t order) const {
    const size_t elements = checkedElementCount(order, order);
    validateRawOutput(precision, workspace_one, temporary_output, elements);
    validateRawOutput(precision, workspace_two, temporary_output, elements);
    validateRawOutput(precision, correction_coordinates, temporary_output,
                      elements);
    validateRawOutput(precision, correction, temporary_output, elements);

    const MatmulParams matmul_params{order, order, order};
    const SquareParams square_params{order};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    const auto encode_quantize = [&](const StorageBuffers &output) {
      if (!precision.requiresQuantizer()) {
        return;
      }
      const QuantizeParams params{
          static_cast<uint32_t>(elements),
          precision.scale_block_elements,
          0u,
          0u,
      };
      const size_t work_items = precision.requiresScalePlane()
                                    ? scaleBytes(precision, elements)
                                    : storageBytes(precision, elements);
      encoder->setComputePipelineState(
          pipelineFor(precision, KernelOperation::Quantize));
      encoder->setBuffer(temporary_output, 0, 0);
      encoder->setBuffer(output.values, 0, 1);
      encoder->setBuffer(output.scales, 0, 2);
      encoder->setBytes(&params, sizeof(params), 3);
      encoder->dispatchThreads(MTL::Size::Make(work_items, 1, 1),
                               MTL::Size::Make(32, 1, 1));
    };
    const auto encode_matmul = [&](const StorageBuffers &left,
                                   const StorageBuffers &right,
                                   const StorageBuffers &output) {
      MTL::Buffer *raw_output =
          precision.requiresQuantizer() ? temporary_output : output.values;
      encode2d(encoder, pipelineFor(precision, KernelOperation::Matmul), order,
               order, [&](MTL::ComputeCommandEncoder *current) {
                 if (precision.requiresScalePlane()) {
                   current->setBuffer(left.values, 0, 0);
                   current->setBuffer(left.scales, 0, 1);
                   current->setBuffer(right.values, 0, 2);
                   current->setBuffer(right.scales, 0, 3);
                   current->setBuffer(raw_output, 0, 4);
                   current->setBytes(&matmul_params, sizeof(matmul_params), 5);
                 } else {
                   current->setBuffer(left.values, 0, 0);
                   current->setBuffer(right.values, 0, 1);
                   current->setBuffer(raw_output, 0, 2);
                   current->setBytes(&matmul_params, sizeof(matmul_params), 3);
                 }
               });
      encode_quantize(output);
    };
    const auto encode_diagonal_sylvester =
        [&](const StorageBuffers &right_hand_side,
            const StorageBuffers &output) {
          MTL::Buffer *raw_output =
              precision.requiresQuantizer() ? temporary_output : output.values;
          encode2d(
              encoder,
              pipelineFor(precision, KernelOperation::DiagonalSylvester), order,
              order, [&](MTL::ComputeCommandEncoder *current) {
                if (precision.requiresScalePlane()) {
                  current->setBuffer(right_hand_side.values, 0, 0);
                  current->setBuffer(right_hand_side.scales, 0, 1);
                  current->setBuffer(root_eigenvalues.values, 0, 2);
                  current->setBuffer(root_eigenvalues.scales, 0, 3);
                  current->setBuffer(raw_output, 0, 4);
                  current->setBytes(&square_params, sizeof(square_params), 5);
                } else {
                  current->setBuffer(right_hand_side.values, 0, 0);
                  current->setBuffer(root_eigenvalues.values, 0, 1);
                  current->setBuffer(raw_output, 0, 2);
                  current->setBytes(&square_params, sizeof(square_params), 3);
                }
              });
          encode_quantize(output);
        };

    encode_matmul(eigenvectors_transposed, residual, workspace_one);
    encode_matmul(workspace_one, eigenvectors, workspace_two);
    encode_diagonal_sylvester(workspace_two, correction_coordinates);
    encode_matmul(eigenvectors, correction_coordinates, workspace_one);
    encode_matmul(workspace_one, eigenvectors_transposed, correction);
    finish(command_buffer, encoder);
  }

  void applySylvesterInverse(const PrecisionSpec &precision,
                             const StorageBuffers &right_hand_side,
                             MTL::Buffer *inverse,
                             const StorageBuffers &correction,
                             uint32_t order) const {
    requireTiledPrecision(precision);
    const SquareParams params{order};
    dispatch2d(pipelineFor(precision, KernelOperation::ApplySylvesterInverse),
               order, order, [&](MTL::ComputeCommandEncoder *encoder) {
                 encoder->setBuffer(right_hand_side.values, 0, 0);
                 encoder->setBuffer(inverse, 0, 1);
                 encoder->setBuffer(correction.values, 0, 2);
                 encoder->setBytes(&params, sizeof(params), 3);
               });
  }

  void reconstructTiled(const PrecisionSpec &precision,
                        const StorageBuffers &eigenvectors,
                        const StorageBuffers &eigenvectors_transposed,
                        const StorageBuffers &root_eigenvalues,
                        const StorageBuffers &workspace,
                        const StorageBuffers &root, uint32_t order,
                        uint32_t batch_count = 1) const {
    requireTiledPrecision(precision);
    const SquareParams square_params{order};
    const MatmulParams matmul_params{order, order, order};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encode3d(encoder, pipelineFor(precision, KernelOperation::ScaleColumns),
             order, order, batch_count,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(eigenvectors.values, 0, 0);
               current->setBuffer(root_eigenvalues.values, 0, 1);
               current->setBuffer(workspace.values, 0, 2);
               current->setBytes(&square_params, sizeof(square_params), 3);
             });
    encode3d(encoder, pipelineFor(precision, KernelOperation::MatmulTiled),
             order, order, batch_count,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(workspace.values, 0, 0);
               current->setBuffer(eigenvectors_transposed.values, 0, 1);
               current->setBuffer(root.values, 0, 2);
               current->setBytes(&matmul_params, sizeof(matmul_params), 3);
             });
    finish(command_buffer, encoder);
  }

  void refineTiled(const PrecisionSpec &precision,
                   const StorageBuffers &eigenvectors,
                   const StorageBuffers &eigenvectors_transposed,
                   const StorageBuffers &residual,
                   const StorageBuffers &workspace_one, MTL::Buffer *inverse,
                   const StorageBuffers &correction_coordinates,
                   const StorageBuffers &correction, uint32_t order,
                   uint32_t batch_count = 1) const {
    requireTiledPrecision(precision);
    const MatmulParams matmul_params{order, order, order};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    const auto encode_matmul = [&](const StorageBuffers &left,
                                   const StorageBuffers &right,
                                   const StorageBuffers &output) {
      encode3d(encoder, pipelineFor(precision, KernelOperation::MatmulTiled),
               order, order, batch_count,
               [&](MTL::ComputeCommandEncoder *current) {
                 current->setBuffer(left.values, 0, 0);
                 current->setBuffer(right.values, 0, 1);
                 current->setBuffer(output.values, 0, 2);
                 current->setBytes(&matmul_params, sizeof(matmul_params), 3);
               });
    };
    encode_matmul(eigenvectors_transposed, residual, workspace_one);
    encode3d(encoder,
             pipelineFor(precision, KernelOperation::MatmulTiledScaled), order,
             order, batch_count, [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(workspace_one.values, 0, 0);
               current->setBuffer(eigenvectors.values, 0, 1);
               current->setBuffer(inverse, 0, 2);
               current->setBuffer(correction_coordinates.values, 0, 3);
               current->setBytes(&matmul_params, sizeof(matmul_params), 4);
             });
    encode_matmul(eigenvectors, correction_coordinates, workspace_one);
    encode_matmul(workspace_one, eigenvectors_transposed, correction);
    finish(command_buffer, encoder);
  }
  void reconstructLowFactorTiled(const PrecisionSpec &precision,
                                 const StorageBuffers &vectors,
                                 MTL::Buffer *root_eigenvalues,
                                 MTL::Buffer *workspace, MTL::Buffer *root,
                                 uint32_t order) const {
    if (precision.precision == Precision::Fp32 || precision.usesLogSpectrum()) {
      fail("Low-factor reconstruction requires a linear low-precision basis.");
    }
    const KernelPipelines &pipelines = formatPipelines(precision);
    const SquareParams square_params{order};
    const LowFactorMatmulParams matmul_params{order, order, order, 1, 0};
    MTL::Buffer *scales =
        vectors.scales == nullptr ? vectors.values : vectors.scales;
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encode2d(encoder, pipelines.scale_low_columns_f32, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(vectors.values, 0, 0);
               current->setBuffer(scales, 0, 1);
               current->setBuffer(root_eigenvalues, 0, 2);
               current->setBuffer(workspace, 0, 3);
               current->setBytes(&square_params, sizeof(square_params), 4);
             });
    encode3d(encoder, pipelines.matmul_f32_low_right, order, order, 1,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(workspace, 0, 0);
               current->setBuffer(vectors.values, 0, 1);
               current->setBuffer(scales, 0, 2);
               current->setBuffer(root_eigenvalues, 0, 3);
               current->setBuffer(root, 0, 4);
               current->setBytes(&matmul_params, sizeof(matmul_params), 5);
             });
    finish(command_buffer, encoder);
  }

  void refineLowFactorTiled(const PrecisionSpec &precision,
                            const StorageBuffers &vectors,
                            MTL::Buffer *residual, MTL::Buffer *workspace,
                            MTL::Buffer *inverse,
                            MTL::Buffer *correction_coordinates,
                            MTL::Buffer *correction, uint32_t order) const {
    if (precision.precision == Precision::Fp32 || precision.usesLogSpectrum()) {
      fail("Low-factor refinement requires a linear low-precision basis.");
    }
    const KernelPipelines &pipelines = formatPipelines(precision);
    MTL::Buffer *scales =
        vectors.scales == nullptr ? vectors.values : vectors.scales;
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    const auto encode_left = [&](MTL::Buffer *right, MTL::Buffer *output,
                                 uint32_t transpose) {
      const LowFactorMatmulParams params{order, order, order, transpose, 0};
      encode3d(encoder, pipelines.matmul_low_left_f32, order, order, 1,
               [&](MTL::ComputeCommandEncoder *current) {
                 current->setBuffer(vectors.values, 0, 0);
                 current->setBuffer(scales, 0, 1);
                 current->setBuffer(right, 0, 2);
                 current->setBuffer(output, 0, 3);
                 current->setBytes(&params, sizeof(params), 4);
               });
    };
    const auto encode_right = [&](MTL::Buffer *left, MTL::Buffer *output,
                                  uint32_t transpose, uint32_t apply_inverse) {
      const LowFactorMatmulParams params{order, order, order, transpose,
                                         apply_inverse};
      encode3d(encoder, pipelines.matmul_f32_low_right, order, order, 1,
               [&](MTL::ComputeCommandEncoder *current) {
                 current->setBuffer(left, 0, 0);
                 current->setBuffer(vectors.values, 0, 1);
                 current->setBuffer(scales, 0, 2);
                 current->setBuffer(inverse, 0, 3);
                 current->setBuffer(output, 0, 4);
                 current->setBytes(&params, sizeof(params), 5);
               });
    };
    encode_left(residual, workspace, 1);
    encode_right(workspace, correction_coordinates, 0, 1);
    encode_left(correction_coordinates, workspace, 0);
    encode_right(workspace, correction, 1, 0);
    finish(command_buffer, encoder);
  }

  void symmetricJacobi(MTL::Buffer *matrix, MTL::Buffer *matrix_scratch,
                       MTL::Buffer *vectors, MTL::Buffer *vectors_scratch,
                       MTL::Buffer *partners, MTL::Buffer *pair_indices,
                       MTL::Buffer *pair_members, MTL::Buffer *rotations,
                       MTL::Buffer *root_eigenvalues, uint32_t order,
                       uint32_t padded_order, uint32_t sweeps) const {
    if (sweeps == 0 || sweeps % 2 != 0 || padded_order % 2 != 0 ||
        padded_order < order) {
      fail("The GPU Jacobi schedule is invalid.");
    }
    const uint32_t pair_count = padded_order / 2;
    const uint32_t round_count = padded_order - 1;
    const SquareParams square_params{order};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encode2d(encoder, resident_.jacobi_identity, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(vectors, 0, 0);
               current->setBytes(&square_params, sizeof(square_params), 1);
             });
    MTL::Buffer *matrix_source = matrix;
    MTL::Buffer *matrix_destination = matrix_scratch;
    MTL::Buffer *vectors_source = vectors;
    MTL::Buffer *vectors_destination = vectors_scratch;
    for (uint32_t sweep = 0; sweep < sweeps; ++sweep) {
      for (uint32_t round = 0; round < round_count; ++round) {
        const JacobiParams params{order, padded_order, pair_count, round};
        encoder->setComputePipelineState(resident_.jacobi_rotations);
        encoder->setBuffer(matrix_source, 0, 0);
        encoder->setBuffer(pair_members, 0, 1);
        encoder->setBuffer(rotations, 0, 2);
        encoder->setBytes(&params, sizeof(params), 3);
        encoder->dispatchThreads(
            MTL::Size::Make(pair_count, 1, 1),
            MTL::Size::Make(std::min<uint32_t>(pair_count, 64), 1, 1));
        encode3d(encoder, resident_.jacobi_similarity, order, order, 2,
                 [&](MTL::ComputeCommandEncoder *current) {
                   current->setBuffer(matrix_source, 0, 0);
                   current->setBuffer(matrix_destination, 0, 1);
                   current->setBuffer(vectors_source, 0, 2);
                   current->setBuffer(vectors_destination, 0, 3);
                   current->setBuffer(partners, 0, 4);
                   current->setBuffer(pair_indices, 0, 5);
                   current->setBuffer(rotations, 0, 6);
                   current->setBytes(&params, sizeof(params), 7);
                 });
        std::swap(matrix_source, matrix_destination);
        std::swap(vectors_source, vectors_destination);
      }
    }
    if (matrix_source != matrix || vectors_source != vectors) {
      fail("The GPU Jacobi output parity is invalid.");
    }
    encoder->setComputePipelineState(resident_.jacobi_root_eigenvalues);
    encoder->setBuffer(matrix, 0, 0);
    encoder->setBuffer(root_eigenvalues, 0, 1);
    encoder->setBytes(&square_params, sizeof(square_params), 2);
    encoder->dispatchThreads(
        MTL::Size::Make(order, 1, 1),
        MTL::Size::Make(std::min<uint32_t>(order, 64), 1, 1));
    finish(command_buffer, encoder);
  }

  void prepareFrozenSpectralFactor(MTL::Buffer *vectors,
                                   MTL::Buffer *vectors_transposed,
                                   MTL::Buffer *root_eigenvalues,
                                   MTL::Buffer *inverse_sylvester,
                                   uint32_t order) const {
    const SquareParams params{order};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encode2d(encoder, resident_.transpose_square, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(vectors, 0, 0);
               current->setBuffer(vectors_transposed, 0, 1);
               current->setBytes(&params, sizeof(params), 2);
             });
    encode2d(encoder, resident_.sylvester_inverse, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(root_eigenvalues, 0, 0);
               current->setBuffer(inverse_sylvester, 0, 1);
               current->setBytes(&params, sizeof(params), 2);
             });
    finish(command_buffer, encoder);
  }

  void quantizeAndDecode(const PrecisionSpec &precision, MTL::Buffer *source,
                         const StorageBuffers &encoded, MTL::Buffer *decoded,
                         uint32_t elements, uint32_t rounding_stream) const {
    if (precision.precision == Precision::Fp32) {
      fail("FP32 does not require device quantization.");
    }
    const size_t required_values = storageBytes(precision, elements);
    const size_t required_metadata =
        precision.usesLogSpectrum() ? spectrumMetadataBytes(precision, elements)
                                    : scaleBytes(precision, elements);
    if (encoded.values == nullptr ||
        encoded.values->length() < required_values ||
        (required_metadata != 0 &&
         (encoded.scales == nullptr ||
          encoded.scales->length() < required_metadata))) {
      fail("The encoded frozen-factor buffer is too small.");
    }
    const KernelPipelines &pipelines = formatPipelines(precision);
    const uint32_t block_elements = precision.usesLogSpectrum()
                                        ? precision.spectral_block_elements
                                        : precision.scale_block_elements;
    const QuantizeParams params{
        elements,
        block_elements,
        rounding_seed_,
        rounding_stream,
    };
    const size_t quantize_items =
        precision.requiresScalePlane()  ? required_metadata
        : precision.usesPackedStorage() ? required_values
                                        : elements;
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    if (precision.usesLogSpectrum()) {
      const size_t blocks =
          elements / block_elements + (elements % block_elements != 0 ? 1 : 0);
      encoder->setComputePipelineState(resident_.log_spectrum_metadata);
      encoder->setBuffer(source, 0, 0);
      encoder->setBuffer(encoded.scales, 0, 1);
      encoder->setBytes(&params, sizeof(params), 2);
      encoder->dispatchThreads(
          MTL::Size::Make(blocks, 1, 1),
          MTL::Size::Make(std::min<size_t>(blocks, 64), 1, 1));
    }
    encoder->setComputePipelineState(rounding_mode_ == RoundingMode::Stochastic
                                         ? pipelines.quantize_stochastic
                                         : pipelines.quantize);
    encoder->setBuffer(source, 0, 0);
    encoder->setBuffer(encoded.values, 0, 1);
    if (encoded.scales != nullptr) {
      encoder->setBuffer(encoded.scales, 0, 2);
    }
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreads(
        MTL::Size::Make(quantize_items, 1, 1),
        MTL::Size::Make(std::min<size_t>(quantize_items, 64), 1, 1));
    encoder->setComputePipelineState(pipelines.decode_to_f32);
    encoder->setBuffer(encoded.values, 0, 0);
    if (encoded.scales != nullptr) {
      encoder->setBuffer(encoded.scales, 0, 1);
    }
    encoder->setBuffer(decoded, 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreads(
        MTL::Size::Make(elements, 1, 1),
        MTL::Size::Make(std::min<uint32_t>(elements, 256), 1, 1));
    finish(command_buffer, encoder);
  }

  double measureF32Orthogonality(MTL::Buffer *vectors,
                                 MTL::Buffer *vectors_transposed,
                                 MTL::Buffer *gram, uint32_t order) const {
    const PrecisionSpec &fp32 = precisionSpec(Precision::Fp32);
    const SquareParams square_params{order};
    const MatmulParams matmul_params{order, order, order};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encode2d(encoder, resident_.transpose_square, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(vectors, 0, 0);
               current->setBuffer(vectors_transposed, 0, 1);
               current->setBytes(&square_params, sizeof(square_params), 2);
             });
    encode3d(encoder, pipelineFor(fp32, KernelOperation::MatmulTiled), order,
             order, 1, [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(vectors_transposed, 0, 0);
               current->setBuffer(vectors, 0, 1);
               current->setBuffer(gram, 0, 2);
               current->setBytes(&matmul_params, sizeof(matmul_params), 3);
             });
    finish(command_buffer, encoder);
    const float *values = static_cast<const float *>(gram->contents());
    double difference_squared = 0.0;
    for (uint32_t column = 0; column < order; ++column) {
      for (uint32_t row = 0; row < order; ++row) {
        const double expected = row == column ? 1.0 : 0.0;
        const double difference =
            static_cast<double>(values[row + column * order]) - expected;
        difference_squared += difference * difference;
      }
    }
    return std::sqrt(difference_squared / static_cast<double>(order));
  }

  void initializeWorkingRoot(MTL::Buffer *root_f32, MTL::Buffer *root_working,
                             uint32_t order) const {
    const SquareParams params{order};
    dispatch2d(resident_.working_from_f32, order, order,
               [&](MTL::ComputeCommandEncoder *encoder) {
                 encoder->setBuffer(root_f32, 0, 0);
                 encoder->setBuffer(root_working, 0, 1);
                 encoder->setBytes(&params, sizeof(params), 2);
               });
  }

  double computeWorkingResidual(MTL::Buffer *matrix, MTL::Buffer *root,
                                MTL::Buffer *residual,
                                MTL::Buffer *residual_f32, MTL::Buffer *norm,
                                uint32_t order) const {
    const size_t element_count = checkedElementCount(order, order);
    if (element_count > std::numeric_limits<uint32_t>::max()) {
      fail("The working residual exceeds the kernel element limit.");
    }
    const uint32_t elements = static_cast<uint32_t>(element_count);
    const SquareParams square_params{order};
    const QuantizeParams element_params{elements, 0u, 0u, 0u};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encode2d(encoder, resident_.working_residual, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(matrix, 0, 0);
               current->setBuffer(root, 0, 1);
               current->setBuffer(residual, 0, 2);
               current->setBytes(&square_params, sizeof(square_params), 3);
             });
    encoder->setComputePipelineState(resident_.working_residual_to_f32);
    encoder->setBuffer(residual, 0, 0);
    encoder->setBuffer(residual_f32, 0, 1);
    encoder->setBytes(&element_params, sizeof(element_params), 2);
    encoder->dispatchThreads(
        MTL::Size::Make(elements, 1, 1),
        MTL::Size::Make(std::min<uint32_t>(elements, 256), 1, 1));
    encoder->setComputePipelineState(resident_.working_norm);
    encoder->setBuffer(matrix, 0, 0);
    encoder->setBuffer(residual, 0, 1);
    encoder->setBuffer(norm, 0, 2);
    encoder->setBytes(&element_params, sizeof(element_params), 3);
    encoder->dispatchThreads(MTL::Size::Make(256, 1, 1),
                             MTL::Size::Make(256, 1, 1));
    finish(command_buffer, encoder);
    const WorkingNorm measured =
        *static_cast<const WorkingNorm *>(norm->contents());
    if (!(measured.matrix_squared > 0.0f) ||
        !std::isfinite(measured.residual_squared) ||
        !std::isfinite(measured.matrix_squared)) {
      return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(static_cast<double>(measured.residual_squared) /
                     static_cast<double>(measured.matrix_squared));
  }

  void addWorkingCorrection(MTL::Buffer *root, MTL::Buffer *correction,
                            uint32_t order) const {
    const SquareParams params{order};
    dispatch2d(resident_.working_add_symmetric_correction, order, order,
               [&](MTL::ComputeCommandEncoder *encoder) {
                 encoder->setBuffer(root, 0, 0);
                 encoder->setBuffer(correction, 0, 1);
                 encoder->setBytes(&params, sizeof(params), 2);
               });
  }

  double refineUpdateAndMeasure(const PrecisionSpec &precision,
                                const StorageBuffers &vectors,
                                const StorageBuffers &vectors_transposed,
                                MTL::Buffer *matrix, MTL::Buffer *root,
                                MTL::Buffer *residual_working,
                                MTL::Buffer *residual_f32, MTL::Buffer *norm,
                                MTL::Buffer *workspace, MTL::Buffer *inverse,
                                MTL::Buffer *correction_coordinates,
                                MTL::Buffer *correction, uint32_t order) const {
    const size_t element_count = checkedElementCount(order, order);
    if (element_count > std::numeric_limits<uint32_t>::max()) {
      fail("The fused refinement exceeds the kernel element limit.");
    }
    const uint32_t elements = static_cast<uint32_t>(element_count);
    const SquareParams square_params{order};
    const MatmulParams matmul_params{order, order, order};
    const QuantizeParams element_params{elements, 0u, 0u, 0u};
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    if (precision.precision == Precision::Fp32) {
      const auto encode_matmul = [&](MTL::Buffer *left, MTL::Buffer *right,
                                     MTL::Buffer *output) {
        encode3d(encoder, pipelineFor(precision, KernelOperation::MatmulTiled),
                 order, order, 1, [&](MTL::ComputeCommandEncoder *current) {
                   current->setBuffer(left, 0, 0);
                   current->setBuffer(right, 0, 1);
                   current->setBuffer(output, 0, 2);
                   current->setBytes(&matmul_params, sizeof(matmul_params), 3);
                 });
      };
      encode_matmul(vectors_transposed.values, residual_f32, workspace);
      encode3d(encoder,
               pipelineFor(precision, KernelOperation::MatmulTiledScaled),
               order, order, 1, [&](MTL::ComputeCommandEncoder *current) {
                 current->setBuffer(workspace, 0, 0);
                 current->setBuffer(vectors.values, 0, 1);
                 current->setBuffer(inverse, 0, 2);
                 current->setBuffer(correction_coordinates, 0, 3);
                 current->setBytes(&matmul_params, sizeof(matmul_params), 4);
               });
      encode_matmul(vectors.values, correction_coordinates, workspace);
      encode_matmul(workspace, vectors_transposed.values, correction);
    } else {
      if (precision.usesLogSpectrum()) {
        fail("The frozen basis requires linear storage.");
      }
      const KernelPipelines &pipelines = formatPipelines(precision);
      MTL::Buffer *scales =
          vectors.scales == nullptr ? vectors.values : vectors.scales;
      const auto encode_left = [&](MTL::Buffer *right, MTL::Buffer *output,
                                   uint32_t transpose) {
        const LowFactorMatmulParams params{order, order, order, transpose, 0};
        encode3d(encoder, pipelines.matmul_low_left_f32, order, order, 1,
                 [&](MTL::ComputeCommandEncoder *current) {
                   current->setBuffer(vectors.values, 0, 0);
                   current->setBuffer(scales, 0, 1);
                   current->setBuffer(right, 0, 2);
                   current->setBuffer(output, 0, 3);
                   current->setBytes(&params, sizeof(params), 4);
                 });
      };
      const auto encode_right = [&](MTL::Buffer *left, MTL::Buffer *output,
                                    uint32_t transpose,
                                    uint32_t apply_inverse) {
        const LowFactorMatmulParams params{order, order, order, transpose,
                                           apply_inverse};
        encode3d(encoder, pipelines.matmul_f32_low_right, order, order, 1,
                 [&](MTL::ComputeCommandEncoder *current) {
                   current->setBuffer(left, 0, 0);
                   current->setBuffer(vectors.values, 0, 1);
                   current->setBuffer(scales, 0, 2);
                   current->setBuffer(inverse, 0, 3);
                   current->setBuffer(output, 0, 4);
                   current->setBytes(&params, sizeof(params), 5);
                 });
      };
      encode_left(residual_f32, workspace, 1);
      encode_right(workspace, correction_coordinates, 0, 1);
      encode_left(correction_coordinates, workspace, 0);
      encode_right(workspace, correction, 1, 0);
    }
    encode2d(encoder, resident_.working_add_symmetric_correction, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(root, 0, 0);
               current->setBuffer(correction, 0, 1);
               current->setBytes(&square_params, sizeof(square_params), 2);
             });
    encode2d(encoder, resident_.working_residual, order, order,
             [&](MTL::ComputeCommandEncoder *current) {
               current->setBuffer(matrix, 0, 0);
               current->setBuffer(root, 0, 1);
               current->setBuffer(residual_working, 0, 2);
               current->setBytes(&square_params, sizeof(square_params), 3);
             });
    encoder->setComputePipelineState(resident_.working_residual_to_f32);
    encoder->setBuffer(residual_working, 0, 0);
    encoder->setBuffer(residual_f32, 0, 1);
    encoder->setBytes(&element_params, sizeof(element_params), 2);
    encoder->dispatchThreads(
        MTL::Size::Make(elements, 1, 1),
        MTL::Size::Make(std::min<uint32_t>(elements, 256), 1, 1));
    encoder->setComputePipelineState(resident_.working_norm);
    encoder->setBuffer(matrix, 0, 0);
    encoder->setBuffer(residual_working, 0, 1);
    encoder->setBuffer(norm, 0, 2);
    encoder->setBytes(&element_params, sizeof(element_params), 3);
    encoder->dispatchThreads(MTL::Size::Make(256, 1, 1),
                             MTL::Size::Make(256, 1, 1));
    finish(command_buffer, encoder);
    const WorkingNorm measured =
        *static_cast<const WorkingNorm *>(norm->contents());
    if (!(measured.matrix_squared > 0.0f) ||
        !std::isfinite(measured.residual_squared) ||
        !std::isfinite(measured.matrix_squared)) {
      return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(static_cast<double>(measured.residual_squared) /
                     static_cast<double>(measured.matrix_squared));
  }

 private:
  struct KernelPipelines {
    Precision precision = Precision::Fp32;
    MTL::ComputePipelineState *scale_columns = nullptr;
    MTL::ComputePipelineState *matmul = nullptr;
    MTL::ComputePipelineState *diagonal_sylvester = nullptr;
    MTL::ComputePipelineState *apply_sylvester_inverse = nullptr;
    MTL::ComputePipelineState *matmul_tiled = nullptr;
    MTL::ComputePipelineState *matmul_tiled_scaled = nullptr;
    MTL::ComputePipelineState *diagonal_sylvester_coalesced = nullptr;
    MTL::ComputePipelineState *quantize = nullptr;
    MTL::ComputePipelineState *quantize_stochastic = nullptr;
    MTL::ComputePipelineState *decode_to_f32 = nullptr;
    MTL::ComputePipelineState *matmul_low_left_f32 = nullptr;
    MTL::ComputePipelineState *matmul_f32_low_right = nullptr;
    MTL::ComputePipelineState *scale_low_columns_f32 = nullptr;
  };

  struct ResidentPipelines {
    MTL::ComputePipelineState *jacobi_identity = nullptr;
    MTL::ComputePipelineState *jacobi_rotations = nullptr;
    MTL::ComputePipelineState *jacobi_similarity = nullptr;
    MTL::ComputePipelineState *jacobi_root_eigenvalues = nullptr;
    MTL::ComputePipelineState *transpose_square = nullptr;
    MTL::ComputePipelineState *sylvester_inverse = nullptr;
    MTL::ComputePipelineState *log_spectrum_metadata = nullptr;
    MTL::ComputePipelineState *working_from_f32 = nullptr;
    MTL::ComputePipelineState *working_residual = nullptr;
    MTL::ComputePipelineState *working_residual_to_f32 = nullptr;
    MTL::ComputePipelineState *working_add_symmetric_correction = nullptr;
    MTL::ComputePipelineState *working_norm = nullptr;

    void release() const {
      MTL::ComputePipelineState *states[] = {
          jacobi_identity,
          jacobi_rotations,
          jacobi_similarity,
          jacobi_root_eigenvalues,
          transpose_square,
          sylvester_inverse,
          log_spectrum_metadata,
          working_from_f32,
          working_residual,
          working_residual_to_f32,
          working_add_symmetric_correction,
          working_norm,
      };
      for (MTL::ComputePipelineState *state : states) {
        if (state != nullptr) {
          state->release();
        }
      }
    }
  };

  void requireTiledPrecision(const PrecisionSpec &precision) const {
    if (precision.precision != Precision::Fp32 &&
        precision.precision != Precision::Bf16) {
      fail("Tiled kernels require FP32 or BF16 correction storage.");
    }
  }

  template <typename Bind>
  void encode2d(MTL::ComputeCommandEncoder *encoder,
                MTL::ComputePipelineState *pipeline, uint32_t columns,
                uint32_t rows, Bind &&bind) const {
    encoder->setComputePipelineState(pipeline);
    bind(encoder);
    encoder->dispatchThreads(MTL::Size::Make(columns, rows, 1),
                             MTL::Size::Make(16, 16, 1));
  }

  template <typename Bind>
  void encode3d(MTL::ComputeCommandEncoder *encoder,
                MTL::ComputePipelineState *pipeline, uint32_t columns,
                uint32_t rows, uint32_t depth, Bind &&bind) const {
    encoder->setComputePipelineState(pipeline);
    bind(encoder);
    encoder->dispatchThreads(MTL::Size::Make(columns, rows, depth),
                             MTL::Size::Make(16, 16, 1));
  }

  void finish(MTL::CommandBuffer *command_buffer,
              MTL::ComputeCommandEncoder *encoder) const {
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    if (command_buffer->status() == MTL::CommandBufferStatusError) {
      fail("Metal command failed: " + errorText(command_buffer->error()));
    }
  }
  void validateRawOutput(const PrecisionSpec &precision,
                         const StorageBuffers &output,
                         MTL::Buffer *temporary_output, size_t elements) const {
    const size_t value_bytes = storageBytes(precision, elements);
    if (output.values == nullptr || output.values->length() < value_bytes) {
      fail("Raw Metal output value buffer is too small.");
    }
    if (precision.requiresScalePlane() &&
        (output.scales == nullptr ||
         output.scales->length() < scaleBytes(precision, elements))) {
      fail("Raw Metal output scale buffer is too small.");
    }
    if (!precision.requiresQuantizer()) {
      if (temporary_output != nullptr) {
        fail(
            "Direct-storage raw Metal operations must not use a float "
            "temporary.");
      }
      return;
    }
    if (temporary_output == nullptr) {
      fail(
          "Packed storage operation requires a float temporary output buffer.");
    }
    if (elements > std::numeric_limits<uint32_t>::max()) {
      fail("Quantizer element count overflow.");
    }
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        temporary_output->length() < elements * sizeof(float)) {
      fail("Raw Metal float producer buffer is too small.");
    }
  }

  void quantizeOutput(const PrecisionSpec &precision,
                      MTL::Buffer *temporary_output,
                      const StorageBuffers &output, size_t elements) const {
    if (!precision.requiresQuantizer()) {
      return;
    }
    const QuantizeParams params{
        static_cast<uint32_t>(elements),
        precision.scale_block_elements,
        0u,
        0u,
    };
    const size_t work_items = precision.requiresScalePlane()
                                  ? scaleBytes(precision, elements)
                                  : storageBytes(precision, elements);
    dispatch1d(pipelineFor(precision, KernelOperation::Quantize), work_items,
               [&](MTL::ComputeCommandEncoder *encoder) {
                 encoder->setBuffer(temporary_output, 0, 0);
                 encoder->setBuffer(output.values, 0, 1);
                 encoder->setBuffer(output.scales, 0, 2);
                 encoder->setBytes(&params, sizeof(params), 3);
               });
  }

  std::string kernelName(const PrecisionSpec &precision,
                         KernelOperation operation) const {
    return std::string(kernelOperationName(operation)) + "_" +
           precision.kernel_suffix;
  }

  const KernelPipelines &formatPipelines(const PrecisionSpec &precision) const {
    requireEnabledKernels(precision);
    for (const KernelPipelines &pipelines : pipelines_) {
      if (pipelines.precision == precision.precision) {
        return pipelines;
      }
    }
    fail("No Metal pipelines loaded for precision: " +
         std::string(precision.cli_name));
  }

  MTL::ComputePipelineState *pipelineFor(const PrecisionSpec &precision,
                                         KernelOperation operation) const {
    requireEnabledKernels(precision);
    for (const KernelPipelines &pipelines : pipelines_) {
      if (pipelines.precision != precision.precision) {
        continue;
      }
      switch (operation) {
        case KernelOperation::ScaleColumns:
          return pipelines.scale_columns;
        case KernelOperation::Matmul:
          return pipelines.matmul;
        case KernelOperation::DiagonalSylvester:
          return pipelines.diagonal_sylvester;
        case KernelOperation::ApplySylvesterInverse:
          return pipelines.apply_sylvester_inverse;
        case KernelOperation::MatmulTiled:
          return pipelines.matmul_tiled;
        case KernelOperation::MatmulTiledScaled:
          return pipelines.matmul_tiled_scaled;
        case KernelOperation::DiagonalSylvesterCoalesced:
          return pipelines.diagonal_sylvester_coalesced;
        case KernelOperation::Quantize:
          return pipelines.quantize;
      }
    }
    fail("No Metal pipeline loaded for precision: " +
         std::string(precision.cli_name));
  }

  MTL::ComputePipelineState *makePipeline(const char *name) {
    NS::String *function_name =
        NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function *function = library_->newFunction(function_name);
    if (function == nullptr) {
      fail(std::string("Cannot find Metal kernel: ") + name);
    }
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pipeline =
        device_->newComputePipelineState(function, &error);
    function->release();
    if (pipeline == nullptr) {
      fail("Cannot create Metal pipeline: " + errorText(error));
    }
    return pipeline;
  }

  template <typename Bind>
  void dispatch2d(MTL::ComputePipelineState *pipeline, uint32_t columns,
                  uint32_t rows, Bind &&bind) const {
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    bind(encoder);
    const MTL::Size threads = MTL::Size::Make(16, 16, 1);
    const MTL::Size grid = MTL::Size::Make(columns, rows, 1);
    encoder->dispatchThreads(grid, threads);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    if (command_buffer->status() == MTL::CommandBufferStatusError) {
      fail("Metal command failed: " + errorText(command_buffer->error()));
    }
  }

  template <typename Bind>
  void dispatch1d(MTL::ComputePipelineState *pipeline, size_t elements,
                  Bind &&bind) const {
    MTL::CommandBuffer *command_buffer = queue_->commandBuffer();
    MTL::ComputeCommandEncoder *encoder =
        command_buffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    bind(encoder);
    const MTL::Size threads = MTL::Size::Make(32, 1, 1);
    const MTL::Size grid = MTL::Size::Make(elements, 1, 1);
    encoder->dispatchThreads(grid, threads);
    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    if (command_buffer->status() == MTL::CommandBufferStatusError) {
      fail("Metal command failed: " + errorText(command_buffer->error()));
    }
  }

  RoundingMode rounding_mode_;
  uint32_t rounding_seed_;
  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *queue_ = nullptr;
  MTL::Library *library_ = nullptr;
  std::vector<KernelPipelines> pipelines_;
  ResidentPipelines resident_;
};

MetalRuntime::MetalRuntime(const char *shader_path, RoundingMode rounding_mode,
                           uint32_t rounding_seed)
    : impl_(std::make_unique<Impl>(shader_path, rounding_mode, rounding_seed)) {
}
MetalRuntime::~MetalRuntime() = default;
void MetalRuntime::requireEnabledKernels(const PrecisionSpec &precision) const {
  impl_->requireEnabledKernels(precision);
}

RoundingMode MetalRuntime::roundingMode() const {
  return impl_->roundingMode();
}

uint32_t MetalRuntime::roundingSeed() const {
  return impl_->roundingSeed();
}

MTL::Buffer *MetalRuntime::makeSharedBuffer(size_t bytes) const {
  return impl_->makeSharedBuffer(bytes);
}

void MetalRuntime::probeF32Edges(MTL::Buffer *source, MTL::Buffer *output,
                                 size_t elements) const {
  impl_->probeF32Edges(source, output, elements);
}

void MetalRuntime::scaleColumns(const PrecisionSpec &precision,
                                const StorageBuffers &matrix,
                                const StorageBuffers &diagonal,
                                const StorageBuffers &output,
                                MTL::Buffer *raw_output, uint32_t order) const {
  impl_->scaleColumns(precision, matrix, diagonal, output, raw_output, order);
}

void MetalRuntime::matmul(const PrecisionSpec &precision,
                          const StorageBuffers &left,
                          const StorageBuffers &right,
                          const StorageBuffers &output, MTL::Buffer *raw_output,
                          uint32_t rows, uint32_t columns,
                          uint32_t inner) const {
  impl_->matmul(precision, left, right, output, raw_output, rows, columns,
                inner);
}

void MetalRuntime::diagonalSylvester(const PrecisionSpec &precision,
                                     const StorageBuffers &diagonal,
                                     const StorageBuffers &right_hand_side,
                                     const StorageBuffers &output,
                                     MTL::Buffer *raw_output,
                                     uint32_t order) const {
  impl_->diagonalSylvester(precision, diagonal, right_hand_side, output,
                           raw_output, order);
}

void MetalRuntime::refineNativePerStage(
    const PrecisionSpec &precision, const StorageBuffers &matrix,
    const StorageBuffers &root, const StorageBuffers &eigenvectors,
    const StorageBuffers &eigenvectors_transposed,
    const StorageBuffers &root_eigenvalues, const StorageBuffers &residual,
    const StorageBuffers &workspace_one, const StorageBuffers &workspace_two,
    MTL::Buffer *raw_output, uint32_t order) const {
  impl_->refineNativePerStage(precision, matrix, root, eigenvectors,
                              eigenvectors_transposed, root_eigenvalues,
                              residual, workspace_one, workspace_two,
                              raw_output, order);
}

void MetalRuntime::refineNativeStaged(
    const PrecisionSpec &precision, const StorageBuffers &matrix,
    const StorageBuffers &root, const StorageBuffers &eigenvectors,
    const StorageBuffers &eigenvectors_transposed,
    const StorageBuffers &root_eigenvalues, const StorageBuffers &residual,
    const StorageBuffers &workspace_one, const StorageBuffers &workspace_two,
    MTL::Buffer *raw_output, uint32_t order) const {
  impl_->refineNativeStaged(precision, matrix, root, eigenvectors,
                            eigenvectors_transposed, root_eigenvalues, residual,
                            workspace_one, workspace_two, raw_output, order);
}

void MetalRuntime::applySylvesterInverse(const PrecisionSpec &precision,
                                         const StorageBuffers &root_eigenvalues,
                                         MTL::Buffer *input,
                                         const StorageBuffers &output,
                                         uint32_t order) const {
  impl_->applySylvesterInverse(precision, root_eigenvalues, input, output,
                               order);
}

void MetalRuntime::reconstructTiled(
    const PrecisionSpec &precision, const StorageBuffers &eigenvectors,
    const StorageBuffers &eigenvectors_transposed,
    const StorageBuffers &root_eigenvalues, const StorageBuffers &workspace,
    const StorageBuffers &root, uint32_t order, uint32_t batch_count) const {
  impl_->reconstructTiled(precision, eigenvectors, eigenvectors_transposed,
                          root_eigenvalues, workspace, root, order,
                          batch_count);
}

void MetalRuntime::refineTiled(const PrecisionSpec &precision,
                               const StorageBuffers &eigenvectors,
                               const StorageBuffers &eigenvectors_transposed,
                               const StorageBuffers &residual,
                               const StorageBuffers &workspace,
                               MTL::Buffer *sylvester_inverse,
                               const StorageBuffers &correction_coordinates,
                               const StorageBuffers &correction, uint32_t order,
                               uint32_t batch_count) const {
  impl_->refineTiled(precision, eigenvectors, eigenvectors_transposed, residual,
                     workspace, sylvester_inverse, correction_coordinates,
                     correction, order, batch_count);
}

void MetalRuntime::reconstructLowFactorTiled(const PrecisionSpec &precision,
                                             const StorageBuffers &eigenvectors,
                                             MTL::Buffer *eigenvalues,
                                             MTL::Buffer *workspace,
                                             MTL::Buffer *root,
                                             uint32_t order) const {
  impl_->reconstructLowFactorTiled(precision, eigenvectors, eigenvalues,
                                   workspace, root, order);
}

void MetalRuntime::refineLowFactorTiled(
    const PrecisionSpec &precision, const StorageBuffers &eigenvectors,
    MTL::Buffer *root_eigenvalues, MTL::Buffer *residual,
    MTL::Buffer *workspace_one, MTL::Buffer *workspace_two,
    MTL::Buffer *correction, uint32_t order) const {
  impl_->refineLowFactorTiled(precision, eigenvectors, root_eigenvalues,
                              residual, workspace_one, workspace_two,
                              correction, order);
}

void MetalRuntime::symmetricJacobi(
    MTL::Buffer *matrix, MTL::Buffer *eigenvectors, MTL::Buffer *partners,
    MTL::Buffer *pair_indices, MTL::Buffer *pair_members, MTL::Buffer *cosines,
    MTL::Buffer *sines, MTL::Buffer *workspace, MTL::Buffer *convergence,
    uint32_t order, uint32_t padded_order, uint32_t sweeps) const {
  impl_->symmetricJacobi(matrix, eigenvectors, partners, pair_indices,
                         pair_members, cosines, sines, workspace, convergence,
                         order, padded_order, sweeps);
}

void MetalRuntime::prepareFrozenSpectralFactor(MTL::Buffer *eigenvalues,
                                               MTL::Buffer *root_eigenvalues,
                                               MTL::Buffer *sylvester_inverse,
                                               MTL::Buffer *status,
                                               uint32_t order) const {
  impl_->prepareFrozenSpectralFactor(eigenvalues, root_eigenvalues,
                                     sylvester_inverse, status, order);
}

void MetalRuntime::quantizeAndDecode(const PrecisionSpec &precision,
                                     MTL::Buffer *source,
                                     const StorageBuffers &encoded,
                                     MTL::Buffer *decoded, uint32_t elements,
                                     uint32_t stream) const {
  impl_->quantizeAndDecode(precision, source, encoded, decoded, elements,
                           stream);
}

double MetalRuntime::measureF32Orthogonality(MTL::Buffer *basis,
                                             MTL::Buffer *transpose,
                                             MTL::Buffer *workspace,
                                             uint32_t order) const {
  return impl_->measureF32Orthogonality(basis, transpose, workspace, order);
}

void MetalRuntime::initializeWorkingRoot(MTL::Buffer *root_f32,
                                         MTL::Buffer *root_working,
                                         uint32_t order) const {
  impl_->initializeWorkingRoot(root_f32, root_working, order);
}

double MetalRuntime::computeWorkingResidual(
    MTL::Buffer *matrix, MTL::Buffer *root, MTL::Buffer *residual_working,
    MTL::Buffer *residual_f32, MTL::Buffer *norm, uint32_t order) const {
  return impl_->computeWorkingResidual(matrix, root, residual_working,
                                       residual_f32, norm, order);
}

void MetalRuntime::addWorkingCorrection(MTL::Buffer *root,
                                        MTL::Buffer *correction,
                                        uint32_t order) const {
  impl_->addWorkingCorrection(root, correction, order);
}

double MetalRuntime::refineUpdateAndMeasure(
    const PrecisionSpec &precision, const StorageBuffers &eigenvectors,
    const StorageBuffers &eigenvectors_transposed, MTL::Buffer *matrix,
    MTL::Buffer *root, MTL::Buffer *residual_working, MTL::Buffer *residual_f32,
    MTL::Buffer *norm, MTL::Buffer *workspace, MTL::Buffer *sylvester_inverse,
    MTL::Buffer *correction_coordinates, MTL::Buffer *correction,
    uint32_t order) const {
  return impl_->refineUpdateAndMeasure(
      precision, eigenvectors, eigenvectors_transposed, matrix, root,
      residual_working, residual_f32, norm, workspace, sylvester_inverse,
      correction_coordinates, correction, order);
}

void releaseBuffers(const std::vector<MTL::Buffer *> &buffers) {
  for (MTL::Buffer *buffer : buffers) {
    if (buffer != nullptr) {
      buffer->release();
    }
  }
}

}  // namespace mixed_sqrt
