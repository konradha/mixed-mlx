#pragma once

#include "precision.hpp"
#include "storage_codecs.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace MTL {
class Buffer;
}

namespace mixed_sqrt {

struct MatmulParams {
  uint32_t rows;
  uint32_t columns;
  uint32_t inner;
};
struct LowFactorMatmulParams {
  uint32_t rows;
  uint32_t columns;
  uint32_t inner;
  uint32_t transpose_low;
  uint32_t apply_inverse;
};
struct SquareParams {
  uint32_t order;
};
struct JacobiParams {
  uint32_t order;
  uint32_t padded_order;
  uint32_t pair_count;
  uint32_t round;
};
struct WorkingFloat {
  float high;
  float low;
};
struct WorkingNorm {
  float residual_squared;
  float matrix_squared;
};
struct FloatEdgeProbe {
  uint32_t loaded_bits;
  uint32_t bfloat_bits;
  uint32_t log2_bits;
  uint32_t exp2_log2_bits;
};
struct QuantizeParams {
  uint32_t elements;
  uint32_t block_elements;
  uint32_t rounding_seed;
  uint32_t rounding_stream;
};
struct LogSpectrumParams {
  uint32_t elements;
  uint32_t block_elements;
  uint32_t rounding_seed;
  uint32_t rounding_stream;
};

static_assert(sizeof(WorkingFloat) == 2 * sizeof(float));
static_assert(sizeof(WorkingNorm) == 2 * sizeof(float));
static_assert(sizeof(FloatEdgeProbe) == 4 * sizeof(uint32_t));

class MetalRuntime {
public:
  MetalRuntime(const char *shader_path, RoundingMode rounding_mode,
               uint32_t rounding_seed);
  ~MetalRuntime();
  MetalRuntime(const MetalRuntime &) = delete;
  MetalRuntime &operator=(const MetalRuntime &) = delete;
  void requireEnabledKernels(const PrecisionSpec &precision) const;

  RoundingMode roundingMode() const;
  uint32_t roundingSeed() const;
  MTL::Buffer *makeSharedBuffer(size_t bytes) const;
  void probeF32Edges(MTL::Buffer *source, MTL::Buffer *output,
                     size_t elements) const;
  void scaleColumns(const PrecisionSpec &precision,
                    const StorageBuffers &matrix,
                    const StorageBuffers &diagonal,
                    const StorageBuffers &output, MTL::Buffer *raw_output,
                    uint32_t order) const;
  void matmul(const PrecisionSpec &precision, const StorageBuffers &left,
              const StorageBuffers &right, const StorageBuffers &output,
              MTL::Buffer *raw_output, uint32_t rows, uint32_t columns,
              uint32_t inner) const;
  void diagonalSylvester(const PrecisionSpec &precision,
                         const StorageBuffers &diagonal,
                         const StorageBuffers &right_hand_side,
                         const StorageBuffers &output, MTL::Buffer *raw_output,
                         uint32_t order) const;
  void refineNativePerStage(
      const PrecisionSpec &precision, const StorageBuffers &matrix,
      const StorageBuffers &root, const StorageBuffers &eigenvectors,
      const StorageBuffers &eigenvectors_transposed,
      const StorageBuffers &root_eigenvalues, const StorageBuffers &residual,
      const StorageBuffers &workspace_one, const StorageBuffers &workspace_two,
      MTL::Buffer *raw_output, uint32_t order) const;
  void refineNativeStaged(
      const PrecisionSpec &precision, const StorageBuffers &matrix,
      const StorageBuffers &root, const StorageBuffers &eigenvectors,
      const StorageBuffers &eigenvectors_transposed,
      const StorageBuffers &root_eigenvalues, const StorageBuffers &residual,
      const StorageBuffers &workspace_one, const StorageBuffers &workspace_two,
      MTL::Buffer *raw_output, uint32_t order) const;
  void applySylvesterInverse(const PrecisionSpec &precision,
                             const StorageBuffers &root_eigenvalues,
                             MTL::Buffer *input, const StorageBuffers &output,
                             uint32_t order) const;
  void reconstructTiled(const PrecisionSpec &precision,
                        const StorageBuffers &eigenvectors,
                        const StorageBuffers &eigenvectors_transposed,
                        const StorageBuffers &root_eigenvalues,
                        const StorageBuffers &workspace,
                        const StorageBuffers &root, uint32_t order,
                        uint32_t batch_count = 1) const;
  void refineTiled(const PrecisionSpec &precision,
                   const StorageBuffers &eigenvectors,
                   const StorageBuffers &eigenvectors_transposed,
                   const StorageBuffers &residual,
                   const StorageBuffers &workspace,
                   MTL::Buffer *sylvester_inverse,
                   const StorageBuffers &correction_coordinates,
                   const StorageBuffers &correction, uint32_t order,
                   uint32_t batch_count = 1) const;
  void reconstructLowFactorTiled(const PrecisionSpec &precision,
                                 const StorageBuffers &eigenvectors,
                                 MTL::Buffer *eigenvalues,
                                 MTL::Buffer *workspace, MTL::Buffer *root,
                                 uint32_t order) const;
  void refineLowFactorTiled(const PrecisionSpec &precision,
                            const StorageBuffers &eigenvectors,
                            MTL::Buffer *root_eigenvalues,
                            MTL::Buffer *residual, MTL::Buffer *workspace_one,
                            MTL::Buffer *workspace_two, MTL::Buffer *correction,
                            uint32_t order) const;
  void symmetricJacobi(MTL::Buffer *matrix, MTL::Buffer *eigenvectors,
                       MTL::Buffer *partners, MTL::Buffer *pair_indices,
                       MTL::Buffer *pair_members, MTL::Buffer *cosines,
                       MTL::Buffer *sines, MTL::Buffer *workspace,
                       MTL::Buffer *convergence, uint32_t order,
                       uint32_t padded_order, uint32_t sweeps) const;
  void prepareFrozenSpectralFactor(MTL::Buffer *eigenvalues,
                                   MTL::Buffer *root_eigenvalues,
                                   MTL::Buffer *sylvester_inverse,
                                   MTL::Buffer *status, uint32_t order) const;
  void quantizeAndDecode(const PrecisionSpec &precision, MTL::Buffer *source,
                         const StorageBuffers &encoded, MTL::Buffer *decoded,
                         uint32_t elements, uint32_t stream) const;
  double measureF32Orthogonality(MTL::Buffer *basis, MTL::Buffer *transpose,
                                 MTL::Buffer *workspace, uint32_t order) const;
  void initializeWorkingRoot(MTL::Buffer *root_f32, MTL::Buffer *root_working,
                             uint32_t order) const;
  double computeWorkingResidual(MTL::Buffer *matrix, MTL::Buffer *root,
                                MTL::Buffer *residual_working,
                                MTL::Buffer *residual_f32, MTL::Buffer *norm,
                                uint32_t order) const;
  void addWorkingCorrection(MTL::Buffer *root, MTL::Buffer *correction,
                            uint32_t order) const;
  double refineUpdateAndMeasure(
      const PrecisionSpec &precision, const StorageBuffers &eigenvectors,
      const StorageBuffers &eigenvectors_transposed, MTL::Buffer *matrix,
      MTL::Buffer *root, MTL::Buffer *residual_working,
      MTL::Buffer *residual_f32, MTL::Buffer *norm, MTL::Buffer *workspace,
      MTL::Buffer *sylvester_inverse, MTL::Buffer *correction_coordinates,
      MTL::Buffer *correction, uint32_t order) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

void releaseBuffers(const std::vector<MTL::Buffer *> &buffers);

} // namespace mixed_sqrt
