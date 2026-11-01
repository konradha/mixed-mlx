#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace msqrt {

enum class LowbitDataType {
  Float16,
  Float8E4M3,
  Float8E5M2,
  MxFloat8E4M3,
  MxFloat4E2M1,
};
enum class LowbitMatmulSignature {
  Float16Float16ToFloat32,
  Float16Float32ToFloat32,
  Float32Float16ToFloat32,
};

enum class LowbitMatmulTile {
  Rows32Columns32,
  Rows64Columns32,
  Rows64Columns64,
};

enum class LowbitAccumulationType {
  Float32,
};

enum class LowbitMatmulStatus {
  Encoded,
  InvalidArgument,
  UnsupportedDataType,
  UnsupportedPipeline,
  Available,
  ResourceCreationFailed,
  TensorCreationFailed,
  EncoderCreationFailed,
};

struct LowbitMatmulResult {
  LowbitMatmulStatus status = LowbitMatmulStatus::InvalidArgument;
  std::string message;

  [[nodiscard]] bool encoded() const noexcept {
    return status == LowbitMatmulStatus::Encoded;
  }
};

// TransposedColumnMajor stores the logical transpose: column + row*ld.
enum class LowbitMatrixStorage {
  ColumnMajor,
  TransposedColumnMajor,
};

struct LowbitMatmulOperandLayout {
  LowbitMatrixStorage storage = LowbitMatrixStorage::ColumnMajor;
  std::size_t data_bytes = 0;
  std::size_t data_leading_dimension_elements = 0;
  std::size_t scale_bytes = 0;
  std::size_t scale_block_elements = 0;
  std::size_t scale_leading_dimension_bytes = 0;

  [[nodiscard]] bool has_scales() const noexcept { return scale_bytes != 0; }
};

struct MatrixCodecParams {
  std::uint32_t order;
  std::uint32_t leading_dimension;
  std::uint32_t scale_stride;
  std::uint32_t row_major;
};

struct LowbitMatmulBuffers {
  MTL::Buffer *left = nullptr;
  MTL::Buffer *right = nullptr;

  // MX only: UE8M0 scales, one per 32 contiguous data elements.
  MTL::Buffer *left_scales = nullptr;
  MTL::Buffer *right_scales = nullptr;

  MTL::Buffer *output = nullptr;

  // MTL4 does no implicit residency tracking. Must hold every non-null buffer.
  MTL::ResidencySet *residency_set = nullptr;
};

// MPP needs the scaled right operand transposed, so MX A planes use
// TransposedColumnMajor and B stays ColumnMajor. Query the layouts.
class LowbitMatmulEncoderResources {
 public:
  LowbitMatmulEncoderResources() = default;
  ~LowbitMatmulEncoderResources();
  LowbitMatmulEncoderResources(const LowbitMatmulEncoderResources &) = delete;
  LowbitMatmulEncoderResources &operator=(
      const LowbitMatmulEncoderResources &) = delete;

 private:
  friend class LowbitMatmul;
  void retain(NS::Object *object);
  std::vector<NS::Object *> objects_;
  MTL::ResidencySet *tensor_residency_set_ = nullptr;
};

// The command buffer must be open between begin and end.

class LowbitMatmul {
 public:
  LowbitMatmul(MTL::Device *device, MTL::Library *library, std::size_t order,
               LowbitDataType data_type, LowbitMatmulTile tile);
  ~LowbitMatmul();
  LowbitMatmul(MTL::Device *device, MTL::Library *library, std::size_t order,
               LowbitMatmulSignature signature, LowbitMatmulTile tile);

  LowbitMatmul(const LowbitMatmul &) = delete;
  LowbitMatmul &operator=(const LowbitMatmul &) = delete;

  [[nodiscard]] std::size_t order() const noexcept;
  [[nodiscard]] LowbitDataType data_type() const noexcept;
  [[nodiscard]] LowbitMatmulTile tile() const noexcept;
  [[nodiscard]] std::size_t tile_rows() const noexcept;
  [[nodiscard]] std::size_t tile_columns() const noexcept;
  [[nodiscard]] LowbitAccumulationType accumulation_type() const noexcept;
  [[nodiscard]] const LowbitMatmulOperandLayout &left_layout() const noexcept;
  [[nodiscard]] const LowbitMatmulOperandLayout &right_layout() const noexcept;
  [[nodiscard]] std::size_t output_bytes() const noexcept;

  [[nodiscard]] LowbitMatmulResult availability() const;

  // Keep buffers, residency set and resources alive through completion.
  [[nodiscard]] LowbitMatmulResult encode(
      MTL4::CommandBuffer *command, const LowbitMatmulBuffers &buffers,
      LowbitMatmulEncoderResources &resources) const;
  [[nodiscard]] LowbitMatmulResult encode(
      MTL4::ComputeCommandEncoder *encoder, const LowbitMatmulBuffers &buffers,
      LowbitMatmulEncoderResources &resources) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msqrt
