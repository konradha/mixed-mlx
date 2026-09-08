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

// Storage orientation of one logical input matrix. ColumnMajor stores element
// (row, column) at row + column*leading_dimension. TransposedColumnMajor stores
// the logical transpose at column + row*leading_dimension.
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
  // Data planes. For MX formats the left plane follows left_layout().storage;
  // the right plane follows right_layout().storage.
  MTL::Buffer *left = nullptr;
  MTL::Buffer *right = nullptr;

  // Required only for MX formats. Each is a UE8M0 plane with one scale per
  // 32 contiguous data elements and the row stride reported by the layout.
  MTL::Buffer *left_scales = nullptr;
  MTL::Buffer *right_scales = nullptr;

  // Dense column-major FP32 output, with leading dimension equal to order().
  MTL::Buffer *output = nullptr;

  // MTL4 performs no implicit residency tracking. This set must contain every
  // non-null buffer above. The caller-owned encoder path requires its command
  // buffer to use this set before encoder creation.
  MTL::ResidencySet *residency_set = nullptr;
};

// A square product C=A*B with column-major logical matrices and FP32 output.
// FP16 and raw FP8 inputs use column-major storage for both operands. MPP
// requires a scaled right operand to be transposed; for MX formats the logical
// A data and scale planes therefore use TransposedColumnMajor storage, while B
// remains ColumnMajor. Query the layouts rather than assuming dense packing.
//
// Keeps transient tensor views and argument tables alive until a caller-owned
// encoder has ended.
class LowbitMatmulEncoderResources {
public:
  LowbitMatmulEncoderResources() = default;
  ~LowbitMatmulEncoderResources();
  LowbitMatmulEncoderResources(const LowbitMatmulEncoderResources &) = delete;
  LowbitMatmulEncoderResources &
  operator=(const LowbitMatmulEncoderResources &) = delete;

private:
  friend class LowbitMatmul;
  void retain(NS::Object *object);
  std::vector<NS::Object *> objects_;
};

// The library must contain the matching functions compiled as Metal 4.1. The
// command buffer must already be between beginCommandBuffer and
// endCommandBuffer. The command-buffer overload creates and ends one MTL4
// compute encoder; the encoder overload records into a caller-owned encoder.

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

  // Includes runtime creation of the requested pipeline and representative
  // tensors on this device. No GPU-family inference or FP32 fallback is used.
  [[nodiscard]] LowbitMatmulResult availability() const;

  // Records exactly one complete FP32 C=A*B operation. The caller retains all
  // buffers and the residency set through GPU completion.
  [[nodiscard]] LowbitMatmulResult
  encode(MTL4::CommandBuffer *command,
         const LowbitMatmulBuffers &buffers) const;
  [[nodiscard]] LowbitMatmulResult
  encode(MTL4::ComputeCommandEncoder *encoder,
         const LowbitMatmulBuffers &buffers,
         LowbitMatmulEncoderResources &resources) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace msqrt
