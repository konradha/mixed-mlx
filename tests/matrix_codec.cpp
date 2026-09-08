#include "../3_spectral/lowbit_matmul.hpp"
#include "../3_spectral/metal.hpp"
#include "../3_spectral/spectral.hpp"
#include "../common/buffers.hpp"

#include <Foundation/Foundation.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Format {
  msqrt::LowbitDataType type;
  const char *codec;
  std::size_t bits;
  bool mx;
};

constexpr Format formats[] = {
    {msqrt::LowbitDataType::Float16, "fp16", 16, false},
    {msqrt::LowbitDataType::Float8E4M3, "fp8_e4m3", 8, false},
    {msqrt::LowbitDataType::Float8E5M2, "fp8_e5m2", 8, false},
    {msqrt::LowbitDataType::MxFloat8E4M3, "mxfp8_e4m3", 8, true},
    {msqrt::LowbitDataType::MxFloat4E2M1, "mxfp4_e2m1", 4, true},
};

struct CodecParams {
  std::uint32_t elements;
  std::uint32_t blocks;
};

std::vector<float> source_matrix(std::size_t order) {
  constexpr float values[] = {
      0.0f,  -0.0f,  0.125f, 0.25f,          0.375f,         0.5f,    0.625f,
      0.75f, 0.875f, 1.0f,   1.00048828125f, 1.00146484375f, 1.0625f, 1.1875f,
      1.25f, 1.375f, 1.75f,  1.875f,         2.0f,           2.125f,  2.5f,
      2.75f, 3.0f,   3.5f,   4.0f,           4.5f,           5.0f,    5.5f,
      6.0f,  6.5f,   7.0f,   7.5f,
  };
  std::vector<float> source(order * order);
  for (std::size_t column = 1; column < order; ++column) {
    for (std::size_t row = 1; row < order; ++row) {
      const int exponent = static_cast<int>(row % 7) - 3 +
                           2 * static_cast<int>(column / 32) +
                           static_cast<int>(column % 3) - 1;
      float value = std::ldexp(values[(13 * row + 7 * column) % 32], exponent);
      if ((row + column) % 2)
        value = -value;
      source[row + column * order] = value;
    }
  }
  // Row and column zero exercise zero scales in both native orientations.
  const float boundaries[] = {
      -0.0f,
      std::ldexp(1.0f, -25),
      std::ldexp(3.0f, -25),
      std::ldexp(1.0f, -17),
      std::ldexp(3.0f, -17),
      std::ldexp(1.0f, -10),
      std::ldexp(3.0f, -10),
      448.0f,
      464.0f,
      -464.0f,
      57344.0f,
      61440.0f,
      65504.0f,
      65520.0f,
  };
  for (std::size_t i = 0; i < sizeof(boundaries) / sizeof(boundaries[0]); ++i)
    source[(i + 1) + (i + 2) * order] = boundaries[i];
  return source;
}

void vector_rows(msqrt::Metal &metal, MTL::ComputeCommandEncoder *encoder,
                 const Format &format, std::size_t order,
                 const msqrt::LowbitMatmulOperandLayout &layout,
                 MTL::Buffer *floats, MTL::Buffer *encoded, MTL::Buffer *scales,
                 bool decode) {
  const CodecParams params{static_cast<std::uint32_t>(order),
                           static_cast<std::uint32_t>(order / 32)};
  auto *pipeline = metal.pipeline(std::string(decode ? "decode_" : "encode_") +
                                  format.codec);
  const std::size_t float_row =
      layout.data_leading_dimension_elements * sizeof(float);
  const std::size_t data_row =
      layout.data_leading_dimension_elements * format.bits / 8;
  const std::size_t width = format.bits == 16 ? 1 : format.bits == 8 ? 4 : 8;
  encoder->setBytes(&params, sizeof(params), format.mx ? 3 : 2);
  for (std::size_t row = 0; row < order; ++row) {
    if (decode) {
      encoder->setBuffer(encoded, row * data_row, 0);
      encoder->setBuffer(floats, row * float_row, format.mx ? 2 : 1);
      if (format.mx)
        encoder->setBuffer(scales, row * layout.scale_leading_dimension_bytes,
                           1);
    } else {
      encoder->setBuffer(floats, row * float_row, 0);
      encoder->setBuffer(encoded, row * data_row, 1);
      if (format.mx)
        encoder->setBuffer(scales, row * layout.scale_leading_dimension_bytes,
                           2);
    }
    msqrt::Metal::dispatch1d(encoder, pipeline,
                             format.mx && !decode ? order : order / width,
                             format.mx && !decode ? 32 : 256);
  }
}

void encode_matrix(msqrt::Metal &metal, MTL::ComputeCommandEncoder *encoder,
                   const Format &format, std::size_t order,
                   const msqrt::LowbitMatmulOperandLayout &layout,
                   MTL::Buffer *input, MTL::Buffer *encoded,
                   MTL::Buffer *scales) {
  const msqrt::MatrixCodecParams params{
      static_cast<std::uint32_t>(order),
      static_cast<std::uint32_t>(layout.data_leading_dimension_elements),
      static_cast<std::uint32_t>(layout.scale_leading_dimension_bytes),
      static_cast<std::uint32_t>(
          layout.storage == msqrt::LowbitMatrixStorage::TransposedColumnMajor)};
  auto *pipeline = metal.pipeline(std::string("encode_matrix_") + format.codec);
  encoder->setBuffer(input, 0, 0);
  encoder->setBuffer(encoded, 0, 1);
  if (format.mx)
    encoder->setBuffer(scales, 0, 2);
  encoder->setBytes(&params, sizeof(params), format.mx ? 3 : 2);
  const std::size_t threads = format.mx
                                  ? order * order
                                  : layout.data_leading_dimension_elements *
                                        order / (format.bits == 16 ? 1 : 4);
  msqrt::Metal::dispatch1d(encoder, pipeline, threads, format.mx ? 32 : 256);
}

void check_operand(msqrt::Metal &metal, const Format &format, std::size_t order,
                   const std::vector<float> &source,
                   const msqrt::LowbitMatmulOperandLayout &layout,
                   const char *operand) {
  const std::size_t leading = layout.data_leading_dimension_elements;
  const std::size_t float_bytes = leading * order * sizeof(float);
  const bool row_major =
      layout.storage == msqrt::LowbitMatrixStorage::TransposedColumnMajor;
  std::vector<float> physical(leading * order, 123456.0f);
  for (std::size_t column = 0; column < order; ++column)
    for (std::size_t row = 0; row < order; ++row)
      physical[row_major ? column + row * leading : row + column * leading] =
          source[row + column * order];

  msqrt::common::BufferGuard<msqrt::Metal> owned(metal);
  const auto make = [&](std::size_t bytes, int fill) {
    auto *buffer = owned.make(bytes);
    std::memset(buffer->contents(), fill, bytes);
    return buffer;
  };
  auto *input = owned.make(source.size() * sizeof(float));
  auto *reference_input = owned.make(float_bytes);
  std::memcpy(input->contents(), source.data(), source.size() * sizeof(float));
  std::memcpy(reference_input->contents(), physical.data(), float_bytes);
  auto *encoded = make(layout.data_bytes, 0xa5);
  auto *reference_encoded = make(layout.data_bytes, 0x5a);
  auto *scales = format.mx ? make(layout.scale_bytes, 0xa5) : nullptr;
  auto *reference_scales = format.mx ? make(layout.scale_bytes, 0x5a) : nullptr;
  auto *decoded = make(float_bytes, 0xa5);
  auto *reference_decoded = make(float_bytes, 0x5a);
  metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
    encode_matrix(metal, encoder, format, order, layout, input, encoded,
                  scales);
    vector_rows(metal, encoder, format, order, layout, reference_input,
                reference_encoded, reference_scales, false);
    vector_rows(metal, encoder, format, order, layout, decoded, encoded, scales,
                true);
    vector_rows(metal, encoder, format, order, layout, reference_decoded,
                reference_encoded, reference_scales, true);
  });

  const std::string label = std::string(format.codec) +
                            " order=" + std::to_string(order) + " " + operand;
  const auto *actual = static_cast<const float *>(decoded->contents());
  const auto *expected =
      static_cast<const float *>(reference_decoded->contents());
  for (std::size_t column = 0; column < order; ++column) {
    for (std::size_t row = 0; row < order; ++row) {
      const std::size_t index =
          row_major ? column + row * leading : row + column * leading;
      if (std::memcmp(actual + index, expected + index, sizeof(float)) != 0)
        throw std::runtime_error(label + " decoded mismatch at (" +
                                 std::to_string(row) + "," +
                                 std::to_string(column) + "): expected " +
                                 std::to_string(expected[index]) + ", got " +
                                 std::to_string(actual[index]));
    }
  }
  if (format.mx) {
    const auto *actual_scales =
        static_cast<const std::uint8_t *>(scales->contents());
    const auto *expected_scales =
        static_cast<const std::uint8_t *>(reference_scales->contents());
    for (std::size_t row = 0; row < order; ++row) {
      for (std::size_t block = 0; block < order / 32; ++block) {
        const std::size_t index =
            row * layout.scale_leading_dimension_bytes + block;
        if (actual_scales[index] != expected_scales[index])
          throw std::runtime_error(
              label + " scale mismatch at physical row " + std::to_string(row) +
              ", block " + std::to_string(block) + ": expected " +
              std::to_string(expected_scales[index]) + ", got " +
              std::to_string(actual_scales[index]));
      }
    }
  }
}

std::size_t check_products(msqrt::Metal &metal, std::size_t order) {
  const std::size_t count = order * order;
  std::vector<float> left(count), right(count), expected(count);
  for (std::size_t column = 0; column < order; ++column)
    for (std::size_t row = 0; row < order; ++row)
      right[row + column * order] =
          0.25f *
          (static_cast<int>(
               (13 * row + 7 * column + row / 32 + 3 * (column / 32)) % 9) -
           4);
  // A signed permutation selects one exact entry of B per output element.
  // The nonsymmetric pattern also changes across padded 32-element blocks.
  for (std::size_t row = 0; row < order; ++row) {
    const std::size_t selected = (5 * row + 7) % order;
    const float sign = row % 3 == 0 ? -1.0f : 1.0f;
    left[row + selected * order] = sign;
    for (std::size_t column = 0; column < order; ++column)
      expected[row + column * order] = sign * right[selected + column * order];
  }
  msqrt::common::BufferGuard<msqrt::Metal> inputs(metal);
  auto *left_source = inputs.make(count * sizeof(float));
  auto *right_source = inputs.make(count * sizeof(float));
  std::memcpy(left_source->contents(), left.data(), count * sizeof(float));
  std::memcpy(right_source->contents(), right.data(), count * sizeof(float));

  const auto check = [&](msqrt::LowbitMatmul &matmul, const Format *left_format,
                         const Format *right_format, const char *name) {
    const std::string label =
        std::string(name) + " order=" + std::to_string(order);
    const auto available = matmul.availability();
    if (available.status != msqrt::LowbitMatmulStatus::Available)
      throw std::runtime_error(label + ": " + available.message);
    const auto &left_layout = matmul.left_layout();
    const auto &right_layout = matmul.right_layout();
    msqrt::common::BufferGuard<msqrt::Metal> owned(metal);
    const auto make = [&](std::size_t bytes) {
      auto *buffer = owned.make(bytes);
      std::memset(buffer->contents(), 0, bytes);
      return buffer;
    };
    // A null format selects the original dense FP32 operand.
    auto *a = left_format ? make(left_layout.data_bytes) : left_source;
    auto *b = right_format ? make(right_layout.data_bytes) : right_source;
    auto *as =
        left_layout.has_scales() ? make(left_layout.scale_bytes) : nullptr;
    auto *bs =
        right_layout.has_scales() ? make(right_layout.scale_bytes) : nullptr;
    auto *output = make(matmul.output_bytes());
    metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
      if (left_format)
        encode_matrix(metal, encoder, *left_format, order, left_layout,
                      left_source, a, as);
      if (right_format)
        encode_matrix(metal, encoder, *right_format, order, right_layout,
                      right_source, b, bs);
    });
    std::vector<MTL::Buffer *> resident_buffers{a, b, output};
    if (as)
      resident_buffers.push_back(as);
    if (bs)
      resident_buffers.push_back(bs);
    auto release = [](MTL::ResidencySet *set) { set->release(); };
    std::unique_ptr<MTL::ResidencySet, decltype(release)> residency(
        metal.residency_set(resident_buffers), release);
    const msqrt::LowbitMatmulBuffers buffers{a,  b,      as,
                                             bs, output, residency.get()};
    msqrt::LowbitMatmulEncoderResources resources;
    metal.submit_mtl4([&](MTL4::CommandBuffer *command) {
      const auto encoded = matmul.encode(command, buffers, resources);
      if (!encoded.encoded())
        throw std::runtime_error(label + ": " + encoded.message);
    });
    const auto *actual = static_cast<const float *>(output->contents());
    for (std::size_t index = 0; index < count; ++index)
      if (actual[index] != expected[index])
        throw std::runtime_error(
            label + " product mismatch at (" + std::to_string(index % order) +
            "," + std::to_string(index / order) + "): expected " +
            std::to_string(expected[index]) + ", got " +
            std::to_string(actual[index]));
  };

  constexpr auto tile = msqrt::LowbitMatmulTile::Rows32Columns32;
  std::size_t checks = 0;
  for (const auto &format : formats) {
    msqrt::LowbitMatmul matmul(metal.device(), metal.library(), order,
                               format.type, tile);
    check(matmul, &format, &format, format.codec);
    ++checks;
  }
  for (const auto signature :
       {msqrt::LowbitMatmulSignature::Float16Float32ToFloat32,
        msqrt::LowbitMatmulSignature::Float32Float16ToFloat32}) {
    msqrt::LowbitMatmul matmul(metal.device(), metal.library(), order,
                               signature, tile);
    const bool left_fp32 =
        signature == msqrt::LowbitMatmulSignature::Float32Float16ToFloat32;
    check(matmul, left_fp32 ? nullptr : &formats[0],
          left_fp32 ? &formats[0] : nullptr,
          left_fp32 ? "fp32_fp16" : "fp16_fp32");
    ++checks;
  }
  return checks;
}

void check_spectral(msqrt::Metal &metal) {
  const auto problem = msqrt::controlled_problem(96, 100.0);
  constexpr double tolerance = 1.0e-12;
  const msqrt::PrecisionPlan plans[] = {
      {"basis_fp16", msqrt::FactorFormat::Fp16, msqrt::FactorFormat::Fp32},
      {"basis_mxfp8", msqrt::FactorFormat::MxFp8E4M3,
       msqrt::FactorFormat::Fp32}};
  for (const auto &plan : plans) {
    const auto result =
        msqrt::run_spectral(metal, problem, plan, 30, tolerance);
    if (!result.accepted || !result.converged ||
        !std::isfinite(result.final_residual) ||
        result.final_residual > tolerance)
      throw std::runtime_error(plan.name +
                               " did not reach the spectral residual target: " +
                               std::to_string(result.final_residual));
    // An adaptive fallback must not hide a zero native initial product.
    if (!(result.initial_error < 1.0))
      throw std::runtime_error(plan.name +
                               " produced an invalid initial spectral root.");
  }
}

} // namespace

int main() {
  auto *pool = NS::AutoreleasePool::alloc()->init();
  int result = 0;
  try {
    msqrt::Metal metal("build/3_spectral/kernels.metallib");
    std::size_t checks = 0;
    for (std::size_t order : {32u, 96u, 128u}) {
      const auto source = source_matrix(order);
      for (const auto &format : formats) {
        msqrt::LowbitMatmul matmul(metal.device(), metal.library(), order,
                                   format.type,
                                   msqrt::LowbitMatmulTile::Rows32Columns32);
        const auto available = matmul.availability();
        if (available.status != msqrt::LowbitMatmulStatus::Available)
          throw std::runtime_error(std::string(format.codec) +
                                   " order=" + std::to_string(order) + ": " +
                                   available.message);
        check_operand(metal, format, order, source, matmul.left_layout(),
                      "left");
        ++checks;
        if (format.mx) {
          check_operand(metal, format, order, source, matmul.right_layout(),
                        "right");
          ++checks;
        }
      }
    }
    std::cout << "Matrix codec checks passed: " << checks
              << " native operands on " << metal.device_name() << '\n';
    std::size_t product_checks = 0;
    for (std::size_t order : {32u, 96u})
      product_checks += check_products(metal, order);
    std::cout << "Native product checks passed: " << product_checks << '\n';
    check_spectral(metal);
    std::cout << "Spectral convergence checks passed: FP16 and MXFP8 bases\n";
  } catch (const std::exception &error) {
    std::cerr << "Matrix numerical check failed: " << error.what() << '\n';
    result = 1;
  }
  pool->release();
  return result;
}
