#include "metal_checks.hpp"

#include "metal_runtime.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core.hpp"
#include "numerics.hpp"
#include "storage_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace mixed_sqrt {
namespace {

void checkRawStorageKernels(MetalRuntime &metal) {
  const std::vector<Precision> precisions{
      Precision::Bf16,      Precision::Fp8E4M3Fn,   Precision::Fp8E5M2,
      Precision::Fp4E2M1,   Precision::Mxfp8E4M3Fn, Precision::Mxfp8E5M2,
      Precision::Mxfp4E2M1,
  };
  constexpr size_t order = 2;
  constexpr size_t matrix_elements = order * order;
  const std::vector<float> identity{1.0f, 0.0f, 0.0f, 1.0f};
  const std::vector<float> roots{1.0f, 1.0f};
  const std::vector<float> rhs_values(matrix_elements, 2.0f);
  for (Precision precision : precisions) {
    const PrecisionSpec &spec = precisionSpec(precision);
    const auto make_storage = [&](size_t elements) {
      return StorageBuffers{
          metal.makeSharedBuffer(storageBytes(spec, elements)),
          spec.requiresScalePlane()
              ? metal.makeSharedBuffer(scaleBytes(spec, elements))
              : nullptr,
      };
    };
    const StorageBuffers left = make_storage(matrix_elements);
    const StorageBuffers right = make_storage(matrix_elements);
    const StorageBuffers root_eigenvalues = make_storage(order);
    const StorageBuffers rhs_storage = make_storage(matrix_elements);
    const StorageBuffers output = make_storage(matrix_elements);
    MTL::Buffer *temporary_output = nullptr;
    if (spec.requiresQuantizer()) {
      temporary_output =
          metal.makeSharedBuffer(matrix_elements * sizeof(float));
    }
    const std::vector<MTL::Buffer *> buffers{
        left.values,
        left.scales,
        right.values,
        right.scales,
        root_eigenvalues.values,
        root_eigenvalues.scales,
        rhs_storage.values,
        rhs_storage.scales,
        output.values,
        output.scales,
        temporary_output,
    };

    copyToStorage(spec, identity, left.values, left.scales);
    copyToStorage(spec, identity, right.values, right.scales);
    copyToStorage(spec, roots, root_eigenvalues.values,
                  root_eigenvalues.scales);
    copyToStorage(spec, rhs_values, rhs_storage.values, rhs_storage.scales);
    std::vector<double> decoded(matrix_elements);

    metal.scaleColumns(spec, left, root_eigenvalues, output, temporary_output,
                       order);
    unpackFromStorage(spec, output.values, output.scales, decoded);
    requireCondition(decoded == std::vector<double>{1.0, 0.0, 0.0, 1.0},
                     "Raw scale kernel check failed.");

    metal.matmul(spec, left, right, output, temporary_output, order, order,
                 order);
    unpackFromStorage(spec, output.values, output.scales, decoded);
    requireCondition(decoded == std::vector<double>{1.0, 0.0, 0.0, 1.0},
                     "Raw matmul kernel check failed.");

    metal.diagonalSylvester(spec, rhs_storage, root_eigenvalues, output,
                            temporary_output, order);
    unpackFromStorage(spec, output.values, output.scales, decoded);
    requireCondition(decoded == std::vector<double>(matrix_elements, 1.0),
                     "Raw diagonal Sylvester kernel check failed.");
    MTL::Buffer *roots_f32 = metal.makeSharedBuffer(order * sizeof(float));
    MTL::Buffer *workspace_f32 =
        metal.makeSharedBuffer(matrix_elements * sizeof(float));
    MTL::Buffer *root_f32 =
        metal.makeSharedBuffer(matrix_elements * sizeof(float));
    MTL::Buffer *residual_f32 =
        metal.makeSharedBuffer(matrix_elements * sizeof(float));
    MTL::Buffer *inverse_f32 =
        metal.makeSharedBuffer(matrix_elements * sizeof(float));
    MTL::Buffer *coordinates_f32 =
        metal.makeSharedBuffer(matrix_elements * sizeof(float));
    MTL::Buffer *correction_f32 =
        metal.makeSharedBuffer(matrix_elements * sizeof(float));
    std::memcpy(roots_f32->contents(), roots.data(), order * sizeof(float));
    std::memcpy(residual_f32->contents(), rhs_values.data(),
                matrix_elements * sizeof(float));
    const std::vector<float> inverse_values(matrix_elements, 0.5f);
    std::memcpy(inverse_f32->contents(), inverse_values.data(),
                matrix_elements * sizeof(float));
    metal.reconstructLowFactorTiled(spec, left, roots_f32, workspace_f32,
                                    root_f32, order);
    const float *reconstructed =
        static_cast<const float *>(root_f32->contents());
    requireCondition(
        std::equal(identity.begin(), identity.end(), reconstructed),
        "Low-factor reconstruction kernel check failed.");
    metal.refineLowFactorTiled(spec, left, residual_f32, workspace_f32,
                               inverse_f32, coordinates_f32, correction_f32,
                               order);
    const float *correction_values =
        static_cast<const float *>(correction_f32->contents());
    requireCondition(std::all_of(correction_values,
                                 correction_values + matrix_elements,
                                 [](float value) { return value == 1.0f; }),
                     "Low-factor refinement kernel check failed.");
    releaseBuffers({roots_f32, workspace_f32, root_f32, residual_f32,
                    inverse_f32, coordinates_f32, correction_f32});
    releaseBuffers(buffers);
  }
}

void checkQuantizationEdgeBehavior(MetalRuntime &metal) {
  size_t failures = 0;
  const auto check = [&](bool condition, const char *format, const char *edge) {
    if (!condition) {
      ++failures;
      std::cout << "edge_check_format=" << format << " edge=" << edge
                << " status=failed\n";
    }
  };
  const auto is_signed_zero = [](float value, bool negative) {
    return value == 0.0f && std::signbit(value) == negative;
  };
  const auto is_underflow_neighbor = [&](float value, float minimum,
                                         bool negative) {
    return (value == 0.0f || std::fabs(value) == minimum) &&
           std::signbit(value) == negative;
  };
  {
    const std::vector<float> source{
        std::ldexp(1.0f, -133),
        std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max(),
    };
    MTL::Buffer *source_buffer =
        metal.makeSharedBuffer(source.size() * sizeof(float));
    MTL::Buffer *output_buffer =
        metal.makeSharedBuffer(source.size() * sizeof(FloatEdgeProbe));
    std::memcpy(source_buffer->contents(), source.data(),
                source.size() * sizeof(float));
    metal.probeF32Edges(source_buffer, output_buffer, source.size());
    const FloatEdgeProbe *probes =
        static_cast<const FloatEdgeProbe *>(output_buffer->contents());
    for (size_t index = 0; index < source.size(); ++index) {
      uint32_t source_bits = 0;
      std::memcpy(&source_bits, &source[index], sizeof(source_bits));
      check(probes[index].loaded_bits == source_bits, "f32", "raw_load_bits");
      float loaded = 0.0f;
      float bfloat_value = 0.0f;
      float logarithm = 0.0f;
      float round_trip = 0.0f;
      std::memcpy(&loaded, &probes[index].loaded_bits, sizeof(float));
      std::memcpy(&bfloat_value, &probes[index].bfloat_bits, sizeof(float));
      std::memcpy(&logarithm, &probes[index].log2_bits, sizeof(float));
      std::memcpy(&round_trip, &probes[index].exp2_log2_bits, sizeof(float));
      std::cout << "edge_probe_index=" << index << " source=" << source[index]
                << " loaded=" << loaded << " bfloat=" << bfloat_value
                << " log2=" << logarithm << " exp2_log2=" << round_trip << '\n';
    }
    releaseBuffers({source_buffer, output_buffer});
  }
  enum class InfinityBehavior : uint8_t {
    Preserve,
    Saturate,
  };
  struct ScalarEdgeSpec {
    Precision precision;
    float minimum;
    float maximum;
    InfinityBehavior infinity_behavior;
    bool preserves_nan;
  };
  const std::vector<ScalarEdgeSpec> scalar_specs{
      {Precision::Bf16, std::numeric_limits<float>::min(),
       std::ldexp(255.0f, 120), InfinityBehavior::Preserve, true},
      {Precision::Fp8E4M3Fn, 0.001953125f, 448.0f, InfinityBehavior::Saturate,
       true},
      {Precision::Fp8E5M2, 0.0000152587890625f, 57344.0f,
       InfinityBehavior::Preserve, true},
      {Precision::Fp4E2M1, 0.5f, 6.0f, InfinityBehavior::Saturate, false},
  };
  for (const ScalarEdgeSpec &edge_spec : scalar_specs) {
    const PrecisionSpec &spec = precisionSpec(edge_spec.precision);
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> source{
        0.0f,
        -0.0f,
        edge_spec.minimum,
        -edge_spec.minimum,
        0.25f * edge_spec.minimum,
        -0.25f * edge_spec.minimum,
        edge_spec.maximum,
        -edge_spec.maximum,
        std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        infinity,
        -infinity,
        nan,
    };
    const size_t bytes = source.size() * sizeof(float);
    MTL::Buffer *source_buffer = metal.makeSharedBuffer(bytes);
    MTL::Buffer *decoded_buffer = metal.makeSharedBuffer(bytes);
    const StorageBuffers encoded{
        metal.makeSharedBuffer(storageBytes(spec, source.size())),
        nullptr,
    };
    std::memcpy(source_buffer->contents(), source.data(), bytes);
    metal.quantizeAndDecode(spec, source_buffer, encoded, decoded_buffer,
                            static_cast<uint32_t>(source.size()), 0x45444745u);
    const float *decoded =
        static_cast<const float *>(decoded_buffer->contents());
    check(is_signed_zero(decoded[0], false), spec.cli_name, "positive_zero");
    check(is_signed_zero(decoded[1], true), spec.cli_name, "negative_zero");
    check(decoded[2] == edge_spec.minimum, spec.cli_name, "minimum_positive");
    check(decoded[3] == -edge_spec.minimum, spec.cli_name, "minimum_negative");
    check(is_underflow_neighbor(decoded[4], edge_spec.minimum, false),
          spec.cli_name, "positive_underflow");
    check(is_underflow_neighbor(decoded[5], edge_spec.minimum, true),
          spec.cli_name, "negative_underflow");
    check(decoded[6] == edge_spec.maximum, spec.cli_name, "maximum_positive");
    check(decoded[7] == -edge_spec.maximum, spec.cli_name, "maximum_negative");
    if (edge_spec.precision == Precision::Bf16) {
      check(decoded[8] == edge_spec.maximum || std::isinf(decoded[8]),
            spec.cli_name, "finite_overflow");
      check(decoded[9] == -edge_spec.maximum || std::isinf(decoded[9]),
            spec.cli_name, "negative_finite_overflow");
    } else {
      check(decoded[8] == edge_spec.maximum, spec.cli_name,
            "finite_saturation");
      check(decoded[9] == -edge_spec.maximum, spec.cli_name,
            "negative_finite_saturation");
    }
    if (edge_spec.infinity_behavior == InfinityBehavior::Preserve) {
      check(std::isinf(decoded[10]) && !std::signbit(decoded[10]),
            spec.cli_name, "positive_infinity");
      check(std::isinf(decoded[11]) && std::signbit(decoded[11]), spec.cli_name,
            "negative_infinity");
    } else {
      check(decoded[10] == edge_spec.maximum, spec.cli_name,
            "positive_infinity");
      check(decoded[11] == -edge_spec.maximum, spec.cli_name,
            "negative_infinity");
    }
    check(edge_spec.preserves_nan ? std::isnan(decoded[12])
                                  : decoded[12] == 0.0f,
          spec.cli_name, "nan");
    std::cout << "edge_check_format=" << spec.cli_name
              << " finite_overflow=" << decoded[8]
              << " positive_infinity=" << decoded[10]
              << " nan_class=" << (std::isnan(decoded[12]) ? "nan" : "finite")
              << '\n';
    releaseBuffers(
        {source_buffer, decoded_buffer, encoded.values, encoded.scales});
  }

  struct BlockEdgeSpec {
    Precision precision;
    int first_scale_exponent;
    float first_minimum;
    InfinityBehavior infinity_behavior;
    bool preserves_nan;
  };
  const std::vector<BlockEdgeSpec> block_specs{
      {Precision::Mxfp8E4M3Fn, -8, std::ldexp(1.0f, -17),
       InfinityBehavior::Saturate, true},
      {Precision::Mxfp8E5M2, -15, std::ldexp(1.0f, -31),
       InfinityBehavior::Preserve, true},
      {Precision::Mxfp4E2M1, -2, 0.125f, InfinityBehavior::Saturate, false},
  };
  for (const BlockEdgeSpec &edge_spec : block_specs) {
    const PrecisionSpec &spec = precisionSpec(edge_spec.precision);
    constexpr uint32_t elements = 64;
    std::vector<float> source(elements, 0.0f);
    source[0] = 1.0f;
    source[2] = -0.0f;
    source[3] = edge_spec.first_minimum;
    source[4] = 0.25f * edge_spec.first_minimum;
    source[5] = std::numeric_limits<float>::infinity();
    source[6] = std::numeric_limits<float>::quiet_NaN();
    source[32] = 8.0f;
    source[34] = -0.0f;
    source[35] = std::ldexp(edge_spec.first_minimum, 3);
    source[36] = std::ldexp(0.25f * edge_spec.first_minimum, 3);
    const size_t bytes = source.size() * sizeof(float);
    MTL::Buffer *source_buffer = metal.makeSharedBuffer(bytes);
    MTL::Buffer *decoded_buffer = metal.makeSharedBuffer(bytes);
    const StorageBuffers encoded{
        metal.makeSharedBuffer(storageBytes(spec, elements)),
        metal.makeSharedBuffer(scaleBytes(spec, elements)),
    };
    std::memcpy(source_buffer->contents(), source.data(), bytes);
    metal.quantizeAndDecode(spec, source_buffer, encoded, decoded_buffer,
                            elements, 0x424c4f43u);
    const float *decoded =
        static_cast<const float *>(decoded_buffer->contents());
    const uint8_t *scales =
        static_cast<const uint8_t *>(encoded.scales->contents());
    check(decoded[0] == 1.0f, spec.cli_name, "first_block_maximum");
    check(decoded[32] == 8.0f, spec.cli_name, "second_block_maximum");
    check(scales[0] ==
              static_cast<uint8_t>(edge_spec.first_scale_exponent + 127),
          spec.cli_name, "first_block_scale");
    check(scales[1] ==
              static_cast<uint8_t>(edge_spec.first_scale_exponent + 3 + 127),
          spec.cli_name, "second_block_scale");
    check(is_signed_zero(decoded[1], false), spec.cli_name,
          "block_positive_zero");
    check(is_signed_zero(decoded[2], true), spec.cli_name,
          "block_negative_zero");
    check(decoded[3] == edge_spec.first_minimum, spec.cli_name,
          "block_minimum");
    check(is_underflow_neighbor(decoded[4], edge_spec.first_minimum, false),
          spec.cli_name, "block_underflow");
    if (edge_spec.infinity_behavior == InfinityBehavior::Preserve) {
      check(std::isinf(decoded[5]), spec.cli_name, "block_infinity");
    } else {
      check(std::isfinite(decoded[5]) && decoded[5] > decoded[0], spec.cli_name,
            "block_infinity_saturation");
    }
    check(edge_spec.preserves_nan ? std::isnan(decoded[6]) : decoded[6] == 0.0f,
          spec.cli_name, "block_nan");
    std::cout << "edge_check_format=" << spec.cli_name
              << " first_scale_exponent=" << int(scales[0]) - 127
              << " second_scale_exponent=" << int(scales[1]) - 127
              << " positive_infinity=" << decoded[5]
              << " nan_class=" << (std::isnan(decoded[6]) ? "nan" : "finite")
              << '\n';
    releaseBuffers(
        {source_buffer, decoded_buffer, encoded.values, encoded.scales});
  }

  for (Precision precision :
       {Precision::Fp8E4M3FnLogS, Precision::Fp4E2M1LogS}) {
    const PrecisionSpec &spec = precisionSpec(precision);
    const std::vector<float> source{
        std::numeric_limits<float>::min(),
        std::ldexp(1.0f, -100),
        std::ldexp(1.0f, -64),
        std::ldexp(1.0f, -32),
        1.0f,
        std::ldexp(1.0f, 32),
        std::ldexp(1.0f, 64),
        std::numeric_limits<float>::max(),
        3.0f,
        3.0f,
        3.0f,
        3.0f,
        3.0f,
        3.0f,
        3.0f,
        3.0f,
    };
    const size_t bytes = source.size() * sizeof(float);
    MTL::Buffer *source_buffer = metal.makeSharedBuffer(bytes);
    MTL::Buffer *decoded_buffer = metal.makeSharedBuffer(bytes);
    const StorageBuffers encoded{
        metal.makeSharedBuffer(storageBytes(spec, source.size())),
        metal.makeSharedBuffer(spectrumMetadataBytes(spec, source.size())),
    };
    std::memcpy(source_buffer->contents(), source.data(), bytes);
    metal.quantizeAndDecode(spec, source_buffer, encoded, decoded_buffer,
                            static_cast<uint32_t>(source.size()), 0x4c4f4753u);
    const float *decoded =
        static_cast<const float *>(decoded_buffer->contents());
    check(decoded[0] > 0.0f && std::isfinite(decoded[0]), spec.cli_name,
          "log_smallest_positive");
    check(decoded[7] > 0.0f && std::isfinite(decoded[7]), spec.cli_name,
          "log_largest_finite");
    check(std::all_of(decoded + 8, decoded + 16,
                      [](float value) {
                        return std::isfinite(value) &&
                               std::fabs(value / 3.0f - 1.0f) <= 1.0e-6f;
                      }),
          spec.cli_name, "log_constant_block");
    std::cout << "edge_check_format=" << spec.cli_name
              << " smallest_positive=" << decoded[0]
              << " largest_finite=" << decoded[7]
              << " constant_block=" << decoded[8] << '\n';
    releaseBuffers(
        {source_buffer, decoded_buffer, encoded.values, encoded.scales});
  }
  requireCondition(failures == 0,
                   "One or more quantization edge checks failed.");
}

void checkStochasticQuantization(MetalRuntime &metal) {
  if (metal.roundingMode() != RoundingMode::Stochastic) {
    return;
  }
  constexpr uint32_t elements = 65536;
  constexpr size_t bytes = static_cast<size_t>(elements) * sizeof(float);
  MTL::Buffer *source_buffer = metal.makeSharedBuffer(bytes);
  MTL::Buffer *decoded_one = metal.makeSharedBuffer(bytes);
  MTL::Buffer *decoded_two = metal.makeSharedBuffer(bytes);
  std::vector<float> source(elements);
  const std::vector<Precision> precisions{
      Precision::Bf16,        Precision::Fp8E4M3Fn,     Precision::Fp8E5M2,
      Precision::Mxfp8E4M3Fn, Precision::Mxfp8E5M2,     Precision::Fp4E2M1,
      Precision::Mxfp4E2M1,   Precision::Fp8E4M3FnLogS, Precision::Fp4E2M1LogS,
  };
  for (Precision precision : precisions) {
    const PrecisionSpec &spec = precisionSpec(precision);
    if (spec.usesLogSpectrum()) {
      for (uint32_t index = 0; index < elements; ++index) {
        const float linear_position =
            float(index % spec.spectral_block_elements) /
            float(spec.spectral_block_elements - 1u);
        const float position = powf(linear_position, 1.3f);
        source[index] = exp2f(-2.0f + 4.0f * position);
      }
    } else {
      for (uint32_t index = 0; index < elements; ++index) {
        source[index] = index < elements / 2u ? 0.3f : -0.3f;
      }
    }
    std::memcpy(source_buffer->contents(), source.data(), bytes);
    const size_t metadata_bytes = spec.usesLogSpectrum()
                                      ? spectrumMetadataBytes(spec, elements)
                                      : scaleBytes(spec, elements);
    const StorageBuffers encoded{
        metal.makeSharedBuffer(storageBytes(spec, elements)),
        metadata_bytes == 0 ? nullptr : metal.makeSharedBuffer(metadata_bytes),
    };
    metal.quantizeAndDecode(spec, source_buffer, encoded, decoded_one, elements,
                            0x53520000u);
    std::vector<float> first(elements);
    std::memcpy(first.data(), decoded_one->contents(), bytes);
    metal.quantizeAndDecode(spec, source_buffer, encoded, decoded_two, elements,
                            0x53520000u);
    const auto *second = static_cast<const float *>(decoded_two->contents());
    const auto equal_bits = [](float left, float right) {
      uint32_t left_bits = 0;
      uint32_t right_bits = 0;
      std::memcpy(&left_bits, &left, sizeof(left));
      std::memcpy(&right_bits, &right, sizeof(right));
      return left_bits == right_bits;
    };
    requireCondition(std::equal(first.begin(), first.end(), second, equal_bits),
                     "Stochastic quantization is not reproducible.");
    metal.quantizeAndDecode(spec, source_buffer, encoded, decoded_two, elements,
                            0x53520001u);
    requireCondition(
        !std::equal(first.begin(), first.end(), second, equal_bits),
        "The stochastic stream does not change quantized values.");

    double mean_error = 0.0;
    double negative_mean_error = 0.0;
    if (spec.usesLogSpectrum()) {
      for (uint32_t index = 0; index < elements; ++index) {
        mean_error += std::log2(static_cast<double>(first[index])) -
                      std::log2(static_cast<double>(source[index]));
      }
      mean_error /= static_cast<double>(elements);
    } else {
      for (uint32_t index = 0; index < elements / 2u; ++index) {
        mean_error += static_cast<double>(first[index]) -
                      static_cast<double>(source[index]);
      }
      for (uint32_t index = elements / 2u; index < elements; ++index) {
        negative_mean_error += static_cast<double>(first[index]) -
                               static_cast<double>(source[index]);
      }
      mean_error /= static_cast<double>(elements) / 2.0;
      negative_mean_error /= static_cast<double>(elements) / 2.0;
    }
    const double maximum_mean_error =
        std::max(std::abs(mean_error), std::abs(negative_mean_error));
    const double tolerance = precision == Precision::Bf16 ? 5.0e-5
                             : (precision == Precision::Fp4E2M1 ||
                                precision == Precision::Mxfp4E2M1 ||
                                precision == Precision::Fp4E2M1LogS)
                                 ? 1.0e-2
                                 : 1.0e-3;
    requireCondition(maximum_mean_error <= tolerance,
                     "Stochastic quantization has a biased sample mean.");
    std::cout << "stochastic_rounding_format=" << spec.cli_name
              << " maximum_sample_mean_error=" << maximum_mean_error << '\n';
    releaseBuffers({encoded.values, encoded.scales});
  }
  releaseBuffers({source_buffer, decoded_one, decoded_two});
}

void checkLogSpectrumStorage(MetalRuntime &metal) {
  const std::vector<float> roots{
      1.0f, 1.5f, 2.0f, 3.0f, 10.0f, 31.622776f, 100.0f, 1000.0f,
  };
  for (Precision precision :
       {Precision::Fp8E4M3FnLogS, Precision::Fp4E2M1LogS}) {
    const PrecisionSpec &spec = precisionSpec(precision);
    const StorageBuffers spectrum{
        metal.makeSharedBuffer(storageBytes(spec, roots.size())),
        metal.makeSharedBuffer(spectrumMetadataBytes(spec, roots.size())),
    };
    copySpectrumToStorage(spec, roots, spectrum.values, spectrum.scales);
    double maximum_relative_error = 0.0;
    for (size_t index = 0; index < roots.size(); ++index) {
      const float decoded =
          spectrumValueAt(spec, spectrum.values, spectrum.scales, index);
      requireCondition(std::isfinite(decoded) && decoded > 0.0f,
                       "Log-spectrum decode check failed.");
      maximum_relative_error = std::max(

          maximum_relative_error,
          std::fabs(static_cast<double>(decoded) / roots[index] - 1.0));
    }
    const double error_bound =
        precision == Precision::Fp8E4M3FnLogS ? 0.5 : 4.0;
    requireCondition(maximum_relative_error < error_bound,
                     "Log-spectrum relative-error check failed.");
    releaseBuffers({spectrum.values, spectrum.scales});
  }
}

void checkTiledKernels(MetalRuntime &metal) {
  constexpr size_t order = 16;
  constexpr size_t matrix_elements = order * order;
  for (size_t batch_count : {size_t{1}, size_t{2}}) {
    const size_t elements = batch_count * matrix_elements;
    std::vector<float> identity(elements, 0.0f);
    std::vector<double> expected_root(elements, 0.0);
    for (size_t batch = 0; batch < batch_count; ++batch) {
      const size_t offset = batch * matrix_elements;
      for (size_t index = 0; index < order; ++index) {
        identity[offset + index + index * order] = 1.0f;
        expected_root[offset + index + index * order] = 1.0;
      }
    }
    const std::vector<float> roots(batch_count * order, 1.0f);
    const std::vector<double> rhs(elements, 2.0);
    for (Precision precision : {Precision::Fp32, Precision::Bf16}) {
      const PrecisionSpec &spec = precisionSpec(precision);
      const auto make_storage = [&](size_t count) {
        return StorageBuffers{
            metal.makeSharedBuffer(storageBytes(spec, count)),
            nullptr,
        };
      };
      const StorageBuffers q = make_storage(elements);
      const StorageBuffers qt = make_storage(elements);
      const StorageBuffers s = make_storage(batch_count * order);
      const StorageBuffers workspace_one = make_storage(elements);
      const StorageBuffers root = make_storage(elements);
      const StorageBuffers residual = make_storage(elements);
      const StorageBuffers coordinates = make_storage(elements);
      const StorageBuffers correction = make_storage(elements);
      MTL::Buffer *inverse = metal.makeSharedBuffer(elements * sizeof(float));
      std::fill_n(static_cast<float *>(inverse->contents()), elements, 0.5f);
      const std::vector<MTL::Buffer *> buffers{
          q.values,
          qt.values,
          s.values,
          workspace_one.values,
          root.values,
          residual.values,
          coordinates.values,
          correction.values,
          inverse,
      };
      copyToStorage(spec, identity, q.values, q.scales);
      copyToStorage(spec, identity, qt.values, qt.scales);
      copyToStorage(spec, roots, s.values, s.scales);
      metal.reconstructTiled(spec, q, qt, s, workspace_one, root, order,
                             static_cast<uint32_t>(batch_count));
      std::vector<double> decoded(elements);
      unpackFromStorage(spec, root.values, root.scales, decoded);
      requireCondition(decoded == expected_root,
                       "Tiled reconstruction kernel check failed.");
      copyResidualToStorage(spec, rhs, residual.values, residual.scales);
      metal.refineTiled(spec, q, qt, residual, workspace_one, inverse,
                        coordinates, correction, order,
                        static_cast<uint32_t>(batch_count));
      unpackFromStorage(spec, correction.values, correction.scales, decoded);
      requireCondition(decoded == std::vector<double>(elements, 1.0),
                       "Tiled correction kernel check failed.");
      releaseBuffers(buffers);
    }
  }
}

void checkNativeStagedCorrection(MetalRuntime &metal) {
  constexpr size_t order = 4;
  constexpr size_t matrix_elements = order * order;
  const std::vector<float> identity{
      1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
  };
  const std::vector<float> roots{1.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<double> residual{
      2.0,  -0.5,   0.25, -0.125, -0.5,   1.75, -0.375, 0.25,
      0.25, -0.375, 1.5,  -0.25,  -0.125, 0.25, -0.25,  1.25,
  };
  for (Precision precision :
       {Precision::Fp8E4M3Fn, Precision::Fp8E5M2, Precision::Mxfp8E4M3Fn,
        Precision::Mxfp8E5M2, Precision::Fp4E2M1, Precision::Mxfp4E2M1}) {
    const PrecisionSpec &spec = precisionSpec(precision);
    const auto make_storage = [&](size_t elements) {
      return StorageBuffers{
          metal.makeSharedBuffer(storageBytes(spec, elements)),
          spec.requiresScalePlane()
              ? metal.makeSharedBuffer(scaleBytes(spec, elements))
              : nullptr,
      };
    };
    const auto require_storage = [&](const StorageBuffers &storage,
                                     size_t elements) {
      requireCondition(
          storage.values != nullptr &&
              storage.values->length() == storageBytes(spec, elements),
          "Native staged correction used an incorrect value buffer format.");
      requireCondition(
          spec.requiresScalePlane()
              ? storage.scales != nullptr &&
                    storage.scales->length() == scaleBytes(spec, elements)
              : storage.scales == nullptr,
          "Native staged correction used an incorrect scale buffer format.");
    };
    const StorageBuffers q = make_storage(matrix_elements);
    const StorageBuffers qt = make_storage(matrix_elements);
    const StorageBuffers s = make_storage(order);
    const StorageBuffers r = make_storage(matrix_elements);
    const StorageBuffers u = make_storage(matrix_elements);
    const StorageBuffers c = make_storage(matrix_elements);
    const StorageBuffers y = make_storage(matrix_elements);
    const StorageBuffers v = make_storage(matrix_elements);
    const StorageBuffers delta = make_storage(matrix_elements);
    MTL::Buffer *temporary_output =
        spec.requiresQuantizer()
            ? metal.makeSharedBuffer(matrix_elements * sizeof(float))
            : nullptr;
    require_storage(q, matrix_elements);
    require_storage(qt, matrix_elements);
    require_storage(s, order);
    require_storage(r, matrix_elements);
    require_storage(u, matrix_elements);
    require_storage(c, matrix_elements);
    require_storage(y, matrix_elements);
    require_storage(v, matrix_elements);
    require_storage(delta, matrix_elements);
    requireCondition(
        (spec.requiresQuantizer() && temporary_output != nullptr) ||
            (!spec.requiresQuantizer() && temporary_output == nullptr),
        "Native staged correction used an incorrect transient quantizer "
        "buffer.");

    copyToStorage(spec, identity, q.values, q.scales);
    copyToStorage(spec, identity, qt.values, qt.scales);
    copyToStorage(spec, roots, s.values, s.scales);
    copyResidualToStorage(spec, residual, r.values, r.scales);
    metal.refineNativeStaged(spec, q, qt, s, r, u, c, y, delta,
                             temporary_output, order);
    std::vector<double> decoded_delta(matrix_elements);
    unpackFromStorage(spec, delta.values, delta.scales, decoded_delta);
    requireCondition(allFinite(decoded_delta),
                     "Native staged correction produced a nonfinite result.");

    const auto decode = [&](const StorageBuffers &storage, size_t elements) {
      std::vector<double> decoded(elements);
      unpackFromStorage(spec, storage.values, storage.scales, decoded);
      return decoded;
    };
    const auto quantize_host = [&](const std::vector<float> &values,
                                   const StorageBuffers &storage) {
      copyToStorage(spec, values, storage.values, storage.scales);
      return decode(storage, values.size());
    };
    const auto host_matmul = [&](const std::vector<double> &left,
                                 const std::vector<double> &right,
                                 const StorageBuffers &storage) {
      std::vector<float> values(matrix_elements, 0.0f);
      for (size_t column = 0; column < order; ++column) {
        for (size_t row = 0; row < order; ++row) {
          float sum = 0.0f;
          for (size_t inner = 0; inner < order; ++inner) {
            sum = std::fma(static_cast<float>(left[row + inner * order]),
                           static_cast<float>(right[inner + column * order]),
                           sum);
          }
          values[row + column * order] = sum;
        }
      }
      return quantize_host(values, storage);
    };
    const std::vector<double> expected_u =
        host_matmul(decode(qt, matrix_elements), decode(r, matrix_elements), u);
    const std::vector<double> expected_c =
        host_matmul(expected_u, decode(q, matrix_elements), c);
    std::vector<float> expected_y_values(matrix_elements);
    const std::vector<double> expected_s = decode(s, order);
    for (size_t column = 0; column < order; ++column) {
      for (size_t row = 0; row < order; ++row) {
        expected_y_values[row + column * order] =
            static_cast<float>(expected_c[row + column * order] /
                               (expected_s[row] + expected_s[column]));
      }
    }
    const std::vector<double> expected_y = quantize_host(expected_y_values, y);
    const std::vector<double> expected_v =
        host_matmul(decode(q, matrix_elements), expected_y, v);
    const std::vector<double> expected_delta =
        host_matmul(expected_v, decode(qt, matrix_elements), delta);
    requireCondition(
        decoded_delta == expected_delta,
        "Native staged correction disagrees with the independent host "
        "reference.");
    releaseBuffers({
        q.values, q.scales,     qt.values,    qt.scales,        s.values,
        s.scales, r.values,     r.scales,     u.values,         u.scales,
        c.values, c.scales,     y.values,     y.scales,         v.values,
        v.scales, delta.values, delta.scales, temporary_output,
    });
  }
}

} // namespace

void runMetalKernelChecks(MetalRuntime &metal) {
  checkRawStorageKernels(metal);
  checkLogSpectrumStorage(metal);
  checkQuantizationEdgeBehavior(metal);
  checkStochasticQuantization(metal);
  checkTiledKernels(metal);
  checkNativeStagedCorrection(metal);
  std::cout << "raw_kernel_checks=passed\n";
}

} // namespace mixed_sqrt
