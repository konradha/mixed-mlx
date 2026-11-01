#include "storage_codecs.hpp"

#include "core.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace mixed_sqrt {

uint16_t floatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  uint32_t rounded = bits >> 16;
  const uint32_t remainder = bits & 0xffffu;
  if (remainder > 0x8000u || (remainder == 0x8000u && (rounded & 1u) != 0)) {
    ++rounded;
  }
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
    rounded |= 0x0040u;
  }
  return static_cast<uint16_t>(rounded);
}

float bf16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float fp8E4M3FnToFloat(uint8_t value) {
  const float sign = (value & 0x80u) == 0 ? 1.0f : -1.0f;
  const uint8_t exponent = (value >> 3) & 0x0fu;
  const uint8_t fraction = value & 0x07u;
  if (exponent == 0x0fu && fraction == 0x07u) {
    return std::copysign(std::numeric_limits<float>::quiet_NaN(), sign);
  }
  const float magnitude =
      exponent == 0 ? std::ldexp(static_cast<float>(fraction), -9)
                    : std::ldexp(1.0f + static_cast<float>(fraction) / 8.0f,
                                 exponent - 7);
  return std::copysign(magnitude, sign);
}

float fp8E5M2ToFloat(uint8_t value) {
  const float sign = (value & 0x80u) == 0 ? 1.0f : -1.0f;
  const uint8_t exponent = (value >> 2) & 0x1fu;
  const uint8_t fraction = value & 0x03u;
  if (exponent == 0x1fu) {
    if (fraction == 0) {
      return std::copysign(std::numeric_limits<float>::infinity(), sign);
    }
    return std::copysign(std::numeric_limits<float>::quiet_NaN(), sign);
  }
  const float magnitude =
      exponent == 0 ? std::ldexp(static_cast<float>(fraction), -16)
                    : std::ldexp(1.0f + static_cast<float>(fraction) / 4.0f,
                                 exponent - 15);
  return std::copysign(magnitude, sign);
}

float fp4E2M1ToFloat(uint8_t value) {
  const float sign = (value & 0x08u) == 0 ? 1.0f : -1.0f;
  const uint8_t exponent = (value >> 1) & 0x03u;
  const uint8_t fraction = value & 0x01u;
  const float magnitude =
      exponent == 0 ? static_cast<float>(fraction) * 0.5f
                    : std::ldexp(1.0f + static_cast<float>(fraction) * 0.5f,
                                 exponent - 1);
  return std::copysign(magnitude, sign);
}

uint8_t nearestFiniteCode(float magnitude, uint8_t maximum_code,
                          float (*decode)(uint8_t)) {
  uint8_t nearest = 0;
  float nearest_distance = std::fabs(magnitude - decode(nearest));
  for (uint16_t code = 1; code <= maximum_code; ++code) {
    const uint8_t candidate = static_cast<uint8_t>(code);
    const float distance = std::fabs(magnitude - decode(candidate));
    if (distance < nearest_distance ||
        (distance == nearest_distance && (candidate & 1u) == 0 &&
         (nearest & 1u) != 0)) {
      nearest = candidate;
      nearest_distance = distance;
    }
  }
  return nearest;
}

uint8_t floatToFp8E4M3Fn(float value) {
  const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
  if (std::isnan(value)) {
    return sign | 0x7fu;
  }
  if (value == 0.0f) {
    return sign;
  }
  if (std::isinf(value) || std::fabs(value) > 448.0f) {
    return sign | 0x7eu;
  }
  return sign | nearestFiniteCode(std::fabs(value), 0x7eu, fp8E4M3FnToFloat);
}

uint8_t floatToFp8E5M2(float value) {
  const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
  if (std::isnan(value)) {
    return sign | 0x7eu;
  }
  if (value == 0.0f) {
    return sign;
  }
  if (std::isinf(value)) {
    return sign | 0x7cu;
  }
  if (std::fabs(value) > 57344.0f) {
    return sign | 0x7bu;
  }
  return sign | nearestFiniteCode(std::fabs(value), 0x7bu, fp8E5M2ToFloat);
}

uint8_t floatToFp4E2M1(float value) {
  const uint8_t sign = std::signbit(value) ? 0x08u : 0u;
  if (std::isnan(value) || value == 0.0f) {
    return sign;
  }
  if (std::isinf(value) || std::fabs(value) > 6.0f) {
    return sign | 0x07u;
  }
  return sign | nearestFiniteCode(std::fabs(value), 0x07u, fp4E2M1ToFloat);
}

template <typename ValueAt>
void packFp4(size_t elements, ValueAt &&value_at, uint8_t *destination) {
  for (size_t index = 0; index < elements; index += 2) {
    const uint8_t lower = floatToFp4E2M1(value_at(index));
    const uint8_t upper =
        index + 1 < elements ? floatToFp4E2M1(value_at(index + 1)) : 0u;
    destination[index / 2] = lower | static_cast<uint8_t>(upper << 4);
  }
}

template <typename ValueAt>
int mxScaleExponent(ValueAt &value_at, size_t block_start, size_t block_end,
                    int element_maximum_exponent) {
  float maximum = 0.0f;
  for (size_t index = block_start; index < block_end; ++index) {
    const float value = value_at(index);
    if (std::isfinite(value)) {
      maximum = std::max(maximum, std::fabs(value));
    }
  }
  if (maximum == 0.0f) {
    return 0;
  }
  return std::clamp(std::ilogb(maximum) - element_maximum_exponent, -127, 128);
}

template <typename ValueAt>
void packMxfp8(size_t elements, ValueAt &&value_at, uint8_t *destination,
               uint8_t *scales, int element_maximum_exponent,
               uint8_t (*encode)(float)) {
  for (size_t block_start = 0, block = 0; block_start < elements;
       block_start += 32, ++block) {
    const size_t block_end = std::min(elements, block_start + size_t{32});
    const int exponent = mxScaleExponent(value_at, block_start, block_end,
                                         element_maximum_exponent);
    scales[block] = static_cast<uint8_t>(exponent + 127);
    const float scale = std::ldexp(1.0f, exponent);
    for (size_t index = block_start; index < block_end; ++index) {
      destination[index] = encode(value_at(index) / scale);
    }
  }
}

template <typename ValueAt>
void packMxfp4(size_t elements, ValueAt &&value_at, uint8_t *destination,
               uint8_t *scales) {
  for (size_t block_start = 0, block = 0; block_start < elements;
       block_start += 32, ++block) {
    const size_t block_end = std::min(elements, block_start + size_t{32});
    const int exponent = mxScaleExponent(value_at, block_start, block_end, 2);
    scales[block] = static_cast<uint8_t>(exponent + 127);
    const float scale = std::ldexp(1.0f, exponent);
    for (size_t index = block_start; index < block_end; index += 2) {
      const uint8_t lower = floatToFp4E2M1(value_at(index) / scale);
      const uint8_t upper = index + 1 < block_end
                                ? floatToFp4E2M1(value_at(index + 1) / scale)
                                : 0u;
      destination[index / 2] = lower | static_cast<uint8_t>(upper << 4);
    }
  }
}

void checkLowPrecisionCodecs() {
  requireCondition(floatToFp8E4M3Fn(448.0f) == 0x7eu &&
                       std::isnan(fp8E4M3FnToFloat(floatToFp8E4M3Fn(
                           std::numeric_limits<float>::quiet_NaN()))),
                   "FP8 E4M3FN codec check failed.");
  requireCondition(
      floatToFp8E5M2(std::numeric_limits<float>::infinity()) == 0x7cu &&
          std::isinf(fp8E5M2ToFloat(0x7cu)),
      "FP8 E5M2 codec check failed.");
  requireCondition(
      floatToFp8E4M3Fn(1.0625f) == 0x38u && floatToFp8E4M3Fn(1.1875f) == 0x3au,
      "FP8 round-to-nearest-even check failed.");

  std::vector<float> fp4_values(33, 0.0f);
  fp4_values.front() = 6.1f;
  std::vector<uint8_t> fp4_storage(
      storageBytes(precisionSpec(Precision::Mxfp4E2M1), 33));
  std::vector<uint8_t> fp4_scales(
      scaleBytes(precisionSpec(Precision::Mxfp4E2M1), 33));
  auto fp4_value_at = [&](size_t index) { return fp4_values[index]; };
  packMxfp4(33, fp4_value_at, fp4_storage.data(), fp4_scales.data());
  requireCondition(fp4_scales.size() == 2 && fp4_scales[0] == 127u &&
                       fp4_scales[1] == 127u &&
                       (fp4_storage[0] & 0x0fu) == 0x07u,
                   "MXFP4 block scale check failed.");

  std::vector<float> e4m3_values(33, 0.0f);
  e4m3_values.front() = 512.0f;
  std::vector<uint8_t> e4m3_storage(33);
  std::vector<uint8_t> e4m3_scales(2);
  auto e4m3_value_at = [&](size_t index) { return e4m3_values[index]; };
  packMxfp8(33, e4m3_value_at, e4m3_storage.data(), e4m3_scales.data(), 8,
            floatToFp8E4M3Fn);
  requireCondition(
      e4m3_scales[0] == 128u && e4m3_scales[1] == 127u &&
          std::ldexp(fp8E4M3FnToFloat(e4m3_storage[0]), 1) == 512.0f,
      "MXFP8 E4M3FN block scale check failed.");

  std::vector<float> e5m2_values(32, 1.0f);
  std::vector<uint8_t> e5m2_storage(32);
  std::vector<uint8_t> e5m2_scales(1);
  auto e5m2_value_at = [&](size_t index) { return e5m2_values[index]; };
  packMxfp8(32, e5m2_value_at, e5m2_storage.data(), e5m2_scales.data(), 15,
            floatToFp8E5M2);
  requireCondition(e5m2_scales[0] == 112u &&
                       std::ldexp(fp8E5M2ToFloat(e5m2_storage[0]), -15) == 1.0f,
                   "MXFP8 E5M2 block scale check failed.");
}

template <typename ValueAt>
void copyValuesToStorage(const PrecisionSpec &precision, size_t elements,
                         ValueAt &&value_at, MTL::Buffer *destination,
                         MTL::Buffer *scale_destination) {
  switch (precision.host_storage) {
    case HostStorageCodec::Fp32: {
      float *values = static_cast<float *>(destination->contents());
      for (size_t index = 0; index < elements; ++index) {
        values[index] = value_at(index);
      }
      return;
    }
    case HostStorageCodec::Bf16: {
      uint16_t *values = static_cast<uint16_t *>(destination->contents());
      for (size_t index = 0; index < elements; ++index) {
        values[index] = floatToBf16(value_at(index));
      }
      return;
    }
    case HostStorageCodec::Fp8E4M3Fn: {
      uint8_t *values = static_cast<uint8_t *>(destination->contents());
      for (size_t index = 0; index < elements; ++index) {
        values[index] = floatToFp8E4M3Fn(value_at(index));
      }
      return;
    }
    case HostStorageCodec::Fp8E5M2: {
      uint8_t *values = static_cast<uint8_t *>(destination->contents());
      for (size_t index = 0; index < elements; ++index) {
        values[index] = floatToFp8E5M2(value_at(index));
      }
      return;
    }
    case HostStorageCodec::Mxfp8E4M3Fn:
      if (scale_destination == nullptr) {
        fail("MXFP8 E4M3FN storage requires a scale plane.");
      }
      packMxfp8(elements, std::forward<ValueAt>(value_at),
                static_cast<uint8_t *>(destination->contents()),
                static_cast<uint8_t *>(scale_destination->contents()), 8,
                floatToFp8E4M3Fn);
      return;
    case HostStorageCodec::Mxfp8E5M2:
      if (scale_destination == nullptr) {
        fail("MXFP8 E5M2 storage requires a scale plane.");
      }
      packMxfp8(elements, std::forward<ValueAt>(value_at),
                static_cast<uint8_t *>(destination->contents()),
                static_cast<uint8_t *>(scale_destination->contents()), 15,
                floatToFp8E5M2);
      return;
    case HostStorageCodec::Fp4E2M1:
      packFp4(elements, std::forward<ValueAt>(value_at),
              static_cast<uint8_t *>(destination->contents()));
      return;
    case HostStorageCodec::Mxfp4E2M1:
      if (scale_destination == nullptr) {
        fail("MXFP4 storage requires a scale plane.");
      }
      packMxfp4(elements, std::forward<ValueAt>(value_at),
                static_cast<uint8_t *>(destination->contents()),
                static_cast<uint8_t *>(scale_destination->contents()));
      return;
  }
}

void copyToStorage(const PrecisionSpec &precision,
                   const std::vector<float> &source, MTL::Buffer *destination,
                   MTL::Buffer *scale_destination) {
  copyValuesToStorage(
      precision, source.size(), [&](size_t index) { return source[index]; },
      destination, scale_destination);
}

float storageValueAt(const PrecisionSpec &precision, MTL::Buffer *source,
                     MTL::Buffer *scale_source, size_t index) {
  switch (precision.host_storage) {
    case HostStorageCodec::Fp32:
      return static_cast<const float *>(source->contents())[index];
    case HostStorageCodec::Bf16:
      return bf16ToFloat(
          static_cast<const uint16_t *>(source->contents())[index]);
    case HostStorageCodec::Fp8E4M3Fn:
      return fp8E4M3FnToFloat(
          static_cast<const uint8_t *>(source->contents())[index]);
    case HostStorageCodec::Fp8E5M2:
      return fp8E5M2ToFloat(
          static_cast<const uint8_t *>(source->contents())[index]);
    case HostStorageCodec::Mxfp8E4M3Fn: {
      if (scale_source == nullptr) {
        fail("MXFP8 E4M3FN storage requires a scale plane.");
      }
      const uint8_t scale =
          static_cast<const uint8_t *>(scale_source->contents())[index / 32];
      return std::ldexp(fp8E4M3FnToFloat(static_cast<const uint8_t *>(
                            source->contents())[index]),
                        static_cast<int>(scale) - 127);
    }
    case HostStorageCodec::Mxfp8E5M2: {
      if (scale_source == nullptr) {
        fail("MXFP8 E5M2 storage requires a scale plane.");
      }
      const uint8_t scale =
          static_cast<const uint8_t *>(scale_source->contents())[index / 32];
      return std::ldexp(fp8E5M2ToFloat(static_cast<const uint8_t *>(
                            source->contents())[index]),
                        static_cast<int>(scale) - 127);
    }
    case HostStorageCodec::Fp4E2M1: {
      const uint8_t packed =
          static_cast<const uint8_t *>(source->contents())[index / 2];
      return fp4E2M1ToFloat((packed >> ((index & 1u) * 4)) & 0x0fu);
    }
    case HostStorageCodec::Mxfp4E2M1: {
      if (scale_source == nullptr) {
        fail("MXFP4 storage requires a scale plane.");
      }
      const uint8_t packed =
          static_cast<const uint8_t *>(source->contents())[index / 2];
      const uint8_t value = (packed >> ((index & 1u) * 4)) & 0x0fu;
      const uint8_t scale =
          static_cast<const uint8_t *>(scale_source->contents())[index / 32];
      return std::ldexp(fp4E2M1ToFloat(value), static_cast<int>(scale) - 127);
    }
  }
}

float maximumFiniteStorageValue(const PrecisionSpec &precision) {
  switch (precision.host_storage) {
    case HostStorageCodec::Fp8E4M3Fn:
      return 448.0f;
    case HostStorageCodec::Fp4E2M1:
      return 6.0f;
    default:
      fail("Log-spectrum storage requires a finite scalar FP8 or FP4 codec.");
  }
}

size_t spectrumMetadataBytes(const PrecisionSpec &precision, size_t elements) {
  if (!precision.usesLogSpectrum()) {
    return scaleBytes(precision, elements);
  }
  if (precision.spectral_block_elements == 0) {
    fail("Log-spectrum storage requires a nonzero block size.");
  }
  const size_t blocks =
      elements / precision.spectral_block_elements +
      (elements % precision.spectral_block_elements != 0 ? 1 : 0);
  return 2 * blocks * sizeof(float);
}

void copySpectrumToStorage(const PrecisionSpec &precision,
                           const std::vector<float> &source,
                           MTL::Buffer *destination,
                           MTL::Buffer *metadata_destination) {
  if (!precision.usesLogSpectrum()) {
    copyToStorage(precision, source, destination, metadata_destination);
    return;
  }
  if (precision.requiresScalePlane() || metadata_destination == nullptr) {
    fail("Log-spectrum storage requires scalar values and metadata.");
  }
  const float maximum = maximumFiniteStorageValue(precision);
  const size_t block_elements = precision.spectral_block_elements;
  float *metadata = static_cast<float *>(metadata_destination->contents());
  const auto normalizedLog = [&](size_t index) {
    const size_t block = index / block_elements;
    const float log_value = std::log2(source[index]);
    const float minimum = metadata[2 * block];
    const float span = metadata[2 * block + 1];
    return span == 0.0f ? 0.0f : maximum * (log_value - minimum) / span;
  };
  for (size_t block_start = 0, block = 0; block_start < source.size();
       block_start += block_elements, ++block) {
    const size_t block_end =
        std::min(source.size(), block_start + block_elements);
    float minimum = std::numeric_limits<float>::infinity();
    float maximum_log = -std::numeric_limits<float>::infinity();
    for (size_t index = block_start; index < block_end; ++index) {
      if (!(source[index] > 0.0f) || !std::isfinite(source[index])) {
        fail("Log-spectrum storage requires finite positive root eigenvalues.");
      }
      const float log_value = std::log2(source[index]);
      minimum = std::min(minimum, log_value);
      maximum_log = std::max(maximum_log, log_value);
    }
    metadata[2 * block] = minimum;
    metadata[2 * block + 1] = maximum_log - minimum;
  }
  copyValuesToStorage(precision, source.size(), normalizedLog, destination,
                      nullptr);
}

float spectrumValueAt(const PrecisionSpec &precision, MTL::Buffer *source,
                      MTL::Buffer *metadata_source, size_t index) {
  if (!precision.usesLogSpectrum()) {
    return storageValueAt(precision, source, metadata_source, index);
  }
  if (metadata_source == nullptr) {
    fail("Log-spectrum storage requires metadata.");
  }
  const size_t block = index / precision.spectral_block_elements;
  const float *metadata =
      static_cast<const float *>(metadata_source->contents());
  const float normalized = storageValueAt(precision, source, nullptr, index);
  const float log_value =
      metadata[2 * block] +
      metadata[2 * block + 1] *
          (normalized / maximumFiniteStorageValue(precision));
  const float value = std::exp2(log_value);
  if (!(value > 0.0f) || !std::isfinite(value)) {
    fail("Log-spectrum decode produced an invalid root eigenvalue.");
  }
  return value;
}

void unpackFromStorage(const PrecisionSpec &precision, MTL::Buffer *source,
                       MTL::Buffer *scale_source,
                       std::vector<double> &destination) {
  for (size_t index = 0; index < destination.size(); ++index) {
    destination[index] = static_cast<double>(
        storageValueAt(precision, source, scale_source, index));
  }
}

void copyResidualToStorage(const PrecisionSpec &precision,
                           const std::vector<double> &source,
                           MTL::Buffer *destination,
                           MTL::Buffer *scale_destination) {
  copyValuesToStorage(
      precision, source.size(),
      [&](size_t index) { return static_cast<float>(source[index]); },
      destination, scale_destination);
}

void addCorrectionFromStorage(const PrecisionSpec &precision,
                              MTL::Buffer *source, MTL::Buffer *scale_source,
                              std::vector<double> &destination) {
  for (size_t index = 0; index < destination.size(); ++index) {
    destination[index] += static_cast<double>(
        storageValueAt(precision, source, scale_source, index));
  }
}

double orthogonalityError(const PrecisionSpec &precision,
                          const StorageBuffers &basis, size_t order) {
  double difference_squared = 0.0;
  const double identity_squared = static_cast<double>(order);
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = 0; row < order; ++row) {
      double dot = 0.0;
      for (size_t index = 0; index < order; ++index) {
        dot +=
            static_cast<double>(storageValueAt(
                precision, basis.values, basis.scales, index + row * order)) *
            static_cast<double>(storageValueAt(
                precision, basis.values, basis.scales, index + column * order));
      }
      const double expected = row == column ? 1.0 : 0.0;
      const double difference = dot - expected;
      difference_squared += difference * difference;
    }
  }
  return std::sqrt(difference_squared / identity_squared);
}

}  // namespace mixed_sqrt
