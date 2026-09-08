#include "formats.hpp"
#include "../common/buffers.hpp"
#include "../common/clock.hpp"
#include "../common/median.hpp"
#include "tensor_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace msqrt {
namespace {

using Clock = common::Clock;

struct CodecParams {
  std::uint32_t elements;
  std::uint32_t blocks;
};

struct Metrics {
  float source_squared;
  float error_squared;
  float reference_dot;
  float decoded_dot;
};

class Generator {
public:
  explicit Generator(std::uint64_t seed) : state_(seed) {}

  float uniform() {
    std::uint64_t value = state_;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state_ = value;
    const std::uint64_t bits = (value * UINT64_C(2685821657736338717)) >> 40;
    return static_cast<float>(bits) * 0x1.0p-24f;
  }

  float normal() {
    const float first = std::max(uniform(), std::numeric_limits<float>::min());
    return std::sqrt(-2.0f * std::log(first)) *
           std::cos(6.283185307179586f * uniform());
  }

private:
  std::uint64_t state_;
};

const char *format_name(Format format) {
  switch (format) {
  case Format::Fp32:
    return "fp32";
  case Format::Fp16:
    return "fp16";
  case Format::Bf16:
    return "bf16";
  case Format::Fp8E4M3:
    return "fp8_e4m3";
  case Format::Fp8E5M2:
    return "fp8_e5m2";
  case Format::MxFp8E4M3:
    return "mxfp8_e4m3";
  case Format::MxFp4E2M1:
    return "mxfp4_e2m1";
  }
  throw std::runtime_error("Unknown native format.");
}

const char *distribution_name(Distribution distribution) {
  return distribution == Distribution::Normal ? "normal" : "wide";
}

std::size_t value_bytes(Format format, std::size_t elements) {
  switch (format) {
  case Format::Fp32:
    return 4 * elements;
  case Format::Fp16:
  case Format::Bf16:
    return 2 * elements;
  case Format::Fp8E4M3:
  case Format::Fp8E5M2:
  case Format::MxFp8E4M3:
    return elements;
  case Format::MxFp4E2M1:
    return (elements + 1) / 2;
  }
  throw std::runtime_error("Unknown native format.");
}

std::size_t scale_bytes(Format format, std::size_t blocks) {
  return format == Format::MxFp8E4M3 || format == Format::MxFp4E2M1 ? blocks
                                                                    : 0;
}

// Vector kernels load 4 bytes at once. Pad alloc so tail vector stays in
// bounds. Bits math still uses exact value_bytes.
std::size_t encoded_alloc_bytes(Format format, std::size_t elements) {
  switch (format) {
  case Format::Fp8E4M3:
  case Format::Fp8E5M2:
  case Format::MxFp8E4M3:
    return ((elements + 3) / 4) * 4;
  case Format::MxFp4E2M1:
    return ((elements + 7) / 8) * 4;
  default:
    return value_bytes(format, elements);
  }
}

const char *encode_kernel(Format format) {
  switch (format) {
  case Format::Fp32:
    return "encode_fp32";
  case Format::Fp16:
    return "encode_fp16";
  case Format::Bf16:
    return "encode_bf16";
  case Format::Fp8E4M3:
    return "encode_fp8_e4m3";
  case Format::Fp8E5M2:
    return "encode_fp8_e5m2";
  case Format::MxFp8E4M3:
  case Format::MxFp4E2M1:
    throw std::runtime_error("An MX format requires the fused round trip.");
  }
  __builtin_unreachable();
}

const char *decode_kernel(Format format) {
  switch (format) {
  case Format::Fp32:
    return "decode_fp32";
  case Format::Fp16:
    return "decode_fp16";
  case Format::Bf16:
    return "decode_bf16";
  case Format::Fp8E4M3:
    return "decode_fp8_e4m3";
  case Format::Fp8E5M2:
    return "decode_fp8_e5m2";
  case Format::MxFp8E4M3:
  case Format::MxFp4E2M1:
    throw std::runtime_error("An MX format requires the fused round trip.");
  }

  __builtin_unreachable();
}

void encode(Metal &metal, MTL::ComputeCommandEncoder *encoder, Format format,
            MTL::Buffer *source, MTL::Buffer *values,
            const CodecParams &params) {
  auto name = encode_kernel(format);
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(values, 0, 1);
  encoder->setBytes(&params, sizeof(params), 2);
  const std::size_t width =
      format == Format::Fp8E4M3 || format == Format::Fp8E5M2 ? 4 : 1;
  Metal::dispatch1d(encoder, metal.pipeline(name),
                    (params.elements + width - 1) / width);
}

void decode(Metal &metal, MTL::ComputeCommandEncoder *encoder, Format format,
            MTL::Buffer *values, MTL::Buffer *output,
            const CodecParams &params) {
  const char *name = decode_kernel(format);
  encoder->setBuffer(values, 0, 0);
  encoder->setBuffer(output, 0, 1);
  encoder->setBytes(&params, sizeof(params), 2);
  const std::size_t width =
      format == Format::Fp8E4M3 || format == Format::Fp8E5M2 ? 4 : 1;
  Metal::dispatch1d(encoder, metal.pipeline(name),
                    (params.elements + width - 1) / width);
}

void roundtrip(Metal &metal, MTL::ComputeCommandEncoder *encoder, Format format,
               MTL::Buffer *source, MTL::Buffer *values, MTL::Buffer *scales,
               MTL::Buffer *output, const CodecParams &params) {
  const char *name = nullptr;
  if (format == Format::MxFp8E4M3) {
    name = "roundtrip_mxfp8_e4m3";
  } else if (format == Format::MxFp4E2M1) {
    name = "roundtrip_mxfp4_e2m1";
  } else {
    encode(metal, encoder, format, source, values, params);
    decode(metal, encoder, format, values, output, params);
    return;
  }

  encoder->setComputePipelineState(metal.pipeline(name));
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(values, 0, 1);
  encoder->setBuffer(scales, 0, 2);
  encoder->setBuffer(output, 0, 3);
  encoder->dispatchThreadgroups(MTL::Size::Make(params.blocks, 1, 1),
                                MTL::Size::Make(32, 1, 1));
}

} // namespace

FormatResult measure_format(Metal &metal, Format format,
                            Distribution distribution, std::size_t elements,
                            std::size_t warmups, std::size_t evaluations) {
  if (elements == 0 || elements > std::numeric_limits<std::uint32_t>::max() ||
      warmups == 0 || evaluations == 0) {
    throw std::runtime_error("Format measurement dimensions are invalid.");
  }
  const std::size_t blocks = (elements + 31) / 32;
  const CodecParams params{static_cast<std::uint32_t>(elements),
                           static_cast<std::uint32_t>(blocks)};
  const std::size_t encoded_bytes = value_bytes(format, elements);
  const std::size_t metadata_bytes = scale_bytes(format, blocks);
  common::BufferGuard<Metal> owned(metal);
  const auto make = [&](std::size_t bytes) {
    return owned.make(std::max<std::size_t>(bytes, 1));
  };
  MTL::Buffer *source = make(elements * sizeof(float));
  MTL::Buffer *multiplier = make(elements * sizeof(float));
  MTL::Buffer *values = make(encoded_alloc_bytes(format, elements));
  MTL::Buffer *scales = make(metadata_bytes);
  if (format == Format::MxFp8E4M3 || format == Format::MxFp4E2M1) {
    validate_mx_tensor_layout(values, scales, elements,
                              format == Format::MxFp4E2M1);
  }
  MTL::Buffer *decoded = make(elements * sizeof(float));
  MTL::Buffer *metrics = make(sizeof(Metrics));

  Generator generator(UINT64_C(0x9e3779b97f4a7c15) ^ elements ^
                      static_cast<std::uint64_t>(format));
  float *source_values = static_cast<float *>(source->contents());
  float *multiplier_values = static_cast<float *>(multiplier->contents());
  for (std::size_t index = 0; index < elements; ++index) {
    source_values[index] =
        distribution == Distribution::Normal
            ? generator.normal()
            : std::copysign(std::exp2(-8.0f + 16.0f * generator.uniform()),
                            generator.uniform() - 0.5f);
    multiplier_values[index] =
        generator.normal() + 0.25f * source_values[index];
  }

  metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
    for (std::size_t warmup = 0; warmup < warmups; ++warmup) {
      roundtrip(metal, encoder, format, source, values, scales, decoded,
                params);
    }
  });
  std::vector<double> gpu_timings;
  std::vector<double> wall_timings;
  gpu_timings.reserve(evaluations);
  wall_timings.reserve(evaluations);
  for (std::size_t evaluation = 0; evaluation < evaluations; ++evaluation) {
    const auto start = Clock::now();
    const double gpu_seconds =
        metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
          roundtrip(metal, encoder, format, source, values, scales, decoded,
                    params);
        });
    gpu_timings.push_back(gpu_seconds);
    wall_timings.push_back(
        std::chrono::duration<double>(Clock::now() - start).count());
  }
  std::vector<double> ordered_timings = gpu_timings;
  const double seconds = common::median_time(ordered_timings);

  metal.submit([&](MTL::ComputeCommandEncoder *encoder) {
    MTL::ComputePipelineState *pipeline = metal.pipeline("measure_error");
    encoder->setBuffer(source, 0, 0);
    encoder->setBuffer(multiplier, 0, 1);
    encoder->setBuffer(decoded, 0, 2);
    encoder->setBuffer(metrics, 0, 3);
    encoder->setBytes(&params, sizeof(params), 4);
    Metal::dispatch1d(encoder, pipeline, 256, 256);
  });
  const Metrics measured = *static_cast<const Metrics *>(metrics->contents());
  const double storage_bytes =
      static_cast<double>(encoded_bytes + metadata_bytes);
  const double traffic_per_round =
      2.0 * elements * sizeof(float) + 2.0 * storage_bytes;

  FormatResult result;
  result.format = format_name(format);
  result.distribution = distribution_name(distribution);
  result.elements = elements;
  result.bits_per_value = 8.0 * storage_bytes / static_cast<double>(elements);
  result.relative_rmse =
      std::sqrt(static_cast<double>(measured.error_squared) /
                static_cast<double>(measured.source_squared));
  result.relative_dot_error =
      std::abs(static_cast<double>(measured.decoded_dot) -
               static_cast<double>(measured.reference_dot)) /
      std::max(std::abs(static_cast<double>(measured.reference_dot)), 1.0e-30);
  result.roundtrip_gib_per_second =
      traffic_per_round / seconds / static_cast<double>(UINT64_C(1) << 30);
  result.evaluation_seconds = std::move(gpu_timings);
  result.wall_seconds = std::move(wall_timings);
  return result;
}

void write_format_results(const std::string &directory,
                          const std::vector<FormatResult> &results) {
  std::filesystem::create_directories(directory);
  std::ofstream output(directory + "/formats.csv");
  if (!output) {
    throw std::runtime_error("Cannot create the format result file.");
  }
  output << "format,distribution,elements,bits_per_value,relative_rmse,"
            "relative_dot_error,roundtrip_gib_per_second\n";
  output << std::setprecision(17);
  for (const FormatResult &result : results) {
    output << result.format << ',' << result.distribution << ','
           << result.elements << ',' << result.bits_per_value << ','
           << result.relative_rmse << ',' << result.relative_dot_error << ','
           << result.roundtrip_gib_per_second << '\n';
  }
  std::ofstream timings_output(directory + "/timings.csv");
  if (!timings_output) {
    throw std::runtime_error("Cannot create the format timing file.");
  }
  timings_output
      << "format,distribution,elements,evaluation,gpu_seconds,wall_seconds\n";
  timings_output << std::setprecision(17);
  for (const FormatResult &result : results) {
    for (std::size_t evaluation = 0;
         evaluation < result.evaluation_seconds.size(); ++evaluation) {
      timings_output << result.format << ',' << result.distribution << ','
                     << result.elements << ',' << evaluation << ','
                     << result.evaluation_seconds[evaluation] << ','
                     << result.wall_seconds[evaluation] << '\n';
    }
  }
}

} // namespace msqrt
