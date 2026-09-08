#pragma once

#include "metal.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace msqrt {

enum class Format { Fp32, Fp16, Bf16, Fp8E4M3, Fp8E5M2, MxFp8E4M3, MxFp4E2M1 };
enum class Distribution { Normal, Wide };

struct FormatResult {
  std::string format;
  std::string distribution;
  std::size_t elements = 0;
  double bits_per_value = 0.0;
  double relative_rmse = 0.0;
  double relative_dot_error = 0.0;
  double roundtrip_gib_per_second = 0.0;
  std::vector<double> evaluation_seconds;
  std::vector<double> wall_seconds;
};

FormatResult measure_format(Metal &metal, Format format,
                            Distribution distribution, std::size_t elements,
                            std::size_t warmups, std::size_t evaluations);
void write_format_results(const std::string &directory,
                          const std::vector<FormatResult> &results);

} // namespace msqrt
