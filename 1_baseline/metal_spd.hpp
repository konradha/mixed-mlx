#pragma once

#include "metal.hpp"
#include "spd.hpp"

#include <cstddef>
#include <vector>

namespace msqrt {

struct GpuFactor {
  std::vector<float> basis;
  std::vector<float> root_spectrum;
  double seconds = 0.0;
};

GpuFactor shared_eigen(const Problem &problem);
Run metal_fp32(Metal &metal, const Problem &problem,
               std::size_t max_corrections = 12, double tolerance = 1.0e-12);

} // namespace msqrt
