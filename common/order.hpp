#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace msqrt::common {

// Matrix element count for square order. Single range check for all stages.
inline std::size_t elements(std::size_t order) {
  if (order == 0 || order > std::sqrt(static_cast<double>(
                                std::numeric_limits<std::size_t>::max()))) {
    throw std::runtime_error("Matrix order is invalid.");
  }
  return order * order;
}

// Checked narrow to 32 bit for Metal element counts and dispatches.
inline std::uint32_t to_u32(std::size_t value) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("Count exceeds Metal 32-bit range.");
  }
  return static_cast<std::uint32_t>(value);
}

} // namespace msqrt::common
