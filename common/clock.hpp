#pragma once

#include <chrono>

namespace msqrt::common {

// Single clock for all stages. Callers store start, then pass it to elapsed.
using Clock = std::chrono::steady_clock;

inline double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace msqrt::common
