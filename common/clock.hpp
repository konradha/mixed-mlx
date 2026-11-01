#pragma once

#include <chrono>

namespace msqrt::common {

using Clock = std::chrono::steady_clock;

inline double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace msqrt::common
