#include "core.hpp"

#include <stdexcept>

namespace mixed_sqrt {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void requireCondition(bool condition, const char *message) {
  if (!condition) {
    fail(message);
  }
}

double elapsedSeconds(const Clock::time_point &start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace mixed_sqrt
