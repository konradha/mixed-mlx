#pragma once

#include <chrono>
#include <string>

namespace mixed_sqrt {

using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string &message);
void requireCondition(bool condition, const char *message);
double elapsedSeconds(const Clock::time_point &start);

}  // namespace mixed_sqrt
