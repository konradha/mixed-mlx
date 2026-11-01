#include "parsing.hpp"

#include "core.hpp"

#include <cmath>
#include <limits>
#include <sstream>

namespace mixed_sqrt {

size_t parsePositiveSize(const std::string &text, const char *option) {
  if (text.empty() || text.front() == '-') {
    fail(std::string(option) + " requires a positive integer.");
  }
  try {
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
      fail(std::string(option) + " requires a positive integer.");
    }
    return static_cast<size_t>(value);
  } catch (const std::exception &) {
    fail(std::string(option) + " requires a positive integer.");
  }
}

size_t parseNonnegativeSize(const std::string &text, const char *option) {
  if (text.empty() || text.front() == '-') {
    fail(std::string(option) + " requires a nonnegative integer.");
  }
  try {
    size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size() || value > std::numeric_limits<size_t>::max()) {
      fail(std::string(option) + " requires a nonnegative integer.");
    }
    return static_cast<size_t>(value);
  } catch (const std::exception &) {
    fail(std::string(option) + " requires a nonnegative integer.");
  }
}

double parsePositiveFiniteDouble(const std::string &text, const char *option) {
  try {
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !(value > 0.0) || !std::isfinite(value)) {
      fail(std::string(option) + " requires a positive finite real.");
    }
    return value;
  } catch (const std::exception &) {
    fail(std::string(option) + " requires a positive finite real.");
  }
}

double parseNonnegativeFiniteDouble(const std::string &text,
                                    const char *option) {
  try {
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || value < 0.0 || !std::isfinite(value)) {
      fail(std::string(option) + " requires a nonnegative finite real.");
    }
    return value;
  } catch (const std::exception &) {
    fail(std::string(option) + " requires a nonnegative finite real.");
  }
}

std::vector<size_t> parseOrders(const std::string &text) {
  std::vector<size_t> orders;
  std::istringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    orders.push_back(parsePositiveSize(item, "--orders"));
  }
  if (orders.empty() || text.back() == ',') {
    fail("--orders requires comma-separated positive integers.");
  }
  return orders;
}

std::vector<double> parseConditions(const std::string &text) {
  std::vector<double> conditions;
  std::istringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    conditions.push_back(parsePositiveFiniteDouble(item, "--conditions"));
  }
  if (conditions.empty() || text.back() == ',') {
    fail("--conditions requires comma-separated positive finite reals.");
  }
  return conditions;
}

} // namespace mixed_sqrt
