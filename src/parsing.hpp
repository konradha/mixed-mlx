#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace mixed_sqrt {

size_t parsePositiveSize(const std::string &text, const char *option);
size_t parseNonnegativeSize(const std::string &text, const char *option);
double parseNonnegativeFiniteDouble(const std::string &text,
                                    const char *option);
std::vector<size_t> parseOrders(const std::string &text);
std::vector<double> parseConditions(const std::string &text);

} // namespace mixed_sqrt
