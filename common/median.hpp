#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace msqrt::common {

// Upper median of a copy. Callers keep per-run records sorted by time and
// read the median record, so time and error come from the same run.
template <typename T, typename Time>
const T &median_run(std::vector<T> &runs, Time time_of) {
  std::sort(runs.begin(), runs.end(), [&](const T &left, const T &right) {
    return time_of(left) < time_of(right);
  });
  return runs[runs.size() / 2];
}

inline double median_time(std::vector<double> &values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

} // namespace msqrt::common
