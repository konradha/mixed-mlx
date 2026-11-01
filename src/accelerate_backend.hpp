#pragma once

#include "experiment.hpp"

#include <cstddef>
#include <vector>

namespace mixed_sqrt {

std::vector<double> symmetricReferenceRoot(const std::vector<double> &matrix,
                                           size_t order);
CaseResult
runAccelerateCase(const PreparedMatrix &prepared, size_t max_corrections,
                  double tolerance,
                  const std::vector<double> *supplied_reference = nullptr);
double timeReferenceSolve(const PreparedMatrix &prepared,
                          std::vector<double> *root_output = nullptr);
CaseResult runTimedAccelerateCase(const PreparedMatrix &prepared,
                                  size_t warmups, size_t evaluations);

} // namespace mixed_sqrt
