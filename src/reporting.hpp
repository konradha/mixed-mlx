#pragma once

#include "experiment.hpp"

#include <fstream>

namespace mixed_sqrt {

void printResult(const CaseResult &result);
void writeSummaryHeader(std::ofstream &output);
void writeCaseCsv(const CaseResult &result, std::ofstream &summary,
                  std::ofstream &iterations);
void writeIterationsHeader(std::ofstream &output);

} // namespace mixed_sqrt
