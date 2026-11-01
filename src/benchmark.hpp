#pragma once

#include "experiment.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace mixed_sqrt {

TimingSummary summarizeTimingSamples(std::vector<double> samples,
                                     size_t warmups);
PairedRatioSummary summarizePairedRatios(
    const std::vector<double> &mixed_samples,
    const std::vector<double> &reference_samples, size_t expected_count,
    std::vector<double> &ratios);
int runAccelerateSweep(const std::vector<size_t> &orders,
                       const std::vector<double> &conditions, size_t warmups,
                       size_t evaluations,
                       const std::filesystem::path &output_directory,
                       const PreparedMatrix *prepared_matrix = nullptr);

}  // namespace mixed_sqrt
