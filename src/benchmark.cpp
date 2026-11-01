#include "benchmark.hpp"

#include "accelerate_backend.hpp"
#include "core.hpp"
#include "numerics.hpp"
#include "reporting.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

namespace mixed_sqrt {

TimingSummary summarizeTimingSamples(std::vector<double> samples,
                                     size_t warmups) {
  if (samples.empty()) {
    fail("Timing samples must not be empty.");
  }
  for (double sample : samples) {
    if (!(sample >= 0.0) || !std::isfinite(sample)) {
      fail("A timing sample is invalid.");
    }
  }
  std::sort(samples.begin(), samples.end());
  double mean = 0.0;
  for (double sample : samples) {
    mean += sample;
  }
  mean /= static_cast<double>(samples.size());
  double variance = 0.0;
  for (double sample : samples) {
    const double difference = sample - mean;
    variance += difference * difference;
  }
  if (samples.size() > 1) {
    variance /= static_cast<double>(samples.size() - 1);
  }
  return TimingSummary{
      warmups,
      samples.size(),
      samples.front(),
      samples.back(),
      samples[samples.size() / 2],
      mean,
      variance,
  };
}

PairedRatioSummary summarizePairedRatios(
    const std::vector<double> &mixed_samples,
    const std::vector<double> &reference_samples, size_t expected_count,
    std::vector<double> &ratios) {
  if (mixed_samples.size() != expected_count ||
      reference_samples.size() != expected_count || expected_count == 0) {
    fail("Paired timing sample count does not match the evaluation count.");
  }
  ratios.clear();
  ratios.reserve(expected_count);
  double mean = 0.0;
  double sum_squared_deviations = 0.0;
  double minimum = std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();
  for (size_t index = 0; index < expected_count; ++index) {
    const double mixed = mixed_samples[index];
    const double reference = reference_samples[index];
    if (!(mixed > 0.0) || !std::isfinite(mixed) || !(reference > 0.0) ||
        !std::isfinite(reference)) {
      fail("A paired timing sample is nonpositive or nonfinite.");
    }
    const double ratio = mixed / reference;
    if (!(ratio > 0.0) || !std::isfinite(ratio)) {
      fail("A paired timing ratio is nonpositive or nonfinite.");
    }
    ratios.push_back(ratio);
    minimum = std::min(minimum, ratio);
    maximum = std::max(maximum, ratio);
    const double count = static_cast<double>(index + 1);
    const double difference = ratio - mean;
    mean += difference / count;
    sum_squared_deviations += difference * (ratio - mean);
  }
  std::vector<double> sorted_ratios = ratios;
  std::sort(sorted_ratios.begin(), sorted_ratios.end());
  const size_t middle = sorted_ratios.size() / 2;
  const double median =
      sorted_ratios.size() % 2 == 0
          ? 0.5 * (sorted_ratios[middle - 1] + sorted_ratios[middle])
          : sorted_ratios[middle];
  const double variance =
      ratios.size() > 1
          ? sum_squared_deviations / static_cast<double>(ratios.size() - 1)
          : 0.0;
  return PairedRatioSummary{mean, variance, minimum, maximum, median};
}

int runAccelerateSweep(const std::vector<size_t> &orders,
                       const std::vector<double> &conditions, size_t warmups,
                       size_t evaluations,
                       const std::filesystem::path &output_directory,
                       const PreparedMatrix *paper_matrix) {
  std::filesystem::create_directories(output_directory);
  std::ofstream summary(output_directory / "summary.csv");
  std::ofstream iterations(output_directory / "iterations.csv");
  if (!summary || !iterations) {
    fail("Cannot create benchmark CSV files.");
  }
  writeSummaryHeader(summary);
  writeIterationsHeader(iterations);
  size_t failures = 0;
  const auto run_prepared = [&](const PreparedMatrix &prepared) {
    CaseResult result = runTimedAccelerateCase(prepared, warmups, evaluations);
    printResult(result);
    writeCaseCsv(result, summary, iterations);
    if (!result.converged || !result.finite) {
      ++failures;
    }
  };
  if (paper_matrix != nullptr) {
    run_prepared(*paper_matrix);
  } else {
    for (size_t order : orders) {
      for (double condition : conditions) {
        const PreparedMatrix prepared =
            prepareControlledMatrix(order, condition);
        run_prepared(prepared);
      }
    }
  }
  std::cout << "benchmark_failures=" << failures << '\n';
  std::cout << "summary_csv=" << (output_directory / "summary.csv") << '\n';
  std::cout << "iterations_csv=" << (output_directory / "iterations.csv")
            << '\n';
  return EXIT_SUCCESS;
}

}  // namespace mixed_sqrt
