#include "metal_backend.hpp"

#include "metal_checks.hpp"
#include "metal_runtime.hpp"
#include "metal_solver.hpp"

#include "core.hpp"
#include "reporting.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace mixed_sqrt {
namespace {

int runSweep(MetalRuntime &metal, const MetalSolverConfiguration &configuration,
             const std::vector<size_t> &orders,
             const std::vector<double> &conditions, size_t warmups,
             size_t evaluations, const std::filesystem::path &output_directory,
             const PreparedMatrix *prepared_matrix = nullptr) {
  std::filesystem::create_directories(output_directory);
  std::ofstream summary(output_directory / "summary.csv");
  std::ofstream iterations(output_directory / "iterations.csv");
  if (!summary || !iterations) {
    fail("Cannot create benchmark CSV files.");
  }
  writeSummaryHeader(summary);
  writeIterationsHeader(iterations);

  size_t failures = 0;
  if (prepared_matrix != nullptr) {
    const MetalProblem problem{
        prepared_matrix->order,
        prepared_matrix->condition,
        12,
        1.0e-12,
        false,
        prepared_matrix,
        nullptr,
    };
    CaseResult result =
        runTimedMetalCase(metal, configuration, problem, warmups, evaluations);
    printResult(result);
    writeCaseCsv(result, summary, iterations);
    if (!result.converged || !result.finite) {
      ++failures;
    }
  } else {
    for (size_t order : orders) {
      for (double condition : conditions) {
        const MetalProblem problem{
            order, condition, 12, 1.0e-12, false, nullptr, nullptr,
        };
        CaseResult result = runTimedMetalCase(metal, configuration, problem,
                                              warmups, evaluations);
        printResult(result);
        writeCaseCsv(result, summary, iterations);
        if (!result.converged || !result.finite) {
          ++failures;
        }
      }
    }
  }
  std::cout << "benchmark_failures=" << failures << '\n';
  std::cout << "summary_csv=" << (output_directory / "summary.csv") << '\n';
  std::cout << "iterations_csv=" << (output_directory / "iterations.csv")
            << '\n';
  return EXIT_SUCCESS;
}

}  // namespace

int runMetal(const MetalRunOptions &options,
             const PreparedMatrix *paper_matrix_pointer) {
  const MetalSolverConfiguration configuration{
      options.basis_precision,
      options.spectrum_precision,
      options.kernel_path,
      options.correction_mode,
      options.native_stopping_policy,
      options.native_stagnation_window,
      options.native_min_relative_improvement,
      options.native_dispatch_policy,
  };
  validateCorrectionMode(options.correction_mode, options.basis_precision,
                         options.spectrum_precision, options.kernel_path);
  ensureKernelsEnabled(precisionSpec(options.basis_precision));
  ensureKernelsEnabled(precisionSpec(options.spectrum_precision));
  const char *shader_path = "metal_mixed_sqrt.metal";
  MetalRuntime metal(shader_path, options.rounding_mode, options.rounding_seed);
  if (options.kernel_check) {
    runMetalKernelChecks(metal);
    return EXIT_SUCCESS;
  }
  if (options.sweep) {
    return runSweep(metal, configuration, options.orders, options.conditions,
                    options.warmups, options.evaluations,
                    options.output_directory, paper_matrix_pointer);
  }

  const double condition =
      paper_matrix_pointer != nullptr ? paper_matrix_pointer->condition : 4.0;
  const MetalProblem problem{
      options.order,
      condition,
      paper_matrix_pointer != nullptr ? size_t{12} : size_t{8},
      1.0e-12,
      true,
      paper_matrix_pointer,
      nullptr,
  };
  const CaseResult result =
      options.adaptive ? runAdaptiveMetalCase(metal, configuration, problem)
                       : runMetalCase(metal, configuration, problem);
  printResult(result);
  return EXIT_SUCCESS;
}

}  // namespace mixed_sqrt
