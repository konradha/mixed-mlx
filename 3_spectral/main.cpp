#include "spectral.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::size_t order = 128;
  std::size_t warmups = 50;
  std::size_t evaluations = 100;
  std::size_t max_corrections = 30;
  std::string output = "artifacts/spectral";
  bool quick = false;
};

Options parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&](const char *name) {
      if (++index >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name +
                                 '.');
      }
      return std::string(argv[index]);
    };
    if (argument == "--order") {
      options.order = std::stoull(value("--order"));
    } else if (argument == "--warmups") {
      options.warmups = std::stoull(value("--warmups"));
    } else if (argument == "--evaluations") {
      options.evaluations = std::stoull(value("--evaluations"));
    } else if (argument == "--max-corrections") {
      options.max_corrections = std::stoull(value("--max-corrections"));
    } else if (argument == "--output") {
      options.output = value("--output");
    } else if (argument == "--quick") {
      options.quick = true;
      options.order = 32;
      options.warmups = 1;
      options.evaluations = 1;
    } else {
      throw std::runtime_error("Unknown option: " + argument);
    }
  }
  if (options.order < 2 || options.evaluations == 0) {
    throw std::runtime_error("The order and evaluation count are invalid.");
  }
  return options;
}

void print_result(const msqrt::SpectralResult &result) {
  std::cout << std::left << std::setw(16) << result.plan << std::setw(12)
            << result.case_name << std::setw(12) << result.product_backend
            << std::right << std::setw(5) << result.order;
  if (!result.accepted) {
    std::cout << "  unsupported: " << result.reason << '\n';
    return;
  }
  std::cout << std::setw(5) << result.correction_count << std::setw(6)
            << result.accepted << std::setw(6) << result.converged
            << std::scientific << std::setprecision(2) << std::setw(12)
            << result.forward_error << std::setw(12) << result.residual
            << std::fixed << std::setprecision(2) << std::setw(11)
            << 1.0e3 * result.complete_solve_seconds << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse(argc, argv);
    msqrt::Metal metal("build/3_spectral/kernels.metallib");
    std::vector<msqrt::Problem> problems;
    if (options.quick) {
      problems.push_back(msqrt::controlled_problem(options.order, 100.0));
    } else {
      for (double condition : {4.0, 100.0, 10000.0}) {
        problems.push_back(msqrt::controlled_problem(options.order, condition));
      }
      problems.push_back(msqrt::paper_problem('V', options.order));
      problems.push_back(msqrt::paper_problem('W', options.order));
      problems.push_back(msqrt::paper_problem('H', options.order));
    }
    const std::vector<msqrt::PrecisionPlan> plans = msqrt::precision_plans();
    std::vector<msqrt::SpectralResult> results;
    std::cout << "Device: " << metal.device_name() << '\n';
    std::cout << "Warmups: " << options.warmups
              << ", evaluations: " << options.evaluations << '\n';
    std::cout
        << "plan            case        backend     order iter accept  conv"
           "       error    residual   solve_ms\n";
    for (const msqrt::Problem &problem : problems) {
      std::vector<std::vector<msqrt::SpectralResult>> evaluations_by_plan(
          plans.size());
      for (std::vector<msqrt::SpectralResult> &evaluations :
           evaluations_by_plan) {
        evaluations.reserve(options.evaluations);
      }
      for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
        const std::size_t start = warmup % plans.size();
        for (std::size_t offset = 0; offset < plans.size(); ++offset) {
          const std::size_t plan_index = (start + offset) % plans.size();
          (void)msqrt::run_spectral(metal, problem, plans[plan_index],
                                    options.max_corrections);
        }
      }
      for (std::size_t evaluation = 0; evaluation < options.evaluations;
           ++evaluation) {
        const std::size_t start = evaluation % plans.size();
        for (std::size_t offset = 0; offset < plans.size(); ++offset) {
          const std::size_t plan_index = (start + offset) % plans.size();
          msqrt::SpectralResult measured = msqrt::run_spectral(
              metal, problem, plans[plan_index], options.max_corrections);
          measured.evaluation = evaluation;
          evaluations_by_plan[plan_index].push_back(std::move(measured));
        }
      }
      for (std::vector<msqrt::SpectralResult> &evaluations :
           evaluations_by_plan) {
        std::stable_sort(evaluations.begin(), evaluations.end(),
                         [](const auto &left, const auto &right) {
                           const bool left_finite =
                               std::isfinite(left.complete_solve_seconds);
                           const bool right_finite =
                               std::isfinite(right.complete_solve_seconds);
                           if (left_finite != right_finite)
                             return left_finite;
                           return left_finite &&
                                  left.complete_solve_seconds <
                                      right.complete_solve_seconds;
                         });
        print_result(evaluations[evaluations.size() / 2]);
        for (msqrt::SpectralResult &measured : evaluations) {
          results.push_back(std::move(measured));
        }
      }
    }
    msqrt::write_spectral_results(options.output, results);
    std::cout << "CSV directory: " << options.output << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
