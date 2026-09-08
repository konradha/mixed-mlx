#include "../common/median.hpp"
#include "metal_spd.hpp"
#include "spd.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Options {
  std::vector<std::size_t> orders{128, 256, 512};
  std::size_t warmups = 50;
  std::size_t evaluations = 100;
  std::string output = "artifacts/baseline";
  bool quick = false;
};

Options parse(int argc, char **argv) {
  Options options;
  bool order_selected = false;
  bool warmups_selected = false;
  bool evaluations_selected = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() -> const char * {
      if (++index >= argc) {
        throw std::runtime_error("Missing command-line value.");
      }
      return argv[index];
    };
    if (argument == "--order") {
      options.orders = {std::stoul(value())};
      order_selected = true;
    } else if (argument == "--warmups") {
      options.warmups = std::stoul(value());
      warmups_selected = true;
    } else if (argument == "--evaluations") {
      options.evaluations = std::stoul(value());
      evaluations_selected = true;
    } else if (argument == "--output") {
      options.output = value();
    } else if (argument == "--quick") {
      options.quick = true;
    } else {
      throw std::runtime_error("Unknown option: " + argument);
    }
  }
  if (options.quick) {
    if (!order_selected) {
      options.orders = {32};
    }
    if (!warmups_selected) {
      options.warmups = 1;
    }
    if (!evaluations_selected) {
      options.evaluations = 3;
    }
  }
  if (options.evaluations == 0 ||
      std::any_of(options.orders.begin(), options.orders.end(),
                  [](std::size_t order) { return order < 2; })) {
    throw std::runtime_error(
        "Orders must be at least two. Evaluations must be positive.");
  }
  return options;
}

using Key =
    std::tuple<std::string, std::string, std::string, std::size_t, double>;

void print_medians(const std::vector<msqrt::Run> &runs) {
  std::map<Key, std::vector<const msqrt::Run *>> groups;
  for (const msqrt::Run &run : runs) {
    const Key key{run.backend, run.format, run.case_name, run.order,
                  run.condition};
    groups[key].push_back(&run);
  }
  std::cout << std::left << std::setw(18) << "backend" << std::setw(18)
            << "format" << std::setw(14) << "case" << std::right << std::setw(7)
            << "order" << std::setw(12) << "condition" << std::setw(13)
            << "error" << std::setw(13) << "residual" << std::setw(12)
            << "median_ms" << '\n';
  for (auto &[key, values] : groups) {
    const msqrt::Run &run =
        *msqrt::common::median_run(values, [](const msqrt::Run *item) {
          return item->times.total_seconds;
        });
    const msqrt::Step &last = run.steps.back();
    const double median = 1.0e3 * run.times.total_seconds;
    std::cout << std::left << std::setw(18) << run.backend << std::setw(18)
              << run.format << std::setw(14) << run.case_name << std::right
              << std::setw(7) << run.order << std::scientific
              << std::setprecision(2) << std::setw(12) << run.condition
              << std::setw(13) << last.forward_error << std::setw(13)
              << last.residual << std::fixed << std::setprecision(2)
              << std::setw(12) << median << '\n';
  }
}

std::vector<msqrt::Problem> make_problems(const Options &options) {
  std::vector<msqrt::Problem> problems;
  for (std::size_t order : options.orders) {
    if (options.quick) {
      problems.push_back(msqrt::controlled_problem(order, 100.0));
      continue;
    }
    for (double condition : {4.0, 100.0, 10000.0}) {
      problems.push_back(msqrt::controlled_problem(order, condition));
    }
    if (order <= 256) {
      problems.push_back(msqrt::paper_problem('V', order));
      problems.push_back(msqrt::paper_problem('W', order));
      problems.push_back(msqrt::paper_problem('H', order));
    }
    if (order == 128) {
      problems.push_back(msqrt::laplacian_problem(order, 1.0e-4));
    }
  }
  return problems;
}

void execute_once(msqrt::Metal &metal, const msqrt::Problem &problem,
                  std::vector<msqrt::Run> *runs) {
  msqrt::Run schur = msqrt::accelerate_schur(problem);
  msqrt::Run mixed = msqrt::accelerate_mixed(problem);
  msqrt::Run gpu = msqrt::metal_fp32(metal, problem);
  if (runs != nullptr) {
    runs->push_back(std::move(schur));
    runs->push_back(std::move(mixed));
    runs->push_back(std::move(gpu));
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse(argc, argv);
    msqrt::Metal metal("build/1_baseline/kernels.metallib");
    const std::vector<msqrt::Problem> problems = make_problems(options);
    std::vector<msqrt::Run> runs;
    std::cout << "Device: " << metal.device_name() << '\n';
    std::cout << "Warmups: " << options.warmups
              << ", evaluations: " << options.evaluations << '\n';
    for (const msqrt::Problem &problem : problems) {
      for (std::size_t warmup = 0; warmup < options.warmups; ++warmup) {
        execute_once(metal, problem, nullptr);
      }
      for (std::size_t evaluation = 0; evaluation < options.evaluations;
           ++evaluation) {
        execute_once(metal, problem, &runs);
      }
    }
    print_medians(runs);
    msqrt::write_runs(options.output, runs);
    std::cout << "CSV directory: " << options.output << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
