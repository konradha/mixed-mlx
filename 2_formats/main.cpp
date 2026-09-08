#include "formats.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::vector<std::size_t> elements{1u << 18u, 1u << 20u, 1u << 22u};
  std::size_t warmups = 50;
  std::size_t evaluations = 100;
  std::string output = "artifacts/formats";
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
    if (argument == "--elements") {
      options.elements = {std::stoull(value("--elements"))};
    } else if (argument == "--warmups") {
      options.warmups = std::stoull(value("--warmups"));
    } else if (argument == "--evaluations") {
      options.evaluations = std::stoull(value("--evaluations"));
    } else if (argument == "--output") {
      options.output = value("--output");
    } else if (argument == "--quick") {
      options.elements = {1u << 16u};
      options.warmups = 1;
      options.evaluations = 3;
    } else {
      throw std::runtime_error("Unknown option: " + argument);
    }
  }
  if (options.warmups == 0 || options.evaluations == 0) {
    throw std::runtime_error("Warmups and evaluations must be positive.");
  }
  return options;
}

void print_result(const msqrt::FormatResult &result) {
  std::cout << std::left << std::setw(14) << result.format << std::setw(9)
            << result.distribution << std::right << std::setw(10)
            << result.elements << std::fixed << std::setprecision(2)
            << std::setw(9) << result.bits_per_value << std::scientific
            << std::setprecision(2) << std::setw(12) << result.relative_rmse
            << std::setw(12) << result.relative_dot_error << std::fixed
            << std::setprecision(2) << std::setw(12)
            << result.roundtrip_gib_per_second << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse(argc, argv);
    msqrt::Metal metal("build/2_formats/kernels.metallib");
    const std::vector<msqrt::Format> formats{
        msqrt::Format::Fp32,     msqrt::Format::Fp16,
        msqrt::Format::Bf16,     msqrt::Format::Fp8E4M3,
        msqrt::Format::Fp8E5M2,  msqrt::Format::MxFp8E4M3,
        msqrt::Format::MxFp4E2M1};
    const std::vector<msqrt::Distribution> distributions{
        msqrt::Distribution::Normal, msqrt::Distribution::Wide};
    std::vector<msqrt::FormatResult> results;
    std::cout << "Device: " << metal.device_name() << '\n';
    std::cout << "Warmups: " << options.warmups
              << ", evaluations: " << options.evaluations << '\n';
    std::cout
        << "format        input      elements     bits        rmse   dot_error"
           "     GiB_per_s\n";
    for (std::size_t elements : options.elements) {
      for (msqrt::Distribution distribution : distributions) {
        for (msqrt::Format format : formats) {
          results.push_back(msqrt::measure_format(metal, format, distribution,
                                                  elements, options.warmups,
                                                  options.evaluations));
          print_result(results.back());
        }
      }
    }
    msqrt::write_format_results(options.output, results);
    std::cout << "CSV: " << options.output << "/formats.csv\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
