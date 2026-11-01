#pragma once

#include "experiment.hpp"
#include "precision.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace mixed_sqrt {

struct MetalRunOptions {
  Precision basis_precision;
  Precision spectrum_precision;
  KernelPath kernel_path;
  CorrectionMode correction_mode;
  RoundingMode rounding_mode;
  uint32_t rounding_seed;
  NativeStoppingPolicy native_stopping_policy;
  size_t native_stagnation_window;
  double native_min_relative_improvement;
  NativeDispatchPolicy native_dispatch_policy;
  bool sweep;
  bool adaptive;
  bool kernel_check;
  const std::vector<size_t> &orders;
  const std::vector<double> &conditions;
  size_t warmups;
  size_t evaluations;
  size_t order;
  std::filesystem::path output_directory;
};

int runMetal(const MetalRunOptions &options,
             const PreparedMatrix *paper_matrix_pointer);

} // namespace mixed_sqrt
