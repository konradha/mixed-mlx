#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace mixed_sqrt {

enum class PaperTestSet { None, I, II, III, IV, V, VI, VII };

const char *paperTestSetName(PaperTestSet test_set);
PaperTestSet paperTestSetForCliName(const std::string &name);
bool paperTestSetIsSymmetric(PaperTestSet test_set);
size_t paperTestSetOrder(PaperTestSet test_set);

struct PreparedMatrix {
  std::vector<double> values;
  size_t order = 0;
  double condition = 0.0;
  std::string paper_test_set = "controlled";
  std::string matrix_family = "conditioned_spd";
  std::string matrix_type = "symmetric";
  uint64_t rng_seed = 0;
  double diagonal_shift = 0.0;
  std::string shift_rule = "not_applicable";
};

struct TimingSummary {
  size_t warmups = 0;
  size_t evaluations = 1;
  double minimum_seconds = 0.0;
  double maximum_seconds = 0.0;
  double median_seconds = 0.0;
  double mean_seconds = 0.0;
  double variance_seconds2 = 0.0;
};

struct PairedRatioSummary {
  double mean = 0.0;
  double variance = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double median = 0.0;
};

struct IterationMeasurement {
  size_t iteration;
  double residual;
  double forward_error;
};

struct PhaseTimes {
  double reference_seconds = 0.0;
  double low_eigen_seconds = 0.0;
  double initial_gpu_seconds = 0.0;
  double residual_fp64_seconds = 0.0;
  double correction_gpu_seconds = 0.0;
  double update_fp64_seconds = 0.0;
  double mixed_total_seconds = 0.0;
};

struct CaseResult {
  std::string name;
  std::string paper_test_set = "controlled";
  std::string matrix_family = "conditioned_spd";
  std::string matrix_type = "symmetric";
  std::string solver_backend = "metal";
  uint64_t rng_seed = 0;
  double diagonal_shift = 0.0;
  std::string shift_rule = "not_applicable";
  std::string precision;
  std::string basis_precision;
  std::string correction_mode;
  std::string spectrum_precision;
  std::string precision_policy;
  std::string rounding_mode = "nearest";
  uint32_t rounding_seed = 0;
  std::string execution_model;
  std::string kernel_path;
  std::string status;
  std::string stopping_policy = "fixed_budget";
  std::string dispatch_policy = "not_applicable";
  std::string termination_reason;
  std::string selection_policy = "fixed";
  std::string selected_basis_stage = "not_applicable";
  std::string selected_spectrum_stage = "not_applicable";
  std::string correction_stage_history = "not_applicable";
  size_t precision_attempts = 1;
  size_t order = 0;
  size_t low_factor_basis_bytes = 0;
  size_t fp32_basis_bytes = 0;
  size_t best_iteration = 0;
  size_t stagnation_steps = 0;
  size_t command_buffers_per_correction = 0;
  size_t kernel_dispatches_per_correction = 0;
  size_t precision_escalations = 0;
  size_t low_factor_corrections = 0;
  TimingSummary timing;
  TimingSummary reference_timing;
  std::vector<double> paired_ratios;
  PairedRatioSummary paired_ratio_summary;
  double condition = 0.0;
  bool converged = false;
  bool finite = false;
  double symmetry_error = 0.0;
  double reference_residual = 0.0;
  double frozen_factor_relative_error = 0.0;
  double frozen_factor_error = 0.0;
  double frozen_basis_orthogonality = 0.0;
  double quantized_basis_orthogonality =
      std::numeric_limits<double>::quiet_NaN();
  double factor_quantization_seconds = 0.0;
  double basis_reorthogonalization_seconds = 0.0;
  double frozen_sylvester_inverse_norm = 0.0;
  double frozen_contraction_bound = 0.0;
  bool certificate_usable = false;
  double best_residual = std::numeric_limits<double>::quiet_NaN();
  double best_forward_error = std::numeric_limits<double>::quiet_NaN();
  double final_residual = std::numeric_limits<double>::quiet_NaN();
  double final_forward_error = std::numeric_limits<double>::quiet_NaN();
  std::vector<IterationMeasurement> history;
  PhaseTimes times;
};

} // namespace mixed_sqrt
