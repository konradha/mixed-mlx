#pragma once

#include "metal.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace msqrt {

using Matrix = std::vector<double>;

enum class FactorFormat {
  Fp32,
  Fp16,
  Bf16,
  Fp8E4M3,
  Fp8E5M2,
  MxFp8E4M3,
  MxFp4E2M1
};

struct Problem {
  std::string name;
  std::size_t order = 0;
  double condition = 0.0;
  Matrix matrix;
  Matrix reference_root;
};

struct PrecisionPlan {
  std::string name;
  FactorFormat basis = FactorFormat::Fp32;
  FactorFormat spectrum = FactorFormat::Fp32;
};

struct IterationRecord {
  std::size_t iteration = 0;
  double residual = 0.0;
  double forward_error = 0.0;
};

struct SpectralResult {
  std::string plan;
  std::string basis_format;
  std::string spectrum_format;
  std::string product_backend;
  std::string arithmetic_format;
  std::string case_name;
  std::size_t order = 0;
  std::size_t evaluation = 0;
  double condition = 0.0;
  bool accepted = false;
  bool converged = false;
  std::string reason;
  std::size_t iterations = 0;
  double basis_bits_per_value = 0.0;
  double spectrum_bits_per_value = 0.0;
  double weighted_factor_storage_bits = 0.0;
  double basis_relative_error = 0.0;
  double spectrum_relative_error = 0.0;
  double basis_condition = 0.0;
  double initial_error = 0.0;
  double final_error = 0.0;
  double final_residual = 0.0;
  double factor_seconds = 0.0;
  double codec_seconds = 0.0;
  double initial_seconds = 0.0;
  double correction_seconds = 0.0;
  double total_seconds = 0.0;
  double product_kernel_seconds = 0.0;
  double conversion_seconds = 0.0;
  std::size_t correction_count = 0;
  double complete_solve_seconds = 0.0;
  double forward_error = 0.0;
  double residual = 0.0;
  std::vector<IterationRecord> history;
};

Problem controlled_problem(std::size_t order, double condition);
Problem paper_problem(char set, std::size_t order);
std::vector<PrecisionPlan> precision_plans();
SpectralResult run_spectral(Metal &metal, const Problem &problem,
                            const PrecisionPlan &plan,
                            std::size_t max_corrections = 30,
                            double tolerance = 1.0e-12);
void write_spectral_results(const std::string &directory,
                            const std::vector<SpectralResult> &results);

}  // namespace msqrt
