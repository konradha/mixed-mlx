#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace msqrt {

using Matrix = std::vector<double>;

struct Problem {
  std::string name;
  std::size_t order = 0;
  double condition = 0.0;
  Matrix matrix;
  Matrix reference_root;
};

struct Step {
  std::size_t iteration = 0;
  double residual = 0.0;
  double forward_error = 0.0;
};

struct Times {
  double factor_seconds = 0.0;
  double initial_seconds = 0.0;
  double residual_seconds = 0.0;
  double correction_seconds = 0.0;
  double total_seconds = 0.0;
};

struct Run {
  std::string stage;
  std::string backend;
  std::string format = "fp32";
  std::string case_name;
  std::size_t order = 0;
  double condition = 0.0;
  bool converged = false;
  Times times;
  std::vector<Step> steps;
};

Problem controlled_problem(std::size_t order, double condition);
Problem laplacian_problem(std::size_t order, double diagonal_shift);
Problem paper_problem(char set, std::size_t order);
Matrix reference_root(const Matrix &matrix, std::size_t order);
double relative_residual(const Matrix &matrix, const Matrix &root,
                         std::size_t order, Matrix *residual = nullptr);
double relative_error(const Matrix &value, const Matrix &reference);
Run accelerate_mixed(const Problem &problem, std::size_t max_corrections = 12,
                     double tolerance = 1.0e-12);
Run accelerate_schur(const Problem &problem);
void write_runs(const std::string &directory, const std::vector<Run> &runs);

}  // namespace msqrt
