#include "spd.hpp"

#include "../common/clock.hpp"
#include "../common/order.hpp"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace msqrt {
namespace {

using Clock = common::Clock;

std::size_t elements(std::size_t order) {
  return common::elements(order);
}

__LAPACK_int lapack_order(std::size_t order) {
  if (order >
      static_cast<std::size_t>(std::numeric_limits<__LAPACK_int>::max())) {
    throw std::runtime_error("Matrix order exceeds LAPACK range.");
  }
  return static_cast<__LAPACK_int>(order);
}

double seconds(Clock::time_point start) {
  return common::elapsed(start);
}

void dgemm(char ta, char tb, std::size_t n, const double *a, const double *b,
           double *c) {
  const __LAPACK_int dim = lapack_order(n);
  const double alpha = 1.0;
  const double beta = 0.0;
  dgemm_(&ta, &tb, &dim, &dim, &dim, &alpha, a, &dim, b, &dim, &beta, c, &dim);
}

void sgemm(char ta, char tb, std::size_t n, const float *a, const float *b,
           float *c) {
  const __LAPACK_int dim = lapack_order(n);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  sgemm_(&ta, &tb, &dim, &dim, &dim, &alpha, a, &dim, b, &dim, &beta, c, &dim);
}

void eigen_double(Matrix &vectors, std::vector<double> &values) {
  __LAPACK_int n = lapack_order(values.size());
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_size = -1;
  __LAPACK_int info = 0;
  double work_query = 0.0;
  __LAPACK_int integer_query = 0;
  dsyevd_(&job, &triangle, &n, vectors.data(), &n, values.data(), &work_query,
          &work_size, &integer_query, &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate dsyevd workspace query failed.");
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  integer_size = integer_query;
  std::vector<double> work(static_cast<std::size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<std::size_t>(integer_size));
  dsyevd_(&job, &triangle, &n, vectors.data(), &n, values.data(), work.data(),
          &work_size, integer_work.data(), &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate dsyevd failed.");
  }
}

void eigen_float(std::vector<float> &vectors, std::vector<float> &values) {
  __LAPACK_int n = lapack_order(values.size());
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_size = -1;
  __LAPACK_int info = 0;
  float work_query = 0.0f;
  __LAPACK_int integer_query = 0;
  ssyevd_(&job, &triangle, &n, vectors.data(), &n, values.data(), &work_query,
          &work_size, &integer_query, &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate ssyevd workspace query failed.");
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  integer_size = integer_query;
  std::vector<float> work(static_cast<std::size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<std::size_t>(integer_size));
  ssyevd_(&job, &triangle, &n, vectors.data(), &n, values.data(), work.data(),
          &work_size, integer_work.data(), &integer_size, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate ssyevd failed.");
  }
}

Matrix deterministic_basis(std::size_t order) {
  Matrix q(elements(order));
  for (std::size_t column = 0; column < order; ++column) {
    for (std::size_t row = 0; row < order; ++row) {
      q[row + column * order] =
          std::sin(0.17 * static_cast<double>((row + 1) * (column + 2))) +
          std::cos(0.11 * static_cast<double>((row + 3) * (column + 1)));
    }
    for (std::size_t prior = 0; prior < column; ++prior) {
      double projection = 0.0;
      for (std::size_t row = 0; row < order; ++row) {
        projection += q[row + prior * order] * q[row + column * order];
      }
      for (std::size_t row = 0; row < order; ++row) {
        q[row + column * order] -= projection * q[row + prior * order];
      }
    }
    double norm = 0.0;
    for (std::size_t row = 0; row < order; ++row) {
      norm = std::hypot(norm, q[row + column * order]);
    }
    if (!(norm > std::numeric_limits<double>::epsilon())) {
      throw std::runtime_error("Deterministic basis construction failed.");
    }
    for (std::size_t row = 0; row < order; ++row) {
      q[row + column * order] /= norm;
    }
  }
  return q;
}

class Generator {
 public:
  explicit Generator(std::uint64_t seed) : state_(seed) {}

  double uniform() {
    std::uint64_t value = state_;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state_ = value;
    const std::uint64_t bits = (value * UINT64_C(2685821657736338717)) >> 11;
    return static_cast<double>(bits) * 0x1.0p-53;
  }

  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u = std::max(uniform(), std::numeric_limits<double>::min());
    const double angle = 6.2831853071795864769 * uniform();
    const double radius = std::sqrt(-2.0 * std::log(u));
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
  }

 private:
  std::uint64_t state_;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

double infinity_norm(const Matrix &a, std::size_t order) {
  double result = 0.0;
  for (std::size_t row = 0; row < order; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < order; ++column) {
      sum += std::abs(a[row + column * order]);
    }
    result = std::max(result, sum);
  }
  return result;
}

void symmetrize(Matrix &a, std::size_t order) {
  for (std::size_t column = 0; column < order; ++column) {
    for (std::size_t row = 0; row < column; ++row) {
      const double value =
          0.5 * (a[row + column * order] + a[column + row * order]);
      a[row + column * order] = value;
      a[column + row * order] = value;
    }
  }
}

void validate_positive(const std::vector<float> &values) {
  for (float value : values) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
      throw std::runtime_error(
          "The FP32 eigensolver returned a nonpositive eigenvalue.");
    }
  }
}

}  // namespace

Problem controlled_problem(std::size_t order, double condition) {
  if (!(condition >= 1.0) || !std::isfinite(condition)) {
    throw std::runtime_error(
        "Condition number must be finite and at least one.");
  }
  const Matrix q = deterministic_basis(order);
  Matrix weighted = q;
  const double log_condition = std::log(condition);
  for (std::size_t column = 0; column < order; ++column) {
    const double exponent = order == 1 ? 0.0
                                       : static_cast<double>(column) /
                                             static_cast<double>(order - 1);
    const double lambda = std::exp(exponent * log_condition);
    for (std::size_t row = 0; row < order; ++row) {
      weighted[row + column * order] *= lambda;
    }
  }
  Matrix matrix(elements(order));
  dgemm('N', 'T', order, weighted.data(), q.data(), matrix.data());
  Problem problem{"controlled", order, condition, std::move(matrix), {}};
  problem.reference_root = reference_root(problem.matrix, order);
  return problem;
}

Problem laplacian_problem(std::size_t order, double diagonal_shift) {
  if (order == 0 || !(diagonal_shift > 0.0) || !std::isfinite(diagonal_shift)) {
    throw std::runtime_error(
        "The Laplacian order and diagonal shift must be positive.");
  }
  Matrix matrix(elements(order), 0.0);
  for (std::size_t index = 0; index < order; ++index) {
    matrix[index + index * order] = 2.0 + diagonal_shift;
    if (index + 1 < order) {
      matrix[index + (index + 1) * order] = -1.0;
      matrix[index + 1 + index * order] = -1.0;
    }
  }
  Matrix discovery = matrix;
  std::vector<double> eigenvalues(order);
  eigen_double(discovery, eigenvalues);
  const double condition = eigenvalues.back() / eigenvalues.front();
  Problem problem{"laplacian", order, condition, std::move(matrix), {}};
  problem.reference_root = reference_root(problem.matrix, order);
  return problem;
}

Problem paper_problem(char set, std::size_t order) {
  if (set != 'V' && set != 'W' && set != 'H') {
    throw std::runtime_error("SPD paper set must be V, W (VI), or H (VII).");
  }
  Matrix matrix(elements(order), 0.0);
  std::string name;
  if (set == 'H') {
    name = "paper_VII";
    for (std::size_t column = 0; column < order; ++column) {
      for (std::size_t row = 0; row < order; ++row) {
        matrix[row + column * order] =
            1.0 / static_cast<double>(row + column + 1);
      }
      matrix[column + column * order] += 1.0e-6;
    }
  } else {
    name = set == 'V' ? "paper_V" : "paper_VI";
    const std::uint64_t ordinal = set == 'V' ? 5u : 6u;
    Generator generator(UINT64_C(0x243f6a8885a308d3) ^
                        (UINT64_C(0x9e3779b97f4a7c15) * ordinal));
    for (std::size_t column = 0; column < order; ++column) {
      for (std::size_t row = column; row < order; ++row) {
        const double value =
            set == 'V' ? generator.uniform() : generator.normal();
        matrix[row + column * order] = value;
        matrix[column + row * order] = value;
      }
    }
    Matrix discovery = matrix;
    std::vector<double> eigenvalues(order);
    eigen_double(discovery, eigenvalues);
    const double separation =
        std::max(1.0, 0.01 * infinity_norm(matrix, order));
    const double shift = std::max(0.0, separation - eigenvalues.front());
    for (std::size_t index = 0; index < order; ++index) {
      matrix[index + index * order] += shift;
    }
  }
  Matrix discovery = matrix;
  std::vector<double> eigenvalues(order);
  eigen_double(discovery, eigenvalues);
  const double condition = eigenvalues.back() / eigenvalues.front();
  Problem problem{name, order, condition, std::move(matrix), {}};
  problem.reference_root = reference_root(problem.matrix, order);
  return problem;
}

Matrix reference_root(const Matrix &matrix, std::size_t order) {
  Matrix vectors = matrix;
  std::vector<double> values(order);
  eigen_double(vectors, values);
  Matrix weighted = vectors;
  for (std::size_t column = 0; column < order; ++column) {
    if (!(values[column] > 0.0)) {
      throw std::runtime_error("Reference matrix is not positive definite.");
    }
    const double scale = std::sqrt(values[column]);
    for (std::size_t row = 0; row < order; ++row) {
      weighted[row + column * order] *= scale;
    }
  }
  Matrix root(elements(order));
  dgemm('N', 'T', order, weighted.data(), vectors.data(), root.data());
  symmetrize(root, order);
  return root;
}

double relative_residual(const Matrix &matrix, const Matrix &root,
                         std::size_t order, Matrix *residual_output) {
  Matrix product(elements(order));
  dgemm('N', 'N', order, root.data(), root.data(), product.data());
  Matrix residual(elements(order));
  double matrix_norm = 0.0;
  double residual_norm = 0.0;
  for (std::size_t index = 0; index < residual.size(); ++index) {
    residual[index] = matrix[index] - product[index];
    matrix_norm = std::hypot(matrix_norm, matrix[index]);
    residual_norm = std::hypot(residual_norm, residual[index]);
  }
  if (residual_output != nullptr) {
    *residual_output = std::move(residual);
  }
  return residual_norm / matrix_norm;
}

double relative_error(const Matrix &value, const Matrix &reference) {
  if (value.size() != reference.size()) {
    throw std::runtime_error("Matrix sizes do not match.");
  }
  double difference = 0.0;
  double norm = 0.0;
  for (std::size_t index = 0; index < value.size(); ++index) {
    difference = std::hypot(difference, value[index] - reference[index]);
    norm = std::hypot(norm, reference[index]);
  }
  return difference / norm;
}

Run accelerate_mixed(const Problem &problem, std::size_t max_corrections,
                     double tolerance) {
  const std::size_t order = problem.order;
  const std::size_t count = elements(order);
  Run run;
  run.stage = "baseline";
  run.backend = "accelerate";
  run.format = "fp32";
  run.case_name = problem.name;
  run.order = order;
  run.condition = problem.condition;
  const auto total_start = Clock::now();

  std::vector<float> q(problem.matrix.begin(), problem.matrix.end());
  std::vector<float> lambda(order);
  auto start = Clock::now();
  eigen_float(q, lambda);
  validate_positive(lambda);
  for (float &value : lambda) {
    value = std::sqrt(value);
  }
  run.times.factor_seconds = seconds(start);

  std::vector<float> scaled = q;
  std::vector<float> root_f32(count);
  start = Clock::now();
  for (std::size_t column = 0; column < order; ++column) {
    for (std::size_t row = 0; row < order; ++row) {
      scaled[row + column * order] *= lambda[column];
    }
  }
  sgemm('N', 'T', order, scaled.data(), q.data(), root_f32.data());
  Matrix root(root_f32.begin(), root_f32.end());
  symmetrize(root, order);
  run.times.initial_seconds = seconds(start);

  Matrix residual;
  start = Clock::now();
  double residual_norm =
      relative_residual(problem.matrix, root, order, &residual);
  run.times.residual_seconds += seconds(start);
  run.steps.push_back(
      {0, residual_norm, relative_error(root, problem.reference_root)});

  std::vector<float> residual_f32(count);
  std::vector<float> workspace(count);
  std::vector<float> coordinates(count);
  std::vector<float> correction(count);
  for (std::size_t iteration = 0;
       residual_norm > tolerance && iteration < max_corrections; ++iteration) {
    start = Clock::now();
    std::transform(residual.begin(), residual.end(), residual_f32.begin(),
                   [](double value) { return static_cast<float>(value); });
    sgemm('T', 'N', order, q.data(), residual_f32.data(), workspace.data());
    sgemm('N', 'N', order, workspace.data(), q.data(), coordinates.data());
    for (std::size_t column = 0; column < order; ++column) {
      for (std::size_t row = 0; row <= column; ++row) {
        const std::size_t upper = row + column * order;
        const std::size_t lower = column + row * order;
        float value = 0.5f * (coordinates[upper] + coordinates[lower]);
        value /= lambda[row] + lambda[column];
        coordinates[upper] = value;
        coordinates[lower] = value;
      }
    }
    sgemm('N', 'N', order, q.data(), coordinates.data(), workspace.data());
    sgemm('N', 'T', order, workspace.data(), q.data(), correction.data());
    for (std::size_t index = 0; index < count; ++index) {
      root[index] += static_cast<double>(correction[index]);
    }
    symmetrize(root, order);
    run.times.correction_seconds += seconds(start);

    start = Clock::now();
    residual_norm = relative_residual(problem.matrix, root, order, &residual);
    run.times.residual_seconds += seconds(start);
    run.steps.push_back({iteration + 1, residual_norm,
                         relative_error(root, problem.reference_root)});
  }
  run.converged = residual_norm <= tolerance;
  run.times.total_seconds = seconds(total_start);
  return run;
}

Run accelerate_schur(const Problem &problem) {
  Run run;
  run.stage = "baseline";
  run.backend = "accelerate_schur";
  run.format = "fp64";
  run.case_name = problem.name;
  run.order = problem.order;
  run.condition = problem.condition;
  const auto total_start = Clock::now();
  Matrix triangular = problem.matrix;
  Matrix vectors(elements(problem.order));
  std::vector<double> real(problem.order);
  std::vector<double> imaginary(problem.order);
  __LAPACK_int order = lapack_order(problem.order);
  const char compute_vectors = 'V';
  const char no_sort = 'N';
  __LAPACK_int selected = 0;
  __LAPACK_int work_size = -1;
  __LAPACK_int info = 0;
  double work_query = 0.0;
  auto start = Clock::now();
  dgees_(&compute_vectors, &no_sort, nullptr, &order, triangular.data(), &order,
         &selected, real.data(), imaginary.data(), vectors.data(), &order,
         &work_query, &work_size, nullptr, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate dgees workspace query failed.");
  }
  work_size = static_cast<__LAPACK_int>(std::ceil(work_query));
  std::vector<double> work(static_cast<std::size_t>(work_size));
  dgees_(&compute_vectors, &no_sort, nullptr, &order, triangular.data(), &order,
         &selected, real.data(), imaginary.data(), vectors.data(), &order,
         work.data(), &work_size, nullptr, &info);
  if (info != 0) {
    throw std::runtime_error("Accelerate dgees failed.");
  }
  run.times.factor_seconds = seconds(start);

  start = Clock::now();
  Matrix triangular_root(elements(problem.order), 0.0);
  for (std::size_t index = 0; index < problem.order; ++index) {
    if (std::abs(imaginary[index]) > 1.0e-10 ||
        !(triangular[index + index * problem.order] > 0.0)) {
      throw std::runtime_error(
          "The SPD Schur diagonal is invalid at index " +
          std::to_string(index) + ": real=" +
          std::to_string(triangular[index + index * problem.order]) +
          ", imaginary=" + std::to_string(imaginary[index]) + '.');
    }
    triangular_root[index + index * problem.order] =
        std::sqrt(triangular[index + index * problem.order]);
  }
  for (std::size_t distance = 1; distance < problem.order; ++distance) {
    for (std::size_t row = 0; row + distance < problem.order; ++row) {
      const std::size_t column = row + distance;
      double value = triangular[row + column * problem.order];
      for (std::size_t inner = row + 1; inner < column; ++inner) {
        value -= triangular_root[row + inner * problem.order] *
                 triangular_root[inner + column * problem.order];
      }
      value /= triangular_root[row + row * problem.order] +
               triangular_root[column + column * problem.order];
      triangular_root[row + column * problem.order] = value;
    }
  }
  Matrix workspace(elements(problem.order));
  Matrix root(elements(problem.order));
  dgemm('N', 'N', problem.order, vectors.data(), triangular_root.data(),
        workspace.data());
  dgemm('N', 'T', problem.order, workspace.data(), vectors.data(), root.data());
  run.times.initial_seconds = seconds(start);

  start = Clock::now();
  const double residual =
      relative_residual(problem.matrix, root, problem.order, nullptr);
  run.times.residual_seconds = seconds(start);
  run.steps.push_back(
      {0, residual, relative_error(root, problem.reference_root)});
  run.converged = residual <= 1.0e-12;
  run.times.total_seconds = seconds(total_start);
  return run;
}

void write_runs(const std::string &directory, const std::vector<Run> &runs) {
  std::filesystem::create_directories(directory);
  std::ofstream summary(directory + "/summary.csv");
  std::ofstream steps(directory + "/iterations.csv");
  if (!summary || !steps) {
    throw std::runtime_error("Cannot create experiment CSV files.");
  }
  summary << "stage,backend,format,case,order,condition,converged,iterations,"
             "initial_error,final_error,final_residual,total_ms,factor_ms,"
             "initial_ms,residual_ms,correction_ms\n";
  steps << "stage,backend,format,case,order,condition,iteration,residual,"
           "forward_error\n";
  summary << std::setprecision(17);
  steps << std::setprecision(17);
  for (const Run &run : runs) {
    const Step &first = run.steps.front();
    const Step &last = run.steps.back();
    summary << run.stage << ',' << run.backend << ',' << run.format << ','
            << run.case_name << ',' << run.order << ',' << run.condition << ','
            << (run.converged ? 1 : 0) << ',' << last.iteration << ','
            << first.forward_error << ',' << last.forward_error << ','
            << last.residual << ',' << 1.0e3 * run.times.total_seconds << ','
            << 1.0e3 * run.times.factor_seconds << ','
            << 1.0e3 * run.times.initial_seconds << ','
            << 1.0e3 * run.times.residual_seconds << ','
            << 1.0e3 * run.times.correction_seconds << '\n';
    for (const Step &step : run.steps) {
      steps << run.stage << ',' << run.backend << ',' << run.format << ','
            << run.case_name << ',' << run.order << ',' << run.condition << ','
            << step.iteration << ',' << step.residual << ','
            << step.forward_error << '\n';
    }
  }
}

}  // namespace msqrt
