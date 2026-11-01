#include "numerics.hpp"

#include "core.hpp"

#include <vecLib/blas_new.h>
#include <vecLib/lapack.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace mixed_sqrt {

__LAPACK_int upwardSafeFloatWorkspaceSize(float query, const char *routine) {
  if (!(query > 0.0f) || !std::isfinite(query)) {
    fail(std::string("Accelerate ") + routine +
         " returned an invalid workspace size.");
  }
  const float upward_query =
      std::nextafter(query, std::numeric_limits<float>::infinity());
  const double rounded_up = std::ceil(static_cast<double>(upward_query));
  if (!std::isfinite(rounded_up) ||
      rounded_up >
          static_cast<double>(std::numeric_limits<__LAPACK_int>::max())) {
    fail(std::string("Accelerate ") + routine +
         " workspace size exceeds the LAPACK integer range.");
  }
  return static_cast<__LAPACK_int>(rounded_up);
}

void dgemm(char transpose_left, char transpose_right, size_t rows,
           size_t columns, size_t inner, const double *left,
           const double *right, double *output) {
  const __LAPACK_int m = static_cast<__LAPACK_int>(rows);
  const __LAPACK_int n = static_cast<__LAPACK_int>(columns);
  const __LAPACK_int k = static_cast<__LAPACK_int>(inner);
  const __LAPACK_int leading_left = transpose_left == 'N' ? m : k;
  const __LAPACK_int leading_right = transpose_right == 'N' ? k : n;
  const __LAPACK_int leading_output = m;
  const double alpha = 1.0;
  const double beta = 0.0;
  dgemm_(&transpose_left, &transpose_right, &m, &n, &k, &alpha, left,
         &leading_left, right, &leading_right, &beta, output, &leading_output);
}

std::vector<double> symmetricSquare(const std::vector<double> &matrix,
                                    size_t order) {
  const __LAPACK_int dimension = static_cast<__LAPACK_int>(order);
  const char triangle = 'U';
  const char transpose = 'N';
  const double alpha = 1.0;
  const double beta = 0.0;
  std::vector<double> product(order * order, 0.0);
  dsyrk_(&triangle, &transpose, &dimension, &dimension, &alpha, matrix.data(),
         &dimension, &beta, product.data(), &dimension);
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = column + 1; row < order; ++row) {
      product[row + column * order] = product[column + row * order];
    }
  }
  return product;
}

void symmetrizeInPlace(std::vector<double> &matrix, size_t order) {
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = column + 1; row < order; ++row) {
      const double value =
          0.5 * (matrix[row + column * order] + matrix[column + row * order]);
      matrix[row + column * order] = value;
      matrix[column + row * order] = value;
    }
  }
}

void symmetricEigenDouble(std::vector<double> &matrix,
                          std::vector<double> &values) {
  const __LAPACK_int order = static_cast<__LAPACK_int>(values.size());
  const __LAPACK_int leading_dimension = order;
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int info = 0;
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_work_size = -1;
  double work_query = 0.0;
  __LAPACK_int integer_work_query = 0;

  dsyevd_(&job, &triangle, &order, matrix.data(), &leading_dimension,
          values.data(), &work_query, &work_size, &integer_work_query,
          &integer_work_size, &info);
  if (info != 0) {
    fail("Accelerate dsyevd workspace query failed.");
  }

  work_size = static_cast<__LAPACK_int>(work_query);
  integer_work_size = integer_work_query;
  std::vector<double> work(static_cast<size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<size_t>(integer_work_size));
  dsyevd_(&job, &triangle, &order, matrix.data(), &leading_dimension,
          values.data(), work.data(), &work_size, integer_work.data(),
          &integer_work_size, &info);
  if (info != 0) {
    fail("Accelerate dsyevd failed.");
  }
}

void symmetricEigenFloat(std::vector<float> &matrix,
                         std::vector<float> &values) {
  const __LAPACK_int order = static_cast<__LAPACK_int>(values.size());
  const __LAPACK_int leading_dimension = order;
  const char job = 'V';
  const char triangle = 'U';
  __LAPACK_int info = 0;
  __LAPACK_int work_size = -1;
  __LAPACK_int integer_work_size = -1;
  float work_query = 0.0f;
  __LAPACK_int integer_work_query = 0;

  ssyevd_(&job, &triangle, &order, matrix.data(), &leading_dimension,
          values.data(), &work_query, &work_size, &integer_work_query,
          &integer_work_size, &info);
  if (info != 0) {
    fail("Accelerate ssyevd workspace query failed.");
  }

  work_size = upwardSafeFloatWorkspaceSize(work_query, "ssyevd");
  integer_work_size = integer_work_query;
  std::vector<float> work(static_cast<size_t>(work_size));
  std::vector<__LAPACK_int> integer_work(
      static_cast<size_t>(integer_work_size));
  ssyevd_(&job, &triangle, &order, matrix.data(), &leading_dimension,
          values.data(), work.data(), &work_size, integer_work.data(),
          &integer_work_size, &info);
  if (info != 0) {
    fail("Accelerate ssyevd failed.");
  }
}

std::vector<double> makeOrthogonal(size_t order) {
  std::vector<double> vectors(order * order);
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = 0; row < order; ++row) {
      vectors[row + column * order] =
          std::sin(0.173 * static_cast<double>((row + 1) * (column + 2))) +
          std::cos(0.117 * static_cast<double>((row + 3) * (column + 1)));
    }
    for (size_t prior = 0; prior < column; ++prior) {
      double projection = 0.0;
      for (size_t row = 0; row < order; ++row) {
        projection +=
            vectors[row + prior * order] * vectors[row + column * order];
      }
      for (size_t row = 0; row < order; ++row) {
        vectors[row + column * order] -=
            projection * vectors[row + prior * order];
      }
    }
    double norm = 0.0;
    for (size_t row = 0; row < order; ++row) {
      norm += vectors[row + column * order] * vectors[row + column * order];
    }
    norm = std::sqrt(norm);
    if (norm <= std::numeric_limits<double>::epsilon()) {
      fail("The deterministic orthogonal basis is singular.");
    }
    for (size_t row = 0; row < order; ++row) {
      vectors[row + column * order] /= norm;
    }
  }
  return vectors;
}

std::vector<double> makeSpdMatrix(size_t order, double condition) {
  if (!std::isfinite(condition) || condition < 1.0) {
    fail("The SPD condition target must be finite and at least one.");
  }
  const std::vector<double> vectors = makeOrthogonal(order);
  std::vector<double> scaled_vectors = vectors;
  const double log_condition = std::log(condition);
  for (size_t column = 0; column < order; ++column) {
    const double fraction = static_cast<double>(column) /
                            static_cast<double>(std::max<size_t>(1, order - 1));
    const double eigenvalue = std::exp(log_condition * fraction);
    for (size_t row = 0; row < order; ++row) {
      scaled_vectors[row + column * order] *= eigenvalue;
    }
  }

  std::vector<double> matrix(order * order, 0.0);
  dgemm('N', 'T', order, order, order, scaled_vectors.data(), vectors.data(),
        matrix.data());
  return matrix;
}

std::vector<double> rootFromEigen(const std::vector<double> &eigenvectors,
                                  const std::vector<double> &eigenvalues,
                                  size_t order) {
  std::vector<double> weighted = eigenvectors;
  for (size_t column = 0; column < order; ++column) {
    const double root = std::sqrt(eigenvalues[column]);
    for (size_t row = 0; row < order; ++row) {
      weighted[row + column * order] *= root;
    }
  }
  std::vector<double> root(order * order, 0.0);
  dgemm('N', 'T', order, order, order, weighted.data(), eigenvectors.data(),
        root.data());
  return root;
}

double relativeResidual(const std::vector<double> &matrix,
                        const std::vector<double> &root, size_t order,
                        std::vector<double> *residual_output) {
  std::vector<double> product = symmetricSquare(root, order);
  double matrix_norm_squared = 0.0;
  double residual_norm_squared = 0.0;
  for (size_t index = 0; index < matrix.size(); ++index) {
    const double residual = matrix[index] - product[index];
    matrix_norm_squared += matrix[index] * matrix[index];
    residual_norm_squared += residual * residual;
    if (residual_output != nullptr) {
      (*residual_output)[index] = residual;
    }
  }
  return std::sqrt(residual_norm_squared / matrix_norm_squared);
}

double relativeError(const std::vector<double> &root,
                     const std::vector<double> &reference_root) {
  double reference_norm_squared = 0.0;
  double error_norm_squared = 0.0;
  for (size_t index = 0; index < root.size(); ++index) {
    const double difference = root[index] - reference_root[index];
    reference_norm_squared += reference_root[index] * reference_root[index];
    error_norm_squared += difference * difference;
  }
  return std::sqrt(error_norm_squared / reference_norm_squared);
}

double frobeniusDifference(const std::vector<double> &left,
                           const std::vector<double> &right) {
  double squared_norm = 0.0;
  for (size_t index = 0; index < left.size(); ++index) {
    const double difference = left[index] - right[index];
    squared_norm += difference * difference;
  }
  return std::sqrt(squared_norm);
}

double relativeSymmetryError(const std::vector<double> &matrix, size_t order) {
  double matrix_norm_squared = 0.0;
  double difference_norm_squared = 0.0;
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = 0; row < order; ++row) {
      const double value = matrix[row + column * order];
      const double difference = value - matrix[column + row * order];
      matrix_norm_squared += value * value;
      difference_norm_squared += difference * difference;
    }
  }
  return std::sqrt(difference_norm_squared / matrix_norm_squared);
}

bool allFinite(const std::vector<double> &values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

std::vector<float> transposeF32(const std::vector<float> &matrix,
                                size_t order) {
  std::vector<float> transposed(order * order);
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = 0; row < order; ++row) {
      transposed[row + column * order] = matrix[column + row * order];
    }
  }
  return transposed;
}

bool orthonormalizeColumnsF32(float *matrix, size_t order) {
  for (size_t column = 0; column < order; ++column) {
    float *current = matrix + column * order;
    for (size_t pass = 0; pass < 2; ++pass) {
      for (size_t previous = 0; previous < column; ++previous) {
        const float *basis = matrix + previous * order;
        double dot = 0.0;
        for (size_t row = 0; row < order; ++row) {
          dot += static_cast<double>(basis[row]) * current[row];
        }
        for (size_t row = 0; row < order; ++row) {
          current[row] -= static_cast<float>(dot * basis[row]);
        }
      }
    }
    double norm_squared = 0.0;
    for (size_t row = 0; row < order; ++row) {
      norm_squared += static_cast<double>(current[row]) * current[row];
    }
    const double norm = std::sqrt(norm_squared);
    if (!std::isfinite(norm) || norm <= std::numeric_limits<float>::min()) {
      return false;
    }
    for (size_t row = 0; row < order; ++row) {
      current[row] = static_cast<float>(current[row] / norm);
    }
  }
  return true;
}

void transposeF32(const float *matrix, float *transposed, size_t order) {
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = 0; row < order; ++row) {
      transposed[row + column * order] = matrix[column + row * order];
    }
  }
}

std::string conditionLabel(double condition) {
  std::ostringstream output;
  output << std::scientific << std::setprecision(0) << condition;
  std::string label = output.str();
  std::replace(label.begin(), label.end(), '+', 'p');
  std::replace(label.begin(), label.end(), '-', 'm');
  std::replace(label.begin(), label.end(), '.', '_');
  return label;
}

} // namespace mixed_sqrt
