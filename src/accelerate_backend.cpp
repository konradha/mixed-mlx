#include "accelerate_backend.hpp"

#include "benchmark.hpp"
#include "core.hpp"
#include "numerics.hpp"
#include "precision.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <type_traits>
#include <vecLib/blas_new.h>
#include <vecLib/lapack.h>

namespace mixed_sqrt {

__LAPACK_int checkedLapackOrder(size_t order) {
  if (order < 2 ||
      order > static_cast<size_t>(std::numeric_limits<__LAPACK_int>::max())) {
    fail("The matrix order is outside the LAPACK dimension range.");
  }
  const size_t elements = checkedElementCount(order, order);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
    fail("The matrix order exceeds safe host dimensions.");
  }
  return static_cast<__LAPACK_int>(order);
}

void sgemm(char transpose_left, char transpose_right, size_t rows,
           size_t columns, size_t inner, const float *left, const float *right,
           float *output) {
  const __LAPACK_int m = checkedLapackOrder(rows);
  const __LAPACK_int n = checkedLapackOrder(columns);
  const __LAPACK_int k = checkedLapackOrder(inner);
  const __LAPACK_int leading_left = transpose_left == 'N' ? m : k;
  const __LAPACK_int leading_right = transpose_right == 'N' ? k : n;
  const __LAPACK_int leading_output = m;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  sgemm_(&transpose_left, &transpose_right, &m, &n, &k, &alpha, left,
         &leading_left, right, &leading_right, &beta, output, &leading_output);
}

void ssymm(char side, char triangle, size_t order, const float *symmetric,
           const float *other, float *output) {
  const __LAPACK_int dimension = checkedLapackOrder(order);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  ssymm_(&side, &triangle, &dimension, &dimension, &alpha, symmetric,
         &dimension, other, &dimension, &beta, output, &dimension);
}

template <typename Real> struct RealSchurFactors {
  std::vector<Real> form;
  std::vector<Real> vectors;
  std::vector<Real> real_eigenvalues;
  std::vector<Real> imaginary_eigenvalues;
};

template <typename Real> __LAPACK_bool selectNoEigenvalues(Real *, Real *) {
  return static_cast<__LAPACK_bool>(0);
}

template <typename Real>
RealSchurFactors<Real> realSchur(const std::vector<double> &matrix,
                                 size_t order) {
  const __LAPACK_int n = checkedLapackOrder(order);
  const __LAPACK_int leading_dimension = n;
  const char job_vectors = 'V';
  const char no_sort = 'N';
  __LAPACK_int selected_dimension = 0;
  __LAPACK_int info = 0;
  __LAPACK_int work_size = -1;
  Real work_query = Real{0};
  RealSchurFactors<Real> factors{
      std::vector<Real>(matrix.begin(), matrix.end()),
      std::vector<Real>(checkedElementCount(order, order)),
      std::vector<Real>(order),
      std::vector<Real>(order),
  };
  const auto compute = [&](Real *work) {
    if constexpr (std::is_same_v<Real, float>) {
      sgees_(&job_vectors, &no_sort, selectNoEigenvalues<float>, &n,
             factors.form.data(), &leading_dimension, &selected_dimension,
             factors.real_eigenvalues.data(),
             factors.imaginary_eigenvalues.data(), factors.vectors.data(),
             &leading_dimension, work, &work_size, nullptr, &info);
    } else {
      dgees_(&job_vectors, &no_sort, selectNoEigenvalues<double>, &n,
             factors.form.data(), &leading_dimension, &selected_dimension,
             factors.real_eigenvalues.data(),
             factors.imaginary_eigenvalues.data(), factors.vectors.data(),
             &leading_dimension, work, &work_size, nullptr, &info);
    }
  };
  compute(&work_query);
  if (info != 0 || !std::isfinite(work_query)) {
    fail(std::is_same_v<Real, float>
             ? "Accelerate sgees workspace query failed."
             : "Accelerate dgees workspace query failed.");
  }
  if constexpr (std::is_same_v<Real, float>) {
    work_size = upwardSafeFloatWorkspaceSize(work_query, "sgees");
  } else {
    work_size =
        std::max<__LAPACK_int>(1, static_cast<__LAPACK_int>(work_query));
  }
  std::vector<Real> work(static_cast<size_t>(work_size));
  compute(work.data());
  const auto finite = [](const std::vector<Real> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](Real value) { return std::isfinite(value); });
  };
  if (info != 0 || !finite(factors.form) || !finite(factors.vectors) ||
      !finite(factors.real_eigenvalues) ||
      !finite(factors.imaginary_eigenvalues)) {
    fail(std::is_same_v<Real, float> ? "Accelerate sgees failed."
                                     : "Accelerate dgees failed.");
  }
  return factors;
}

struct SchurBlock {
  size_t start;
  size_t size;
};

template <typename Real>
std::vector<SchurBlock> schurBlocks(const std::vector<Real> &form,
                                    size_t order) {
  std::vector<SchurBlock> blocks;
  blocks.reserve(order);
  size_t index = 0;
  while (index < order) {
    const bool pair =
        index + 1 < order && form[(index + 1) + index * order] != Real{0};
    blocks.push_back({index, pair ? size_t{2} : size_t{1}});
    index += pair ? 2 : 1;
  }
  return blocks;
}

constexpr size_t schur_tile_limit = 32;

template <typename Real>
std::vector<SchurBlock> schurTiles(const std::vector<Real> &form,
                                   size_t order) {
  constexpr size_t tile_limit = schur_tile_limit;
  const std::vector<SchurBlock> atomic_blocks = schurBlocks(form, order);
  std::vector<SchurBlock> tiles;
  tiles.reserve((order + tile_limit - 1) / tile_limit);
  for (const SchurBlock &atomic : atomic_blocks) {
    if (tiles.empty() || tiles.back().size + atomic.size > tile_limit) {
      tiles.push_back(atomic);
    } else {
      tiles.back().size += atomic.size;
    }
  }
  return tiles;
}

template <typename Real>
void solveSmallSylvester(const Real *left, size_t leading_left,
                         size_t left_order, const Real *right,
                         size_t leading_right, size_t right_order,
                         const Real *constant, size_t leading_constant,
                         Real *solution, size_t leading_solution) {
  const size_t unknowns = left_order * right_order;
  Real equations[4][5] = {};
  for (size_t column = 0; column < right_order; ++column) {
    for (size_t row = 0; row < left_order; ++row) {
      const size_t equation = row + column * left_order;
      for (size_t inner = 0; inner < left_order; ++inner) {
        equations[equation][inner + column * left_order] +=
            left[row + inner * leading_left];
      }
      for (size_t inner = 0; inner < right_order; ++inner) {
        equations[equation][row + inner * left_order] +=
            right[inner + column * leading_right];
      }
      equations[equation][unknowns] = constant[row + column * leading_constant];
    }
  }
  for (size_t pivot = 0; pivot < unknowns; ++pivot) {
    size_t best = pivot;
    for (size_t row = pivot + 1; row < unknowns; ++row) {
      if (std::abs(equations[row][pivot]) > std::abs(equations[best][pivot])) {
        best = row;
      }
    }
    if (!(std::abs(equations[best][pivot]) >
          Real{16} * std::numeric_limits<Real>::epsilon())) {
      fail("The real Schur Sylvester operator is singular.");
    }
    if (best != pivot) {
      for (size_t column = pivot; column <= unknowns; ++column) {
        std::swap(equations[pivot][column], equations[best][column]);
      }
    }
    const Real diagonal = equations[pivot][pivot];
    for (size_t column = pivot; column <= unknowns; ++column) {
      equations[pivot][column] /= diagonal;
    }
    for (size_t row = 0; row < unknowns; ++row) {
      if (row == pivot) {
        continue;
      }
      const Real multiplier = equations[row][pivot];
      for (size_t column = pivot; column <= unknowns; ++column) {
        equations[row][column] -= multiplier * equations[pivot][column];
      }
    }
  }
  for (size_t column = 0; column < right_order; ++column) {
    for (size_t row = 0; row < left_order; ++row) {
      const Real value = equations[row + column * left_order][unknowns];
      if (!std::isfinite(value)) {
        fail("The real Schur square root is nonfinite.");
      }
      solution[row + column * leading_solution] = value;
    }
  }
}

template <typename Real>
std::vector<Real> quasiTriangularSquareRoot(const std::vector<Real> &form,
                                            size_t order) {
  const std::vector<SchurBlock> blocks = schurBlocks(form, order);
  std::vector<Real> root(checkedElementCount(order, order), Real{0});
  for (const SchurBlock &block : blocks) {
    const size_t index = block.start;
    if (block.size == 1) {
      const Real value = form[index + index * order];
      if (!(value > Real{0}) || !std::isfinite(value)) {
        fail("A real Schur block is on the closed negative real axis.");
      }
      root[index + index * order] = std::sqrt(value);
      continue;
    }
    const Real a = form[index + index * order];
    const Real b = form[index + (index + 1) * order];
    const Real c = form[(index + 1) + index * order];
    const Real d = form[(index + 1) + (index + 1) * order];
    const Real determinant = a * d - b * c;
    if (!(determinant > Real{0}) || !std::isfinite(determinant)) {
      fail("A real 2 by 2 Schur block has no real principal square root.");
    }
    const Real scalar_root = std::sqrt(determinant);
    const Real denominator_squared = a + d + Real{2} * scalar_root;
    if (!(denominator_squared > Real{0}) ||
        !std::isfinite(denominator_squared)) {
      fail("A real 2 by 2 Schur block is on the principal branch cut.");
    }
    const Real denominator = std::sqrt(denominator_squared);
    root[index + index * order] = (a + scalar_root) / denominator;
    root[index + (index + 1) * order] = b / denominator;
    root[(index + 1) + index * order] = c / denominator;
    root[(index + 1) + (index + 1) * order] = (d + scalar_root) / denominator;
  }
  for (size_t distance = 1; distance < blocks.size(); ++distance) {
    for (size_t left_block = 0; left_block + distance < blocks.size();
         ++left_block) {
      const size_t right_block = left_block + distance;
      const SchurBlock &left = blocks[left_block];
      const SchurBlock &right = blocks[right_block];
      Real constant[4] = {};
      for (size_t column = 0; column < right.size; ++column) {
        for (size_t row = 0; row < left.size; ++row) {
          Real value =
              form[(left.start + row) + (right.start + column) * order];
          for (size_t inner = left.start + left.size; inner < right.start;
               ++inner) {
            value -= root[(left.start + row) + inner * order] *
                     root[inner + (right.start + column) * order];
          }
          constant[row + column * left.size] = value;
        }
      }
      solveSmallSylvester(
          root.data() + left.start + left.start * order, order, left.size,
          root.data() + right.start + right.start * order, order, right.size,
          constant, left.size, root.data() + left.start + right.start * order,
          order);
    }
  }
  return root;
}

template <typename Real>
void subtractSchurProduct(size_t rows, size_t columns, size_t inner,
                          const Real *left, size_t leading_left,
                          const Real *right, size_t leading_right, Real *output,
                          size_t leading_output) {
  const __LAPACK_int m = static_cast<__LAPACK_int>(rows);
  const __LAPACK_int n = static_cast<__LAPACK_int>(columns);
  const __LAPACK_int k = static_cast<__LAPACK_int>(inner);
  const __LAPACK_int lda = static_cast<__LAPACK_int>(leading_left);
  const __LAPACK_int ldb = static_cast<__LAPACK_int>(leading_right);
  const __LAPACK_int ldc = static_cast<__LAPACK_int>(leading_output);
  const char no_transpose = 'N';
  const Real alpha = Real{-1};
  const Real beta = Real{1};
  if constexpr (std::is_same_v<Real, float>) {
    sgemm_(&no_transpose, &no_transpose, &m, &n, &k, &alpha, left, &lda, right,
           &ldb, &beta, output, &ldc);
  } else {
    dgemm_(&no_transpose, &no_transpose, &m, &n, &k, &alpha, left, &lda, right,
           &ldb, &beta, output, &ldc);
  }
}

template <typename Real>
void solveSchurTile(const Real *left, size_t leading_left, size_t left_order,
                    const Real *right, size_t leading_right, size_t right_order,
                    Real *solution, size_t leading_solution) {
  const char no_transpose = 'N';
  const __LAPACK_int positive = 1;
  const __LAPACK_int m = static_cast<__LAPACK_int>(left_order);
  const __LAPACK_int n = static_cast<__LAPACK_int>(right_order);
  const __LAPACK_int lda = static_cast<__LAPACK_int>(leading_left);
  const __LAPACK_int ldb = static_cast<__LAPACK_int>(leading_right);
  const __LAPACK_int ldc = static_cast<__LAPACK_int>(leading_solution);
  Real scale = Real{1};
  __LAPACK_int info = 0;
  if constexpr (std::is_same_v<Real, float>) {
    strsyl_(&no_transpose, &no_transpose, &positive, &m, &n, left, &lda, right,
            &ldb, solution, &ldc, &scale, &info);
  } else {
    dtrsyl_(&no_transpose, &no_transpose, &positive, &m, &n, left, &lda, right,
            &ldb, solution, &ldc, &scale, &info);
  }
  if (info != 0 || !(scale > Real{0}) || !std::isfinite(scale)) {
    fail(std::is_same_v<Real, float>
             ? "Accelerate strsyl failed for a Schur square-root tile."
             : "Accelerate dtrsyl failed for a Schur square-root tile.");
  }
  if (scale == Real{1}) {
    return;
  }
  const Real inverse_scale = Real{1} / scale;
  if (!std::isfinite(inverse_scale)) {
    fail("The Schur tile scaling is not recoverable.");
  }
  for (size_t column = 0; column < right_order; ++column) {
    for (size_t row = 0; row < left_order; ++row) {
      Real &value = solution[row + column * leading_solution];
      value *= inverse_scale;
      if (!std::isfinite(value)) {
        fail("The Schur tile scaling overflowed.");
      }
    }
  }
}

size_t schurTileSpan(const std::vector<SchurBlock> &tiles, size_t begin,
                     size_t end) {
  return tiles[end - 1].start + tiles[end - 1].size - tiles[begin].start;
}

void solveSchurRecursive(const float *left, size_t leading_left,
                         const std::vector<SchurBlock> &left_tiles,
                         size_t left_begin, size_t left_end, const float *right,
                         size_t leading_right,
                         const std::vector<SchurBlock> &right_tiles,
                         size_t right_begin, size_t right_end, float *solution,
                         size_t leading_solution) {
  const size_t left_tile_count = left_end - left_begin;
  const size_t right_tile_count = right_end - right_begin;
  if (left_tile_count == 1 && right_tile_count == 1) {
    solveSchurTile(left, leading_left,
                   schurTileSpan(left_tiles, left_begin, left_end), right,
                   leading_right,
                   schurTileSpan(right_tiles, right_begin, right_end), solution,
                   leading_solution);
    return;
  }

  if (left_tile_count > 1 && right_tile_count > 1) {
    const size_t left_middle = left_begin + left_tile_count / 2;
    const size_t right_middle = right_begin + right_tile_count / 2;
    const size_t left_first =
        left_tiles[left_middle].start - left_tiles[left_begin].start;
    const size_t left_second = schurTileSpan(left_tiles, left_middle, left_end);
    const size_t right_first =
        right_tiles[right_middle].start - right_tiles[right_begin].start;
    const size_t right_second =
        schurTileSpan(right_tiles, right_middle, right_end);

    solveSchurRecursive(left + left_first + left_first * leading_left,
                        leading_left, left_tiles, left_middle, left_end, right,
                        leading_right, right_tiles, right_begin, right_middle,
                        solution + left_first, leading_solution);
    subtractSchurProduct(left_first, right_first, left_second,
                         left + left_first * leading_left, leading_left,
                         solution + left_first, leading_solution, solution,
                         leading_solution);
    subtractSchurProduct(left_second, right_second, right_first,
                         solution + left_first, leading_solution,
                         right + right_first * leading_right, leading_right,
                         solution + left_first + right_first * leading_solution,
                         leading_solution);
    solveSchurRecursive(left, leading_left, left_tiles, left_begin, left_middle,
                        right, leading_right, right_tiles, right_begin,
                        right_middle, solution, leading_solution);
    solveSchurRecursive(left + left_first + left_first * leading_left,
                        leading_left, left_tiles, left_middle, left_end,
                        right + right_first + right_first * leading_right,
                        leading_right, right_tiles, right_middle, right_end,
                        solution + left_first + right_first * leading_solution,
                        leading_solution);
    subtractSchurProduct(
        left_first, right_second, left_second, left + left_first * leading_left,
        leading_left, solution + left_first + right_first * leading_solution,
        leading_solution, solution + right_first * leading_solution,
        leading_solution);
    subtractSchurProduct(
        left_first, right_second, right_first, solution, leading_solution,
        right + right_first * leading_right, leading_right,
        solution + right_first * leading_solution, leading_solution);
    solveSchurRecursive(left, leading_left, left_tiles, left_begin, left_middle,
                        right + right_first + right_first * leading_right,
                        leading_right, right_tiles, right_middle, right_end,
                        solution + right_first * leading_solution,
                        leading_solution);
    return;
  }

  if (left_tile_count > 1) {
    const size_t left_middle = left_begin + left_tile_count / 2;
    const size_t left_first =
        left_tiles[left_middle].start - left_tiles[left_begin].start;
    const size_t left_second = schurTileSpan(left_tiles, left_middle, left_end);
    const size_t right_order =
        schurTileSpan(right_tiles, right_begin, right_end);
    solveSchurRecursive(left + left_first + left_first * leading_left,
                        leading_left, left_tiles, left_middle, left_end, right,
                        leading_right, right_tiles, right_begin, right_end,
                        solution + left_first, leading_solution);
    subtractSchurProduct(left_first, right_order, left_second,
                         left + left_first * leading_left, leading_left,
                         solution + left_first, leading_solution, solution,
                         leading_solution);
    solveSchurRecursive(left, leading_left, left_tiles, left_begin, left_middle,
                        right, leading_right, right_tiles, right_begin,
                        right_end, solution, leading_solution);
    return;
  }

  const size_t right_middle = right_begin + right_tile_count / 2;
  const size_t left_order = schurTileSpan(left_tiles, left_begin, left_end);
  const size_t right_first =
      right_tiles[right_middle].start - right_tiles[right_begin].start;
  const size_t right_second =
      schurTileSpan(right_tiles, right_middle, right_end);
  solveSchurRecursive(left, leading_left, left_tiles, left_begin, left_end,
                      right, leading_right, right_tiles, right_begin,
                      right_middle, solution, leading_solution);
  subtractSchurProduct(left_order, right_second, right_first, solution,
                       leading_solution, right + right_first * leading_right,
                       leading_right, solution + right_first * leading_solution,
                       leading_solution);
  solveSchurRecursive(left, leading_left, left_tiles, left_begin, left_end,
                      right + right_first + right_first * leading_right,
                      leading_right, right_tiles, right_middle, right_end,
                      solution + right_first * leading_solution,
                      leading_solution);
}

template <typename Real>
std::vector<Real>
blockedQuasiTriangularSquareRoot(const std::vector<Real> &form, size_t order) {
  const std::vector<SchurBlock> tiles = schurTiles(form, order);
  std::vector<Real> root(checkedElementCount(order, order), Real{0});
  std::vector<Real> tile_workspace(schur_tile_limit * schur_tile_limit);
  for (const SchurBlock &tile : tiles) {
    std::vector<Real> diagonal(tile.size * tile.size);
    for (size_t column = 0; column < tile.size; ++column) {
      for (size_t row = 0; row < tile.size; ++row) {
        diagonal[row + column * tile.size] =
            form[(tile.start + row) + (tile.start + column) * order];
      }
    }
    const std::vector<Real> diagonal_root =
        quasiTriangularSquareRoot(diagonal, tile.size);
    for (size_t column = 0; column < tile.size; ++column) {
      for (size_t row = 0; row < tile.size; ++row) {
        root[(tile.start + row) + (tile.start + column) * order] =
            diagonal_root[row + column * tile.size];
      }
    }
  }
  for (size_t distance = 1; distance < tiles.size(); ++distance) {
    for (size_t left_index = 0; left_index + distance < tiles.size();
         ++left_index) {
      const size_t right_index = left_index + distance;
      const SchurBlock &left = tiles[left_index];
      const SchurBlock &right = tiles[right_index];
      for (size_t column = 0; column < right.size; ++column) {
        for (size_t row = 0; row < left.size; ++row) {
          tile_workspace[row + column * left.size] =
              form[(left.start + row) + (right.start + column) * order];
        }
      }
      for (size_t middle_index = left_index + 1; middle_index < right_index;
           ++middle_index) {
        const SchurBlock &middle = tiles[middle_index];
        subtractSchurProduct(left.size, right.size, middle.size,
                             root.data() + left.start + middle.start * order,
                             order,
                             root.data() + middle.start + right.start * order,
                             order, tile_workspace.data(), left.size);
      }
      solveSchurTile(root.data() + left.start + left.start * order, order,
                     left.size, root.data() + right.start + right.start * order,
                     order, right.size, tile_workspace.data(), left.size);
      for (size_t column = 0; column < right.size; ++column) {
        for (size_t row = 0; row < left.size; ++row) {
          root[(left.start + row) + (right.start + column) * order] =
              tile_workspace[row + column * left.size];
        }
      }
    }
  }
  return root;
}

void generalResidual(const std::vector<double> &matrix,
                     const std::vector<double> &root, size_t order,
                     std::vector<double> &residual,
                     std::vector<double> &product, double &relative_norm) {
  dgemm('N', 'N', order, order, order, root.data(), root.data(),
        product.data());
  double matrix_norm_squared = 0.0;
  double residual_norm_squared = 0.0;
  for (size_t index = 0; index < matrix.size(); ++index) {
    const double value = matrix[index] - product[index];
    residual[index] = value;
    matrix_norm_squared += matrix[index] * matrix[index];
    residual_norm_squared += value * value;
  }
  if (!(matrix_norm_squared > 0.0) || !std::isfinite(matrix_norm_squared) ||
      !std::isfinite(residual_norm_squared)) {
    relative_norm = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  relative_norm = std::sqrt(residual_norm_squared / matrix_norm_squared);
}

void symmetricResidual(const std::vector<double> &matrix,
                       const std::vector<double> &root, size_t order,
                       std::vector<double> &residual,
                       std::vector<double> &product, double &relative_norm) {
  const __LAPACK_int dimension = static_cast<__LAPACK_int>(order);
  const char triangle = 'U';
  const char no_transpose = 'N';
  const double alpha = 1.0;
  const double beta = 0.0;
  dsyrk_(&triangle, &no_transpose, &dimension, &dimension, &alpha, root.data(),
         &dimension, &beta, product.data(), &dimension);
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = column + 1; row < order; ++row) {
      product[row + column * order] = product[column + row * order];
    }
  }

  double matrix_norm_squared = 0.0;
  double residual_norm_squared = 0.0;
  for (size_t column = 0; column < order; ++column) {
    for (size_t row = 0; row <= column; ++row) {
      const size_t upper = row + column * order;
      const size_t lower = column + row * order;
      const double value = matrix[upper] - product[upper];
      residual[upper] = value;
      residual[lower] = value;
      const double weight = row == column ? 1.0 : 2.0;
      matrix_norm_squared += weight * matrix[upper] * matrix[upper];
      residual_norm_squared += weight * value * value;
    }
  }
  if (!(matrix_norm_squared > 0.0) || !std::isfinite(matrix_norm_squared) ||
      !std::isfinite(residual_norm_squared)) {
    relative_norm = std::numeric_limits<double>::quiet_NaN();
    return;
  }
  relative_norm = std::sqrt(residual_norm_squared / matrix_norm_squared);
}

std::vector<double> symmetricReferenceRoot(const std::vector<double> &matrix,
                                           size_t order) {
  std::vector<double> vectors = matrix;
  std::vector<double> values(order);
  symmetricEigenDouble(vectors, values);
  for (double value : values) {
    if (!(value > 0.0) || !std::isfinite(value)) {
      fail("The FP64 reference has a nonpositive eigenvalue.");
    }
  }
  std::vector<double> root = rootFromEigen(vectors, values, order);
  symmetrizeInPlace(root, order);
  return root;
}

std::vector<double> nonsymmetricReferenceRoot(const std::vector<double> &matrix,
                                              size_t order) {
  RealSchurFactors<double> factors = realSchur<double>(matrix, order);
  std::vector<double> schur_root =
      blockedQuasiTriangularSquareRoot(factors.form, order);
  std::vector<double> workspace(checkedElementCount(order, order));
  std::vector<double> root(checkedElementCount(order, order));
  dgemm('N', 'N', order, order, order, factors.vectors.data(),
        schur_root.data(), workspace.data());
  dgemm('N', 'T', order, order, order, workspace.data(), factors.vectors.data(),
        root.data());
  if (!allFinite(root)) {
    fail("The FP64 real Schur square root is nonfinite.");
  }
  return root;
}

class DeterministicGenerator {
public:
  explicit DeterministicGenerator(uint64_t seed)
      : state_(seed), has_spare_(false) {
    if (state_ == 0) {
      state_ = UINT64_C(0x9e3779b97f4a7c15);
    }
  }

  double uniform() {
    uint64_t value = state_;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    state_ = value;
    const uint64_t bits = (value * UINT64_C(2685821657736338717)) >> 11;
    return static_cast<double>(bits) * 0x1.0p-53;
  }

  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double first =
        std::max(uniform(), std::numeric_limits<double>::min());
    const double second = uniform();
    const double radius = std::sqrt(-2.0 * std::log(first));
    const double angle = 6.283185307179586476925286766559 * second;
    spare_ = radius * std::sin(angle);
    has_spare_ = true;
    return radius * std::cos(angle);
  }

private:
  uint64_t state_;
  double spare_ = 0.0;
  bool has_spare_;
};

double matrixInfinityNorm(const std::vector<double> &matrix, size_t order) {
  double maximum = 0.0;
  for (size_t row = 0; row < order; ++row) {
    double sum = 0.0;
    for (size_t column = 0; column < order; ++column) {
      sum += std::abs(matrix[row + column * order]);
    }
    maximum = std::max(maximum, sum);
  }
  if (!(maximum > 0.0) || !std::isfinite(maximum)) {
    fail("The prepared matrix norm is invalid.");
  }
  return maximum;
}

PreparedMatrix prepareControlledMatrix(size_t order, double condition) {
  checkedLapackOrder(order);
  PreparedMatrix prepared;
  prepared.values = makeSpdMatrix(order, condition);
  prepared.order = order;
  prepared.condition = condition;
  return prepared;
}

PreparedMatrix preparePaperMatrix(PaperTestSet test_set, size_t order) {
  if (test_set == PaperTestSet::None) {
    fail("A paper test set is required.");
  }
  PreparedMatrix prepared;
  checkedLapackOrder(order);
  prepared.order = order;
  prepared.paper_test_set = paperTestSetName(test_set);
  prepared.matrix_type =
      paperTestSetIsSymmetric(test_set) ? "symmetric" : "nonsymmetric";
  const bool uniform = test_set == PaperTestSet::I ||
                       test_set == PaperTestSet::III ||
                       test_set == PaperTestSet::V;
  const bool normal = test_set == PaperTestSet::II ||
                      test_set == PaperTestSet::IV ||
                      test_set == PaperTestSet::VI;
  if (uniform) {
    prepared.matrix_family = "uniform_random";
  } else if (normal) {
    prepared.matrix_family = "normal_random";
  } else {
    prepared.matrix_family = "hilbert_shifted";
  }
  const size_t elements = checkedElementCount(order, order);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
    fail("The paper matrix exceeds safe host dimensions.");
  }
  prepared.values.assign(elements, 0.0);
  prepared.shift_rule = test_set == PaperTestSet::VII
                            ? "fixed_hilbert_tau"
                            : "set_min_real_part_to_max_1_or_0.01_inf_norm";
  if (test_set == PaperTestSet::VII) {
    for (size_t column = 0; column < order; ++column) {
      for (size_t row = 0; row < order; ++row) {
        prepared.values[row + column * order] =
            1.0 / static_cast<double>(row + column + 1);
      }
      prepared.values[column + column * order] += 1.0e-6;
    }
    prepared.diagonal_shift = 1.0e-6;
  } else {
    const uint64_t seed = UINT64_C(0x243f6a8885a308d3) ^
                          (UINT64_C(0x9e3779b97f4a7c15) *
                           static_cast<uint64_t>(static_cast<int>(test_set)));
    prepared.rng_seed = seed;
    DeterministicGenerator generator(seed);
    if (paperTestSetIsSymmetric(test_set)) {
      for (size_t column = 0; column < order; ++column) {
        for (size_t row = column; row < order; ++row) {
          const double value =
              uniform ? generator.uniform() : generator.normal();
          prepared.values[row + column * order] = value;
          prepared.values[column + row * order] = value;
        }
      }
    } else {
      for (double &value : prepared.values) {
        value = uniform ? generator.uniform() : generator.normal();
      }
    }
  }
  if (!allFinite(prepared.values)) {
    fail("The generated paper matrix is nonfinite.");
  }
  if (test_set != PaperTestSet::VII) {
    const double separation =
        std::max(1.0, 0.01 * matrixInfinityNorm(prepared.values, order));
    double minimum_real_part = std::numeric_limits<double>::infinity();
    double maximum_real_part = -std::numeric_limits<double>::infinity();
    if (paperTestSetIsSymmetric(test_set)) {
      std::vector<double> discovery = prepared.values;
      std::vector<double> eigenvalues(order);
      symmetricEigenDouble(discovery, eigenvalues);
      minimum_real_part = eigenvalues.front();
      maximum_real_part = eigenvalues.back();
    } else {
      const RealSchurFactors<double> discovery =
          realSchur<double>(prepared.values, order);
      for (double value : discovery.real_eigenvalues) {
        minimum_real_part = std::min(minimum_real_part, value);
        maximum_real_part = std::max(maximum_real_part, value);
      }
    }
    if (!std::isfinite(minimum_real_part) ||
        !std::isfinite(maximum_real_part)) {
      fail("The paper matrix shift discovery is nonfinite.");
    }
    const double shift = std::max(0.0, separation - minimum_real_part);
    prepared.diagonal_shift = shift;
    for (size_t index = 0; index < order; ++index) {
      prepared.values[index + index * order] += shift;
    }
    if (paperTestSetIsSymmetric(test_set)) {
      const double smallest = minimum_real_part + shift;
      const double largest = maximum_real_part + shift;
      prepared.condition = largest / smallest;
    }
  } else {
    std::vector<double> discovery = prepared.values;
    std::vector<double> eigenvalues(order);
    symmetricEigenDouble(discovery, eigenvalues);
    if (!(eigenvalues.front() > 0.0) || !std::isfinite(eigenvalues.front()) ||
        !std::isfinite(eigenvalues.back())) {
      fail("Paper set VII is not positive definite.");
    }
    prepared.condition = eigenvalues.back() / eigenvalues.front();
  }
  if (!allFinite(prepared.values)) {
    fail("The shifted paper matrix is nonfinite.");
  }
  return prepared;
}

CaseResult runAccelerateCase(const PreparedMatrix &prepared,
                             size_t max_corrections, double tolerance,
                             const std::vector<double> *supplied_reference) {
  if (prepared.values.size() !=
      checkedElementCount(prepared.order, prepared.order)) {
    fail("The prepared matrix dimensions are inconsistent.");
  }
  if (!allFinite(prepared.values)) {
    fail("The prepared matrix is nonfinite.");
  }
  if (!std::isfinite(tolerance) || !(tolerance > 0.0)) {
    fail("The residual tolerance must be positive and finite.");
  }
  checkedLapackOrder(prepared.order);
  const bool symmetric = prepared.matrix_type == "symmetric";
  if (!symmetric && prepared.matrix_type != "nonsymmetric") {
    fail("The prepared matrix type is not supported.");
  }
  const size_t order = prepared.order;
  const size_t elements = checkedElementCount(order, order);

  CaseResult result;
  result.paper_test_set = prepared.paper_test_set;
  result.matrix_family = prepared.matrix_family;
  result.matrix_type = prepared.matrix_type;
  result.solver_backend = solverBackendName(SolverBackend::Accelerate);
  result.rng_seed = prepared.rng_seed;
  result.rounding_mode = "not_applicable";
  result.diagonal_shift = prepared.diagonal_shift;
  result.shift_rule = prepared.shift_rule;
  result.precision = "fp32";
  result.basis_precision = "fp32";
  result.spectrum_precision = "fp32";
  result.correction_mode =
      symmetric ? "paper_algorithm_3" : "paper_algorithm_2";
  result.precision_policy = "required_full_accuracy";
  result.execution_model = symmetric
                               ? "fp32_syevd_frozen_spectral_correction"
                               : "fp32_real_schur_frozen_sylvester_correction";
  result.kernel_path = "not_applicable";
  result.stopping_policy = "residual_or_12_corrections";
  result.dispatch_policy = "not_applicable";
  result.order = order;
  result.condition = prepared.condition;
  result.name = result.paper_test_set == "controlled"
                    ? "spd_n" + std::to_string(order) + "_k" +
                          conditionLabel(prepared.condition) +
                          "_accelerate_fp32"
                    : "paper_" + result.paper_test_set + "_accelerate_fp32";

  std::vector<double> computed_reference;
  if (supplied_reference == nullptr) {
    const Clock::time_point reference_start = Clock::now();
    computed_reference =
        symmetric ? symmetricReferenceRoot(prepared.values, order)
                  : nonsymmetricReferenceRoot(prepared.values, order);
    result.times.reference_seconds = elapsedSeconds(reference_start);
  }
  const std::vector<double> &reference_root =
      supplied_reference == nullptr ? computed_reference : *supplied_reference;
  if (reference_root.size() != elements || !allFinite(reference_root)) {
    fail("The supplied FP64 reference root is invalid.");
  }
  std::vector<double> residual(elements);
  std::vector<double> product(elements);
  if (symmetric) {
    symmetricResidual(prepared.values, reference_root, order, residual, product,
                      result.reference_residual);
  } else {
    generalResidual(prepared.values, reference_root, order, residual, product,
                    result.reference_residual);
  }
  if (!std::isfinite(result.reference_residual)) {
    fail("The FP64 reference residual is nonfinite.");
  }

  std::vector<float> factors;
  std::vector<float> factor_root;
  std::vector<SchurBlock> correction_tiles;
  const Clock::time_point mixed_start = Clock::now();
  const Clock::time_point factorization_start = Clock::now();
  if (symmetric) {
    factors.assign(prepared.values.begin(), prepared.values.end());
    std::vector<float> eigenvalues(order);
    symmetricEigenFloat(factors, eigenvalues);
    factor_root.resize(order);
    for (size_t index = 0; index < order; ++index) {
      if (!(eigenvalues[index] > 0.0f) || !std::isfinite(eigenvalues[index])) {
        fail("Accelerate ssyevd returned a nonpositive eigenvalue.");
      }
      factor_root[index] = std::sqrt(eigenvalues[index]);
    }
  } else {
    RealSchurFactors<float> schur = realSchur<float>(prepared.values, order);
    factors = std::move(schur.vectors);
    correction_tiles = schurTiles(schur.form, order);
    factor_root = blockedQuasiTriangularSquareRoot(schur.form, order);
  }
  result.times.low_eigen_seconds = elapsedSeconds(factorization_start);

  std::vector<float> workspace_one(elements);
  std::vector<float> workspace_two(elements);
  std::vector<float> correction(elements);
  std::vector<double> root(elements);
  const Clock::time_point initial_start = Clock::now();
  if (symmetric) {
    workspace_one = factors;
    for (size_t column = 0; column < order; ++column) {
      const float scale = factor_root[column];
      for (size_t row = 0; row < order; ++row) {
        workspace_one[row + column * order] *= scale;
      }
    }
    sgemm('N', 'T', order, order, order, workspace_one.data(), factors.data(),
          correction.data());
  } else {
    sgemm('N', 'N', order, order, order, factors.data(), factor_root.data(),
          workspace_one.data());
    sgemm('N', 'T', order, order, order, workspace_one.data(), factors.data(),
          correction.data());
  }
  if (!std::all_of(correction.begin(), correction.end(),
                   [](float value) { return std::isfinite(value); })) {
    fail("The FP32 initial square root is nonfinite.");
  }
  for (size_t index = 0; index < elements; ++index) {
    root[index] = static_cast<double>(correction[index]);
  }
  if (symmetric) {
    symmetrizeInPlace(root, order);
  }
  result.times.initial_gpu_seconds = elapsedSeconds(initial_start);
  result.frozen_factor_error = frobeniusDifference(root, reference_root);
  result.frozen_factor_relative_error = relativeError(root, reference_root);
  result.frozen_basis_orthogonality = std::numeric_limits<double>::quiet_NaN();
  result.frozen_sylvester_inverse_norm =
      std::numeric_limits<double>::quiet_NaN();
  result.frozen_contraction_bound = std::numeric_limits<double>::quiet_NaN();
  result.certificate_usable = false;

  const Clock::time_point first_residual_start = Clock::now();
  double residual_norm = 0.0;
  if (symmetric) {
    symmetricResidual(prepared.values, root, order, residual, product,
                      residual_norm);
  } else {
    generalResidual(prepared.values, root, order, residual, product,
                    residual_norm);
  }
  result.times.residual_fp64_seconds += elapsedSeconds(first_residual_start);
  double forward_error = relativeError(root, reference_root);
  result.history.push_back({0, residual_norm, forward_error});
  result.best_residual = residual_norm;
  result.best_forward_error = forward_error;
  result.best_iteration = 0;

  const auto finite_float_matrix = [](const std::vector<float> &values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
  };
  size_t iteration = 0;
  while (residual_norm > tolerance && iteration < max_corrections) {
    if (!allFinite(root) || !std::isfinite(residual_norm) ||
        !std::isfinite(forward_error)) {
      break;
    }
    const Clock::time_point correction_start = Clock::now();
    for (size_t index = 0; index < elements; ++index) {
      workspace_two[index] = static_cast<float>(residual[index]);
    }
    if (!finite_float_matrix(workspace_two)) {
      fail("The FP64 residual is outside the FP32 correction range.");
    }
    if (symmetric) {
      ssymm('L', 'U', order, workspace_two.data(), factors.data(),
            workspace_one.data());
      sgemm('T', 'N', order, order, order, factors.data(), workspace_one.data(),
            workspace_two.data());
      for (size_t column = 0; column < order; ++column) {
        for (size_t row = 0; row <= column; ++row) {
          const float denominator = factor_root[row] + factor_root[column];
          if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
            fail("The FP32 diagonal Sylvester operator is singular.");
          }
          const size_t upper = row + column * order;
          const size_t lower = column + row * order;
          float value = workspace_two[upper];
          if (row != column) {
            value = 0.5f * (value + workspace_two[lower]);
          }
          value /= denominator;
          workspace_two[upper] = value;
          workspace_two[lower] = value;
        }
      }
    } else {
      sgemm('T', 'N', order, order, order, factors.data(), workspace_two.data(),
            workspace_one.data());
      sgemm('N', 'N', order, order, order, workspace_one.data(), factors.data(),
            workspace_two.data());
      solveSchurRecursive(factor_root.data(), order, correction_tiles, 0,
                          correction_tiles.size(), factor_root.data(), order,
                          correction_tiles, 0, correction_tiles.size(),
                          workspace_two.data(), order);
    }
    if (!finite_float_matrix(workspace_two)) {
      fail("The FP32 Sylvester correction is nonfinite.");
    }
    if (symmetric) {
      ssymm('R', 'U', order, workspace_two.data(), factors.data(),
            workspace_one.data());
      sgemm('N', 'T', order, order, order, workspace_one.data(), factors.data(),
            correction.data());
    } else {
      sgemm('N', 'N', order, order, order, factors.data(), workspace_two.data(),
            workspace_one.data());
      sgemm('N', 'T', order, order, order, workspace_one.data(), factors.data(),
            correction.data());
    }
    if (!finite_float_matrix(correction)) {
      fail("The transformed FP32 correction is nonfinite.");
    }
    result.times.correction_gpu_seconds += elapsedSeconds(correction_start);

    const Clock::time_point update_start = Clock::now();
    for (size_t index = 0; index < elements; ++index) {
      root[index] += static_cast<double>(correction[index]);
    }
    if (symmetric) {
      symmetrizeInPlace(root, order);
    }
    result.times.update_fp64_seconds += elapsedSeconds(update_start);

    ++iteration;
    const Clock::time_point residual_start = Clock::now();
    if (symmetric) {
      symmetricResidual(prepared.values, root, order, residual, product,
                        residual_norm);
    } else {
      generalResidual(prepared.values, root, order, residual, product,
                      residual_norm);
    }
    result.times.residual_fp64_seconds += elapsedSeconds(residual_start);
    forward_error = relativeError(root, reference_root);
    result.history.push_back({iteration, residual_norm, forward_error});
    if (std::isfinite(residual_norm) && residual_norm < result.best_residual) {
      result.best_iteration = iteration;
      result.best_residual = residual_norm;
      result.best_forward_error = forward_error;
    }
  }
  result.times.mixed_total_seconds = elapsedSeconds(mixed_start);
  result.final_residual = residual_norm;
  result.final_forward_error = forward_error;
  result.converged = std::isfinite(residual_norm) && residual_norm <= tolerance;
  result.finite = allFinite(root) && std::isfinite(residual_norm) &&
                  std::isfinite(forward_error);
  if (!result.finite) {
    result.termination_reason = "nonfinite";
  } else if (result.converged) {
    result.termination_reason = "converged";
  } else if (result.history.back().residual > result.history.front().residual) {
    result.termination_reason = "diverged";
  } else {
    result.termination_reason = "budget_exhausted";
  }
  result.status = result.termination_reason;
  result.symmetry_error = symmetric ? relativeSymmetryError(root, order)
                                    : std::numeric_limits<double>::quiet_NaN();
  result.timing = TimingSummary{
      0,
      1,
      result.times.mixed_total_seconds,
      result.times.mixed_total_seconds,
      result.times.mixed_total_seconds,
      result.times.mixed_total_seconds,
      0.0,
  };
  result.reference_timing = TimingSummary{
      0,
      1,
      result.times.reference_seconds,
      result.times.reference_seconds,
      result.times.reference_seconds,
      result.times.reference_seconds,
      0.0,
  };
  return result;
}

double timeReferenceSolve(const PreparedMatrix &prepared,
                          std::vector<double> *root_output) {
  const Clock::time_point start = Clock::now();
  std::vector<double> root =
      prepared.matrix_type == "symmetric"
          ? symmetricReferenceRoot(prepared.values, prepared.order)
          : nonsymmetricReferenceRoot(prepared.values, prepared.order);
  const double seconds = elapsedSeconds(start);
  if (!allFinite(root)) {
    fail("The timed FP64 reference root is nonfinite.");
  }
  if (root_output != nullptr) {
    *root_output = std::move(root);
  }
  return seconds;
}

CaseResult runTimedAccelerateCase(const PreparedMatrix &prepared,
                                  size_t warmups, size_t evaluations) {
  if (evaluations == 0) {
    fail("The evaluation count must be nonzero.");
  }
  std::vector<double> accuracy_reference;
  timeReferenceSolve(prepared, &accuracy_reference);
  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    if (warmup % 2 == 0) {
      timeReferenceSolve(prepared);
      runAccelerateCase(prepared, 12, 1.0e-12, &accuracy_reference);
    } else {
      runAccelerateCase(prepared, 12, 1.0e-12, &accuracy_reference);
      timeReferenceSolve(prepared);
    }
  }
  std::vector<CaseResult> results;
  std::vector<double> mixed_samples;
  std::vector<double> reference_samples;
  results.reserve(evaluations);
  mixed_samples.reserve(evaluations);
  reference_samples.reserve(evaluations);
  for (size_t evaluation = 0; evaluation < evaluations; ++evaluation) {
    if (evaluation % 2 == 0) {
      reference_samples.push_back(timeReferenceSolve(prepared));
      results.push_back(
          runAccelerateCase(prepared, 12, 1.0e-12, &accuracy_reference));
    } else {
      results.push_back(
          runAccelerateCase(prepared, 12, 1.0e-12, &accuracy_reference));
      reference_samples.push_back(timeReferenceSolve(prepared));
    }
    mixed_samples.push_back(results.back().times.mixed_total_seconds);
  }
  PhaseTimes mean_times;
  for (const CaseResult &sample : results) {
    mean_times.low_eigen_seconds += sample.times.low_eigen_seconds;
    mean_times.initial_gpu_seconds += sample.times.initial_gpu_seconds;
    mean_times.residual_fp64_seconds += sample.times.residual_fp64_seconds;
    mean_times.correction_gpu_seconds += sample.times.correction_gpu_seconds;
    mean_times.update_fp64_seconds += sample.times.update_fp64_seconds;
    mean_times.mixed_total_seconds += sample.times.mixed_total_seconds;
  }
  const double count = static_cast<double>(results.size());
  mean_times.low_eigen_seconds /= count;
  mean_times.initial_gpu_seconds /= count;
  mean_times.residual_fp64_seconds /= count;
  mean_times.correction_gpu_seconds /= count;
  mean_times.update_fp64_seconds /= count;
  mean_times.mixed_total_seconds /= count;
  const TimingSummary mixed_timing =
      summarizeTimingSamples(mixed_samples, warmups);
  const TimingSummary reference_timing =
      summarizeTimingSamples(reference_samples, warmups);
  std::vector<double> paired_ratios;
  const PairedRatioSummary paired_ratio_summary = summarizePairedRatios(
      mixed_samples, reference_samples, evaluations, paired_ratios);
  std::sort(results.begin(), results.end(),
            [](const CaseResult &left, const CaseResult &right) {
              return left.times.mixed_total_seconds <
                     right.times.mixed_total_seconds;
            });
  CaseResult result = std::move(results[results.size() / 2]);
  mean_times.reference_seconds = reference_timing.mean_seconds;
  result.times = mean_times;
  result.timing = mixed_timing;
  result.reference_timing = reference_timing;
  result.paired_ratios = std::move(paired_ratios);
  result.paired_ratio_summary = paired_ratio_summary;
  return result;
}

} // namespace mixed_sqrt
