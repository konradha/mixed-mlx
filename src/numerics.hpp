#pragma once

#include "experiment.hpp"

#include <vecLib/lapack_types.h>

#include <cstddef>
#include <string>
#include <vector>

namespace mixed_sqrt {

__LAPACK_int upwardSafeFloatWorkspaceSize(float query, const char *routine);
void dgemm(char transpose_left, char transpose_right, size_t rows,
           size_t columns, size_t inner, const double *left,
           const double *right, double *output);
void symmetrizeInPlace(std::vector<double> &matrix, size_t order);
void symmetricEigenDouble(std::vector<double> &matrix,
                          std::vector<double> &values);
void symmetricEigenFloat(std::vector<float> &matrix,
                         std::vector<float> &values);
std::vector<double> makeSpdMatrix(size_t order, double condition);
std::vector<double> rootFromEigen(const std::vector<double> &eigenvectors,
                                  const std::vector<double> &eigenvalues,
                                  size_t order);
double relativeResidual(const std::vector<double> &matrix,
                        const std::vector<double> &root, size_t order,
                        std::vector<double> *residual_output = nullptr);
double relativeError(const std::vector<double> &root,
                     const std::vector<double> &reference_root);
double frobeniusDifference(const std::vector<double> &left,
                           const std::vector<double> &right);
double relativeSymmetryError(const std::vector<double> &matrix, size_t order);
bool allFinite(const std::vector<double> &values);
std::vector<float> transposeF32(const std::vector<float> &matrix, size_t order);
bool orthonormalizeColumnsF32(float *matrix, size_t order);
void transposeF32(const float *matrix, float *transposed, size_t order);
std::string conditionLabel(double condition);
PreparedMatrix prepareControlledMatrix(size_t order, double condition);
PreparedMatrix preparePaperMatrix(PaperTestSet test_set, size_t order);

}  // namespace mixed_sqrt
