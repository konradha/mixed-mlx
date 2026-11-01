#pragma once

#include "experiment.hpp"
#include "precision.hpp"

#include <cstddef>

namespace mixed_sqrt {

class MetalRuntime;

struct MetalSolverConfiguration {
  Precision basis_precision = Precision::Fp32;
  Precision spectrum_precision = Precision::Fp32;
  KernelPath kernel_path = KernelPath::Tiled;
  CorrectionMode correction_mode = CorrectionMode::Fp32Reorthogonalized;
  NativeStoppingPolicy stopping_policy = NativeStoppingPolicy::FixedBudget;
  size_t stagnation_window = 4;
  double minimum_relative_improvement = 1.0e-3;
  NativeDispatchPolicy dispatch_policy = NativeDispatchPolicy::Fused;
};

struct MetalProblem {
  size_t order;
  double condition;
  size_t max_iterations;
  double tolerance;
  bool enforce_policy;
  const PreparedMatrix *prepared_matrix = nullptr;
  const std::vector<double> *supplied_reference = nullptr;
};

CaseResult runMetalCase(MetalRuntime &metal,
                        const MetalSolverConfiguration &configuration,
                        const MetalProblem &problem);
CaseResult runAdaptiveMetalCase(MetalRuntime &metal,
                                const MetalSolverConfiguration &configuration,
                                const MetalProblem &problem);
CaseResult runTimedMetalCase(MetalRuntime &metal,
                             const MetalSolverConfiguration &configuration,
                             const MetalProblem &problem, size_t warmups,
                             size_t evaluations);

}  // namespace mixed_sqrt
