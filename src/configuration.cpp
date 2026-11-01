#include "core.hpp"
#include "experiment.hpp"
#include "precision.hpp"

#include <limits>
#include <utility>

namespace mixed_sqrt {
namespace {

template <typename Enum, size_t Size>
const char *enumName(Enum value,
                     const std::pair<Enum, const char *> (&names)[Size],
                     const char *error) {
  for (const auto &entry : names) {
    if (entry.first == value) {
      return entry.second;
    }
  }
  fail(error);
}

template <typename Enum, size_t Size>
Enum enumForCliName(const std::string &name,
                    const std::pair<Enum, const char *> (&names)[Size],
                    const char *error_prefix) {
  for (const auto &entry : names) {
    if (name == entry.second) {
      return entry.first;
    }
  }
  fail(std::string(error_prefix) + name);
}

constexpr std::pair<RoundingMode, const char *> rounding_names[] = {
    {RoundingMode::Nearest, "nearest"},
    {RoundingMode::Stochastic, "stochastic"},
};
constexpr std::pair<KernelOperation, const char *> kernel_operation_names[] = {
    {KernelOperation::ScaleColumns, "scale_columns"},
    {KernelOperation::Matmul, "matmul"},
    {KernelOperation::DiagonalSylvester, "diagonal_sylvester"},
    {KernelOperation::ApplySylvesterInverse, "apply_sylvester_inverse"},
    {KernelOperation::MatmulTiled, "matmul_tiled"},
    {KernelOperation::MatmulTiledScaled, "matmul_tiled_scaled"},
    {KernelOperation::DiagonalSylvesterCoalesced,
     "diagonal_sylvester_coalesced"},
    {KernelOperation::Quantize, "quantize"},
};
constexpr std::pair<KernelPath, const char *> kernel_path_names[] = {
    {KernelPath::Legacy, "legacy"},
    {KernelPath::Tiled, "tiled"},
};
constexpr std::pair<SolverBackend, const char *> solver_backend_names[] = {
    {SolverBackend::Metal, "metal"},
    {SolverBackend::Accelerate, "accelerate"},
};
constexpr std::pair<PaperTestSet, const char *> paper_test_set_names[] = {
    {PaperTestSet::None, "controlled"}, {PaperTestSet::I, "I"},
    {PaperTestSet::II, "II"},           {PaperTestSet::III, "III"},
    {PaperTestSet::IV, "IV"},           {PaperTestSet::V, "V"},
    {PaperTestSet::VI, "VI"},           {PaperTestSet::VII, "VII"},
};
constexpr std::pair<CorrectionMode, const char *> correction_mode_names[] = {
    {CorrectionMode::Fp32Reorthogonalized, "fp32_reorthogonalized"},
    {CorrectionMode::NativeQuantized, "native_quantized"},
};
constexpr std::pair<NativeStoppingPolicy, const char *>
    stopping_policy_names[] = {
        {NativeStoppingPolicy::FixedBudget, "fixed_budget"},
        {NativeStoppingPolicy::Stagnation, "stagnation"},
};
constexpr std::pair<NativeDispatchPolicy, const char *>
    dispatch_policy_names[] = {
        {NativeDispatchPolicy::PerStage, "per_stage"},
        {NativeDispatchPolicy::Fused, "fused"},
};

} // namespace

const char *roundingModeName(RoundingMode mode) {
  return enumName(mode, rounding_names, "Unknown rounding mode.");
}

RoundingMode roundingModeForCliName(const std::string &name) {
  return enumForCliName(name, rounding_names, "Unsupported rounding mode: ");
}

const PrecisionSpec &precisionSpec(Precision precision) {
  for (const PrecisionSpec &spec : kPrecisionSpecs) {
    if (spec.precision == precision) {
      return spec;
    }
  }
  fail("Unknown precision specification.");
}

double verificationTolerance(Precision precision) {
  const PrecisionSpec &spec = precisionSpec(precision);
  if (precision == Precision::Fp32) {
    return 1.0e-8;
  }
  if (precision == Precision::Bf16) {
    return 1.0e-6;
  }
  return spec.storage_bits == 4 ? 2.5e-1 : 1.0e-2;
}

double basisOrthogonalityTolerance(Precision precision) {
  const PrecisionSpec &spec = precisionSpec(precision);
  if (precision == Precision::Fp32) {
    return 1.0e-3;
  }
  if (precision == Precision::Bf16) {
    return 5.0e-3;
  }
  return spec.storage_bits == 4 ? 5.0e-1 : 1.5e-1;
}

Precision basisPrecision(Precision precision) {
  precisionSpec(precision);
  return precision == Precision::Fp32 ? Precision::Fp32 : Precision::Bf16;
}

bool usesQuantizedFactorFp32Correction(Precision precision) {
  precisionSpec(precision);
  return precision != Precision::Fp32 && precision != Precision::Bf16;
}

bool usesFp32Correction(Precision basis_precision,
                        Precision spectrum_precision) {
  return basis_precision != spectrum_precision ||
         usesQuantizedFactorFp32Correction(basis_precision) ||
         usesQuantizedFactorFp32Correction(spectrum_precision);
}

std::string factorPrecisionLabel(Precision basis_precision,
                                 Precision spectrum_precision) {
  const PrecisionSpec &basis = precisionSpec(basis_precision);
  const PrecisionSpec &spectrum = precisionSpec(spectrum_precision);
  return basis_precision == spectrum_precision
             ? basis.cli_name
             : std::string(basis.cli_name) + "__s_" + spectrum.cli_name;
}

std::string executionModel(Precision basis_precision,
                           Precision spectrum_precision) {
  if (basis_precision != spectrum_precision) {
    return "asymmetric_factor_reorthogonalized_fp32_correction";
  }
  if (precisionSpec(spectrum_precision).usesLogSpectrum()) {
    return "quantized_log_spectral_factor_reorthogonalized_fp32_correction";
  }
  return usesFp32Correction(basis_precision, spectrum_precision)
             ? "quantized_factor_reorthogonalized_fp32_correction"
             : "native_low_correction";
}

PrecisionPolicy factorPrecisionPolicy(Precision basis_precision,
                                      Precision spectrum_precision) {
  return basis_precision == spectrum_precision
             ? precisionSpec(basis_precision).policy
             : PrecisionPolicy::MeasuredControlled;
}

const char *precisionPolicyName(PrecisionPolicy policy) {
  return policy == PrecisionPolicy::RequiredFullAccuracy
             ? "required_full_accuracy"
             : "measured_controlled";
}

const PrecisionSpec &precisionForCliName(const std::string &name) {
  for (const PrecisionSpec &spec : kPrecisionSpecs) {
    if (name == spec.cli_name) {
      return spec;
    }
  }
  fail("Unsupported precision: " + name);
}

void ensureKernelsEnabled(const PrecisionSpec &precision) {
  if (!precision.kernels_enabled) {
    fail("Precision " + std::string(precision.cli_name) +
         " is recognized, but its Metal kernels are not enabled.");
  }
}

size_t storageBytes(const PrecisionSpec &precision, size_t elements) {
  const size_t bits_per_element = precision.storage_bits;
  if (elements > std::numeric_limits<size_t>::max() / bits_per_element) {
    fail("Metal buffer size overflow.");
  }
  const size_t total_bits = elements * bits_per_element;
  if (total_bits > std::numeric_limits<size_t>::max() - 7) {
    fail("Metal buffer size overflow.");
  }
  return (total_bits + 7) / 8;
}

size_t checkedElementCount(size_t rows, size_t columns) {
  if (rows != 0 && columns > std::numeric_limits<size_t>::max() / rows) {
    fail("Metal element count overflow.");
  }
  return rows * columns;
}

size_t scaleBytes(const PrecisionSpec &precision, size_t elements) {
  if (!precision.requiresScalePlane()) {
    return 0;
  }
  const size_t blocks = precision.scale_block_elements;
  return elements / blocks + (elements % blocks != 0 ? 1 : 0);
}

const char *kernelOperationName(KernelOperation operation) {
  return enumName(operation, kernel_operation_names,
                  "Unknown Metal kernel operation.");
}

const char *kernelPathName(KernelPath path) {
  return enumName(path, kernel_path_names, "Unknown kernel path.");
}

KernelPath kernelPathForCliName(const std::string &name) {
  return enumForCliName(name, kernel_path_names, "Unsupported kernel path: ");
}

const char *solverBackendName(SolverBackend backend) {
  return enumName(backend, solver_backend_names, "Unknown solver backend.");
}

SolverBackend solverBackendForCliName(const std::string &name) {
  return enumForCliName(name, solver_backend_names,
                        "Unsupported solver backend: ");
}

const char *paperTestSetName(PaperTestSet test_set) {
  return enumName(test_set, paper_test_set_names, "Unknown paper test set.");
}

PaperTestSet paperTestSetForCliName(const std::string &name) {
  for (const auto &entry : paper_test_set_names) {
    if (entry.first != PaperTestSet::None && name == entry.second) {
      return entry.first;
    }
  }
  fail("Unsupported paper test set: " + name);
}

bool paperTestSetIsSymmetric(PaperTestSet test_set) {
  return test_set == PaperTestSet::V || test_set == PaperTestSet::VI ||
         test_set == PaperTestSet::VII;
}

size_t paperTestSetOrder(PaperTestSet test_set) {
  switch (test_set) {
  case PaperTestSet::I:
  case PaperTestSet::II:
    return 2048;
  case PaperTestSet::III:
  case PaperTestSet::IV:
  case PaperTestSet::V:
  case PaperTestSet::VI:
    return 4096;
  case PaperTestSet::VII:
    return 1024;
  case PaperTestSet::None:
    fail("The controlled matrix family does not have a paper order.");
  }
  fail("Unknown paper test set.");
}

const char *correctionModeName(CorrectionMode mode) {
  return enumName(mode, correction_mode_names, "Unknown correction mode.");
}

CorrectionMode correctionModeForCliName(const std::string &name) {
  return enumForCliName(name, correction_mode_names,
                        "Unsupported correction mode: ");
}

const char *nativeStoppingPolicyName(NativeStoppingPolicy policy) {
  return enumName(policy, stopping_policy_names,
                  "Unknown native stopping policy.");
}

NativeStoppingPolicy nativeStoppingPolicyForCliName(const std::string &name) {
  return enumForCliName(name, stopping_policy_names,
                        "Unsupported native stopping policy: ");
}

void validateNativeStoppingPolicy(CorrectionMode correction_mode,
                                  NativeStoppingPolicy stopping_policy) {
  if (stopping_policy == NativeStoppingPolicy::Stagnation &&
      correction_mode != CorrectionMode::NativeQuantized) {
    fail("--native-stopping stagnation requires --correction-mode "
         "native_quantized.");
  }
}

const char *nativeDispatchPolicyName(NativeDispatchPolicy policy) {
  return enumName(policy, dispatch_policy_names,
                  "Unknown native dispatch policy.");
}

NativeDispatchPolicy nativeDispatchPolicyForCliName(const std::string &name) {
  return enumForCliName(name, dispatch_policy_names,
                        "Unsupported native dispatch policy: ");
}

void validateNativeDispatchPolicy(CorrectionMode correction_mode,
                                  NativeDispatchPolicy dispatch_policy) {
  if (dispatch_policy == NativeDispatchPolicy::PerStage &&
      correction_mode != CorrectionMode::NativeQuantized) {
    fail("--native-dispatch per_stage requires --correction-mode "
         "native_quantized.");
  }
}

void validateCorrectionMode(CorrectionMode mode, Precision basis_precision,
                            Precision spectrum_precision,
                            KernelPath kernel_path) {
  if (mode != CorrectionMode::NativeQuantized) {
    return;
  }
  const PrecisionSpec &basis = precisionSpec(basis_precision);
  const PrecisionSpec &spectrum = precisionSpec(spectrum_precision);
  if (basis.usesLogSpectrum() || spectrum.usesLogSpectrum()) {
    fail("Native quantized correction does not support log-spectrum storage.");
  }
  if (basis_precision != spectrum_precision) {
    fail("Native quantized correction requires matching basis and spectrum "
         "precisions.");
  }
  if (kernel_path != KernelPath::Legacy) {
    fail("Native quantized correction requires the legacy kernel path.");
  }
  if (!usesQuantizedFactorFp32Correction(basis_precision)) {
    fail("Native quantized correction supports FP8 E4M3FN, FP8 E5M2, MXFP8 "
         "E4M3FN, MXFP8 E5M2, FP4 E2M1, or MXFP4 E2M1 storage.");
  }
}

} // namespace mixed_sqrt
