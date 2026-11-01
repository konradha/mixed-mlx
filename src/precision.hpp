#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mixed_sqrt {

enum class Precision {
  Fp32,
  Bf16,
  Fp8E4M3Fn,
  Fp8E5M2,
  Mxfp8E4M3Fn,
  Mxfp8E5M2,
  Fp4E2M1,
  Mxfp4E2M1,
  Fp8E4M3FnLogS,
  Fp4E2M1LogS,
};

enum class RoundingMode { Nearest, Stochastic };
enum class StorageLayout { Scalar, Packed };
enum class HostStorageCodec {
  Fp32,
  Bf16,
  Fp8E4M3Fn,
  Fp8E5M2,
  Mxfp8E4M3Fn,
  Mxfp8E5M2,
  Fp4E2M1,
  Mxfp4E2M1,
};
enum class PrecisionPolicy { RequiredFullAccuracy, MeasuredControlled };
enum class SpectralEncoding { Linear, LogAffine };

struct PrecisionSpec {
  Precision precision;
  const char *cli_name;
  const char *kernel_suffix;
  uint8_t storage_bits;
  StorageLayout storage_layout;
  uint32_t scale_block_elements;
  HostStorageCodec host_storage;
  PrecisionPolicy policy;
  bool kernels_enabled;
  SpectralEncoding spectral_encoding = SpectralEncoding::Linear;
  uint32_t spectral_block_elements = 0;

  constexpr bool usesPackedStorage() const {
    return storage_layout == StorageLayout::Packed;
  }
  constexpr bool requiresScalePlane() const {
    return scale_block_elements != 0;
  }
  constexpr bool requiresQuantizer() const {
    return usesPackedStorage() || requiresScalePlane();
  }
  constexpr bool usesLogSpectrum() const {
    return spectral_encoding == SpectralEncoding::LogAffine;
  }
};

inline constexpr PrecisionSpec kPrecisionSpecs[] = {
    {Precision::Fp32, "fp32", "f32", 32, StorageLayout::Scalar, 0,
     HostStorageCodec::Fp32, PrecisionPolicy::RequiredFullAccuracy, true},
    {Precision::Bf16, "bf16", "bf16", 16, StorageLayout::Scalar, 0,
     HostStorageCodec::Bf16, PrecisionPolicy::RequiredFullAccuracy, true},
    {Precision::Fp8E4M3Fn, "fp8_e4m3fn", "fp8_e4m3fn", 8, StorageLayout::Scalar,
     0, HostStorageCodec::Fp8E4M3Fn, PrecisionPolicy::MeasuredControlled, true},
    {Precision::Fp8E5M2, "fp8_e5m2", "fp8_e5m2", 8, StorageLayout::Scalar, 0,
     HostStorageCodec::Fp8E5M2, PrecisionPolicy::MeasuredControlled, true},
    {Precision::Mxfp8E4M3Fn, "mxfp8_e4m3fn", "mxfp8_e4m3fn", 8,
     StorageLayout::Scalar, 32, HostStorageCodec::Mxfp8E4M3Fn,
     PrecisionPolicy::MeasuredControlled, true},
    {Precision::Mxfp8E5M2, "mxfp8_e5m2", "mxfp8_e5m2", 8, StorageLayout::Scalar,
     32, HostStorageCodec::Mxfp8E5M2, PrecisionPolicy::MeasuredControlled,
     true},
    {Precision::Fp4E2M1, "fp4_e2m1", "fp4_e2m1", 4, StorageLayout::Packed, 0,
     HostStorageCodec::Fp4E2M1, PrecisionPolicy::MeasuredControlled, true},
    {Precision::Mxfp4E2M1, "mxfp4_e2m1", "mxfp4_e2m1", 4, StorageLayout::Packed,
     32, HostStorageCodec::Mxfp4E2M1, PrecisionPolicy::MeasuredControlled,
     true},
    {Precision::Fp8E4M3FnLogS, "fp8_e4m3fn_log_s", "fp8_e4m3fn", 8,
     StorageLayout::Scalar, 0, HostStorageCodec::Fp8E4M3Fn,
     PrecisionPolicy::MeasuredControlled, true, SpectralEncoding::LogAffine, 8},
    {Precision::Fp4E2M1LogS, "fp4_e2m1_log_s", "fp4_e2m1", 4,
     StorageLayout::Packed, 0, HostStorageCodec::Fp4E2M1,
     PrecisionPolicy::MeasuredControlled, true, SpectralEncoding::LogAffine, 8},
};

const char *roundingModeName(RoundingMode mode);
RoundingMode roundingModeForCliName(const std::string &name);
const PrecisionSpec &precisionSpec(Precision precision);
double verificationTolerance(Precision precision);
double basisOrthogonalityTolerance(Precision precision);
Precision basisPrecision(Precision precision);
bool usesQuantizedFactorFp32Correction(Precision precision);
bool usesFp32Correction(Precision basis_precision,
                        Precision spectrum_precision);
std::string factorPrecisionLabel(Precision basis_precision,
                                 Precision spectrum_precision);
std::string executionModel(Precision basis_precision,
                           Precision spectrum_precision);
PrecisionPolicy factorPrecisionPolicy(Precision basis_precision,
                                      Precision spectrum_precision);
const char *precisionPolicyName(PrecisionPolicy policy);
const PrecisionSpec &precisionForCliName(const std::string &name);
void ensureKernelsEnabled(const PrecisionSpec &precision);
size_t storageBytes(const PrecisionSpec &precision, size_t elements);
size_t checkedElementCount(size_t rows, size_t columns);
size_t scaleBytes(const PrecisionSpec &precision, size_t elements);

enum class KernelOperation {
  ScaleColumns,
  Matmul,
  DiagonalSylvester,
  ApplySylvesterInverse,
  MatmulTiled,
  MatmulTiledScaled,
  DiagonalSylvesterCoalesced,
  Quantize,
};
const char *kernelOperationName(KernelOperation operation);

enum class KernelPath { Legacy, Tiled };
const char *kernelPathName(KernelPath path);
KernelPath kernelPathForCliName(const std::string &name);

enum class SolverBackend { Metal, Accelerate };
const char *solverBackendName(SolverBackend backend);
SolverBackend solverBackendForCliName(const std::string &name);

enum class CorrectionMode { Fp32Reorthogonalized, NativeQuantized };
const char *correctionModeName(CorrectionMode mode);
CorrectionMode correctionModeForCliName(const std::string &name);

enum class NativeStoppingPolicy { FixedBudget, Stagnation };
const char *nativeStoppingPolicyName(NativeStoppingPolicy policy);
NativeStoppingPolicy nativeStoppingPolicyForCliName(const std::string &name);
void validateNativeStoppingPolicy(CorrectionMode correction_mode,
                                  NativeStoppingPolicy stopping_policy);

enum class NativeDispatchPolicy { PerStage, Fused };
const char *nativeDispatchPolicyName(NativeDispatchPolicy policy);
NativeDispatchPolicy nativeDispatchPolicyForCliName(const std::string &name);
void validateNativeDispatchPolicy(CorrectionMode correction_mode,
                                  NativeDispatchPolicy dispatch_policy);
void validateCorrectionMode(CorrectionMode mode, Precision basis_precision,
                            Precision spectrum_precision,
                            KernelPath kernel_path);

} // namespace mixed_sqrt
