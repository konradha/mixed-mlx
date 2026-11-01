#include "metal_solver.hpp"

#include "metal_runtime.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "accelerate_backend.hpp"
#include "benchmark.hpp"
#include "core.hpp"
#include "numerics.hpp"
#include "storage_codecs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mixed_sqrt {

struct GpuResidentSpdResult {
  std::vector<double> root;
  std::vector<double> residuals;
  std::vector<double> initial_root;
  std::vector<float> root_eigenvalues;
  double basis_orthogonality = 0.0;
  double quantized_basis_orthogonality =
      std::numeric_limits<double>::quiet_NaN();
  double factor_quantization_seconds = 0.0;
  double basis_reorthogonalization_seconds = 0.0;
  std::string selected_basis_stage;
  std::string selected_spectrum_stage;
  std::vector<std::string> correction_stages;
  size_t precision_escalations = 0;
  size_t low_factor_corrections = 0;
  size_t low_factor_basis_bytes = 0;
  size_t fp32_basis_bytes = 0;
  PhaseTimes times;
};

struct JacobiSchedule {
  uint32_t padded_order = 0;
  std::vector<uint32_t> partners;
  std::vector<uint32_t> pair_indices;
  std::vector<uint32_t> pair_members;
};

JacobiSchedule makeJacobiSchedule(uint32_t order) {
  const uint32_t padded_order = order + order % 2;
  const uint32_t round_count = padded_order - 1;
  std::vector<uint32_t> participants(padded_order);
  for (uint32_t index = 0; index < padded_order; ++index) {
    participants[index] = index;
  }
  JacobiSchedule schedule;
  schedule.padded_order = padded_order;
  schedule.partners.resize(static_cast<size_t>(padded_order) * round_count);
  schedule.pair_indices.resize(schedule.partners.size());
  schedule.pair_members.resize(schedule.partners.size());
  for (uint32_t round = 0; round < round_count; ++round) {
    const size_t offset = static_cast<size_t>(round) * padded_order;
    for (uint32_t pair = 0; pair < padded_order / 2; ++pair) {
      const uint32_t first = participants[pair];
      const uint32_t second = participants[padded_order - 1 - pair];
      const size_t pair_offset = offset + 2 * static_cast<size_t>(pair);
      schedule.pair_members[pair_offset] = std::min(first, second);
      schedule.pair_members[pair_offset + 1] = std::max(first, second);
      schedule.partners[offset + first] = second;
      schedule.partners[offset + second] = first;
      schedule.pair_indices[offset + first] = pair;
      schedule.pair_indices[offset + second] = pair;
    }
    std::rotate(participants.begin() + 1, participants.end() - 1,
                participants.end());
  }
  return schedule;
}

std::vector<double> readWorkingMatrix(MTL::Buffer *buffer, size_t elements) {
  const WorkingFloat *values =
      static_cast<const WorkingFloat *>(buffer->contents());
  std::vector<double> result(elements);
  for (size_t index = 0; index < elements; ++index) {
    result[index] = static_cast<double>(values[index].high) +
                    static_cast<double>(values[index].low);
  }
  return result;
}

GpuResidentSpdResult
solveGpuResidentSpd(MetalRuntime &metal, const std::vector<double> &matrix,
                    Precision basis_precision, Precision spectrum_precision,
                    uint32_t order, size_t max_iterations, double tolerance) {
  const size_t elements = checkedElementCount(order, order);
  if (matrix.size() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    fail("The GPU-resident matrix dimensions are invalid.");
  }
  const PrecisionSpec &fp32 = precisionSpec(Precision::Fp32);
  const PrecisionSpec &basis = precisionSpec(basis_precision);
  const PrecisionSpec &spectrum = precisionSpec(spectrum_precision);
  if (basis.usesLogSpectrum()) {
    fail("A log-spectrum format cannot store the frozen basis.");
  }
  const uint32_t element_count = static_cast<uint32_t>(elements);
  const size_t float_matrix_bytes = elements * sizeof(float);
  const size_t working_matrix_bytes = elements * sizeof(WorkingFloat);
  const JacobiSchedule schedule = makeJacobiSchedule(order);
  const uint32_t pair_count = schedule.padded_order / 2;
  const auto make_buffer = [&](size_t bytes) {
    return metal.makeSharedBuffer(bytes);
  };
  const auto make_storage = [&](const PrecisionSpec &precision, size_t count) {
    const size_t scale_bytes = precision.usesLogSpectrum()
                                   ? spectrumMetadataBytes(precision, count)
                                   : scaleBytes(precision, count);
    return StorageBuffers{
        make_buffer(storageBytes(precision, count)),
        scale_bytes == 0 ? nullptr : make_buffer(scale_bytes),
    };
  };

  MTL::Buffer *jacobi_matrix = make_buffer(float_matrix_bytes);
  MTL::Buffer *gram = make_buffer(float_matrix_bytes);
  MTL::Buffer *stable_vectors = make_buffer(float_matrix_bytes);
  MTL::Buffer *factor_vectors = make_buffer(float_matrix_bytes);
  MTL::Buffer *vectors_transposed = make_buffer(float_matrix_bytes);
  MTL::Buffer *stable_root_eigenvalues = make_buffer(order * sizeof(float));
  MTL::Buffer *factor_root_eigenvalues = make_buffer(order * sizeof(float));
  MTL::Buffer *inverse_sylvester = make_buffer(float_matrix_bytes);
  MTL::Buffer *partners =
      make_buffer(schedule.partners.size() * sizeof(uint32_t));
  MTL::Buffer *pair_indices =
      make_buffer(schedule.pair_indices.size() * sizeof(uint32_t));
  MTL::Buffer *pair_members =
      make_buffer(schedule.pair_members.size() * sizeof(uint32_t));
  MTL::Buffer *rotations =
      make_buffer(static_cast<size_t>(2 * pair_count) * 2 * sizeof(float));
  MTL::Buffer *workspace = make_buffer(float_matrix_bytes);
  MTL::Buffer *root_f32 = make_buffer(float_matrix_bytes);
  MTL::Buffer *residual_f32 = make_buffer(float_matrix_bytes);
  MTL::Buffer *correction_coordinates = make_buffer(float_matrix_bytes);
  MTL::Buffer *correction = make_buffer(float_matrix_bytes);
  MTL::Buffer *matrix_working = make_buffer(working_matrix_bytes);
  MTL::Buffer *root_working = make_buffer(working_matrix_bytes);
  MTL::Buffer *residual_working = make_buffer(working_matrix_bytes);
  MTL::Buffer *norm = make_buffer(sizeof(WorkingNorm));
  const StorageBuffers encoded_basis = basis_precision == Precision::Fp32
                                           ? StorageBuffers{}
                                           : make_storage(basis, elements);
  const StorageBuffers encoded_spectrum = spectrum_precision == Precision::Fp32
                                              ? StorageBuffers{}
                                              : make_storage(spectrum, order);
  const std::vector<MTL::Buffer *> buffers{
      jacobi_matrix,
      gram,
      stable_vectors,
      factor_vectors,
      vectors_transposed,
      stable_root_eigenvalues,
      factor_root_eigenvalues,
      inverse_sylvester,
      partners,
      pair_indices,
      pair_members,
      rotations,
      workspace,
      root_f32,
      residual_f32,
      correction_coordinates,
      correction,
      matrix_working,
      root_working,
      residual_working,
      norm,
      encoded_basis.values,
      encoded_basis.scales,
      encoded_spectrum.values,
      encoded_spectrum.scales,
  };

  float *low_values = static_cast<float *>(jacobi_matrix->contents());
  WorkingFloat *working_values =
      static_cast<WorkingFloat *>(matrix_working->contents());
  for (size_t index = 0; index < elements; ++index) {
    const float high = static_cast<float>(matrix[index]);
    low_values[index] = high;
    working_values[index] = WorkingFloat{
        high,
        static_cast<float>(matrix[index] - static_cast<double>(high)),
    };
  }
  std::memcpy(partners->contents(), schedule.partners.data(),
              schedule.partners.size() * sizeof(uint32_t));
  std::memcpy(pair_indices->contents(), schedule.pair_indices.data(),
              schedule.pair_indices.size() * sizeof(uint32_t));
  std::memcpy(pair_members->contents(), schedule.pair_members.data(),
              schedule.pair_members.size() * sizeof(uint32_t));

  GpuResidentSpdResult result;
  result.low_factor_basis_bytes =
      storageBytes(basis, elements) + scaleBytes(basis, elements);
  result.fp32_basis_bytes = float_matrix_bytes;
  const Clock::time_point mixed_start = Clock::now();
  const Clock::time_point eigen_start = Clock::now();
  metal.symmetricJacobi(jacobi_matrix, gram, stable_vectors, factor_vectors,
                        partners, pair_indices, pair_members, rotations,
                        stable_root_eigenvalues, order, schedule.padded_order,
                        8);
  MTL::Buffer *selected_vectors = stable_vectors;
  double selected_basis_orthogonality =
      std::numeric_limits<double>::quiet_NaN();
  if (basis_precision != Precision::Fp32) {
    const Clock::time_point quantize_start = Clock::now();
    metal.quantizeAndDecode(basis, stable_vectors, encoded_basis,
                            factor_vectors, element_count, 0u);
    result.factor_quantization_seconds += elapsedSeconds(quantize_start);
    result.quantized_basis_orthogonality = metal.measureF32Orthogonality(
        factor_vectors, vectors_transposed, gram, order);
    const double quantized_orthogonality = result.quantized_basis_orthogonality;
    if (std::isfinite(quantized_orthogonality) &&
        quantized_orthogonality <=
            basisOrthogonalityTolerance(basis_precision)) {
      selected_vectors = factor_vectors;
      result.selected_basis_stage = basis.cli_name;
      selected_basis_orthogonality = quantized_orthogonality;
    } else {
      result.selected_basis_stage =
          std::string(basis.cli_name) + "_rejected_to_fp32";
      ++result.precision_escalations;
    }
  } else {
    result.selected_basis_stage = "fp32";
  }

  MTL::Buffer *selected_roots = stable_root_eigenvalues;
  if (spectrum_precision != Precision::Fp32) {
    const Clock::time_point quantize_start = Clock::now();
    metal.quantizeAndDecode(spectrum, stable_root_eigenvalues, encoded_spectrum,
                            factor_root_eigenvalues, order, 1u);
    result.factor_quantization_seconds += elapsedSeconds(quantize_start);
    const float *quantized_roots =
        static_cast<const float *>(factor_root_eigenvalues->contents());
    bool usable = true;
    for (uint32_t index = 0; index < order; ++index) {
      usable = usable && quantized_roots[index] > 0.0f &&
               std::isfinite(quantized_roots[index]);
    }
    if (usable) {
      selected_roots = factor_root_eigenvalues;
      result.selected_spectrum_stage = spectrum.cli_name;
    } else {
      result.selected_spectrum_stage =
          std::string(spectrum.cli_name) + "_rejected_to_fp32";
      ++result.precision_escalations;
    }
  } else {
    result.selected_spectrum_stage = "fp32";
  }
  if (!std::isfinite(selected_basis_orthogonality)) {
    selected_basis_orthogonality = metal.measureF32Orthogonality(
        selected_vectors, vectors_transposed, gram, order);
  }
  metal.prepareFrozenSpectralFactor(selected_vectors, vectors_transposed,
                                    selected_roots, inverse_sylvester, order);
  result.times.low_eigen_seconds = elapsedSeconds(eigen_start);
  result.basis_orthogonality = selected_basis_orthogonality;
  const float *selected_root_values =
      static_cast<const float *>(selected_roots->contents());
  result.root_eigenvalues.assign(selected_root_values,
                                 selected_root_values + order);
  for (float value : result.root_eigenvalues) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
      releaseBuffers(buffers);
      fail("The frozen GPU spectrum contains an invalid value.");
    }
  }

  const StorageBuffers workspace_storage{workspace, nullptr};
  const StorageBuffers root_storage{root_f32, nullptr};
  const Clock::time_point initial_start = Clock::now();
  if (selected_vectors == factor_vectors) {
    metal.reconstructLowFactorTiled(basis, encoded_basis, selected_roots,
                                    workspace, root_f32, order);
  } else {
    metal.reconstructTiled(fp32, StorageBuffers{selected_vectors, nullptr},
                           StorageBuffers{vectors_transposed, nullptr},
                           StorageBuffers{selected_roots, nullptr},
                           workspace_storage, root_storage, order);
  }
  metal.initializeWorkingRoot(root_f32, root_working, order);
  result.times.initial_gpu_seconds = elapsedSeconds(initial_start);

  Clock::time_point residual_start = Clock::now();
  double residual =
      metal.computeWorkingResidual(matrix_working, root_working,
                                   residual_working, residual_f32, norm, order);
  result.times.residual_fp64_seconds += elapsedSeconds(residual_start);
  result.residuals.push_back(residual);
  if (selected_vectors != stable_vectors ||
      selected_roots != stable_root_eigenvalues) {
    selected_vectors = stable_vectors;
    selected_roots = stable_root_eigenvalues;
    metal.prepareFrozenSpectralFactor(selected_vectors, vectors_transposed,
                                      selected_roots, inverse_sylvester, order);
  }
  for (size_t iteration = 0; residual > tolerance && iteration < max_iterations;
       ++iteration) {
    result.correction_stages.push_back("fp32_frozen");
    const Clock::time_point correction_start = Clock::now();
    residual = metal.refineUpdateAndMeasure(
        fp32, StorageBuffers{stable_vectors, nullptr},
        StorageBuffers{vectors_transposed, nullptr}, matrix_working,
        root_working, residual_working, residual_f32, norm, workspace,
        inverse_sylvester, correction_coordinates, correction, order);
    result.times.correction_gpu_seconds += elapsedSeconds(correction_start);
    result.residuals.push_back(residual);
    if (!std::isfinite(residual)) {
      break;
    }
  }
  result.times.mixed_total_seconds = elapsedSeconds(mixed_start);
  result.root = readWorkingMatrix(root_working, elements);
  result.initial_root.resize(elements);
  const float *initial_values =
      static_cast<const float *>(root_f32->contents());
  for (size_t index = 0; index < elements; ++index) {
    result.initial_root[index] = static_cast<double>(initial_values[index]);
  }
  releaseBuffers(buffers);
  return result;
}

CaseResult runMetalCase(MetalRuntime &metal,
                        const MetalSolverConfiguration &configuration,
                        const MetalProblem &problem) {
  const Precision basis_precision = configuration.basis_precision;
  const Precision spectrum_precision = configuration.spectrum_precision;
  const KernelPath kernel_path = configuration.kernel_path;
  const CorrectionMode correction_mode = configuration.correction_mode;
  const NativeStoppingPolicy native_stopping_policy =
      configuration.stopping_policy;
  const size_t native_stagnation_window = configuration.stagnation_window;
  const double native_min_relative_improvement =
      configuration.minimum_relative_improvement;
  const NativeDispatchPolicy native_dispatch_policy =
      configuration.dispatch_policy;
  const size_t order = problem.order;
  double condition = problem.condition;
  const size_t max_iterations = problem.max_iterations;
  const double tolerance = problem.tolerance;
  const bool enforce_policy = problem.enforce_policy;
  const PreparedMatrix *prepared_matrix = problem.prepared_matrix;
  const std::vector<double> *supplied_reference = problem.supplied_reference;
  validateNativeStoppingPolicy(correction_mode, native_stopping_policy);
  validateNativeDispatchPolicy(correction_mode, native_dispatch_policy);
  if (native_stagnation_window == 0) {
    fail("--native-stagnation-window must be a positive integer.");
  }
  if (!std::isfinite(native_min_relative_improvement) ||
      native_min_relative_improvement < 0.0) {
    fail("--native-min-relative-improvement must be finite and nonnegative.");
  }
  validateCorrectionMode(correction_mode, basis_precision, spectrum_precision,
                         kernel_path);
  const PrecisionSpec &basis_spec = precisionSpec(basis_precision);
  const PrecisionSpec &spectrum_spec = precisionSpec(spectrum_precision);
  metal.requireEnabledKernels(basis_spec);
  metal.requireEnabledKernels(spectrum_spec);
  const PrecisionPolicy policy =
      factorPrecisionPolicy(basis_precision, spectrum_precision);
  const size_t refinement_iterations =
      policy == PrecisionPolicy::RequiredFullAccuracy || max_iterations == 1
          ? max_iterations
          : std::max(max_iterations, size_t{64});
  const bool enforce_full_accuracy =
      enforce_policy && policy == PrecisionPolicy::RequiredFullAccuracy;
  const bool native_quantized =
      correction_mode == CorrectionMode::NativeQuantized;
  if (order < 2) {
    fail("The matrix order must be at least two.");
  }
  if (order > std::numeric_limits<uint32_t>::max()) {
    fail("The matrix order exceeds Metal's uint32_t kernel dimensions.");
  }
  const size_t matrix_elements = checkedElementCount(order, order);
  if (matrix_elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
    fail("The matrix order exceeds safe host and Metal buffer dimensions.");
  }
  if (native_quantized && basis_spec.requiresQuantizer() &&
      matrix_elements > std::numeric_limits<uint32_t>::max()) {
    fail(
        "Native quantized correction exceeds the raw quantizer element limit.");
  }

  CaseResult result;
  result.solver_backend = solverBackendName(SolverBackend::Metal);
  result.rounding_mode = roundingModeName(metal.roundingMode());
  result.rounding_seed = metal.roundingSeed();
  if (prepared_matrix != nullptr) {
    if (prepared_matrix->order != order ||
        prepared_matrix->values.size() != matrix_elements ||
        prepared_matrix->matrix_type != "symmetric" ||
        !allFinite(prepared_matrix->values)) {
      fail("Metal requires a finite prepared symmetric matrix of the selected "
           "order.");
    }
    result.paper_test_set = prepared_matrix->paper_test_set;
    result.matrix_family = prepared_matrix->matrix_family;
    result.matrix_type = prepared_matrix->matrix_type;
    result.rng_seed = prepared_matrix->rng_seed;
    result.diagonal_shift = prepared_matrix->diagonal_shift;
    result.shift_rule = prepared_matrix->shift_rule;
    condition = prepared_matrix->condition;
  }
  result.precision = factorPrecisionLabel(basis_precision, spectrum_precision);
  result.basis_precision = basis_spec.cli_name;
  result.spectrum_precision = spectrum_spec.cli_name;
  result.correction_mode = correctionModeName(correction_mode);
  result.execution_model =
      correction_mode == CorrectionMode::NativeQuantized
          ? "native_quantized_correction"
          : executionModel(basis_precision, spectrum_precision);
  result.kernel_path = kernelPathName(kernel_path);
  result.precision_policy = precisionPolicyName(policy);
  result.stopping_policy =
      native_quantized
          ? nativeStoppingPolicyName(native_stopping_policy)
          : nativeStoppingPolicyName(NativeStoppingPolicy::FixedBudget);
  if (native_quantized) {
    result.dispatch_policy = nativeDispatchPolicyName(native_dispatch_policy);
    result.kernel_dispatches_per_correction =
        5 + (basis_spec.requiresQuantizer() ? 5 : 0);
    result.command_buffers_per_correction =
        native_dispatch_policy == NativeDispatchPolicy::Fused
            ? 1
            : result.kernel_dispatches_per_correction;
  }
  result.name =
      result.paper_test_set == "controlled"
          ? "spd_n" + std::to_string(order) + "_k" + conditionLabel(condition) +
                "_" + result.precision + "_" + result.correction_mode
          : "paper_" + result.paper_test_set + "_metal_" + result.precision;
  if (metal.roundingMode() == RoundingMode::Stochastic &&
      (basis_precision != Precision::Fp32 ||
       spectrum_precision != Precision::Fp32)) {
    result.name += "_stochastic";
  }
  result.order = order;
  result.condition = condition;

  std::vector<double> generated_matrix;
  if (prepared_matrix == nullptr) {
    generated_matrix = makeSpdMatrix(order, condition);
  }
  const std::vector<double> &matrix =
      prepared_matrix == nullptr ? generated_matrix : prepared_matrix->values;

  std::vector<double> computed_reference_root;
  if (supplied_reference == nullptr) {
    const Clock::time_point reference_start = Clock::now();
    computed_reference_root = symmetricReferenceRoot(matrix, order);
    result.times.reference_seconds = elapsedSeconds(reference_start);
  }
  const std::vector<double> &reference_root = supplied_reference == nullptr
                                                  ? computed_reference_root
                                                  : *supplied_reference;
  if (reference_root.size() != matrix_elements || !allFinite(reference_root)) {
    fail("The supplied FP64 reference root is invalid.");
  }
  result.reference_residual = relativeResidual(matrix, reference_root, order);
  const bool gpu_resident =
      correction_mode == CorrectionMode::Fp32Reorthogonalized &&
      kernel_path == KernelPath::Tiled;
  if (gpu_resident) {
    const double reference_seconds = result.times.reference_seconds;
    GpuResidentSpdResult gpu = solveGpuResidentSpd(
        metal, matrix, basis_precision, spectrum_precision,
        static_cast<uint32_t>(order), refinement_iterations, tolerance);
    result.times = gpu.times;
    result.times.reference_seconds = reference_seconds;
    result.execution_model =
        "gpu_resident_fp32_jacobi_low_factor_float_float_refinement";
    result.dispatch_policy = "gpu_resident";
    result.command_buffers_per_correction = 1;
    result.kernel_dispatches_per_correction = 8;
    result.selected_basis_stage = gpu.selected_basis_stage;
    result.selected_spectrum_stage = gpu.selected_spectrum_stage;
    result.precision_escalations = gpu.precision_escalations;
    result.low_factor_basis_bytes = gpu.low_factor_basis_bytes;
    result.fp32_basis_bytes = gpu.fp32_basis_bytes;
    result.low_factor_corrections = gpu.low_factor_corrections;
    result.quantized_basis_orthogonality = gpu.quantized_basis_orthogonality;
    result.factor_quantization_seconds = gpu.factor_quantization_seconds;
    result.basis_reorthogonalization_seconds =
        gpu.basis_reorthogonalization_seconds;
    if (!gpu.correction_stages.empty()) {
      std::ostringstream stages;
      for (size_t index = 0; index < gpu.correction_stages.size(); ++index) {
        if (index != 0) {
          stages << ';';
        }
        stages << gpu.correction_stages[index];
      }
      result.correction_stage_history = stages.str();
    }
    result.frozen_basis_orthogonality = gpu.basis_orthogonality;
    result.frozen_factor_error =
        frobeniusDifference(gpu.initial_root, reference_root);
    result.frozen_factor_relative_error =
        relativeError(gpu.initial_root, reference_root);
    const float minimum_root = *std::min_element(gpu.root_eigenvalues.begin(),
                                                 gpu.root_eigenvalues.end());
    result.frozen_sylvester_inverse_norm = 1.0 / (2.0 * minimum_root);
    result.frozen_contraction_bound =
        3.0 * result.frozen_sylvester_inverse_norm * result.frozen_factor_error;
    result.certificate_usable =
        std::isfinite(result.frozen_basis_orthogonality) &&
        result.frozen_basis_orthogonality < 1.0e-3;
    const double initial_error =
        relativeError(gpu.initial_root, reference_root);
    const double final_error = relativeError(gpu.root, reference_root);
    result.history.reserve(gpu.residuals.size());
    for (size_t index = 0; index < gpu.residuals.size(); ++index) {
      double forward_error = std::numeric_limits<double>::quiet_NaN();
      if (index == 0) {
        forward_error = initial_error;
      }
      if (index + 1 == gpu.residuals.size()) {
        forward_error = final_error;
      }
      result.history.push_back({index, gpu.residuals[index], forward_error});
    }
    const auto best =
        std::min_element(gpu.residuals.begin(), gpu.residuals.end());
    result.best_iteration =
        static_cast<size_t>(std::distance(gpu.residuals.begin(), best));
    result.best_residual = *best;
    result.best_forward_error =
        result.best_iteration + 1 == gpu.residuals.size()
            ? final_error
            : std::numeric_limits<double>::quiet_NaN();
    result.final_residual = gpu.residuals.back();
    result.final_forward_error = final_error;
    result.finite = allFinite(gpu.root) &&
                    std::isfinite(result.final_residual) &&
                    std::isfinite(result.final_forward_error);
    result.converged = result.finite && result.final_residual <= tolerance;
    result.symmetry_error = relativeSymmetryError(gpu.root, order);
    if (!result.finite) {
      result.termination_reason = "nonfinite";
    } else if (result.converged) {
      result.termination_reason = "converged";
    } else if (result.final_residual > gpu.residuals.front()) {
      result.termination_reason = "diverged";
    } else {
      result.termination_reason = "budget_exhausted";
    }
    result.status = result.termination_reason;
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
        reference_seconds,
        reference_seconds,
        reference_seconds,
        reference_seconds,
        0.0,
    };
    if (enforce_full_accuracy) {
      if (!result.finite) {
        fail("The GPU-resident solver produced a nonfinite result.");
      }
      if (!result.converged) {
        std::ostringstream message;
        message
            << std::setprecision(17)
            << "The GPU-resident solver did not reach the residual tolerance. "
            << "Residual: " << result.final_residual;
        fail(message.str());
      }
      if (result.final_forward_error >
          verificationTolerance(spectrum_precision)) {
        fail("The GPU-resident result does not match the FP64 reference.");
      }
      if (result.symmetry_error > 1.0e-12) {
        fail("The GPU-resident result lost symmetric structure.");
      }
    }
    return result;
  }

  const Clock::time_point mixed_start = Clock::now();
  const Clock::time_point low_eigen_start = Clock::now();
  std::vector<float> low_matrix(matrix.begin(), matrix.end());
  std::vector<float> low_eigenvalues(order);
  symmetricEigenFloat(low_matrix, low_eigenvalues);
  std::vector<float> root_eigenvalues(order);
  for (size_t index = 0; index < order; ++index) {
    const float value = low_eigenvalues[index];
    if (!(value > 0.0f) || !std::isfinite(value)) {
      fail("The low-precision eigensolver returned a nonpositive eigenvalue.");
    }
    root_eigenvalues[index] = std::sqrt(value);
  }
  const std::vector<float> low_eigenvectors = low_matrix;
  const bool fp32_correction =
      !native_quantized &&
      usesFp32Correction(basis_precision, spectrum_precision);
  const std::vector<float> low_eigenvectors_transposed =
      fp32_correction ? std::vector<float>{}
                      : transposeF32(low_eigenvectors, order);
  result.times.low_eigen_seconds = elapsedSeconds(low_eigen_start);

  const size_t matrix_bytes = storageBytes(basis_spec, matrix_elements);
  const size_t vector_bytes = storageBytes(spectrum_spec, order);
  const size_t matrix_scale_bytes = scaleBytes(basis_spec, matrix_elements);
  const size_t vector_spectrum_metadata_bytes =
      spectrumMetadataBytes(spectrum_spec, order);
  const auto makeStorageBuffers = [&](size_t values_bytes,
                                      size_t scales_bytes) {
    return StorageBuffers{
        metal.makeSharedBuffer(values_bytes),
        scales_bytes == 0 ? nullptr : metal.makeSharedBuffer(scales_bytes),
    };
  };
  const StorageBuffers q = makeStorageBuffers(matrix_bytes, matrix_scale_bytes);
  const StorageBuffers qt =
      fp32_correction ? StorageBuffers{}
                      : makeStorageBuffers(matrix_bytes, matrix_scale_bytes);
  const StorageBuffers s =
      makeStorageBuffers(vector_bytes, vector_spectrum_metadata_bytes);
  const PrecisionSpec &correction_spec =
      fp32_correction ? precisionSpec(Precision::Fp32) : basis_spec;
  const size_t correction_matrix_bytes =
      storageBytes(correction_spec, matrix_elements);
  const size_t correction_vector_bytes = storageBytes(correction_spec, order);
  const size_t correction_matrix_scale_bytes =
      scaleBytes(correction_spec, matrix_elements);
  const size_t correction_vector_scale_bytes =
      scaleBytes(correction_spec, order);
  const StorageBuffers decoded_q =
      fp32_correction ? makeStorageBuffers(correction_matrix_bytes,
                                           correction_matrix_scale_bytes)
                      : StorageBuffers{};
  const StorageBuffers decoded_qt =
      fp32_correction ? makeStorageBuffers(correction_matrix_bytes,
                                           correction_matrix_scale_bytes)
                      : StorageBuffers{};
  const StorageBuffers decoded_s =
      fp32_correction ? makeStorageBuffers(correction_vector_bytes,
                                           correction_vector_scale_bytes)
                      : StorageBuffers{};
  const StorageBuffers workspace_one = makeStorageBuffers(
      correction_matrix_bytes, correction_matrix_scale_bytes);
  const StorageBuffers workspace_two = makeStorageBuffers(
      correction_matrix_bytes, correction_matrix_scale_bytes);
  const StorageBuffers root_low = makeStorageBuffers(
      correction_matrix_bytes, correction_matrix_scale_bytes);
  const StorageBuffers residual_low = makeStorageBuffers(
      correction_matrix_bytes, correction_matrix_scale_bytes);
  const StorageBuffers correction_coordinates = makeStorageBuffers(
      correction_matrix_bytes, correction_matrix_scale_bytes);
  const StorageBuffers correction_low = makeStorageBuffers(
      correction_matrix_bytes, correction_matrix_scale_bytes);
  MTL::Buffer *inverse_sylvester =
      native_quantized
          ? nullptr
          : metal.makeSharedBuffer(matrix_elements * sizeof(float));
  MTL::Buffer *temporary_output =
      correction_spec.requiresQuantizer()
          ? metal.makeSharedBuffer(matrix_elements * sizeof(float))
          : nullptr;
  const std::vector<MTL::Buffer *> buffers{q.values,
                                           q.scales,
                                           qt.values,
                                           qt.scales,
                                           s.values,
                                           s.scales,
                                           decoded_q.values,
                                           decoded_q.scales,
                                           decoded_qt.values,
                                           decoded_qt.scales,
                                           decoded_s.values,
                                           decoded_s.scales,
                                           workspace_one.values,
                                           workspace_one.scales,
                                           workspace_two.values,
                                           workspace_two.scales,
                                           root_low.values,
                                           root_low.scales,
                                           residual_low.values,
                                           residual_low.scales,
                                           correction_coordinates.values,
                                           correction_coordinates.scales,
                                           correction_low.values,
                                           correction_low.scales,
                                           inverse_sylvester,
                                           temporary_output};

  copyToStorage(basis_spec, low_eigenvectors, q.values, q.scales);
  if (!fp32_correction) {
    copyToStorage(basis_spec, low_eigenvectors_transposed, qt.values,
                  qt.scales);
  }
  copySpectrumToStorage(spectrum_spec, root_eigenvalues, s.values, s.scales);
  std::vector<float> frozen_root_eigenvalues(order);
  for (size_t index = 0; index < order; ++index) {
    frozen_root_eigenvalues[index] =
        spectrumValueAt(spectrum_spec, s.values, s.scales, index);
    if (!(frozen_root_eigenvalues[index] > 0.0f) ||
        !std::isfinite(frozen_root_eigenvalues[index])) {
      fail("The frozen root spectrum contains an invalid value.");
    }
  }
  if (!native_quantized) {
    float *inverse_sylvester_values =
        static_cast<float *>(inverse_sylvester->contents());
    for (size_t column = 0; column < order; ++column) {
      for (size_t row = 0; row < order; ++row) {
        const float denominator =
            frozen_root_eigenvalues[row] + frozen_root_eigenvalues[column];
        if (!(denominator > 0.0f) || !std::isfinite(denominator)) {
          fail("The frozen Sylvester operator is singular.");
        }
        inverse_sylvester_values[row + column * order] = 1.0f / denominator;
      }
    }
  }
  bool frozen_factor_usable = true;
  if (fp32_correction) {
    const auto decodeFrozenFactor = [&](const StorageBuffers &source,
                                        MTL::Buffer *destination,
                                        size_t elements) {
      float *decoded = static_cast<float *>(destination->contents());
      for (size_t index = 0; index < elements; ++index) {
        decoded[index] =
            storageValueAt(basis_spec, source.values, source.scales, index);
      }
    };
    decodeFrozenFactor(q, decoded_q.values, matrix_elements);
    frozen_factor_usable = orthonormalizeColumnsF32(
        static_cast<float *>(decoded_q.values->contents()), order);
    if (!frozen_factor_usable) {
      decodeFrozenFactor(q, decoded_q.values, matrix_elements);
    }
    transposeF32(static_cast<const float *>(decoded_q.values->contents()),
                 static_cast<float *>(decoded_qt.values->contents()), order);
    float *decoded_s_values =
        static_cast<float *>(decoded_s.values->contents());
    for (size_t index = 0; index < order; ++index) {
      decoded_s_values[index] = frozen_root_eigenvalues[index];
    }
  }
  const StorageBuffers &correction_q = fp32_correction ? decoded_q : q;
  const StorageBuffers &correction_qt = fp32_correction ? decoded_qt : qt;
  const StorageBuffers &correction_s = fp32_correction ? decoded_s : s;
  result.frozen_basis_orthogonality =
      orthogonalityError(correction_spec, correction_q, order);

  const uint32_t order_u32 = static_cast<uint32_t>(order);
  const Clock::time_point initial_start = Clock::now();
  if (kernel_path == KernelPath::Tiled) {
    metal.reconstructTiled(correction_spec, correction_q, correction_qt,
                           correction_s, workspace_one, root_low, order_u32);
  } else {
    metal.scaleColumns(correction_spec, correction_q, correction_s,
                       workspace_one, temporary_output, order_u32);
    metal.matmul(correction_spec, workspace_one, correction_qt, root_low,
                 temporary_output, order_u32, order_u32, order_u32);
  }
  std::vector<double> root(matrix_elements);
  unpackFromStorage(correction_spec, root_low.values, root_low.scales, root);
  symmetrizeInPlace(root, order);
  result.frozen_factor_error = frobeniusDifference(root, reference_root);
  result.frozen_factor_relative_error = relativeError(root, reference_root);
  if (native_quantized) {
    result.frozen_sylvester_inverse_norm =
        std::numeric_limits<double>::quiet_NaN();
    result.frozen_contraction_bound = std::numeric_limits<double>::quiet_NaN();
    result.certificate_usable = false;
  } else {
    const float minimum_frozen_root = *std::min_element(
        frozen_root_eigenvalues.begin(), frozen_root_eigenvalues.end());
    result.frozen_sylvester_inverse_norm = 1.0 / (2.0 * minimum_frozen_root);
    result.frozen_contraction_bound =
        result.frozen_sylvester_inverse_norm *
        (2.0 * result.frozen_factor_error + result.frozen_factor_error);
    result.certificate_usable = frozen_factor_usable;
  }
  result.times.initial_gpu_seconds = elapsedSeconds(initial_start);

  std::vector<double> residual(matrix_elements);
  const Clock::time_point initial_residual_start = Clock::now();
  double residual_norm = relativeResidual(matrix, root, order, &residual);
  result.times.residual_fp64_seconds += elapsedSeconds(initial_residual_start);
  result.history.push_back(
      {0, residual_norm, relativeError(root, reference_root)});
  const double initial_forward_error = result.history.back().forward_error;
  std::vector<double> best_root;
  bool has_finite_best = allFinite(root) && std::isfinite(residual_norm) &&
                         std::isfinite(initial_forward_error);
  if (has_finite_best) {
    best_root = root;
    result.best_residual = residual_norm;
    result.best_forward_error = initial_forward_error;
  }
  bool stagnated = false;
  bool observed_nonfinite = !has_finite_best;

  size_t iteration = 0;
  while (frozen_factor_usable && residual_norm > tolerance &&
         iteration < refinement_iterations) {
    const Clock::time_point correction_start = Clock::now();
    copyResidualToStorage(correction_spec, residual, residual_low.values,
                          residual_low.scales);
    if (kernel_path == KernelPath::Tiled) {
      metal.refineTiled(correction_spec, correction_q, correction_qt,
                        residual_low, workspace_one, inverse_sylvester,
                        correction_coordinates, correction_low, order_u32);
    } else if (native_quantized &&
               native_dispatch_policy == NativeDispatchPolicy::Fused) {
      metal.refineNativeStaged(correction_spec, correction_q, correction_qt,
                               correction_s, residual_low, workspace_one,
                               workspace_two, correction_coordinates,
                               correction_low, temporary_output, order_u32);
    } else if (native_quantized) {
      metal.refineNativePerStage(correction_spec, correction_q, correction_qt,
                                 correction_s, residual_low, workspace_one,
                                 workspace_two, correction_coordinates,
                                 correction_low, temporary_output, order_u32);
    } else {
      metal.matmul(correction_spec, correction_qt, residual_low, workspace_one,
                   temporary_output, order_u32, order_u32, order_u32);
      metal.matmul(correction_spec, workspace_one, correction_q, workspace_two,
                   temporary_output, order_u32, order_u32, order_u32);
      metal.applySylvesterInverse(correction_spec, workspace_two,
                                  inverse_sylvester, correction_coordinates,
                                  order_u32);
      metal.matmul(correction_spec, correction_q, correction_coordinates,
                   workspace_one, temporary_output, order_u32, order_u32,
                   order_u32);
      metal.matmul(correction_spec, workspace_one, correction_qt,
                   correction_low, temporary_output, order_u32, order_u32,
                   order_u32);
    }
    result.times.correction_gpu_seconds += elapsedSeconds(correction_start);

    const Clock::time_point update_start = Clock::now();
    addCorrectionFromStorage(correction_spec, correction_low.values,
                             correction_low.scales, root);
    symmetrizeInPlace(root, order);
    result.times.update_fp64_seconds += elapsedSeconds(update_start);

    ++iteration;
    const Clock::time_point residual_start = Clock::now();
    residual_norm = relativeResidual(matrix, root, order, &residual);
    result.times.residual_fp64_seconds += elapsedSeconds(residual_start);
    result.history.push_back(
        {iteration, residual_norm, relativeError(root, reference_root)});
    const double forward_error = result.history.back().forward_error;
    if (!allFinite(root) || !std::isfinite(residual_norm) ||
        !std::isfinite(forward_error)) {
      observed_nonfinite = true;
      break;
    }
    const double prior_best_residual = result.best_residual;
    const double relative_improvement =
        prior_best_residual > 0.0
            ? (prior_best_residual - residual_norm) / prior_best_residual
            : 0.0;
    if (residual_norm < result.best_residual) {
      best_root = root;
      result.best_iteration = iteration;
      result.best_residual = residual_norm;
      result.best_forward_error = forward_error;
    }
    if (native_quantized &&
        native_stopping_policy == NativeStoppingPolicy::Stagnation &&
        residual_norm > tolerance) {
      if (relative_improvement > native_min_relative_improvement) {
        result.stagnation_steps = 0;
      } else {
        ++result.stagnation_steps;
      }
      if (result.stagnation_steps >= native_stagnation_window) {
        stagnated = true;
        break;
      }
    }
  }

  result.times.mixed_total_seconds = elapsedSeconds(mixed_start);
  if (native_quantized &&
      native_stopping_policy == NativeStoppingPolicy::Stagnation &&
      has_finite_best) {
    root = std::move(best_root);
    result.final_residual = result.best_residual;
    result.final_forward_error = result.best_forward_error;
    result.converged =
        frozen_factor_usable && result.best_residual <= tolerance;
    result.finite = has_finite_best;
  } else {
    result.final_residual = residual_norm;
    result.final_forward_error = result.history.back().forward_error;
    result.converged = frozen_factor_usable && residual_norm <= tolerance;
    result.finite = allFinite(root) && std::isfinite(residual_norm) &&
                    std::isfinite(result.final_forward_error);
  }
  if (observed_nonfinite) {
    result.termination_reason = "nonfinite";
  } else if (!frozen_factor_usable) {
    result.termination_reason = "frozen_factor_rank_deficient";
  } else if (result.converged) {
    result.termination_reason = "converged";
  } else if (stagnated) {
    result.termination_reason = "stagnated";
  } else if (result.history.back().residual > result.history.front().residual) {
    result.termination_reason = "diverged";
  } else {
    result.termination_reason = "budget_exhausted";
  }
  result.status = result.termination_reason;
  result.symmetry_error = relativeSymmetryError(root, order);
  releaseBuffers(buffers);
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

  if (enforce_full_accuracy) {
    if (!result.finite) {
      fail("The solver produced a nonfinite result.");
    }
    if (!result.converged) {
      fail("The solver did not reach the requested residual tolerance.");
    }
    if (result.final_forward_error >
        verificationTolerance(spectrum_precision)) {
      fail("The result does not match the FP64 Accelerate reference.");
    }
    if (result.symmetry_error > 1.0e-12) {
      fail("The result lost symmetric structure.");
    }
  }
  return result;
}

CaseResult runAdaptiveMetalCase(MetalRuntime &metal,
                                const MetalSolverConfiguration &configuration,
                                const MetalProblem &problem) {
  if (configuration.correction_mode == CorrectionMode::NativeQuantized) {
    fail("--adaptive does not support native quantized correction.");
  }
  MetalSolverConfiguration fp32_configuration = configuration;
  fp32_configuration.basis_precision = Precision::Fp32;
  fp32_configuration.spectrum_precision = Precision::Fp32;
  MetalProblem controlled_problem = problem;
  controlled_problem.enforce_policy = false;
  controlled_problem.prepared_matrix = nullptr;
  controlled_problem.supplied_reference = nullptr;
  CaseResult selected =
      runMetalCase(metal, fp32_configuration, controlled_problem);
  selected.selection_policy = "latency_first";
  selected.precision_attempts = 1;
  selected.execution_model += "_adaptive";
  if (problem.enforce_policy &&
      (!selected.finite || !selected.converged ||
       selected.history.back().forward_error > 1.0e-9 ||
       selected.symmetry_error > 1.0e-12)) {
    fail("The adaptive solver did not meet the full-accuracy contract.");
  }
  return selected;
}

CaseResult runTimedMetalCase(MetalRuntime &metal,
                             const MetalSolverConfiguration &configuration,
                             const MetalProblem &problem, size_t warmups,
                             size_t evaluations) {
  const size_t order = problem.order;
  const double condition = problem.condition;
  const PreparedMatrix *prepared_matrix = problem.prepared_matrix;
  if (evaluations == 0) {
    fail("The evaluation count must be nonzero.");
  }
  PreparedMatrix controlled;
  if (prepared_matrix == nullptr) {
    controlled = prepareControlledMatrix(order, condition);
    prepared_matrix = &controlled;
  }
  if (prepared_matrix->order != order ||
      prepared_matrix->matrix_type != "symmetric") {
    fail("The timed Metal case requires a prepared symmetric matrix.");
  }
  std::vector<double> accuracy_reference;
  timeReferenceSolve(*prepared_matrix, &accuracy_reference);
  MetalProblem measured_problem = problem;
  measured_problem.prepared_matrix = prepared_matrix;
  measured_problem.supplied_reference = &accuracy_reference;
  const auto run_mixed = [&]() {
    return runMetalCase(metal, configuration, measured_problem);
  };
  for (size_t warmup = 0; warmup < warmups; ++warmup) {
    if (warmup % 2 == 0) {
      timeReferenceSolve(*prepared_matrix);
      run_mixed();
    } else {
      run_mixed();
      timeReferenceSolve(*prepared_matrix);
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
      reference_samples.push_back(timeReferenceSolve(*prepared_matrix));
      results.push_back(run_mixed());
    } else {
      results.push_back(run_mixed());
      reference_samples.push_back(timeReferenceSolve(*prepared_matrix));
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
