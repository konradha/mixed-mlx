#include "application.hpp"

#include "accelerate_backend.hpp"
#include "benchmark.hpp"
#include "core.hpp"
#include "experiment.hpp"
#include "metal_backend.hpp"
#include "numerics.hpp"
#include "parsing.hpp"
#include "precision.hpp"
#include "reporting.hpp"
#include "storage_codecs.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace mixed_sqrt {

int runApplication(int argc, char **argv) {
  SolverBackend solver_backend = SolverBackend::Metal;
  PaperTestSet paper_test_set = PaperTestSet::None;
  size_t paper_order = 0;
  bool paper_order_selected = false;
  Precision basis_precision = Precision::Fp32;
  Precision spectrum_precision = Precision::Fp32;
  KernelPath kernel_path = KernelPath::Tiled;
  CorrectionMode correction_mode = CorrectionMode::Fp32Reorthogonalized;
  RoundingMode rounding_mode = RoundingMode::Nearest;
  uint32_t rounding_seed = 0;
  size_t warmups = 50;
  size_t evaluations = 100;
  NativeStoppingPolicy native_stopping_policy =
      NativeStoppingPolicy::FixedBudget;
  size_t native_stagnation_window = 4;
  double native_min_relative_improvement = 1.0e-3;
  NativeDispatchPolicy native_dispatch_policy = NativeDispatchPolicy::Fused;
  std::vector<size_t> orders{16, 32, 64, 96, 128, 192, 256};
  std::vector<double> conditions{4.0, 1.0e2, 1.0e4, 1.0e6};
  bool custom_sweep_grid = false;
  bool sweep = false;
  bool adaptive = false;
  bool codec_check = false;
  bool kernel_check = false;
  bool metal_solver_option = false;
  bool precision_option_selected = false;
  bool rounding_seed_selected = false;
  std::vector<std::string> positional;

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto take = [&](const char *error) {
      if (++index == argc) {
        fail(error);
      }
      return argv[index];
    };
    if (argument == "--solver-backend") {
      solver_backend = solverBackendForCliName(
          take("--solver-backend requires metal or accelerate."));
    } else if (argument == "--paper-test-set") {
      paper_test_set = paperTestSetForCliName(
          take("--paper-test-set requires I, II, III, IV, V, VI, or VII."));
    } else if (argument == "--paper-order") {
      paper_order = parsePositiveSize(
          take("--paper-order requires a positive integer."), "--paper-order");
      paper_order_selected = true;
    } else if (argument == "--precision") {
      precision_option_selected = true;
      const Precision selected =
          precisionForCliName(take("--precision requires fp32, bf16, "
                                   "fp8_e4m3fn, fp8_e5m2, mxfp8_e4m3fn, "
                                   "mxfp8_e5m2, fp4_e2m1, mxfp4_e2m1, "
                                   "fp8_e4m3fn_log_s, or fp4_e2m1_log_s."))
              .precision;
      basis_precision = basisPrecision(selected);
      spectrum_precision = selected;
    } else if (argument == "--basis-precision") {
      metal_solver_option = true;
      const Precision selected =
          precisionForCliName(take("--basis-precision requires a non-log "
                                   "storage precision."))
              .precision;
      if (precisionSpec(selected).usesLogSpectrum()) {
        fail("--basis-precision does not accept a log-spectrum precision.");
      }
      basis_precision = selected;
    } else if (argument == "--spectrum-precision") {
      metal_solver_option = true;
      spectrum_precision =
          precisionForCliName(take("--spectrum-precision requires a storage "
                                   "precision."))
              .precision;
    } else if (argument == "--rounding") {
      metal_solver_option = true;
      rounding_mode = roundingModeForCliName(
          take("--rounding requires nearest or stochastic."));
    } else if (argument == "--rounding-seed") {
      metal_solver_option = true;
      rounding_seed_selected = true;
      const size_t value = parseNonnegativeSize(
          take("--rounding-seed requires a uint32 integer."),
          "--rounding-seed");
      if (value > std::numeric_limits<uint32_t>::max()) {
        fail("--rounding-seed requires a uint32 integer.");
      }
      rounding_seed = static_cast<uint32_t>(value);
    } else if (argument == "--kernel-path") {
      metal_solver_option = true;
      kernel_path =
          kernelPathForCliName(take("--kernel-path requires legacy or tiled."));
    } else if (argument == "--correction-mode") {
      metal_solver_option = true;
      correction_mode = correctionModeForCliName(
          take("--correction-mode requires fp32_reorthogonalized or "
               "native_quantized."));
    } else if (argument == "--native-dispatch") {
      metal_solver_option = true;
      native_dispatch_policy = nativeDispatchPolicyForCliName(
          take("--native-dispatch requires fused or per_stage."));
    } else if (argument == "--native-stopping") {
      metal_solver_option = true;
      native_stopping_policy = nativeStoppingPolicyForCliName(
          take("--native-stopping requires fixed_budget or stagnation."));
    } else if (argument == "--native-stagnation-window") {
      metal_solver_option = true;
      native_stagnation_window = parsePositiveSize(
          take("--native-stagnation-window requires a positive integer."),
          "--native-stagnation-window");
    } else if (argument == "--native-min-relative-improvement") {
      metal_solver_option = true;
      native_min_relative_improvement = parseNonnegativeFiniteDouble(
          take("--native-min-relative-improvement requires a nonnegative "
               "finite real."),
          "--native-min-relative-improvement");
    } else if (argument == "--orders") {
      orders = parseOrders(
          take("--orders requires comma-separated positive integers."));
      custom_sweep_grid = true;
    } else if (argument == "--conditions") {
      conditions = parseConditions(
          take("--conditions requires comma-separated positive finite reals."));
      custom_sweep_grid = true;
    } else if (argument == "--warmups") {
      warmups = parseNonnegativeSize(
          take("--warmups requires a nonnegative integer."), "--warmups");
    } else if (argument == "--evaluations") {
      evaluations = parsePositiveSize(
          take("--evaluations requires a positive integer."), "--evaluations");
    } else if (argument == "--sweep") {
      sweep = true;
    } else if (argument == "--adaptive") {
      metal_solver_option = true;
      adaptive = true;
    } else if (argument == "--codec-check") {
      codec_check = true;
    } else if (argument == "--kernel-check") {
      kernel_check = true;
    } else if (argument.rfind("--", 0) == 0) {
      fail("Unknown option: " + argument);
    } else {
      positional.push_back(argument);
    }
  }

  if (custom_sweep_grid && !sweep) {
    fail("--orders and --conditions require --sweep.");
  }
  const bool has_paper_set = paper_test_set != PaperTestSet::None;
  if (paper_order_selected && !has_paper_set) {
    fail("--paper-order requires --paper-test-set.");
  }
  if (has_paper_set && custom_sweep_grid) {
    fail("--paper-test-set does not accept --orders or --conditions.");
  }
  if (has_paper_set && adaptive) {
    fail("--paper-test-set does not support --adaptive.");
  }
  if (has_paper_set && !sweep && !positional.empty()) {
    fail("A direct paper test set does not accept a matrix order.");
  }
  if (has_paper_set && (codec_check || kernel_check)) {
    fail("--paper-test-set does not support diagnostic check modes.");
  }
  if (solver_backend == SolverBackend::Accelerate &&
      (metal_solver_option || codec_check || kernel_check)) {
    fail("The Accelerate backend does not accept Metal solver options.");
  }
  if (solver_backend == SolverBackend::Accelerate &&
      precision_option_selected &&
      (basis_precision != Precision::Fp32 ||
       spectrum_precision != Precision::Fp32)) {
    fail("The Accelerate backend requires --precision fp32.");
  }
  if (solver_backend == SolverBackend::Metal && has_paper_set &&
      !paperTestSetIsSymmetric(paper_test_set)) {
    fail("The Metal backend does not support nonsymmetric paper test sets.");
  }
  if (solver_backend == SolverBackend::Metal && has_paper_set &&
      (basis_precision != Precision::Fp32 ||
       spectrum_precision != Precision::Fp32 ||
       correction_mode != CorrectionMode::Fp32Reorthogonalized)) {
    fail("Paper test sets require the FP32 mixed correction mode.");
  }
  if (solver_backend == SolverBackend::Metal) {
    validateNativeStoppingPolicy(correction_mode, native_stopping_policy);
    validateNativeDispatchPolicy(correction_mode, native_dispatch_policy);
  }
  if (rounding_seed_selected && rounding_mode != RoundingMode::Stochastic) {
    fail("--rounding-seed requires --rounding stochastic.");
  }
  if (rounding_mode == RoundingMode::Stochastic &&
      correction_mode == CorrectionMode::NativeQuantized) {
    fail(
        "Stochastic rounding supports frozen-factor quantization with FP32 "
        "correction. It does not support native quantized correction.");
  }
  if (codec_check && rounding_mode == RoundingMode::Stochastic) {
    fail(
        "Use --rounding stochastic --kernel-check for the GPU rounding "
        "check.");
  }
  if (codec_check) {
    if (sweep || !positional.empty()) {
      fail("--codec-check does not accept sweep mode or a matrix order.");
    }
    checkLowPrecisionCodecs();
    std::cout << "codec_checks=passed\n";
    return EXIT_SUCCESS;
  }
  if (adaptive && correction_mode == CorrectionMode::NativeQuantized) {
    fail("--adaptive does not support native quantized correction.");
  }
  if (adaptive && sweep) {
    fail("--adaptive does not support sweep mode.");
  }

  PreparedMatrix paper_matrix;
  const PreparedMatrix *paper_matrix_pointer = nullptr;
  if (has_paper_set) {
    const size_t actual_order =
        paper_order_selected ? paper_order : paperTestSetOrder(paper_test_set);
    paper_matrix = preparePaperMatrix(paper_test_set, actual_order);
    paper_matrix_pointer = &paper_matrix;
  }

  if (solver_backend == SolverBackend::Accelerate) {
    if (adaptive) {
      fail("--adaptive requires the Metal backend.");
    }
    if (sweep) {
      if (positional.size() > 1) {
        fail("Sweep mode accepts at most one output directory.");
      }
      const std::filesystem::path output_directory =
          positional.empty() ? "accelerate_benchmark" : positional.front();
      return runAccelerateSweep(orders, conditions, warmups, evaluations,
                                output_directory, paper_matrix_pointer);
    }
    if (positional.size() > 1) {
      fail("Expected at most one matrix order.");
    }
    PreparedMatrix controlled_matrix;
    const PreparedMatrix *selected_matrix = paper_matrix_pointer;
    if (selected_matrix == nullptr) {
      const size_t order =
          positional.empty()
              ? 32
              : parsePositiveSize(positional.front(), "matrix order");
      controlled_matrix = prepareControlledMatrix(order, 4.0);
      selected_matrix = &controlled_matrix;
    }
    printResult(runAccelerateCase(*selected_matrix, 12, 1.0e-12));
    return EXIT_SUCCESS;
  }

  if (kernel_check && (sweep || !positional.empty())) {
    fail("--kernel-check does not accept sweep mode or a matrix order.");
  }
  if (positional.size() > 1) {
    fail(sweep ? "Sweep mode accepts at most one output directory."
               : "Expected at most one matrix order.");
  }
  const size_t metal_order =
      paper_matrix_pointer != nullptr
          ? paper_matrix_pointer->order
          : (!sweep && !kernel_check && !positional.empty()
                 ? parsePositiveSize(positional.front(), "matrix order")
                 : 32);
  const std::filesystem::path metal_output_directory =
      sweep && !positional.empty() ? positional.front() : "metal_benchmark";

  const MetalRunOptions metal_options{
      basis_precision,
      spectrum_precision,
      kernel_path,
      correction_mode,
      rounding_mode,
      rounding_seed,
      native_stopping_policy,
      native_stagnation_window,
      native_min_relative_improvement,
      native_dispatch_policy,
      sweep,
      adaptive,
      kernel_check,
      orders,
      conditions,
      warmups,
      evaluations,
      metal_order,
      metal_output_directory,
  };
  return runMetal(metal_options, paper_matrix_pointer);
}

}  // namespace mixed_sqrt
