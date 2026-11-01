#include "reporting.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

namespace mixed_sqrt {

double updatesAndOthersSeconds(const CaseResult &result) {
  const double accounted =
      result.times.low_eigen_seconds + result.times.initial_gpu_seconds +
      result.times.residual_fp64_seconds + result.times.correction_gpu_seconds;
  return std::max(0.0, result.times.mixed_total_seconds - accounted);
}

template <typename... Values> void printLine(Values &&...values) {
  (std::cout << ... << std::forward<Values>(values)) << '\n';
}

void printResult(const CaseResult &result) {
  printLine("paper_test_set=", result.paper_test_set,
            " matrix_family=", result.matrix_family,
            " matrix_type=", result.matrix_type,
            " solver_backend=", result.solver_backend, " order=", result.order,
            " condition=", result.condition, " rng_seed=", result.rng_seed,
            " rounding_mode=", result.rounding_mode,
            " rounding_seed=", result.rounding_seed,
            " diagonal_shift=", std::setprecision(17), result.diagonal_shift,
            " shift_rule=", result.shift_rule);
  for (const IterationMeasurement &record : result.history) {
    printLine("case=", result.name,
              " selection_policy=", result.selection_policy,
              " precision_attempts=", result.precision_attempts,
              " correction_mode=", result.correction_mode,
              " precision_policy=", result.precision_policy,
              " execution_model=", result.execution_model,
              " kernel_path=", result.kernel_path,
              " dispatch_policy=", result.dispatch_policy,
              " iteration=", record.iteration, " residual=", record.residual,
              " forward_error=", record.forward_error);
  }
  printLine("reference_residual=", result.reference_residual);
  printLine(
      "frozen_factor_error=", result.frozen_factor_error,
      " frozen_factor_relative_error=", result.frozen_factor_relative_error,
      " frozen_basis_orthogonality=", result.frozen_basis_orthogonality,
      " frozen_sylvester_inverse_norm=", result.frozen_sylvester_inverse_norm,
      " frozen_contraction_bound=", result.frozen_contraction_bound,
      " certificate_usable=", result.certificate_usable ? "true" : "false");
  const double basis_storage_ratio =
      result.fp32_basis_bytes > 0
          ? static_cast<double>(result.low_factor_basis_bytes) /
                static_cast<double>(result.fp32_basis_bytes)
          : 0.0;
  printLine(
      "selected_basis_stage=", result.selected_basis_stage,
      " selected_spectrum_stage=", result.selected_spectrum_stage,
      " correction_stage_history=", result.correction_stage_history,
      " precision_escalations=", result.precision_escalations,
      " low_factor_basis_bytes=", result.low_factor_basis_bytes,
      " fp32_basis_bytes=", result.fp32_basis_bytes,
      " basis_storage_ratio=", basis_storage_ratio,
      " low_factor_corrections=", result.low_factor_corrections,
      " quantized_basis_orthogonality=", result.quantized_basis_orthogonality);
  printLine("best_iteration=", result.best_iteration,
            " best_residual=", result.best_residual,
            " best_forward_error=", result.best_forward_error,
            " stagnation_steps=", result.stagnation_steps);
  printLine("final_residual=", result.final_residual);
  printLine("final_forward_error=", result.final_forward_error);
  printLine("symmetry_error=", result.symmetry_error);
  printLine("mixed_total_seconds=", result.times.mixed_total_seconds);
  printLine("factorization_seconds=", result.times.low_eigen_seconds,
            " initial_root_seconds=", result.times.initial_gpu_seconds,
            " single_seconds=",
            result.times.low_eigen_seconds + result.times.initial_gpu_seconds,
            " residual_seconds=", result.times.residual_fp64_seconds,
            " correction_system_seconds=", result.times.correction_gpu_seconds,
            " updates_and_others_seconds=", updatesAndOthersSeconds(result),
            " factor_quantization_seconds=", result.factor_quantization_seconds,
            " basis_reorthogonalization_seconds=",
            result.basis_reorthogonalization_seconds);
  printLine("timing_warmups=", result.timing.warmups,
            " timing_evaluations=", result.timing.evaluations,
            " timing_min_seconds=", result.timing.minimum_seconds,
            " timing_max_seconds=", result.timing.maximum_seconds,
            " timing_median_seconds=", result.timing.median_seconds,
            " timing_mean_seconds=", result.timing.mean_seconds,
            " timing_variance_seconds2=", result.timing.variance_seconds2);
  const double mixed_to_reference =
      result.reference_timing.mean_seconds > 0.0
          ? result.timing.mean_seconds / result.reference_timing.mean_seconds
          : 0.0;
  printLine(
      "reference_timing_warmups=", result.reference_timing.warmups,
      " reference_timing_evaluations=", result.reference_timing.evaluations,
      " reference_timing_min_seconds=", result.reference_timing.minimum_seconds,
      " reference_timing_max_seconds=", result.reference_timing.maximum_seconds,
      " reference_timing_median_seconds=",
      result.reference_timing.median_seconds,
      " reference_timing_mean_seconds=", result.reference_timing.mean_seconds,
      " reference_timing_variance_seconds2=",
      result.reference_timing.variance_seconds2,
      " mixed_to_reference=", mixed_to_reference);
  if (!result.paired_ratios.empty()) {
    printLine("paired_ratio_mean=", result.paired_ratio_summary.mean,
              " paired_ratio_variance=", result.paired_ratio_summary.variance,
              " paired_ratio_min=", result.paired_ratio_summary.minimum,
              " paired_ratio_max=", result.paired_ratio_summary.maximum,
              " paired_ratio_median=", result.paired_ratio_summary.median);
  }
  printLine("converged=", result.converged ? "true" : "false",
            " finite=", result.finite ? "true" : "false",
            " stopping_policy=", result.stopping_policy,
            " termination_reason=", result.termination_reason);
  printLine(
      "command_buffers_per_correction=", result.command_buffers_per_correction,
      " kernel_dispatches_per_correction=",
      result.kernel_dispatches_per_correction);
}

void writeSummaryHeader(std::ofstream &output) {
  output
      << "paper_test_set,matrix_family,matrix_type,solver_backend,rng_seed,"
      << "rounding_mode,rounding_seed,diagonal_shift,shift_rule,"
      << "case,precision,basis_precision,spectrum_precision,correction_mode,"
         "precision_policy,"
      << "stopping_policy,dispatch_policy,termination_reason,selection_policy,"
      << "precision_attempts,execution_model,kernel_path,"
      << "command_buffers_per_correction,kernel_dispatches_per_correction,"
      << "selected_basis_stage,selected_spectrum_stage,"
      << "correction_stage_history,precision_escalations,"
      << "low_factor_corrections,low_factor_basis_bytes,fp32_basis_bytes,"
      << "basis_storage_ratio,quantized_basis_orthogonality,"
      << "factor_quantization_seconds,basis_reorthogonalization_seconds,"
      << "warmups,evaluations,timing_evaluations,"
      << "timing_min_seconds,timing_max_seconds,timing_median_seconds,timing_"
         "mean_seconds,"
      << "timing_variance_seconds2,"
      << "reference_timing_min_seconds,reference_timing_max_seconds,"
      << "reference_timing_median_seconds,reference_timing_mean_seconds,"
      << "reference_timing_variance_seconds2,"
      << "order,condition,iterations,best_iteration,best_residual,best_forward_"
         "error,"
      << "stagnation_steps,converged,finite,status,initial_residual,final_"
         "residual,"
      << "initial_forward_error,final_forward_error,symmetry_error,reference_"
         "residual,"
      << "frozen_factor_error,frozen_factor_relative_error,frozen_basis_"
         "orthogonality,"
      << "frozen_sylvester_inverse_norm,frozen_contraction_bound,certificate_"
         "usable,"
      << "reference_seconds,"
      << "low_eigen_seconds,initial_gpu_seconds,residual_fp64_seconds,"
      << "correction_gpu_seconds,update_fp64_seconds,"
      << "factorization_seconds,initial_root_seconds,single_seconds,residual_"
         "seconds,"
      << "correction_system_seconds,updates_and_others_seconds,mixed_total_"
         "seconds,"
      << "low_eigen_fraction,initial_gpu_fraction,residual_fraction,"
      << "correction_gpu_fraction,update_fraction,"
      << "paired_ratio_mean,paired_ratio_variance,paired_ratio_min,"
      << "paired_ratio_max,paired_ratio_median,mixed_to_reference\n";
}

void writeCaseCsv(const CaseResult &result, std::ofstream &summary,
                  std::ofstream &iterations) {
  const double total = result.times.mixed_total_seconds;
  const auto fraction = [total](double seconds) {
    return total > 0.0 ? seconds / total : 0.0;
  };
  const auto &initial = result.history.front();
  const auto &final = result.history.back();
  summary << std::setprecision(17) << result.paper_test_set << ','
          << result.matrix_family << ',' << result.matrix_type << ','
          << result.solver_backend << ',' << result.rng_seed << ','
          << result.rounding_mode << ',' << result.rounding_seed << ','
          << result.diagonal_shift << ',' << result.shift_rule << ','
          << result.name << ',' << result.precision << ','
          << result.basis_precision << ',' << result.spectrum_precision << ','
          << result.correction_mode << ',' << result.precision_policy << ','
          << result.stopping_policy << ',' << result.dispatch_policy << ','
          << result.termination_reason << ',' << result.selection_policy << ','
          << result.precision_attempts << ',' << result.execution_model << ','
          << result.kernel_path << ',' << result.command_buffers_per_correction
          << ',' << result.kernel_dispatches_per_correction << ','
          << result.selected_basis_stage << ','
          << result.selected_spectrum_stage << ','
          << result.correction_stage_history << ','
          << result.precision_escalations << ','
          << result.low_factor_corrections << ','
          << result.low_factor_basis_bytes << ',' << result.fp32_basis_bytes
          << ','
          << (result.fp32_basis_bytes > 0
                  ? static_cast<double>(result.low_factor_basis_bytes) /
                        static_cast<double>(result.fp32_basis_bytes)
                  : 0.0)
          << ',' << result.quantized_basis_orthogonality << ','
          << result.factor_quantization_seconds << ','
          << result.basis_reorthogonalization_seconds << ','
          << result.timing.warmups << ',' << result.timing.evaluations << ','
          << result.timing.evaluations << ',' << result.timing.minimum_seconds
          << ',' << result.timing.maximum_seconds << ','
          << result.timing.median_seconds << ',' << result.timing.mean_seconds
          << ',' << result.timing.variance_seconds2 << ','
          << result.reference_timing.minimum_seconds << ','
          << result.reference_timing.maximum_seconds << ','
          << result.reference_timing.median_seconds << ','
          << result.reference_timing.mean_seconds << ','
          << result.reference_timing.variance_seconds2 << ',' << result.order
          << ',' << result.condition << ',' << final.iteration << ','
          << result.best_iteration << ',' << result.best_residual << ','
          << result.best_forward_error << ',' << result.stagnation_steps << ','
          << result.converged << ',' << result.finite << ',' << result.status
          << ',' << initial.residual << ',' << result.final_residual << ','
          << initial.forward_error << ',' << result.final_forward_error << ','
          << result.symmetry_error << ',' << result.reference_residual << ','
          << result.frozen_factor_error << ','
          << result.frozen_factor_relative_error << ','
          << result.frozen_basis_orthogonality << ','
          << result.frozen_sylvester_inverse_norm << ','
          << result.frozen_contraction_bound << ',' << result.certificate_usable
          << ',' << result.times.reference_seconds << ','
          << result.times.low_eigen_seconds << ','
          << result.times.initial_gpu_seconds << ','
          << result.times.residual_fp64_seconds << ','
          << result.times.correction_gpu_seconds << ','
          << result.times.update_fp64_seconds << ','
          << result.times.low_eigen_seconds << ','
          << result.times.initial_gpu_seconds << ','
          << result.times.low_eigen_seconds + result.times.initial_gpu_seconds
          << ',' << result.times.residual_fp64_seconds << ','
          << result.times.correction_gpu_seconds << ','
          << updatesAndOthersSeconds(result) << ',' << total << ','
          << fraction(result.times.low_eigen_seconds) << ','
          << fraction(result.times.initial_gpu_seconds) << ','
          << fraction(result.times.residual_fp64_seconds) << ','
          << fraction(result.times.correction_gpu_seconds) << ','
          << fraction(result.times.update_fp64_seconds) << ','
          << result.paired_ratio_summary.mean << ','
          << result.paired_ratio_summary.variance << ','
          << result.paired_ratio_summary.minimum << ','
          << result.paired_ratio_summary.maximum << ','
          << result.paired_ratio_summary.median << ','
          << (result.reference_timing.mean_seconds > 0.0
                  ? result.timing.mean_seconds /
                        result.reference_timing.mean_seconds
                  : 0.0)
          << '\n';
  for (const IterationMeasurement &record : result.history) {
    iterations << std::setprecision(17) << result.paper_test_set << ','
               << result.matrix_family << ',' << result.matrix_type << ','
               << result.solver_backend << ',' << result.rng_seed << ','
               << result.rounding_mode << ',' << result.rounding_seed << ','
               << result.diagonal_shift << ',' << result.shift_rule << ','
               << result.name << ',' << result.precision << ','
               << result.basis_precision << ',' << result.spectrum_precision
               << ',' << result.correction_mode << ','
               << result.precision_policy << ',' << result.stopping_policy
               << ',' << result.dispatch_policy << ','
               << result.termination_reason << ',' << result.selection_policy
               << ',' << result.precision_attempts << ','
               << result.execution_model << ',' << result.kernel_path << ','
               << result.selected_basis_stage << ','
               << result.selected_spectrum_stage << ','
               << result.correction_stage_history << ','
               << result.precision_escalations << ','
               << result.low_factor_corrections << ','
               << result.low_factor_basis_bytes << ','
               << result.fp32_basis_bytes << ','
               << (result.fp32_basis_bytes > 0
                       ? static_cast<double>(result.low_factor_basis_bytes) /
                             static_cast<double>(result.fp32_basis_bytes)
                       : 0.0)
               << ',' << result.quantized_basis_orthogonality << ','
               << result.factor_quantization_seconds << ','
               << result.basis_reorthogonalization_seconds << ','
               << result.order << ',' << result.condition << ','
               << result.timing.warmups << ',' << result.timing.evaluations
               << ',' << result.timing.evaluations << ','
               << result.reference_timing.minimum_seconds << ','
               << result.reference_timing.maximum_seconds << ','
               << result.reference_timing.median_seconds << ','
               << result.reference_timing.mean_seconds << ','
               << result.reference_timing.variance_seconds2 << ','
               << result.paired_ratio_summary.mean << ','
               << result.paired_ratio_summary.variance << ','
               << result.paired_ratio_summary.minimum << ','
               << result.paired_ratio_summary.maximum << ','
               << result.paired_ratio_summary.median << ','
               << result.times.low_eigen_seconds << ','
               << result.times.initial_gpu_seconds << ','
               << result.times.low_eigen_seconds +
                      result.times.initial_gpu_seconds
               << ',' << result.times.residual_fp64_seconds << ','
               << result.times.correction_gpu_seconds << ','
               << updatesAndOthersSeconds(result) << ','
               << result.times.mixed_total_seconds << ',' << record.iteration
               << ',' << record.residual << ',' << record.forward_error << '\n';
  }
}

void writeIterationsHeader(std::ofstream &output) {
  output
      << "paper_test_set,matrix_family,matrix_type,solver_backend,rng_seed,"
      << "rounding_mode,rounding_seed,diagonal_shift,shift_rule,"
      << "case,precision,basis_precision,spectrum_precision,correction_mode,"
      << "precision_policy,stopping_policy,dispatch_policy,termination_reason,"
      << "selection_policy,precision_attempts,execution_model,kernel_path,"
      << "selected_basis_stage,selected_spectrum_stage,"
      << "correction_stage_history,precision_escalations,"
      << "low_factor_corrections,low_factor_basis_bytes,fp32_basis_bytes,"
      << "basis_storage_ratio,quantized_basis_orthogonality,"
      << "factor_quantization_seconds,basis_reorthogonalization_seconds,"
      << "order,condition,warmups,evaluations,timing_evaluations,"
      << "reference_timing_min_seconds,reference_timing_max_seconds,"
      << "reference_timing_median_seconds,reference_timing_mean_seconds,"
      << "reference_timing_variance_seconds2,"
      << "paired_ratio_mean,paired_ratio_variance,paired_ratio_min,"
      << "paired_ratio_max,paired_ratio_median,"
      << "factorization_seconds,initial_root_seconds,single_seconds,residual_"
         "seconds,"
      << "correction_system_seconds,updates_and_others_seconds,mixed_total_"
         "seconds,"
      << "iteration,residual,forward_error\n";
}

} // namespace mixed_sqrt
