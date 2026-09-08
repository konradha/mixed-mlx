#include <metal_stdlib>
using namespace metal;

struct SquareParams {
  uint order;
};

struct NormPair {
  float residual_squared;
  float matrix_squared;
};

struct WorkingFloat {
  float high;
  float low;
};

inline WorkingFloat working_normalize(float high, float low) {
  const float sum = high + low;
  return WorkingFloat{sum, low - (sum - high)};
}

inline WorkingFloat working_add_pair(WorkingFloat left, WorkingFloat right) {
  const float sum = left.high + right.high;
  const float virtual_right = sum - left.high;
  const float error = (left.high - (sum - virtual_right)) +
                      (right.high - virtual_right) + left.low + right.low;
  return working_normalize(sum, error);
}

inline WorkingFloat working_multiply(WorkingFloat left, WorkingFloat right) {
  const float product = left.high * right.high;
  const float error = fma(left.high, right.high, -product) +
                      left.high * right.low + left.low * right.high +
                      left.low * right.low;
  return working_normalize(product, error);
}

inline WorkingFloat working_product_entry(const device WorkingFloat *left,
                                          const device WorkingFloat *right,
                                          uint row, uint column, uint order) {
  WorkingFloat sum{0.0f, 0.0f};
  for (uint inner = 0; inner < order; ++inner) {
    sum =
        working_add_pair(sum, working_multiply(left[row + inner * order],
                                               right[inner + column * order]));
  }
  return sum;
}

kernel void scale_columns_f32(const device float *vectors [[buffer(0)]],
                              const device float *scales [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant SquareParams &params [[buffer(3)]],
                              uint2 p [[thread_position_in_grid]]) {
  if (p.x < params.order && p.y < params.order) {
    output[p.y + p.x * params.order] =
        vectors[p.y + p.x * params.order] * scales[p.x];
  }
}

kernel void scale_columns_f16(const device half *vectors [[buffer(0)]],
                              const device float *scales [[buffer(1)]],
                              device half *output [[buffer(2)]],
                              constant SquareParams &params [[buffer(3)]],
                              uint2 p [[thread_position_in_grid]]) {
  if (p.x < params.order && p.y < params.order) {
    output[p.y + p.x * params.order] =
        half(float(vectors[p.y + p.x * params.order]) * scales[p.x]);
  }
}

kernel void apply_scale_f32(const device float *source [[buffer(0)]],
                            const device float *scale [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant uint &elements [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
  if (index < elements) {
    output[index] = source[index] * scale[index];
  }
}

kernel void working_from_f32(const device float *source [[buffer(0)]],
                             device WorkingFloat *destination [[buffer(1)]],
                             constant SquareParams &params [[buffer(2)]],
                             uint2 p [[thread_position_in_grid]]) {
  if (p.x < params.order && p.y < params.order) {
    const uint index = p.y + p.x * params.order;
    const uint transpose = p.x + p.y * params.order;
    destination[index] =
        WorkingFloat{0.5f * (source[index] + source[transpose]), 0.0f};
  }
}

kernel void working_residual(const device WorkingFloat *matrix [[buffer(0)]],
                             const device WorkingFloat *root [[buffer(1)]],
                             device WorkingFloat *residual [[buffer(2)]],
                             device float *residual_f32 [[buffer(3)]],
                             constant SquareParams &params [[buffer(4)]],
                             uint2 p [[thread_position_in_grid]]) {
  const uint row = p.y;
  const uint column = p.x;
  if (row <= column && column < params.order) {
    const uint index = row + column * params.order;
    const uint transpose = column + row * params.order;
    const WorkingFloat product =
        working_product_entry(root, root, row, column, params.order);
    const WorkingFloat value = working_add_pair(
        matrix[index], WorkingFloat{-product.high, -product.low});
    residual[index] = value;
    residual_f32[index] = value.high + value.low;
    residual[transpose] = value;
    residual_f32[transpose] = value.high + value.low;
  }
}

kernel void working_add(device WorkingFloat *root [[buffer(0)]],
                        const device float *correction [[buffer(1)]],
                        constant SquareParams &params [[buffer(2)]],
                        uint2 p [[thread_position_in_grid]]) {
  if (p.x < params.order && p.y < params.order) {
    const uint index = p.y + p.x * params.order;
    const uint transpose = p.x + p.y * params.order;
    const float value = 0.5f * (correction[index] + correction[transpose]);
    root[index] = working_add_pair(root[index], WorkingFloat{value, 0.0f});
  }
}

kernel void working_norm(const device WorkingFloat *matrix [[buffer(0)]],
                         const device WorkingFloat *residual [[buffer(1)]],
                         device NormPair *result [[buffer(2)]],
                         constant uint &elements [[buffer(3)]],
                         uint index [[thread_index_in_threadgroup]],
                         uint width [[threads_per_threadgroup]]) {
  threadgroup float residual_sums[256];
  threadgroup float matrix_sums[256];
  float residual_sum = 0.0f;
  float matrix_sum = 0.0f;
  for (uint i = index; i < elements; i += width) {
    const float residual_value = residual[i].high + residual[i].low;
    const float matrix_value = matrix[i].high + matrix[i].low;
    residual_sum = fma(residual_value, residual_value, residual_sum);
    matrix_sum = fma(matrix_value, matrix_value, matrix_sum);
  }
  residual_sums[index] = residual_sum;
  matrix_sums[index] = matrix_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = width / 2u; stride != 0u; stride /= 2u) {
    if (index < stride) {
      residual_sums[index] += residual_sums[index + stride];
      matrix_sums[index] += matrix_sums[index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (index == 0u) {
    result[0] = NormPair{residual_sums[0], matrix_sums[0]};
  }
}
