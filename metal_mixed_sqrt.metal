#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
  uint rows;
  uint columns;
  uint inner;
};
struct LowFactorMatmulParams {
  uint rows;
  uint columns;
  uint inner;
  uint transpose_low;
  uint apply_inverse;
};

struct SquareParams {
  uint order;
};

kernel void scale_columns_f32(const device float *eigenvectors [[buffer(0)]],
                              const device float *root_eigenvalues
                              [[buffer(1)]],
                              device float *output [[buffer(2)]],
                              constant SquareParams &params [[buffer(3)]],
                              uint3 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint matrix_offset = position.z * params.order * params.order;
  const uint vector_offset = position.z * params.order;
  output[matrix_offset + row + column * params.order] =
      eigenvectors[matrix_offset + row + column * params.order] *
      root_eigenvalues[vector_offset + column];
}

kernel void matmul_f32(const device float *left [[buffer(0)]],
                       const device float *right [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       constant MatmulParams &params [[buffer(3)]],
                       uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }

  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(left[row + inner * params.rows],
              right[inner + column * params.inner], sum);
  }
  output[row + column * params.rows] = sum;
}

kernel void diagonal_sylvester_f32(const device float *right_hand_side
                                   [[buffer(0)]],
                                   const device float *root_eigenvalues
                                   [[buffer(1)]],
                                   device float *correction [[buffer(2)]],
                                   constant SquareParams &params [[buffer(3)]],
                                   uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const float denominator = root_eigenvalues[row] + root_eigenvalues[column];
  correction[row + column * params.order] =
      right_hand_side[row + column * params.order] / denominator;
}

kernel void
apply_sylvester_inverse_f32(const device float *right_hand_side [[buffer(0)]],
                            const device float *inverse [[buffer(1)]],
                            device float *correction [[buffer(2)]],
                            constant SquareParams &params [[buffer(3)]],
                            uint2 position [[thread_position_in_grid]]) {
  const uint row = position.x;
  const uint column = position.y;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  correction[index] = right_hand_side[index] * inverse[index];
}

kernel void scale_columns_bf16(const device bfloat *eigenvectors [[buffer(0)]],
                               const device bfloat *root_eigenvalues
                               [[buffer(1)]],
                               device bfloat *output [[buffer(2)]],
                               constant SquareParams &params [[buffer(3)]],
                               uint3 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint matrix_offset = position.z * params.order * params.order;
  const uint vector_offset = position.z * params.order;
  const uint index = matrix_offset + row + column * params.order;
  output[index] = bfloat(float(eigenvectors[index]) *
                         float(root_eigenvalues[vector_offset + column]));
}

kernel void matmul_bf16(const device bfloat *left [[buffer(0)]],
                        const device bfloat *right [[buffer(1)]],
                        device bfloat *output [[buffer(2)]],
                        constant MatmulParams &params [[buffer(3)]],
                        uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }

  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(float(left[row + inner * params.rows]),
              float(right[inner + column * params.inner]), sum);
  }
  output[row + column * params.rows] = bfloat(sum);
}

kernel void
diagonal_sylvester_bf16(const device bfloat *right_hand_side [[buffer(0)]],
                        const device bfloat *root_eigenvalues [[buffer(1)]],
                        device bfloat *correction [[buffer(2)]],
                        constant SquareParams &params [[buffer(3)]],
                        uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const float denominator =
      float(root_eigenvalues[row]) + float(root_eigenvalues[column]);
  correction[row + column * params.order] =
      bfloat(float(right_hand_side[row + column * params.order]) / denominator);
}

constant uint kFastTile = 16u;
kernel void
apply_sylvester_inverse_bf16(const device bfloat *right_hand_side [[buffer(0)]],
                             const device float *inverse [[buffer(1)]],
                             device bfloat *correction [[buffer(2)]],
                             constant SquareParams &params [[buffer(3)]],
                             uint2 position [[thread_position_in_grid]]) {
  const uint row = position.x;
  const uint column = position.y;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  correction[index] = bfloat(float(right_hand_side[index]) * inverse[index]);
}

kernel void matmul_tiled_f32(const device float *left [[buffer(0)]],
                             const device float *right [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant MatmulParams &params [[buffer(3)]],
                             uint3 tile [[threadgroup_position_in_grid]],
                             uint3 local [[thread_position_in_threadgroup]]) {
  const uint row = tile.x * kFastTile + local.x;
  const uint column = tile.y * kFastTile + local.y;
  const uint linear = local.y * kFastTile + local.x;
  const uint left_row = linear & (kFastTile - 1u);
  const uint left_inner = linear / kFastTile;
  const uint right_inner = linear & (kFastTile - 1u);
  const uint right_column = linear / kFastTile;
  const uint left_offset = tile.z * params.rows * params.inner;
  const uint right_offset = tile.z * params.inner * params.columns;
  const uint output_offset = tile.z * params.rows * params.columns;
  threadgroup float left_tile[kFastTile][kFastTile];
  threadgroup float right_tile[kFastTile][kFastTile];
  float sum = 0.0f;

  for (uint base = 0; base < params.inner; base += kFastTile) {
    const uint load_left_inner = base + left_inner;
    const uint load_right_inner = base + right_inner;
    const uint global_left_row = tile.x * kFastTile + left_row;
    const uint global_right_column = tile.y * kFastTile + right_column;
    left_tile[left_inner][left_row] =
        global_left_row < params.rows && load_left_inner < params.inner
            ? left[left_offset + global_left_row +
                   load_left_inner * params.rows]
            : 0.0f;
    right_tile[right_column][right_inner] =
        load_right_inner < params.inner && global_right_column < params.columns
            ? right[right_offset + load_right_inner +
                    global_right_column * params.inner]
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < kFastTile; ++inner) {
      sum = fma(left_tile[inner][local.x], right_tile[local.y][inner], sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && column < params.columns) {
    output[output_offset + row + column * params.rows] = sum;
  }
}

kernel void matmul_tiled_scaled_f32(const device float *left [[buffer(0)]],
                                    const device float *right [[buffer(1)]],
                                    const device float *inverse [[buffer(2)]],
                                    device float *output [[buffer(3)]],
                                    constant MatmulParams &params [[buffer(4)]],
                                    uint3 tile [[threadgroup_position_in_grid]],
                                    uint3 local
                                    [[thread_position_in_threadgroup]]) {
  const uint row = tile.x * kFastTile + local.x;
  const uint column = tile.y * kFastTile + local.y;
  const uint linear = local.y * kFastTile + local.x;
  const uint left_row = linear & (kFastTile - 1u);
  const uint left_inner = linear / kFastTile;
  const uint right_inner = linear & (kFastTile - 1u);
  const uint right_column = linear / kFastTile;
  const uint left_offset = tile.z * params.rows * params.inner;
  const uint right_offset = tile.z * params.inner * params.columns;
  const uint output_offset = tile.z * params.rows * params.columns;
  threadgroup float left_tile[kFastTile][kFastTile];
  threadgroup float right_tile[kFastTile][kFastTile];
  float sum = 0.0f;

  for (uint base = 0; base < params.inner; base += kFastTile) {
    const uint load_left_inner = base + left_inner;
    const uint load_right_inner = base + right_inner;
    const uint global_left_row = tile.x * kFastTile + left_row;
    const uint global_right_column = tile.y * kFastTile + right_column;
    left_tile[left_inner][left_row] =
        global_left_row < params.rows && load_left_inner < params.inner
            ? left[left_offset + global_left_row +
                   load_left_inner * params.rows]
            : 0.0f;
    right_tile[right_column][right_inner] =
        load_right_inner < params.inner && global_right_column < params.columns
            ? right[right_offset + load_right_inner +
                    global_right_column * params.inner]
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < kFastTile; ++inner) {
      sum = fma(left_tile[inner][local.x], right_tile[local.y][inner], sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && column < params.columns) {
    const uint index = output_offset + row + column * params.rows;
    output[index] = sum * inverse[index];
  }
}

kernel void diagonal_sylvester_coalesced_f32(
    const device float *right_hand_side [[buffer(0)]],
    const device float *root_eigenvalues [[buffer(1)]],
    device float *correction [[buffer(2)]],
    constant SquareParams &params [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint row = position.x;
  const uint column = position.y;
  if (row >= params.order || column >= params.order) {
    return;
  }
  correction[row + column * params.order] =
      right_hand_side[row + column * params.order] /
      (root_eigenvalues[row] + root_eigenvalues[column]);
}

kernel void matmul_tiled_bf16(const device bfloat *left [[buffer(0)]],
                              const device bfloat *right [[buffer(1)]],
                              device bfloat *output [[buffer(2)]],
                              constant MatmulParams &params [[buffer(3)]],
                              uint3 tile [[threadgroup_position_in_grid]],
                              uint3 local [[thread_position_in_threadgroup]]) {
  const uint row = tile.x * kFastTile + local.x;
  const uint column = tile.y * kFastTile + local.y;
  const uint linear = local.y * kFastTile + local.x;
  const uint left_row = linear & (kFastTile - 1u);
  const uint left_inner = linear / kFastTile;
  const uint right_inner = linear & (kFastTile - 1u);
  const uint right_column = linear / kFastTile;
  const uint left_offset = tile.z * params.rows * params.inner;
  const uint right_offset = tile.z * params.inner * params.columns;
  const uint output_offset = tile.z * params.rows * params.columns;
  threadgroup float left_tile[kFastTile][kFastTile];
  threadgroup float right_tile[kFastTile][kFastTile];
  float sum = 0.0f;

  for (uint base = 0; base < params.inner; base += kFastTile) {
    const uint load_left_inner = base + left_inner;
    const uint load_right_inner = base + right_inner;
    const uint global_left_row = tile.x * kFastTile + left_row;
    const uint global_right_column = tile.y * kFastTile + right_column;
    left_tile[left_inner][left_row] =
        global_left_row < params.rows && load_left_inner < params.inner
            ? float(left[left_offset + global_left_row +
                         load_left_inner * params.rows])
            : 0.0f;
    right_tile[right_column][right_inner] =
        load_right_inner < params.inner && global_right_column < params.columns
            ? float(right[right_offset + load_right_inner +
                          global_right_column * params.inner])
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < kFastTile; ++inner) {
      sum = fma(left_tile[inner][local.x], right_tile[local.y][inner], sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && column < params.columns) {
    output[output_offset + row + column * params.rows] = bfloat(sum);
  }
}

kernel void
matmul_tiled_scaled_bf16(const device bfloat *left [[buffer(0)]],
                         const device bfloat *right [[buffer(1)]],
                         const device float *inverse [[buffer(2)]],
                         device bfloat *output [[buffer(3)]],
                         constant MatmulParams &params [[buffer(4)]],
                         uint3 tile [[threadgroup_position_in_grid]],
                         uint3 local [[thread_position_in_threadgroup]]) {
  const uint row = tile.x * kFastTile + local.x;
  const uint column = tile.y * kFastTile + local.y;
  const uint linear = local.y * kFastTile + local.x;
  const uint left_row = linear & (kFastTile - 1u);
  const uint left_inner = linear / kFastTile;
  const uint right_inner = linear & (kFastTile - 1u);
  const uint right_column = linear / kFastTile;
  const uint left_offset = tile.z * params.rows * params.inner;
  const uint right_offset = tile.z * params.inner * params.columns;
  const uint output_offset = tile.z * params.rows * params.columns;
  threadgroup float left_tile[kFastTile][kFastTile];
  threadgroup float right_tile[kFastTile][kFastTile];
  float sum = 0.0f;

  for (uint base = 0; base < params.inner; base += kFastTile) {
    const uint load_left_inner = base + left_inner;
    const uint load_right_inner = base + right_inner;
    const uint global_left_row = tile.x * kFastTile + left_row;
    const uint global_right_column = tile.y * kFastTile + right_column;
    left_tile[left_inner][left_row] =
        global_left_row < params.rows && load_left_inner < params.inner
            ? float(left[left_offset + global_left_row +
                         load_left_inner * params.rows])
            : 0.0f;
    right_tile[right_column][right_inner] =
        load_right_inner < params.inner && global_right_column < params.columns
            ? float(right[right_offset + load_right_inner +
                          global_right_column * params.inner])
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < kFastTile; ++inner) {
      sum = fma(left_tile[inner][local.x], right_tile[local.y][inner], sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && column < params.columns) {
    const uint index = output_offset + row + column * params.rows;
    output[index] = bfloat(sum * inverse[index]);
  }
}

kernel void diagonal_sylvester_coalesced_bf16(
    const device bfloat *right_hand_side [[buffer(0)]],
    const device bfloat *root_eigenvalues [[buffer(1)]],
    device bfloat *correction [[buffer(2)]],
    constant SquareParams &params [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint row = position.x;
  const uint column = position.y;
  if (row >= params.order || column >= params.order) {
    return;
  }
  correction[row + column * params.order] =
      bfloat(float(right_hand_side[row + column * params.order]) /
             (float(root_eigenvalues[row]) + float(root_eigenvalues[column])));
}

inline uint round_to_nearest_even_nonnegative(float value) {
  const float lower = floor(value);
  const float fractional = value - lower;
  uint rounded = uint(lower);
  if (fractional > 0.5f || (fractional == 0.5f && (rounded & 1u) != 0u)) {
    ++rounded;
  }
  return rounded;
}

inline uint stochastic_hash(uint index, uint seed, uint stream) {
  uint state = index * 747796405u + seed + stream * 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

inline bool stochastic_round_up(float probability, uint index, uint seed,
                                uint stream) {
  const float sample =
      float(stochastic_hash(index, seed, stream) >> 8u) * 0x1.0p-24f;
  return sample < probability;
}

inline float decode_fp8_e4m3fn(uchar encoded) {
  const uint bits = uint(encoded);
  const uint exponent = (bits >> 3u) & 0x0fu;
  const uint mantissa = bits & 0x07u;
  const bool negative = (bits & 0x80u) != 0u;

  if (exponent == 0u) {
    const float magnitude = float(mantissa) * 0.001953125f;
    return negative ? -magnitude : magnitude;
  }
  if (exponent == 0x0fu && mantissa == 0x07u) {
    return as_type<float>((negative ? 0x80000000u : 0u) | 0x7fc00000u);
  }

  const float magnitude =
      (1.0f + float(mantissa) * 0.125f) * exp2(float(int(exponent) - 7));
  return negative ? -magnitude : magnitude;
}

inline uchar encode_fp8_e4m3fn(float value) {
  const bool negative = signbit(value);
  const float magnitude = fabs(value);
  const uint sign = negative ? 0x80u : 0u;

  if (isnan(value)) {
    return uchar(sign | 0x7fu);
  }
  if (magnitude >= 448.0f) {
    return uchar(sign | 0x7eu);
  }
  if (magnitude < 0.015625f) {
    const uint mantissa = round_to_nearest_even_nonnegative(magnitude * 512.0f);
    if (mantissa >= 8u) {
      return uchar(sign | 0x08u);
    }
    return uchar(sign | mantissa);
  }

  int unbiased_exponent = int(floor(log2(magnitude)));
  uint exponent = uint(unbiased_exponent + 7);
  uint mantissa = round_to_nearest_even_nonnegative(
      (magnitude / exp2(float(unbiased_exponent)) - 1.0f) * 8.0f);
  if (mantissa == 8u) {
    mantissa = 0u;
    ++exponent;
  }
  if (exponent > 0x0fu) {
    return uchar(sign | 0x7eu);
  }
  if (exponent == 0x0fu && mantissa == 0x07u) {
    mantissa = 0x06u;
  }
  return uchar(sign | (exponent << 3u) | mantissa);
}

inline uchar encode_fp8_e4m3fn_stochastic(float value, uint index, uint seed,
                                          uint stream) {
  const uint sign = signbit(value) ? 0x80u : 0u;
  const float magnitude = fabs(value);
  if (isnan(value) || magnitude == 0.0f || magnitude >= 448.0f) {
    return encode_fp8_e4m3fn(value);
  }

  const uint nearest = uint(encode_fp8_e4m3fn(magnitude));
  const float nearest_value = decode_fp8_e4m3fn(uchar(nearest));
  if (nearest_value == magnitude) {
    return uchar(sign | nearest);
  }
  const uint lower = nearest_value < magnitude ? nearest : nearest - 1u;
  const uint upper = lower + 1u;
  const float lower_value = decode_fp8_e4m3fn(uchar(lower));
  const float upper_value = decode_fp8_e4m3fn(uchar(upper));
  const float probability =
      (magnitude - lower_value) / (upper_value - lower_value);
  const uint rounded =
      stochastic_round_up(probability, index, seed, stream) ? upper : lower;
  return uchar(sign | rounded);
}

inline float decode_fp8_e5m2(uchar encoded) {
  const uint bits = uint(encoded);
  const uint exponent = (bits >> 2u) & 0x1fu;
  const uint mantissa = bits & 0x03u;
  const bool negative = (bits & 0x80u) != 0u;

  if (exponent == 0u) {
    const float magnitude = float(mantissa) * 0.0000152587890625f;
    return negative ? -magnitude : magnitude;
  }
  if (exponent == 0x1fu) {
    if (mantissa == 0u) {
      return as_type<float>((negative ? 0x80000000u : 0u) | 0x7f800000u);
    }
    return as_type<float>((negative ? 0x80000000u : 0u) | 0x7fc00000u);
  }

  const float magnitude =
      (1.0f + float(mantissa) * 0.25f) * exp2(float(int(exponent) - 15));
  return negative ? -magnitude : magnitude;
}

inline uchar encode_fp8_e5m2(float value) {
  const bool negative = signbit(value);
  const float magnitude = fabs(value);
  const uint sign = negative ? 0x80u : 0u;

  if (isnan(value)) {
    return uchar(sign | 0x7eu);
  }
  if (isinf(value)) {
    return uchar(sign | 0x7cu);
  }
  if (magnitude >= 57344.0f) {
    return uchar(sign | 0x7bu);
  }
  if (magnitude < 0.00006103515625f) {
    const uint mantissa =
        round_to_nearest_even_nonnegative(magnitude * 65536.0f);
    if (mantissa >= 4u) {
      return uchar(sign | 0x04u);
    }
    return uchar(sign | mantissa);
  }

  int unbiased_exponent = int(floor(log2(magnitude)));
  uint exponent = uint(unbiased_exponent + 15);
  uint mantissa = round_to_nearest_even_nonnegative(
      (magnitude / exp2(float(unbiased_exponent)) - 1.0f) * 4.0f);
  if (mantissa == 4u) {
    mantissa = 0u;
    ++exponent;
  }
  if (exponent >= 0x1fu) {
    return uchar(sign | 0x7bu);
  }
  return uchar(sign | (exponent << 2u) | mantissa);
}

inline uchar encode_fp8_e5m2_stochastic(float value, uint index, uint seed,
                                        uint stream) {
  const uint sign = signbit(value) ? 0x80u : 0u;
  const float magnitude = fabs(value);
  if (isnan(value) || isinf(value) || magnitude == 0.0f ||
      magnitude >= 57344.0f) {
    return encode_fp8_e5m2(value);
  }

  const uint nearest = uint(encode_fp8_e5m2(magnitude));
  const float nearest_value = decode_fp8_e5m2(uchar(nearest));
  if (nearest_value == magnitude) {
    return uchar(sign | nearest);
  }
  const uint lower = nearest_value < magnitude ? nearest : nearest - 1u;
  const uint upper = lower + 1u;
  const float lower_value = decode_fp8_e5m2(uchar(lower));
  const float upper_value = decode_fp8_e5m2(uchar(upper));
  const float probability =
      (magnitude - lower_value) / (upper_value - lower_value);
  const uint rounded =
      stochastic_round_up(probability, index, seed, stream) ? upper : lower;
  return uchar(sign | rounded);
}

kernel void
scale_columns_fp8_e4m3fn(const device uchar *eigenvectors [[buffer(0)]],
                         const device uchar *root_eigenvalues [[buffer(1)]],
                         device uchar *output [[buffer(2)]],
                         constant SquareParams &params [[buffer(3)]],
                         uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const uint index = row + column * params.order;
  output[index] =
      encode_fp8_e4m3fn(decode_fp8_e4m3fn(eigenvectors[index]) *
                        decode_fp8_e4m3fn(root_eigenvalues[column]));
}

kernel void matmul_fp8_e4m3fn(const device uchar *left [[buffer(0)]],
                              const device uchar *right [[buffer(1)]],
                              device uchar *output [[buffer(2)]],
                              constant MatmulParams &params [[buffer(3)]],
                              uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }

  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(decode_fp8_e4m3fn(left[row + inner * params.rows]),
              decode_fp8_e4m3fn(right[inner + column * params.inner]), sum);
  }
  output[row + column * params.rows] = encode_fp8_e4m3fn(sum);
}

kernel void
diagonal_sylvester_fp8_e4m3fn(const device uchar *right_hand_side [[buffer(0)]],
                              const device uchar *root_eigenvalues
                              [[buffer(1)]],
                              device uchar *correction [[buffer(2)]],
                              constant SquareParams &params [[buffer(3)]],
                              uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const float denominator = decode_fp8_e4m3fn(root_eigenvalues[row]) +
                            decode_fp8_e4m3fn(root_eigenvalues[column]);
  correction[row + column * params.order] = encode_fp8_e4m3fn(
      decode_fp8_e4m3fn(right_hand_side[row + column * params.order]) /
      denominator);
}

kernel void scale_columns_fp8_e5m2(const device uchar *eigenvectors
                                   [[buffer(0)]],
                                   const device uchar *root_eigenvalues
                                   [[buffer(1)]],
                                   device uchar *output [[buffer(2)]],
                                   constant SquareParams &params [[buffer(3)]],
                                   uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const uint index = row + column * params.order;
  output[index] = encode_fp8_e5m2(decode_fp8_e5m2(eigenvectors[index]) *
                                  decode_fp8_e5m2(root_eigenvalues[column]));
}

kernel void matmul_fp8_e5m2(const device uchar *left [[buffer(0)]],
                            const device uchar *right [[buffer(1)]],
                            device uchar *output [[buffer(2)]],
                            constant MatmulParams &params [[buffer(3)]],
                            uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }

  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(decode_fp8_e5m2(left[row + inner * params.rows]),
              decode_fp8_e5m2(right[inner + column * params.inner]), sum);
  }
  output[row + column * params.rows] = encode_fp8_e5m2(sum);
}

kernel void
diagonal_sylvester_fp8_e5m2(const device uchar *right_hand_side [[buffer(0)]],
                            const device uchar *root_eigenvalues [[buffer(1)]],
                            device uchar *correction [[buffer(2)]],
                            constant SquareParams &params [[buffer(3)]],
                            uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const float denominator = decode_fp8_e5m2(root_eigenvalues[row]) +
                            decode_fp8_e5m2(root_eigenvalues[column]);
  correction[row + column * params.order] = encode_fp8_e5m2(
      decode_fp8_e5m2(right_hand_side[row + column * params.order]) /
      denominator);
}

struct QuantizeParams {
  uint elements;
  uint block_elements;
  uint rounding_seed;
  uint rounding_stream;
};

struct LogSpectrumParams {
  uint elements;
  uint block_elements;
  uint rounding_seed;
  uint rounding_stream;
};

inline uchar unpack_fp4_e2m1(const device uchar *values, uint index) {
  return uchar((uint(values[index >> 1u]) >> ((index & 1u) << 2u)) & 0x0fu);
}

inline float decode_fp4_e2m1(uchar encoded) {
  const uint bits = uint(encoded) & 0x0fu;
  const uint exponent = (bits >> 1u) & 0x03u;
  const uint fraction = bits & 0x01u;
  const float magnitude = exponent == 0u ? float(fraction) * 0.5f
                                         : (1.0f + float(fraction) * 0.5f) *
                                               exp2(float(int(exponent) - 1));
  return (bits & 0x08u) != 0u ? -magnitude : magnitude;
}

inline uchar encode_fp4_e2m1(float value) {
  const uint sign = signbit(value) ? 0x08u : 0u;
  if (isnan(value) || value == 0.0f) {
    return uchar(sign);
  }
  if (isinf(value)) {
    return uchar(sign | 0x07u);
  }

  const float magnitude = fabs(value);
  uint nearest = 0u;
  float nearest_distance = fabs(magnitude - decode_fp4_e2m1(uchar(nearest)));
  for (uint candidate = 1u; candidate <= 0x07u; ++candidate) {
    const float distance = fabs(magnitude - decode_fp4_e2m1(uchar(candidate)));
    if (distance < nearest_distance ||
        (distance == nearest_distance && (candidate & 1u) == 0u &&
         (nearest & 1u) != 0u)) {
      nearest = candidate;
      nearest_distance = distance;
    }
  }
  return uchar(sign | nearest);
}

inline uchar encode_fp4_e2m1_stochastic(float value, uint index, uint seed,
                                        uint stream) {
  const uint sign = signbit(value) ? 0x08u : 0u;
  const float magnitude = fabs(value);
  if (isnan(value) || magnitude == 0.0f || magnitude >= 6.0f) {
    return encode_fp4_e2m1(value);
  }

  uint lower = 0u;
  while (lower < 0x07u && decode_fp4_e2m1(uchar(lower + 1u)) <= magnitude) {
    ++lower;
  }
  const float lower_value = decode_fp4_e2m1(uchar(lower));
  if (lower_value == magnitude || lower == 0x07u) {
    return uchar(sign | lower);
  }
  const uint upper = lower + 1u;
  const float upper_value = decode_fp4_e2m1(uchar(upper));
  const float probability =
      (magnitude - lower_value) / (upper_value - lower_value);
  const uint rounded =
      stochastic_round_up(probability, index, seed, stream) ? upper : lower;
  return uchar(sign | rounded);
}

inline float decode_mxfp4_e2m1(const device uchar *values,
                               const device uchar *scales, uint index) {
  const int scale_exponent = int(uint(scales[index / 32u])) - 127;
  return decode_fp4_e2m1(unpack_fp4_e2m1(values, index)) *
         exp2(float(scale_exponent));
}

kernel void scale_columns_fp4_e2m1(const device uchar *eigenvectors
                                   [[buffer(0)]],
                                   const device uchar *root_eigenvalues
                                   [[buffer(1)]],
                                   device float *output [[buffer(2)]],
                                   constant SquareParams &params [[buffer(3)]],
                                   uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const uint index = row + column * params.order;
  output[index] = decode_fp4_e2m1(unpack_fp4_e2m1(eigenvectors, index)) *
                  decode_fp4_e2m1(unpack_fp4_e2m1(root_eigenvalues, column));
}

kernel void matmul_fp4_e2m1(const device uchar *left [[buffer(0)]],
                            const device uchar *right [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            constant MatmulParams &params [[buffer(3)]],
                            uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }

  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(
        decode_fp4_e2m1(unpack_fp4_e2m1(left, row + inner * params.rows)),
        decode_fp4_e2m1(unpack_fp4_e2m1(right, inner + column * params.inner)),
        sum);
  }
  output[row + column * params.rows] = sum;
}

kernel void
diagonal_sylvester_fp4_e2m1(const device uchar *right_hand_side [[buffer(0)]],
                            const device uchar *root_eigenvalues [[buffer(1)]],
                            device float *correction [[buffer(2)]],
                            constant SquareParams &params [[buffer(3)]],
                            uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const float denominator =
      decode_fp4_e2m1(unpack_fp4_e2m1(root_eigenvalues, row)) +
      decode_fp4_e2m1(unpack_fp4_e2m1(root_eigenvalues, column));
  const uint index = row + column * params.order;
  correction[index] =
      decode_fp4_e2m1(unpack_fp4_e2m1(right_hand_side, index)) / denominator;
}

kernel void quantize_fp4_e2m1(const device float *temporary [[buffer(0)]],
                              device uchar *output [[buffer(1)]],
                              device uchar *scales [[buffer(2)]],
                              constant QuantizeParams &params [[buffer(3)]],
                              uint byte_index [[thread_position_in_grid]]) {
  const uint first = byte_index << 1u;
  if (first >= params.elements) {
    return;
  }

  const uchar lower = encode_fp4_e2m1(temporary[first]);
  const uchar upper = first + 1u < params.elements
                          ? encode_fp4_e2m1(temporary[first + 1u])
                          : uchar(0u);
  output[byte_index] = lower | uchar(uint(upper) << 4u);
}

kernel void
quantize_fp4_e2m1_stochastic(const device float *temporary [[buffer(0)]],
                             device uchar *output [[buffer(1)]],
                             device uchar *scales [[buffer(2)]],
                             constant QuantizeParams &params [[buffer(3)]],
                             uint byte_index [[thread_position_in_grid]]) {
  const uint first = byte_index << 1u;
  if (first >= params.elements) {
    return;
  }

  const uchar lower = encode_fp4_e2m1_stochastic(
      temporary[first], first, params.rounding_seed, params.rounding_stream);
  const uchar upper = first + 1u < params.elements
                          ? encode_fp4_e2m1_stochastic(
                                temporary[first + 1u], first + 1u,
                                params.rounding_seed, params.rounding_stream)
                          : uchar(0u);
  output[byte_index] = lower | uchar(uint(upper) << 4u);
}

kernel void
scale_columns_mxfp4_e2m1(const device uchar *eigenvectors [[buffer(0)]],
                         const device uchar *eigenvector_scales [[buffer(1)]],
                         const device uchar *root_eigenvalues [[buffer(2)]],
                         const device uchar *root_eigenvalue_scales
                         [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         constant SquareParams &params [[buffer(5)]],
                         uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const uint index = row + column * params.order;
  output[index] =
      decode_mxfp4_e2m1(eigenvectors, eigenvector_scales, index) *
      decode_mxfp4_e2m1(root_eigenvalues, root_eigenvalue_scales, column);
}

kernel void matmul_mxfp4_e2m1(const device uchar *left [[buffer(0)]],
                              const device uchar *left_scales [[buffer(1)]],
                              const device uchar *right [[buffer(2)]],
                              const device uchar *right_scales [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              constant MatmulParams &params [[buffer(5)]],
                              uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }

  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(
        decode_mxfp4_e2m1(left, left_scales, row + inner * params.rows),
        decode_mxfp4_e2m1(right, right_scales, inner + column * params.inner),
        sum);
  }
  output[row + column * params.rows] = sum;
}

kernel void diagonal_sylvester_mxfp4_e2m1(
    const device uchar *right_hand_side [[buffer(0)]],
    const device uchar *right_hand_side_scales [[buffer(1)]],
    const device uchar *root_eigenvalues [[buffer(2)]],
    const device uchar *root_eigenvalue_scales [[buffer(3)]],
    device float *correction [[buffer(4)]],
    constant SquareParams &params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }

  const float denominator =
      decode_mxfp4_e2m1(root_eigenvalues, root_eigenvalue_scales, row) +
      decode_mxfp4_e2m1(root_eigenvalues, root_eigenvalue_scales, column);
  const uint index = row + column * params.order;
  correction[index] =
      decode_mxfp4_e2m1(right_hand_side, right_hand_side_scales, index) /
      denominator;
}

kernel void quantize_mxfp4_e2m1(const device float *temporary [[buffer(0)]],
                                device uchar *output [[buffer(1)]],
                                device uchar *scales [[buffer(2)]],
                                constant QuantizeParams &params [[buffer(3)]],
                                uint block [[thread_position_in_grid]]) {
  const uint block_start = block * 32u;
  if (block_start >= params.elements) {
    return;
  }
  const uint block_end = min(block_start + 32u, params.elements);

  float maximum = 0.0f;
  for (uint index = block_start; index < block_end; ++index) {
    const float value = temporary[index];
    if (isfinite(value)) {
      maximum = max(maximum, fabs(value));
    }
  }

  int scale_exponent = 0;
  if (maximum != 0.0f) {
    scale_exponent = clamp(int(floor(log2(maximum))) - 2, -127, 128);
  }
  scales[block] = uchar(scale_exponent + 127);
  const float scale = exp2(float(scale_exponent));

  for (uint index = block_start; index < block_end; index += 2u) {
    const uchar lower = encode_fp4_e2m1(temporary[index] / scale);
    const uchar upper = index + 1u < block_end
                            ? encode_fp4_e2m1(temporary[index + 1u] / scale)
                            : uchar(0u);
    output[index >> 1u] = lower | uchar(uint(upper) << 4u);
  }
}

kernel void
quantize_mxfp4_e2m1_stochastic(const device float *temporary [[buffer(0)]],
                               device uchar *output [[buffer(1)]],
                               device uchar *scales [[buffer(2)]],
                               constant QuantizeParams &params [[buffer(3)]],
                               uint block [[thread_position_in_grid]]) {
  const uint block_start = block * 32u;
  if (block_start >= params.elements) {
    return;
  }
  const uint block_end = min(block_start + 32u, params.elements);

  float maximum = 0.0f;
  for (uint index = block_start; index < block_end; ++index) {
    const float value = temporary[index];
    if (isfinite(value)) {
      maximum = max(maximum, fabs(value));
    }
  }

  int scale_exponent = 0;
  if (maximum != 0.0f) {
    scale_exponent = clamp(int(floor(log2(maximum))) - 2, -127, 128);
  }
  scales[block] = uchar(scale_exponent + 127);
  const float scale = exp2(float(scale_exponent));

  for (uint index = block_start; index < block_end; index += 2u) {
    const uchar lower = encode_fp4_e2m1_stochastic(temporary[index] / scale,
                                                   index, params.rounding_seed,
                                                   params.rounding_stream);
    const uchar upper = index + 1u < block_end
                            ? encode_fp4_e2m1_stochastic(
                                  temporary[index + 1u] / scale, index + 1u,
                                  params.rounding_seed, params.rounding_stream)
                            : uchar(0u);
    output[index >> 1u] = lower | uchar(uint(upper) << 4u);
  }
}

inline float decode_mxfp8_e4m3fn(const device uchar *values,
                                 const device uchar *scales, uint index) {
  const int scale_exponent = int(uint(scales[index / 32u])) - 127;
  return decode_fp8_e4m3fn(values[index]) * exp2(float(scale_exponent));
}

inline float decode_mxfp8_e5m2(const device uchar *values,
                               const device uchar *scales, uint index) {
  const int scale_exponent = int(uint(scales[index / 32u])) - 127;
  return decode_fp8_e5m2(values[index]) * exp2(float(scale_exponent));
}

kernel void
scale_columns_mxfp8_e4m3fn(const device uchar *eigenvectors [[buffer(0)]],
                           const device uchar *eigenvector_scales [[buffer(1)]],
                           const device uchar *root_eigenvalues [[buffer(2)]],
                           const device uchar *root_eigenvalue_scales
                           [[buffer(3)]],
                           device float *output [[buffer(4)]],
                           constant SquareParams &params [[buffer(5)]],
                           uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  output[index] =
      decode_mxfp8_e4m3fn(eigenvectors, eigenvector_scales, index) *
      decode_mxfp8_e4m3fn(root_eigenvalues, root_eigenvalue_scales, column);
}

kernel void matmul_mxfp8_e4m3fn(const device uchar *left [[buffer(0)]],
                                const device uchar *left_scales [[buffer(1)]],
                                const device uchar *right [[buffer(2)]],
                                const device uchar *right_scales [[buffer(3)]],
                                device float *output [[buffer(4)]],
                                constant MatmulParams &params [[buffer(5)]],
                                uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }
  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(
        decode_mxfp8_e4m3fn(left, left_scales, row + inner * params.rows),
        decode_mxfp8_e4m3fn(right, right_scales, inner + column * params.inner),
        sum);
  }
  output[row + column * params.rows] = sum;
}

kernel void diagonal_sylvester_mxfp8_e4m3fn(
    const device uchar *right_hand_side [[buffer(0)]],
    const device uchar *right_hand_side_scales [[buffer(1)]],
    const device uchar *root_eigenvalues [[buffer(2)]],
    const device uchar *root_eigenvalue_scales [[buffer(3)]],
    device float *correction [[buffer(4)]],
    constant SquareParams &params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const float denominator =
      decode_mxfp8_e4m3fn(root_eigenvalues, root_eigenvalue_scales, row) +
      decode_mxfp8_e4m3fn(root_eigenvalues, root_eigenvalue_scales, column);
  const uint index = row + column * params.order;
  correction[index] =
      decode_mxfp8_e4m3fn(right_hand_side, right_hand_side_scales, index) /
      denominator;
}

kernel void quantize_mxfp8_e4m3fn(const device float *temporary [[buffer(0)]],
                                  device uchar *output [[buffer(1)]],
                                  device uchar *scales [[buffer(2)]],
                                  constant QuantizeParams &params [[buffer(3)]],
                                  uint block [[thread_position_in_grid]]) {
  const uint block_start = block * 32u;
  if (block_start >= params.elements) {
    return;
  }
  const uint block_end = min(block_start + 32u, params.elements);
  float maximum = 0.0f;
  for (uint index = block_start; index < block_end; ++index) {
    const float value = temporary[index];
    if (isfinite(value)) {
      maximum = max(maximum, fabs(value));
    }
  }
  const int scale_exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 8, -127, 128);
  scales[block] = uchar(scale_exponent + 127);
  const float scale = exp2(float(scale_exponent));
  for (uint index = block_start; index < block_end; ++index) {
    output[index] = encode_fp8_e4m3fn(temporary[index] / scale);
  }
}

kernel void
quantize_mxfp8_e4m3fn_stochastic(const device float *temporary [[buffer(0)]],
                                 device uchar *output [[buffer(1)]],
                                 device uchar *scales [[buffer(2)]],
                                 constant QuantizeParams &params [[buffer(3)]],
                                 uint block [[thread_position_in_grid]]) {
  const uint block_start = block * 32u;
  if (block_start >= params.elements) {
    return;
  }
  const uint block_end = min(block_start + 32u, params.elements);
  float maximum = 0.0f;
  for (uint index = block_start; index < block_end; ++index) {
    const float value = temporary[index];
    if (isfinite(value)) {
      maximum = max(maximum, fabs(value));
    }
  }
  const int scale_exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 8, -127, 128);
  scales[block] = uchar(scale_exponent + 127);
  const float scale = exp2(float(scale_exponent));
  for (uint index = block_start; index < block_end; ++index) {
    output[index] = encode_fp8_e4m3fn_stochastic(temporary[index] / scale,
                                                 index, params.rounding_seed,
                                                 params.rounding_stream);
  }
}

kernel void
scale_columns_mxfp8_e5m2(const device uchar *eigenvectors [[buffer(0)]],
                         const device uchar *eigenvector_scales [[buffer(1)]],
                         const device uchar *root_eigenvalues [[buffer(2)]],
                         const device uchar *root_eigenvalue_scales
                         [[buffer(3)]],
                         device float *output [[buffer(4)]],
                         constant SquareParams &params [[buffer(5)]],
                         uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  output[index] =
      decode_mxfp8_e5m2(eigenvectors, eigenvector_scales, index) *
      decode_mxfp8_e5m2(root_eigenvalues, root_eigenvalue_scales, column);
}

kernel void matmul_mxfp8_e5m2(const device uchar *left [[buffer(0)]],
                              const device uchar *left_scales [[buffer(1)]],
                              const device uchar *right [[buffer(2)]],
                              const device uchar *right_scales [[buffer(3)]],
                              device float *output [[buffer(4)]],
                              constant MatmulParams &params [[buffer(5)]],
                              uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.rows || column >= params.columns) {
    return;
  }
  float sum = 0.0f;
  for (uint inner = 0; inner < params.inner; ++inner) {
    sum = fma(
        decode_mxfp8_e5m2(left, left_scales, row + inner * params.rows),
        decode_mxfp8_e5m2(right, right_scales, inner + column * params.inner),
        sum);
  }
  output[row + column * params.rows] = sum;
}

kernel void diagonal_sylvester_mxfp8_e5m2(
    const device uchar *right_hand_side [[buffer(0)]],
    const device uchar *right_hand_side_scales [[buffer(1)]],
    const device uchar *root_eigenvalues [[buffer(2)]],
    const device uchar *root_eigenvalue_scales [[buffer(3)]],
    device float *correction [[buffer(4)]],
    constant SquareParams &params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const float denominator =
      decode_mxfp8_e5m2(root_eigenvalues, root_eigenvalue_scales, row) +
      decode_mxfp8_e5m2(root_eigenvalues, root_eigenvalue_scales, column);
  const uint index = row + column * params.order;
  correction[index] =
      decode_mxfp8_e5m2(right_hand_side, right_hand_side_scales, index) /
      denominator;
}

kernel void quantize_mxfp8_e5m2(const device float *temporary [[buffer(0)]],
                                device uchar *output [[buffer(1)]],
                                device uchar *scales [[buffer(2)]],
                                constant QuantizeParams &params [[buffer(3)]],
                                uint block [[thread_position_in_grid]]) {
  const uint block_start = block * 32u;
  if (block_start >= params.elements) {
    return;
  }
  const uint block_end = min(block_start + 32u, params.elements);
  float maximum = 0.0f;
  for (uint index = block_start; index < block_end; ++index) {
    const float value = temporary[index];
    if (isfinite(value)) {
      maximum = max(maximum, fabs(value));
    }
  }
  const int scale_exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 15, -127, 128);
  scales[block] = uchar(scale_exponent + 127);
  const float scale = exp2(float(scale_exponent));
  for (uint index = block_start; index < block_end; ++index) {
    output[index] = encode_fp8_e5m2(temporary[index] / scale);
  }
}

kernel void
quantize_mxfp8_e5m2_stochastic(const device float *temporary [[buffer(0)]],
                               device uchar *output [[buffer(1)]],
                               device uchar *scales [[buffer(2)]],
                               constant QuantizeParams &params [[buffer(3)]],
                               uint block [[thread_position_in_grid]]) {
  const uint block_start = block * 32u;
  if (block_start >= params.elements) {
    return;
  }
  const uint block_end = min(block_start + 32u, params.elements);
  float maximum = 0.0f;
  for (uint index = block_start; index < block_end; ++index) {
    const float value = temporary[index];
    if (isfinite(value)) {
      maximum = max(maximum, fabs(value));
    }
  }
  const int scale_exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 15, -127, 128);
  scales[block] = uchar(scale_exponent + 127);
  const float scale = exp2(float(scale_exponent));
  for (uint index = block_start; index < block_end; ++index) {
    output[index] = encode_fp8_e5m2_stochastic(temporary[index] / scale, index,
                                               params.rounding_seed,
                                               params.rounding_stream);
  }
}

struct JacobiParams {
  uint order;
  uint padded_order;
  uint pair_count;
  uint round;
};

struct WorkingFloat {
  float high;
  float low;
};

struct WorkingNorm {
  float residual_squared;
  float matrix_squared;
};

// Metal has no binary64 type. This pair stores a leading float and its
// correction. Error-free sums and FMA products give about 48 useful bits.
inline WorkingFloat working_normalize(float high, float low) {
  const float sum = high + low;
  return WorkingFloat{sum, low - (sum - high)};
}

inline WorkingFloat working_add(WorkingFloat left, WorkingFloat right) {
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

template <typename Value>
inline Value matrix_product_entry(const device Value *left,
                                  const device Value *right, uint row,
                                  uint column, uint order);

template <>
inline WorkingFloat
matrix_product_entry<WorkingFloat>(const device WorkingFloat *left,
                                   const device WorkingFloat *right, uint row,
                                   uint column, uint order) {
  WorkingFloat sum{0.0f, 0.0f};
  for (uint inner = 0; inner < order; ++inner) {
    sum = working_add(sum, working_multiply(left[row + inner * order],
                                            right[inner + column * order]));
  }
  return sum;
}

kernel void jacobi_identity_f32(device float *vectors [[buffer(0)]],
                                constant SquareParams &params [[buffer(1)]],
                                uint2 position [[thread_position_in_grid]]) {
  if (position.x >= params.order || position.y >= params.order) {
    return;
  }
  vectors[position.y + position.x * params.order] =
      position.x == position.y ? 1.0f : 0.0f;
}

// One round contains disjoint index pairs. The next two kernels apply all
// rotations without write conflicts. A complete sweep uses order-1 rounds.
kernel void jacobi_rotations_f32(const device float *matrix [[buffer(0)]],
                                 const device uint *pair_members [[buffer(1)]],
                                 device float2 *rotations [[buffer(2)]],
                                 constant JacobiParams &params [[buffer(3)]],
                                 uint pair [[thread_position_in_grid]]) {
  if (pair >= params.pair_count) {
    return;
  }
  const uint map_offset = params.round * params.padded_order;
  const uint first = 2u * pair;
  const uint second = first + 1u;
  const uint p = pair_members[map_offset + first];
  const uint q = pair_members[map_offset + second];
  float cosine = 1.0f;
  float sine = 0.0f;
  if (p < params.order && q < params.order) {
    const float app = matrix[p + p * params.order];
    const float aqq = matrix[q + q * params.order];
    const float apq =
        0.5f * (matrix[p + q * params.order] + matrix[q + p * params.order]);
    if (apq != 0.0f) {
      const float tau = (aqq - app) / (2.0f * apq);
      const float tangent =
          copysign(1.0f / (fabs(tau) + sqrt(1.0f + tau * tau)), tau);
      cosine = rsqrt(1.0f + tangent * tangent);
      sine = tangent * cosine;
    }
  }
  rotations[first] = float2(cosine, -sine);
  rotations[second] = float2(cosine, sine);
}

kernel void
jacobi_similarity_vectors_f32(const device float *matrix_source [[buffer(0)]],
                              device float *matrix_destination [[buffer(1)]],
                              const device float *vectors_source [[buffer(2)]],
                              device float *vectors_destination [[buffer(3)]],
                              const device uint *partners [[buffer(4)]],
                              const device uint *pair_indices [[buffer(5)]],
                              const device float2 *pair_rotations [[buffer(6)]],
                              constant JacobiParams &params [[buffer(7)]],
                              uint3 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint map_offset = params.round * params.padded_order;
  const uint index = row + column * params.order;
  if (position.z == 1u) {
    const uint partner = partners[map_offset + column];
    const float2 rotation =
        pair_rotations[2u * pair_indices[map_offset + column] +
                       uint(partner < column)];
    vectors_destination[index] =
        partner < params.order
            ? rotation.x * vectors_source[index] +
                  rotation.y * vectors_source[row + partner * params.order]
            : vectors_source[index];
    return;
  }
  const uint row_partner = partners[map_offset + row];
  const uint column_partner = partners[map_offset + column];
  const float2 row_rotation =
      pair_rotations[2u * pair_indices[map_offset + row] +
                     uint(row_partner < row)];
  const float2 column_rotation =
      pair_rotations[2u * pair_indices[map_offset + column] +
                     uint(column_partner < column)];
  const uint safe_row_partner = row_partner < params.order ? row_partner : row;
  const uint safe_column_partner =
      column_partner < params.order ? column_partner : column;
  matrix_destination[index] =
      row_rotation.x * column_rotation.x * matrix_source[index] +
      row_rotation.y * column_rotation.x *
          matrix_source[safe_row_partner + column * params.order] +
      row_rotation.x * column_rotation.y *
          matrix_source[row + safe_column_partner * params.order] +
      row_rotation.y * column_rotation.y *
          matrix_source[safe_row_partner + safe_column_partner * params.order];
}

kernel void transpose_square_f32(const device float *source [[buffer(0)]],
                                 device float *destination [[buffer(1)]],
                                 constant SquareParams &params [[buffer(2)]],
                                 uint2 position [[thread_position_in_grid]]) {
  if (position.x < params.order && position.y < params.order) {
    destination[position.y + position.x * params.order] =
        source[position.x + position.y * params.order];
  }
}

kernel void sylvester_inverse_from_spectrum_f32(
    const device float *root_eigenvalues [[buffer(0)]],
    device float *inverse [[buffer(1)]],
    constant SquareParams &params [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row < params.order && column < params.order) {
    inverse[row + column * params.order] =
        1.0f / (root_eigenvalues[row] + root_eigenvalues[column]);
  }
}

kernel void
jacobi_root_eigenvalues_f32(const device float *matrix [[buffer(0)]],
                            device float *root_eigenvalues [[buffer(1)]],
                            constant SquareParams &params [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  if (index < params.order) {
    root_eigenvalues[index] = sqrt(matrix[index + index * params.order]);
  }
}
struct DecodeBf16 {
  static float load(const device uchar *values, const device uchar *,
                    uint index) {
    return float(reinterpret_cast<const device bfloat *>(values)[index]);
  }
};

struct DecodeFp8E4M3Fn {
  static float load(const device uchar *values, const device uchar *,
                    uint index) {
    return decode_fp8_e4m3fn(values[index]);
  }
};

struct DecodeFp8E5M2 {
  static float load(const device uchar *values, const device uchar *,
                    uint index) {
    return decode_fp8_e5m2(values[index]);
  }
};

struct DecodeMxfp8E4M3Fn {
  static float load(const device uchar *values, const device uchar *scales,
                    uint index) {
    return decode_mxfp8_e4m3fn(values, scales, index);
  }
};

struct DecodeMxfp8E5M2 {
  static float load(const device uchar *values, const device uchar *scales,
                    uint index) {
    return decode_mxfp8_e5m2(values, scales, index);
  }
};

struct DecodeFp4E2M1 {
  static float load(const device uchar *values, const device uchar *,
                    uint index) {
    return decode_fp4_e2m1(unpack_fp4_e2m1(values, index));
  }
};

struct DecodeMxfp4E2M1 {
  static float load(const device uchar *values, const device uchar *scales,
                    uint index) {
    return decode_mxfp4_e2m1(values, scales, index);
  }
};
template <typename Decoder>
inline void
matmul_low_left_f32(const device uchar *left, const device uchar *left_scales,
                    const device float *right, device float *output,
                    constant LowFactorMatmulParams &params, uint3 tile,
                    uint3 local, threadgroup float *left_tile,
                    threadgroup float *right_tile) {
  const uint row = tile.x * kFastTile + local.x;
  const uint column = tile.y * kFastTile + local.y;
  const uint linear = local.y * kFastTile + local.x;
  const uint left_row = linear & (kFastTile - 1u);
  const uint left_inner = linear / kFastTile;
  const uint right_inner = linear & (kFastTile - 1u);
  const uint right_column = linear / kFastTile;
  float sum = 0.0f;

  for (uint base = 0; base < params.inner; base += kFastTile) {
    const uint load_left_inner = base + left_inner;
    const uint load_right_inner = base + right_inner;
    const uint global_left_row = tile.x * kFastTile + left_row;
    const uint global_right_column = tile.y * kFastTile + right_column;
    uint left_index = 0u;
    if (params.transpose_low == 0u) {
      left_index = global_left_row + load_left_inner * params.rows;
    } else {
      left_index = load_left_inner + global_left_row * params.inner;
    }
    left_tile[left_inner * kFastTile + left_row] =
        global_left_row < params.rows && load_left_inner < params.inner
            ? Decoder::load(left, left_scales, left_index)
            : 0.0f;
    right_tile[right_column * kFastTile + right_inner] =
        load_right_inner < params.inner && global_right_column < params.columns
            ? right[load_right_inner + global_right_column * params.inner]
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < kFastTile; ++inner) {
      sum = fma(left_tile[inner * kFastTile + local.x],
                right_tile[local.y * kFastTile + inner], sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && column < params.columns) {
    output[row + column * params.rows] = sum;
  }
}

template <typename Decoder>
inline void matmul_f32_low_right(
    const device float *left, const device uchar *right,
    const device uchar *right_scales, const device float *inverse,
    device float *output, constant LowFactorMatmulParams &params, uint3 tile,
    uint3 local, threadgroup float *left_tile, threadgroup float *right_tile) {
  const uint row = tile.x * kFastTile + local.x;
  const uint column = tile.y * kFastTile + local.y;
  const uint linear = local.y * kFastTile + local.x;
  const uint left_row = linear & (kFastTile - 1u);
  const uint left_inner = linear / kFastTile;
  const uint right_inner = linear & (kFastTile - 1u);
  const uint right_column = linear / kFastTile;
  float sum = 0.0f;

  for (uint base = 0; base < params.inner; base += kFastTile) {
    const uint load_left_inner = base + left_inner;
    const uint load_right_inner = base + right_inner;
    const uint global_left_row = tile.x * kFastTile + left_row;
    const uint global_right_column = tile.y * kFastTile + right_column;
    left_tile[left_inner * kFastTile + left_row] =
        global_left_row < params.rows && load_left_inner < params.inner
            ? left[global_left_row + load_left_inner * params.rows]
            : 0.0f;
    uint right_index = 0u;
    if (params.transpose_low == 0u) {
      right_index = load_right_inner + global_right_column * params.inner;
    } else {
      right_index = global_right_column + load_right_inner * params.columns;
    }
    right_tile[right_column * kFastTile + right_inner] =
        load_right_inner < params.inner && global_right_column < params.columns
            ? Decoder::load(right, right_scales, right_index)
            : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < kFastTile; ++inner) {
      sum = fma(left_tile[inner * kFastTile + local.x],
                right_tile[local.y * kFastTile + inner], sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.rows && column < params.columns) {
    const uint index = row + column * params.rows;
    output[index] = params.apply_inverse == 0u ? sum : sum * inverse[index];
  }
}

template <typename Decoder>
inline void
scale_low_columns_f32(const device uchar *vectors,
                      const device uchar *vector_scales,
                      const device float *roots, device float *output,
                      constant SquareParams &params, uint2 position) {
  const uint row = position.y;
  const uint column = position.x;
  if (row < params.order && column < params.order) {
    const uint index = row + column * params.order;
    output[index] =
        Decoder::load(vectors, vector_scales, index) * roots[column];
  }
}

#define DEFINE_LOW_FACTOR_KERNELS(LEFT_NAME, RIGHT_NAME, SCALE_NAME, DECODER)  \
  kernel void LEFT_NAME(const device uchar *left [[buffer(0)]],                \
                        const device uchar *left_scales [[buffer(1)]],         \
                        const device float *right [[buffer(2)]],               \
                        device float *output [[buffer(3)]],                    \
                        constant LowFactorMatmulParams &params [[buffer(4)]],  \
                        uint3 tile [[threadgroup_position_in_grid]],           \
                        uint3 local [[thread_position_in_threadgroup]]) {      \
    threadgroup float left_tile[kFastTile * kFastTile];                        \
    threadgroup float right_tile[kFastTile * kFastTile];                       \
    matmul_low_left_f32<DECODER>(left, left_scales, right, output, params,     \
                                 tile, local, left_tile, right_tile);          \
  }                                                                            \
  kernel void RIGHT_NAME(const device float *left [[buffer(0)]],               \
                         const device uchar *right [[buffer(1)]],              \
                         const device uchar *right_scales [[buffer(2)]],       \
                         const device float *inverse [[buffer(3)]],            \
                         device float *output [[buffer(4)]],                   \
                         constant LowFactorMatmulParams &params [[buffer(5)]], \
                         uint3 tile [[threadgroup_position_in_grid]],          \
                         uint3 local [[thread_position_in_threadgroup]]) {     \
    threadgroup float left_tile[kFastTile * kFastTile];                        \
    threadgroup float right_tile[kFastTile * kFastTile];                       \
    matmul_f32_low_right<DECODER>(left, right, right_scales, inverse, output,  \
                                  params, tile, local, left_tile, right_tile); \
  }                                                                            \
  kernel void SCALE_NAME(const device uchar *vectors [[buffer(0)]],            \
                         const device uchar *vector_scales [[buffer(1)]],      \
                         const device float *roots [[buffer(2)]],              \
                         device float *output [[buffer(3)]],                   \
                         constant SquareParams &params [[buffer(4)]],          \
                         uint2 position [[thread_position_in_grid]]) {         \
    scale_low_columns_f32<DECODER>(vectors, vector_scales, roots, output,      \
                                   params, position);                          \
  }

DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_bf16, matmul_f32_low_right_bf16,
                          scale_low_columns_f32_bf16, DecodeBf16)
DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_fp8_e4m3fn,
                          matmul_f32_low_right_fp8_e4m3fn,
                          scale_low_columns_f32_fp8_e4m3fn, DecodeFp8E4M3Fn)
DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_fp8_e5m2,
                          matmul_f32_low_right_fp8_e5m2,
                          scale_low_columns_f32_fp8_e5m2, DecodeFp8E5M2)
DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_mxfp8_e4m3fn,
                          matmul_f32_low_right_mxfp8_e4m3fn,
                          scale_low_columns_f32_mxfp8_e4m3fn, DecodeMxfp8E4M3Fn)
DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_mxfp8_e5m2,
                          matmul_f32_low_right_mxfp8_e5m2,
                          scale_low_columns_f32_mxfp8_e5m2, DecodeMxfp8E5M2)
DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_fp4_e2m1,
                          matmul_f32_low_right_fp4_e2m1,
                          scale_low_columns_f32_fp4_e2m1, DecodeFp4E2M1)
DEFINE_LOW_FACTOR_KERNELS(matmul_low_left_f32_mxfp4_e2m1,
                          matmul_f32_low_right_mxfp4_e2m1,
                          scale_low_columns_f32_mxfp4_e2m1, DecodeMxfp4E2M1)

#undef DEFINE_LOW_FACTOR_KERNELS

template <typename Decoder>
inline void decode_to_f32(const device uchar *values,
                          const device uchar *scales, device float *output,
                          constant QuantizeParams &params, uint index) {
  if (index < params.elements) {
    output[index] = Decoder::load(values, scales, index);
  }
}

#define DEFINE_DECODE_KERNEL(NAME, DECODER)                                    \
  kernel void NAME(const device uchar *values [[buffer(0)]],                   \
                   const device uchar *scales [[buffer(1)]],                   \
                   device float *output [[buffer(2)]],                         \
                   constant QuantizeParams &params [[buffer(3)]],              \
                   uint index [[thread_position_in_grid]]) {                   \
    decode_to_f32<DECODER>(values, scales, output, params, index);             \
  }

DEFINE_DECODE_KERNEL(decode_bf16_to_f32, DecodeBf16)
DEFINE_DECODE_KERNEL(decode_fp8_e4m3fn_to_f32, DecodeFp8E4M3Fn)
DEFINE_DECODE_KERNEL(decode_fp8_e5m2_to_f32, DecodeFp8E5M2)
DEFINE_DECODE_KERNEL(decode_mxfp8_e4m3fn_to_f32, DecodeMxfp8E4M3Fn)
DEFINE_DECODE_KERNEL(decode_mxfp8_e5m2_to_f32, DecodeMxfp8E5M2)
DEFINE_DECODE_KERNEL(decode_fp4_e2m1_to_f32, DecodeFp4E2M1)
DEFINE_DECODE_KERNEL(decode_mxfp4_e2m1_to_f32, DecodeMxfp4E2M1)

#undef DEFINE_DECODE_KERNEL

kernel void log_spectrum_metadata(const device float *spectrum [[buffer(0)]],
                                  device float2 *metadata [[buffer(1)]],
                                  constant LogSpectrumParams &params
                                  [[buffer(2)]],
                                  uint block [[thread_position_in_grid]]) {
  const uint begin = block * params.block_elements;
  if (begin >= params.elements) {
    return;
  }
  const uint end = min(begin + params.block_elements, params.elements);
  float minimum = INFINITY;
  float maximum = -INFINITY;
  for (uint index = begin; index < end; ++index) {
    const float value = log2(spectrum[index]);
    minimum = min(minimum, value);
    maximum = max(maximum, value);
  }
  metadata[block] = float2(minimum, maximum - minimum);
}

inline float normalized_log_spectrum(const device float *spectrum,
                                     const device float2 *metadata,
                                     constant LogSpectrumParams &params,
                                     uint index, float maximum) {
  const float2 block = metadata[index / params.block_elements];
  return block.y == 0.0f
             ? 0.0f
             : maximum * (log2(spectrum[index]) - block.x) / block.y;
}

kernel void
quantize_fp8_e4m3fn_log_s(const device float *spectrum [[buffer(0)]],
                          device uchar *output [[buffer(1)]],
                          const device float2 *metadata [[buffer(2)]],
                          constant LogSpectrumParams &params [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = encode_fp8_e4m3fn(
        normalized_log_spectrum(spectrum, metadata, params, index, 448.0f));
  }
}

kernel void quantize_fp8_e4m3fn_log_s_stochastic(
    const device float *spectrum [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    const device float2 *metadata [[buffer(2)]],
    constant LogSpectrumParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = encode_fp8_e4m3fn_stochastic(
        normalized_log_spectrum(spectrum, metadata, params, index, 448.0f),
        index, params.rounding_seed, params.rounding_stream);
  }
}

kernel void quantize_fp4_e2m1_log_s(const device float *spectrum [[buffer(0)]],
                                    device uchar *output [[buffer(1)]],
                                    const device float2 *metadata [[buffer(2)]],
                                    constant LogSpectrumParams &params
                                    [[buffer(3)]],
                                    uint byte_index
                                    [[thread_position_in_grid]]) {
  const uint first = 2u * byte_index;
  if (first >= params.elements) {
    return;
  }
  const uchar low = encode_fp4_e2m1(
      normalized_log_spectrum(spectrum, metadata, params, first, 6.0f));
  uchar high = 0;
  if (first + 1u < params.elements) {
    high = encode_fp4_e2m1(
        normalized_log_spectrum(spectrum, metadata, params, first + 1u, 6.0f));
  }
  output[byte_index] = uchar(uint(low) | (uint(high) << 4u));
}

kernel void quantize_fp4_e2m1_log_s_stochastic(
    const device float *spectrum [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    const device float2 *metadata [[buffer(2)]],
    constant LogSpectrumParams &params [[buffer(3)]],
    uint byte_index [[thread_position_in_grid]]) {
  const uint first = 2u * byte_index;
  if (first >= params.elements) {
    return;
  }
  const uchar low = encode_fp4_e2m1_stochastic(
      normalized_log_spectrum(spectrum, metadata, params, first, 6.0f), first,
      params.rounding_seed, params.rounding_stream);
  uchar high = 0;
  if (first + 1u < params.elements) {
    high = encode_fp4_e2m1_stochastic(
        normalized_log_spectrum(spectrum, metadata, params, first + 1u, 6.0f),
        first + 1u, params.rounding_seed, params.rounding_stream);
  }
  output[byte_index] = uchar(uint(low) | (uint(high) << 4u));
}

kernel void
decode_fp8_e4m3fn_log_s_to_f32(const device uchar *values [[buffer(0)]],
                               const device float2 *metadata [[buffer(1)]],
                               device float *output [[buffer(2)]],
                               constant LogSpectrumParams &params [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    const float2 block = metadata[index / params.block_elements];
    const float normalized = decode_fp8_e4m3fn(values[index]);
    output[index] = exp2(block.x + block.y * normalized / 448.0f);
  }
}

kernel void
decode_fp4_e2m1_log_s_to_f32(const device uchar *values [[buffer(0)]],
                             const device float2 *metadata [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant LogSpectrumParams &params [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    const float2 block = metadata[index / params.block_elements];
    const float normalized = decode_fp4_e2m1(unpack_fp4_e2m1(values, index));
    output[index] = exp2(block.x + block.y * normalized / 6.0f);
  }
}

struct FloatEdgeProbe {
  uint loaded_bits;
  uint bfloat_bits;
  uint log2_bits;
  uint exp2_log2_bits;
};

kernel void probe_f32_edges(const device float *source [[buffer(0)]],
                            device FloatEdgeProbe *output [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
  const float value = source[index];
  const float logarithm = log2(value);
  output[index] = FloatEdgeProbe{
      as_type<uint>(value),
      as_type<uint>(float(bfloat(value))),
      as_type<uint>(logarithm),
      as_type<uint>(exp2(logarithm)),
  };
}

kernel void quantize_fp8_e4m3fn(const device float *source [[buffer(0)]],
                                device uchar *output [[buffer(1)]],
                                constant QuantizeParams &params [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = encode_fp8_e4m3fn(source[index]);
  }
}

kernel void quantize_fp8_e5m2(const device float *source [[buffer(0)]],
                              device uchar *output [[buffer(1)]],
                              constant QuantizeParams &params [[buffer(3)]],
                              uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = encode_fp8_e5m2(source[index]);
  }
}

kernel void
quantize_fp8_e4m3fn_stochastic(const device float *source [[buffer(0)]],
                               device uchar *output [[buffer(1)]],
                               constant QuantizeParams &params [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = encode_fp8_e4m3fn_stochastic(
        source[index], index, params.rounding_seed, params.rounding_stream);
  }
}

kernel void
quantize_fp8_e5m2_stochastic(const device float *source [[buffer(0)]],
                             device uchar *output [[buffer(1)]],
                             constant QuantizeParams &params [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = encode_fp8_e5m2_stochastic(
        source[index], index, params.rounding_seed, params.rounding_stream);
  }
}

kernel void quantize_bf16(const device float *source [[buffer(0)]],
                          device bfloat *output [[buffer(1)]],
                          constant QuantizeParams &params [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = bfloat(source[index]);
  }
}

kernel void quantize_bf16_stochastic(const device float *source [[buffer(0)]],
                                     device bfloat *output [[buffer(1)]],
                                     constant QuantizeParams &params
                                     [[buffer(3)]],
                                     uint index [[thread_position_in_grid]]) {
  if (index >= params.elements) {
    return;
  }
  const float value = source[index];
  if (!isfinite(value) || value == 0.0f) {
    output[index] = bfloat(value);
    return;
  }
  const uint bits = as_type<uint>(value);
  const uint discarded = bits & 0xffffu;
  uint rounded = bits & 0xffff0000u;
  if ((stochastic_hash(index, params.rounding_seed, params.rounding_stream) &
       0xffffu) < discarded) {
    rounded += 0x00010000u;
  }
  output[index] = bfloat(as_type<float>(rounded));
}

kernel void
working_from_symmetric_f32(const device float *source [[buffer(0)]],
                           device WorkingFloat *destination [[buffer(1)]],
                           constant SquareParams &params [[buffer(2)]],
                           uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  const float value =
      0.5f * (source[index] + source[column + row * params.order]);
  destination[index] = WorkingFloat{value, 0.0f};
}

kernel void working_residual(const device WorkingFloat *matrix [[buffer(0)]],
                             const device WorkingFloat *root [[buffer(1)]],
                             device WorkingFloat *residual [[buffer(2)]],
                             constant SquareParams &params [[buffer(3)]],
                             uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  const WorkingFloat product =
      matrix_product_entry<WorkingFloat>(root, root, row, column, params.order);
  residual[index] =
      working_add(matrix[index], WorkingFloat{-product.high, -product.low});
}

kernel void working_residual_to_f32(const device WorkingFloat *residual
                                    [[buffer(0)]],
                                    device float *output [[buffer(1)]],
                                    constant QuantizeParams &params
                                    [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    output[index] = residual[index].high + residual[index].low;
  }
}

kernel void
working_add_symmetric_correction(device WorkingFloat *root [[buffer(0)]],
                                 const device float *correction [[buffer(1)]],
                                 constant SquareParams &params [[buffer(2)]],
                                 uint2 position [[thread_position_in_grid]]) {
  const uint row = position.y;
  const uint column = position.x;
  if (row >= params.order || column >= params.order) {
    return;
  }
  const uint index = row + column * params.order;
  const float symmetric_correction =
      0.5f * (correction[index] + correction[column + row * params.order]);
  root[index] =
      working_add(root[index], WorkingFloat{symmetric_correction, 0.0f});
}

kernel void working_norm(const device WorkingFloat *matrix [[buffer(0)]],
                         const device WorkingFloat *residual [[buffer(1)]],
                         device WorkingNorm *norm [[buffer(2)]],
                         constant QuantizeParams &params [[buffer(3)]],
                         uint local_index [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
  threadgroup float residual_sums[256];
  threadgroup float matrix_sums[256];
  float residual_sum = 0.0f;
  float matrix_sum = 0.0f;
  for (uint index = local_index; index < params.elements; index += threads) {
    const float residual_value = residual[index].high + residual[index].low;
    const float matrix_value = matrix[index].high + matrix[index].low;
    residual_sum = fma(residual_value, residual_value, residual_sum);
    matrix_sum = fma(matrix_value, matrix_value, matrix_sum);
  }
  residual_sums[local_index] = residual_sum;
  matrix_sums[local_index] = matrix_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = threads / 2u; stride != 0u; stride /= 2u) {
    if (local_index < stride) {
      residual_sums[local_index] += residual_sums[local_index + stride];
      matrix_sums[local_index] += matrix_sums[local_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0u) {
    norm[0] = WorkingNorm{residual_sums[0], matrix_sums[0]};
  }
}
