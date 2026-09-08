#include <metal_packed_numeric>
#include <metal_stdlib>
using namespace metal;

struct CodecParams {
  uint elements;
  uint blocks;
};

struct Metrics {
  float source_squared;
  float error_squared;
  float reference_dot;
  float decoded_dot;
};

kernel void encode_fp32(const device float *source [[buffer(0)]],
                        device float *encoded [[buffer(1)]],
                        constant CodecParams &params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    encoded[index] = source[index];
  }
}

kernel void decode_fp32(const device float *encoded [[buffer(0)]],
                        device float *decoded [[buffer(1)]],
                        constant CodecParams &params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    decoded[index] = encoded[index];
  }
}

kernel void encode_fp16(const device float *source [[buffer(0)]],
                        device half *encoded [[buffer(1)]],
                        constant CodecParams &params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    encoded[index] = half(source[index]);
  }
}

kernel void decode_fp16(const device half *encoded [[buffer(0)]],
                        device float *decoded [[buffer(1)]],
                        constant CodecParams &params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    decoded[index] = float(encoded[index]);
  }
}

kernel void encode_bf16(const device float *source [[buffer(0)]],
                        device bfloat *encoded [[buffer(1)]],
                        constant CodecParams &params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    encoded[index] = bfloat(source[index]);
  }
}

kernel void decode_bf16(const device bfloat *encoded [[buffer(0)]],
                        device float *decoded [[buffer(1)]],
                        constant CodecParams &params [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index < params.elements) {
    decoded[index] = float(encoded[index]);
  }
}

#if __METAL_VERSION__ >= 401
using fp8_e4x4 = packed_numeric_type<metal_fp8_e4m3_format, 4>;
using fp8_e5x4 = packed_numeric_type<metal_fp8_e5m2_format, 4>;
using fp4_e2x8 = packed_numeric_type<metal_fp4_e2m1_format, 8>;
using packed4_storage = fp8_e4x4::storage_type;
inline float max_abs(float4 values) {
  const float4 magnitude = fabs(values);
  return max(max(magnitude.x, magnitude.y), max(magnitude.z, magnitude.w));
}

inline float4 load_float4(const device float *source, uint begin,
                          uint elements) {
  float4 values(0.0f);
  for (uint lane = 0; lane < 4u; ++lane) {
    if (begin + lane < elements) {
      values[lane] = source[begin + lane];
    }
  }
  return values;
}

kernel void encode_fp8_e4m3(const device float *source [[buffer(0)]],
                            device packed4_storage *encoded [[buffer(1)]],
                            constant CodecParams &params [[buffer(2)]],
                            uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.elements) {
    encoded[vector_index] =
        pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(
            load_float4(source, begin, params.elements))
            .as_storage_type();
  }
}

kernel void decode_fp8_e4m3(const device packed4_storage *encoded [[buffer(0)]],
                            device float *decoded [[buffer(1)]],
                            constant CodecParams &params [[buffer(2)]],
                            uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.elements) {
    const float4 values = unpack<float>(fp8_e4x4(encoded[vector_index]));
    for (uint lane = 0; lane < 4u && begin + lane < params.elements; ++lane) {
      decoded[begin + lane] = values[lane];
    }
  }
}

kernel void encode_fp8_e5m2(const device float *source [[buffer(0)]],
                            device packed4_storage *encoded [[buffer(1)]],
                            constant CodecParams &params [[buffer(2)]],
                            uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.elements) {
    encoded[vector_index] =
        pack<metal_fp8_e5m2_format, rounding_mode::to_nearest_even,
             saturation_mode::none>(load_float4(source, begin, params.elements))
            .as_storage_type();
  }
}

kernel void decode_fp8_e5m2(const device packed4_storage *encoded [[buffer(0)]],
                            device float *decoded [[buffer(1)]],
                            constant CodecParams &params [[buffer(2)]],
                            uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.elements) {
    const float4 values = unpack<float>(fp8_e5x4(encoded[vector_index]));
    for (uint lane = 0; lane < 4u && begin + lane < params.elements; ++lane) {
      decoded[begin + lane] = values[lane];
    }
  }
}

kernel void roundtrip_mxfp8_e4m3(const device float4 *source [[buffer(0)]],
                                 device volatile packed4_storage *encoded
                                 [[buffer(1)]],
                                 device volatile uchar *scales [[buffer(2)]],
                                 device float4 *decoded [[buffer(3)]],
                                 uint block [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint vector_index = block * 8u + lane;
  const float4 values = lane < 8u ? source[vector_index] : float4(0.0f);
  const float maximum = simd_max(max_abs(values));

  float encode_scale = 1.0f;
  if (lane == 0u) {
    const int exponent =
        maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 8, -127, 128);
    scales[block] = uchar(exponent + 127);
    encode_scale = exp2(float(exponent));
  }
  encode_scale = simd_broadcast_first(encode_scale);
  if (lane < 8u) {
    encoded[vector_index] =
        pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values / encode_scale)
            .as_storage_type();
  }

  // force through device mem
  simdgroup_barrier(mem_flags::mem_device);

  float decode_scale = 1.0f;
  if (lane == 0u) {
    decode_scale = exp2(float(int(scales[block]) - 127));
  }
  decode_scale = simd_broadcast_first(decode_scale);
  if (lane < 8u) {
    const packed4_storage stored = encoded[vector_index];
    decoded[vector_index] = unpack<float>(fp8_e4x4(stored)) * decode_scale;
  }
}

kernel void roundtrip_mxfp4_e2m1(const device float4 *source [[buffer(0)]],
                                 device volatile packed4_storage *encoded
                                 [[buffer(1)]],
                                 device volatile uchar *scales [[buffer(2)]],
                                 device float4 *decoded [[buffer(3)]],
                                 uint block [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint first_vector = block * 8u + lane * 2u;
  const float4 low = lane < 4u ? source[first_vector] : float4(0.0f);
  const float4 high = lane < 4u ? source[first_vector + 1u] : float4(0.0f);
  const float maximum = simd_max(max(max_abs(low), max_abs(high)));

  float encode_scale = 1.0f;
  if (lane == 0u) {
    const int exponent =
        maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 2, -127, 128);
    scales[block] = uchar(exponent + 127);
    encode_scale = exp2(float(exponent));
  }
  encode_scale = simd_broadcast_first(encode_scale);
  const uint packed_index = block * 4u + lane;
  if (lane < 4u) {
    const vec<float, 8> values(low.x, low.y, low.z, low.w, high.x, high.y,
                               high.z, high.w);
    encoded[packed_index] =
        pack<metal_fp4_e2m1_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values / encode_scale)
            .as_storage_type();
  }
  simdgroup_barrier(mem_flags::mem_device);

  float decode_scale = 1.0f;
  if (lane == 0u) {
    decode_scale = exp2(float(int(scales[block]) - 127));
  }
  decode_scale = simd_broadcast_first(decode_scale);
  if (lane < 4u) {
    const packed4_storage stored = encoded[packed_index];
    const vec<float, 8> values = unpack<float>(fp4_e2x8(stored)) * decode_scale;
    decoded[first_vector] = float4(values[0], values[1], values[2], values[3]);
    decoded[first_vector + 1u] =
        float4(values[4], values[5], values[6], values[7]);
  }
}
#endif

kernel void measure_error(const device float *source [[buffer(0)]],
                          const device float *multiplier [[buffer(1)]],
                          const device float *decoded [[buffer(2)]],
                          device Metrics *metrics [[buffer(3)]],
                          constant CodecParams &params [[buffer(4)]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint width [[threads_per_threadgroup]]) {
  threadgroup float source_sum[256];
  threadgroup float error_sum[256];
  threadgroup float reference_sum[256];
  threadgroup float decoded_sum[256];
  float source_squared = 0.0f;
  float error_squared = 0.0f;
  float reference_dot = 0.0f;
  float decoded_dot = 0.0f;
  for (uint index = lane; index < params.elements; index += width) {
    const float error = decoded[index] - source[index];
    source_squared = fma(source[index], source[index], source_squared);
    error_squared = fma(error, error, error_squared);
    reference_dot = fma(source[index], multiplier[index], reference_dot);
    decoded_dot = fma(decoded[index], multiplier[index], decoded_dot);
  }
  source_sum[lane] = source_squared;
  error_sum[lane] = error_squared;
  reference_sum[lane] = reference_dot;
  decoded_sum[lane] = decoded_dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = width / 2u; stride != 0u; stride /= 2u) {
    if (lane < stride) {
      source_sum[lane] += source_sum[lane + stride];
      error_sum[lane] += error_sum[lane + stride];
      reference_sum[lane] += reference_sum[lane + stride];
      decoded_sum[lane] += decoded_sum[lane + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    metrics[0] =
        Metrics{source_sum[0], error_sum[0], reference_sum[0], decoded_sum[0]};
  }
}
