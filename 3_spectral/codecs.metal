#include <metal_stdlib>
#if __METAL_VERSION__ >= 401
#include <metal_packed_numeric>
#endif
using namespace metal;

struct CodecParams {
  uint elements;
  uint blocks;
};

struct MatrixCodecParams {
  uint order;
  uint leading_dimension;
  uint scale_stride;
  uint row_major;
};

inline float load_matrix_element(const device float *source, uint row,
                                 uint column,
                                 constant MatrixCodecParams &params) {
  const uint index = params.row_major ? row + column * params.order
                                      : column + row * params.order;
  return source[index];
}

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

kernel void encode_matrix_fp16(const device float *source [[buffer(0)]],
                               device half *encoded [[buffer(1)]],
                               constant MatrixCodecParams &params [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  if (index < params.order * params.leading_dimension) {
    const uint row = index / params.leading_dimension;
    const uint column = index % params.leading_dimension;
    encoded[index] =
        column < params.order
            ? half(load_matrix_element(source, row, column, params))
            : half(0.0f);
  }
}

#if __METAL_VERSION__ >= 401
using fp8_e4x4 = packed_numeric_type<metal_fp8_e4m3_format, 4>;
using fp8_e5x4 = packed_numeric_type<metal_fp8_e5m2_format, 4>;
using fp4_e2x8 = packed_numeric_type<metal_fp4_e2m1_format, 8>;
using packed4_storage = fp8_e4x4::storage_type;

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

inline float4 load_matrix_float4(const device float *source, uint row,
                                 uint column,
                                 constant MatrixCodecParams &params) {
  float4 values;
  for (uint item = 0; item < 4u; ++item) {
    values[item] = load_matrix_element(source, row, column + item, params);
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

kernel void encode_mxfp8_e4m3(const device float *source [[buffer(0)]],
                              device packed4_storage *encoded [[buffer(1)]],
                              device uchar *scales [[buffer(2)]],
                              constant CodecParams &params [[buffer(3)]],
                              uint block [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint begin = block * 32u + lane * 4u;
  const float4 values =
      lane < 8u ? load_float4(source, begin, params.elements) : float4(0.0f);
  const float maximum = simd_max(max(max(fabs(values.x), fabs(values.y)),
                                     max(fabs(values.z), fabs(values.w))));
  const int exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 8, -127, 128);
  if (lane == 0u) {
    scales[block] = uchar(exponent + 127);
  }
  if (lane < 8u && begin < params.elements) {
    encoded[block * 8u + lane] =
        pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values / exp2(float(exponent)))
            .as_storage_type();
  }
}

kernel void decode_mxfp8_e4m3(const device packed4_storage *encoded
                              [[buffer(0)]],
                              const device uchar *scales [[buffer(1)]],
                              device float *decoded [[buffer(2)]],
                              constant CodecParams &params [[buffer(3)]],
                              uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.elements) {
    const float scale = exp2(float(int(scales[begin / 32u]) - 127));
    const float4 values =
        unpack<float>(fp8_e4x4(encoded[vector_index])) * scale;
    for (uint lane = 0; lane < 4u && begin + lane < params.elements; ++lane) {
      decoded[begin + lane] = values[lane];
    }
  }
}

kernel void encode_mxfp4_e2m1(const device float *source [[buffer(0)]],
                              device packed4_storage *encoded [[buffer(1)]],
                              device uchar *scales [[buffer(2)]],
                              constant CodecParams &params [[buffer(3)]],
                              uint block [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const uint begin = block * 32u + lane * 8u;
  vec<float, 8> values(0.0f);
  float local_maximum = 0.0f;
  if (lane < 4u) {
    for (uint item = 0; item < 8u; ++item) {
      if (begin + item < params.elements) {
        values[item] = source[begin + item];
        local_maximum = max(local_maximum, fabs(values[item]));
      }
    }
  }
  const float maximum = simd_max(local_maximum);
  const int exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 2, -127, 128);
  if (lane == 0u) {
    scales[block] = uchar(exponent + 127);
  }
  if (lane < 4u && begin < params.elements) {
    encoded[block * 4u + lane] =
        pack<metal_fp4_e2m1_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values / exp2(float(exponent)))
            .as_storage_type();
  }
}

kernel void decode_mxfp4_e2m1(const device packed4_storage *encoded
                              [[buffer(0)]],
                              const device uchar *scales [[buffer(1)]],
                              device float *decoded [[buffer(2)]],
                              constant CodecParams &params [[buffer(3)]],
                              uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 8u * vector_index;
  if (begin < params.elements) {
    const float scale = exp2(float(int(scales[begin / 32u]) - 127));
    const vec<float, 8> values =
        unpack<float>(fp4_e2x8(encoded[vector_index])) * scale;
    for (uint lane = 0; lane < 8u && begin + lane < params.elements; ++lane) {
      decoded[begin + lane] = values[lane];
    }
  }
}

kernel void
encode_matrix_fp8_e4m3(const device float *source [[buffer(0)]],
                       device packed4_storage *encoded [[buffer(1)]],
                       constant MatrixCodecParams &params [[buffer(2)]],
                       uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.order * params.leading_dimension) {
    const uint row = begin / params.leading_dimension;
    const uint column = begin % params.leading_dimension;
    const float4 values = column < params.order
                              ? load_matrix_float4(source, row, column, params)
                              : float4(0.0f);
    encoded[vector_index] =
        pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values)
            .as_storage_type();
  }
}

kernel void
encode_matrix_fp8_e5m2(const device float *source [[buffer(0)]],
                       device packed4_storage *encoded [[buffer(1)]],
                       constant MatrixCodecParams &params [[buffer(2)]],
                       uint vector_index [[thread_position_in_grid]]) {
  const uint begin = 4u * vector_index;
  if (begin < params.order * params.leading_dimension) {
    const uint row = begin / params.leading_dimension;
    const uint column = begin % params.leading_dimension;
    const float4 values = column < params.order
                              ? load_matrix_float4(source, row, column, params)
                              : float4(0.0f);
    encoded[vector_index] =
        pack<metal_fp8_e5m2_format, rounding_mode::to_nearest_even,
             saturation_mode::none>(values)
            .as_storage_type();
  }
}

kernel void
encode_matrix_mxfp8_e4m3(const device float *source [[buffer(0)]],
                         device packed4_storage *encoded [[buffer(1)]],
                         device uchar *scales [[buffer(2)]],
                         constant MatrixCodecParams &params [[buffer(3)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint blocks_per_row = params.order / 32u;
  if (group >= params.order * blocks_per_row) {
    return;
  }
  const uint row = group / blocks_per_row;
  const uint block = group % blocks_per_row;
  const uint column = block * 32u + lane * 4u;
  const float4 values = lane < 8u
                            ? load_matrix_float4(source, row, column, params)
                            : float4(0.0f);
  const float maximum = simd_max(max(max(fabs(values.x), fabs(values.y)),
                                     max(fabs(values.z), fabs(values.w))));
  const int exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 8, -127, 128);
  if (lane == 0u) {
    scales[row * params.scale_stride + block] = uchar(exponent + 127);
  }
  if (lane < 8u) {
    encoded[row * (params.leading_dimension / 4u) + block * 8u + lane] =
        pack<metal_fp8_e4m3_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values / exp2(float(exponent)))
            .as_storage_type();
  }
}

kernel void
encode_matrix_mxfp4_e2m1(const device float *source [[buffer(0)]],
                         device packed4_storage *encoded [[buffer(1)]],
                         device uchar *scales [[buffer(2)]],
                         constant MatrixCodecParams &params [[buffer(3)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint blocks_per_row = params.order / 32u;
  if (group >= params.order * blocks_per_row) {
    return;
  }
  const uint row = group / blocks_per_row;
  const uint block = group % blocks_per_row;
  const uint column = block * 32u + lane * 8u;
  vec<float, 8> values(0.0f);
  float local_maximum = 0.0f;
  if (lane < 4u) {
    for (uint item = 0; item < 8u; ++item) {
      values[item] = load_matrix_element(source, row, column + item, params);
      local_maximum = max(local_maximum, fabs(values[item]));
    }
  }
  const float maximum = simd_max(local_maximum);
  const int exponent =
      maximum == 0.0f ? 0 : clamp(int(floor(log2(maximum))) - 2, -127, 128);
  if (lane == 0u) {
    scales[row * params.scale_stride + block] = uchar(exponent + 127);
  }
  if (lane < 4u) {
    encoded[row * (params.leading_dimension / 8u) + block * 4u + lane] =
        pack<metal_fp4_e2m1_format, rounding_mode::to_nearest_even,
             saturation_mode::saturate>(values / exp2(float(exponent)))
            .as_storage_type();
  }
}
#endif
