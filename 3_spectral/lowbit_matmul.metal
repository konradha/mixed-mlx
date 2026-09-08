#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

#if !defined(__METAL_VERSION__) || __METAL_VERSION__ < 410
#error "lowbit_matmul.metal requires Metal language 4.1"
#endif

namespace {

constant constexpr auto kDenseMatmul32x32 =
    matmul2d_descriptor(32, 32, static_cast<int>(dynamic_extent), false, false,
                        false, matmul2d_descriptor::mode::multiply);
constant constexpr auto kDenseMatmul64x32 =
    matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, false,
                        false, matmul2d_descriptor::mode::multiply);
constant constexpr auto kDenseMatmul64x64 =
    matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, false,
                        false, matmul2d_descriptor::mode::multiply);
constant constexpr auto kScaledMatmul32x32 =
    matmul2d_descriptor(32, 32, static_cast<int>(dynamic_extent), false, true,
                        false, matmul2d_descriptor::mode::multiply);
constant constexpr auto kScaledMatmul64x32 =
    matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true,
                        false, matmul2d_descriptor::mode::multiply);
constant constexpr auto kScaledMatmul64x64 =
    matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true,
                        false, matmul2d_descriptor::mode::multiply);

using mx_scale =
    tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;

template <typename ExternalLeft, typename ExternalRight>
inline void run_dense_32x32(ExternalLeft left, ExternalRight right,
                            tensor<device float, dextents<int32_t, 2>> output,
                            uint2 position) {
  matmul2d<kDenseMatmul32x32, execution_simdgroups<4>> operation;
  auto mpp_left = right.slice(0, position.y * 32);
  auto mpp_right = left.slice(position.x * 32, 0);
  auto destination = output.slice(position.x * 32, position.y * 32);
  operation.run(mpp_left, mpp_right, destination);
}

template <typename ExternalLeft, typename ExternalRight>
inline void run_dense_64x32(ExternalLeft left, ExternalRight right,
                            tensor<device float, dextents<int32_t, 2>> output,
                            uint2 position) {
  matmul2d<kDenseMatmul64x32, execution_simdgroups<4>> operation;

  // A column-major host matrix appears transposed through a tensor whose
  // innermost dimension has stride one. Computing B^T*A^T therefore produces
  // C^T in the output buffer, which is the column-major representation of A*B.
  auto mpp_left = right.slice(0, position.y * 64);
  auto mpp_right = left.slice(position.x * 32, 0);
  auto destination = output.slice(position.x * 32, position.y * 64);
  operation.run(mpp_left, mpp_right, destination);
}

template <typename ExternalLeft, typename ExternalRight>
inline void run_dense_64x64(ExternalLeft left, ExternalRight right,
                            tensor<device float, dextents<int32_t, 2>> output,
                            uint2 position) {
  matmul2d<kDenseMatmul64x64, execution_simdgroups<4>> operation;
  auto mpp_left = right.slice(0, position.y * 64);
  auto mpp_right = left.slice(position.x * 64, 0);
  auto destination = output.slice(position.x * 64, position.y * 64);
  operation.run(mpp_left, mpp_right, destination);
}

template <typename T>
inline void run_scaled_32x32(T left, T right,
                             tensor<device float, dextents<int32_t, 2>> output,
                             uint2 position) {
  matmul2d<kScaledMatmul32x32, execution_simdgroups<4>> operation;
  auto mpp_left = right.slice(0, position.y * 32);
  auto mpp_right = left.slice(0, position.x * 32);
  auto destination = output.slice(position.x * 32, position.y * 32);
  operation.run(mpp_left, mpp_right, destination);
}

template <typename T>
inline void run_scaled_64x32(T left, T right,
                             tensor<device float, dextents<int32_t, 2>> output,
                             uint2 position) {
  matmul2d<kScaledMatmul64x32, execution_simdgroups<4>> operation;

  // MPP requires a scaled right operand to use transpose_right. The external
  // left matrix is consequently supplied in transposed-column-major storage;
  // its tensor view is A, and transpose_right makes the MPP operand A^T.
  // The external right matrix remains column-major and its tensor view is B^T.
  auto mpp_left = right.slice(0, position.y * 64);
  auto mpp_right = left.slice(0, position.x * 32);
  auto destination = output.slice(position.x * 32, position.y * 64);
  operation.run(mpp_left, mpp_right, destination);
}

template <typename T>
inline void run_scaled_64x64(T left, T right,
                             tensor<device float, dextents<int32_t, 2>> output,
                             uint2 position) {
  matmul2d<kScaledMatmul64x64, execution_simdgroups<4>> operation;
  auto mpp_left = right.slice(0, position.y * 64);
  auto mpp_right = left.slice(0, position.x * 64);
  auto destination = output.slice(position.x * 64, position.y * 64);
  operation.run(mpp_left, mpp_right, destination);
}

} // namespace

using fp16_tensor = tensor<device half, dextents<int32_t, 2>>;
using fp32_tensor = tensor<device float, dextents<int32_t, 2>>;

#define DEFINE_DENSE_KERNEL(NAME, LEFT, RIGHT, RUN)                            \
  kernel void NAME(LEFT left, RIGHT right, fp32_tensor output,                 \
                   uint2 position [[threadgroup_position_in_grid]]) {          \
    RUN(left, right, output, position);                                        \
  }

#define DEFINE_SCALED_KERNEL(NAME, INPUT, RUN)                                 \
  kernel void NAME(INPUT left, INPUT right, fp32_tensor output,                \
                   uint2 position [[threadgroup_position_in_grid]]) {          \
    RUN(left, right, output, position);                                        \
  }

DEFINE_DENSE_KERNEL(lowbit_matmul_fp16_32x32, fp16_tensor, fp16_tensor,
                    run_dense_32x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp16_64x32, fp16_tensor, fp16_tensor,
                    run_dense_64x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp16_64x64, fp16_tensor, fp16_tensor,
                    run_dense_64x64)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp16_fp32_32x32, fp16_tensor, fp32_tensor,
                    run_dense_32x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp16_fp32_64x32, fp16_tensor, fp32_tensor,
                    run_dense_64x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp16_fp32_64x64, fp16_tensor, fp32_tensor,
                    run_dense_64x64)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp32_fp16_32x32, fp32_tensor, fp16_tensor,
                    run_dense_32x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp32_fp16_64x32, fp32_tensor, fp16_tensor,
                    run_dense_64x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_fp32_fp16_64x64, fp32_tensor, fp16_tensor,
                    run_dense_64x64)

#if __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__
using e4m3_tensor = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>>;
DEFINE_DENSE_KERNEL(lowbit_matmul_e4m3_32x32, e4m3_tensor, e4m3_tensor,
                    run_dense_32x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_e4m3_64x32, e4m3_tensor, e4m3_tensor,
                    run_dense_64x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_e4m3_64x64, e4m3_tensor, e4m3_tensor,
                    run_dense_64x64)
#endif

#if __HAVE_METAL_FP8_E5M2_FORMAT_TYPE__
using e5m2_tensor = tensor<device metal_fp8_e5m2_format, dextents<int32_t, 2>>;
DEFINE_DENSE_KERNEL(lowbit_matmul_e5m2_32x32, e5m2_tensor, e5m2_tensor,
                    run_dense_32x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_e5m2_64x32, e5m2_tensor, e5m2_tensor,
                    run_dense_64x32)
DEFINE_DENSE_KERNEL(lowbit_matmul_e5m2_64x64, e5m2_tensor, e5m2_tensor,
                    run_dense_64x64)
#endif

#if __HAVE_TENSOR_MULTIPLANE__ && __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__ &&       \
    __HAVE_METAL_FP8_UE8M0_FORMAT_TYPE__
using mxfp8_tensor = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>,
                            tensor_handle, mx_scale>;
DEFINE_SCALED_KERNEL(lowbit_matmul_mxfp8_e4m3_32x32, mxfp8_tensor,
                     run_scaled_32x32)
DEFINE_SCALED_KERNEL(lowbit_matmul_mxfp8_e4m3_64x32, mxfp8_tensor,
                     run_scaled_64x32)
DEFINE_SCALED_KERNEL(lowbit_matmul_mxfp8_e4m3_64x64, mxfp8_tensor,
                     run_scaled_64x64)
#endif

#if __HAVE_TENSOR_MULTIPLANE__ && __HAVE_METAL_FP4_E2M1_FORMAT_TYPE__ &&       \
    __HAVE_METAL_FP8_UE8M0_FORMAT_TYPE__
using mxfp4_tensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>,
                            tensor_handle, mx_scale>;
DEFINE_SCALED_KERNEL(lowbit_matmul_mxfp4_e2m1_32x32, mxfp4_tensor,
                     run_scaled_32x32)
DEFINE_SCALED_KERNEL(lowbit_matmul_mxfp4_e2m1_64x32, mxfp4_tensor,
                     run_scaled_64x32)
DEFINE_SCALED_KERNEL(lowbit_matmul_mxfp4_e2m1_64x64, mxfp4_tensor,
                     run_scaled_64x64)
#endif

#undef DEFINE_SCALED_KERNEL
#undef DEFINE_DENSE_KERNEL
