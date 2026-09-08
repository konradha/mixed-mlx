#include "lowbit_matmul.hpp"

#include <Foundation/Foundation.hpp>

#include <limits>
#include <utility>

namespace msqrt {
namespace {

constexpr std::size_t kBindingCount = 3;
constexpr std::size_t kSimdgroups = 4;
constexpr std::size_t kLowbitStrideBytes = 128;
constexpr std::size_t kScaleBlock = 32;

template <typename T> class Owned {
public:
  Owned() = default;
  explicit Owned(T *value) : value_(value) {}
  ~Owned() {
    if (value_)
      value_->release();
  }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  Owned(Owned &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}
  Owned &operator=(Owned &&other) noexcept {
    if (this != &other) {
      if (value_)
        value_->release();
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }
  T *get() const noexcept { return value_; }
  explicit operator bool() const noexcept { return value_ != nullptr; }

private:
  T *value_ = nullptr;
};

LowbitMatmulResult make_result(LowbitMatmulStatus status, std::string message) {
  return {status, std::move(message)};
}

std::string error_text(NS::Error *error) {
  if (!error || !error->localizedDescription())
    return "unknown Metal error";
  return error->localizedDescription()->utf8String();
}

bool multiply(std::size_t a, std::size_t b, std::size_t &out) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    return false;
  out = a * b;
  return true;
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t &out) {
  if (!alignment ||
      value > std::numeric_limits<std::size_t>::max() - (alignment - 1))
    return false;
  out = ((value + alignment - 1) / alignment) * alignment;
  return true;
}

Owned<MTL::TensorExtents> extents(NS::Integer first, NS::Integer second) {
  const NS::Integer values[] = {first, second};
  return Owned<MTL::TensorExtents>(
      MTL::TensorExtents::alloc()->init(2, values));
}

} // namespace

class LowbitMatmul::Impl {
public:
  Impl(MTL::Device *input_device, MTL::Library *input_library,
       std::size_t input_order, LowbitDataType type, LowbitMatmulTile tile)
      : matrix_order(input_order), input_type(type), input_tile(tile) {
    if (!initialize_device(input_device, input_library) || !validate_order() ||
        !configure_tile()) {
      return;
    }
    const char *function_base = nullptr;
    switch (input_type) {
    case LowbitDataType::Float16:
      left_bits = right_bits = 16;
      left_data_type = right_data_type = MTL::TensorDataTypeFloat16;
      function_base = "lowbit_matmul_fp16";
      break;
    case LowbitDataType::Float8E4M3:
      left_bits = right_bits = 8;
      left_data_type = right_data_type = MTL::TensorDataTypeMetalFloat8E4M3;
      function_base = "lowbit_matmul_e4m3";
      break;
    case LowbitDataType::Float8E5M2:
      left_bits = right_bits = 8;
      left_data_type = right_data_type = MTL::TensorDataTypeMetalFloat8E5M2;
      function_base = "lowbit_matmul_e5m2";
      break;
    case LowbitDataType::MxFloat8E4M3:
      left_bits = right_bits = 8;
      left_data_type = right_data_type = MTL::TensorDataTypeMetalFloat8E4M3;
      function_base = "lowbit_matmul_mxfp8_e4m3";
      has_scales = true;
      break;
    case LowbitDataType::MxFloat4E2M1:
      left_bits = right_bits = 4;
      left_data_type = right_data_type = MTL::TensorDataTypeMetalFloat4E2M1;
      function_base = "lowbit_matmul_mxfp4_e2m1";
      has_scales = true;
      break;
    default:
      state = make_result(LowbitMatmulStatus::UnsupportedDataType,
                          "No native MPP kernel exists for this format.");
      return;
    }
    finish_initialization(function_base);
  }

  Impl(MTL::Device *input_device, MTL::Library *input_library,
       std::size_t input_order, LowbitMatmulSignature signature,
       LowbitMatmulTile tile)
      : matrix_order(input_order), input_type(LowbitDataType::Float16),
        input_tile(tile) {
    if (!initialize_device(input_device, input_library) || !validate_order() ||
        !configure_tile()) {
      return;
    }
    const char *function_base = nullptr;
    switch (signature) {
    case LowbitMatmulSignature::Float16Float16ToFloat32:
      left_bits = right_bits = 16;
      left_data_type = right_data_type = MTL::TensorDataTypeFloat16;
      function_base = "lowbit_matmul_fp16";
      break;
    case LowbitMatmulSignature::Float16Float32ToFloat32:
      left_bits = 16;
      right_bits = 32;
      left_data_type = MTL::TensorDataTypeFloat16;
      right_data_type = MTL::TensorDataTypeFloat32;
      function_base = "lowbit_matmul_fp16_fp32";
      break;
    case LowbitMatmulSignature::Float32Float16ToFloat32:
      left_bits = 32;
      right_bits = 16;
      left_data_type = MTL::TensorDataTypeFloat32;
      right_data_type = MTL::TensorDataTypeFloat16;
      function_base = "lowbit_matmul_fp32_fp16";
      break;
    default:
      state = make_result(LowbitMatmulStatus::UnsupportedDataType,
                          "No native MPP kernel exists for this signature.");
      return;
    }
    finish_initialization(function_base);
  }

  ~Impl() {
    if (pipeline)
      pipeline->release();
    if (output_descriptor)
      output_descriptor->release();
    if (right_descriptor)
      right_descriptor->release();
    if (left_descriptor)
      left_descriptor->release();
    if (library)
      library->release();
    if (device)
      device->release();
  }

  bool initialize_device(MTL::Device *input_device,
                         MTL::Library *input_library) {
    if (!input_device || !input_library) {
      state = make_result(LowbitMatmulStatus::InvalidArgument,
                          "A Metal device and library are required.");
      return false;
    }
    device = input_device;
    library = input_library;
    device->retain();
    library->retain();
    return true;
  }

  bool validate_order() {
    if (!matrix_order ||
        matrix_order >
            static_cast<std::size_t>(std::numeric_limits<NS::Integer>::max())) {
      state = make_result(LowbitMatmulStatus::InvalidArgument,
                          "Matrix order is outside the tensor index range.");
      return false;
    }
    return true;
  }

  bool configure_tile() {
    switch (input_tile) {
    case LowbitMatmulTile::Rows32Columns32:
      tile_rows = 32;
      tile_columns = 32;
      tile_suffix = "32x32";
      return true;
    case LowbitMatmulTile::Rows64Columns32:
      tile_rows = 64;
      tile_columns = 32;
      tile_suffix = "64x32";
      return true;
    case LowbitMatmulTile::Rows64Columns64:
      tile_rows = 64;
      tile_columns = 64;
      tile_suffix = "64x64";
      return true;
    default:
      state = make_result(LowbitMatmulStatus::InvalidArgument,
                          "Unknown MPP tile configuration.");
      return false;
    }
  }

  void finish_initialization(const char *function_base) {
    function_name = std::string(function_base) + "_" + tile_suffix;
    state = build_layouts();
    if (state.status == LowbitMatmulStatus::Available) {
      state = build_descriptors();
    }
    if (state.status == LowbitMatmulStatus::Available) {
      state = probe_tensors();
    }
    if (state.status == LowbitMatmulStatus::Available) {
      state = build_pipeline();
    }
  }

  LowbitMatmulResult
  build_data_layout(std::size_t bits, LowbitMatmulOperandLayout &layout) const {
    const std::size_t stride_alignment = bits == 8   ? kLowbitStrideBytes
                                         : bits == 4 ? 2 * kLowbitStrideBytes
                                                     : 1;
    std::size_t leading = 0;
    std::size_t element_count = 0;
    std::size_t total_bits = 0;
    if (!align_up(matrix_order, stride_alignment, leading) ||
        !multiply(leading, matrix_order, element_count) ||
        !multiply(element_count, bits, total_bits)) {
      return make_result(LowbitMatmulStatus::InvalidArgument,
                         "Input tensor layout overflows size_t.");
    }
    layout.data_bytes = total_bits / 8;
    layout.data_leading_dimension_elements = leading;
    return make_result(LowbitMatmulStatus::Available, "Layout is valid.");
  }

  LowbitMatmulResult build_layouts() {
    if ((left_bits < 16 || right_bits < 16) && matrix_order % kScaleBlock) {
      return make_result(
          LowbitMatmulStatus::InvalidArgument,
          "FP8 and FP4 tensor dimensions must be multiples of 32.");
    }
    LowbitMatmulResult layout = build_data_layout(left_bits, left);
    if (layout.status != LowbitMatmulStatus::Available)
      return layout;
    layout = build_data_layout(right_bits, right);
    if (layout.status != LowbitMatmulStatus::Available)
      return layout;

    if (has_scales) {
      const std::size_t scale_width = matrix_order / kScaleBlock;
      const std::size_t scale_leading =
          left.data_leading_dimension_elements / kScaleBlock;
      if (!multiply(scale_leading, matrix_order - 1, left.scale_bytes) ||
          left.scale_bytes >
              std::numeric_limits<std::size_t>::max() - scale_width) {
        return make_result(LowbitMatmulStatus::InvalidArgument,
                           "Scale tensor layout overflows size_t.");
      }
      left.scale_bytes += scale_width;
      left.storage = LowbitMatrixStorage::TransposedColumnMajor;
      left.scale_block_elements = kScaleBlock;
      left.scale_leading_dimension_bytes = scale_leading;
      right.scale_bytes = left.scale_bytes;
      right.scale_block_elements = kScaleBlock;
      right.scale_leading_dimension_bytes = scale_leading;
    }

    std::size_t count = 0;
    if (!multiply(matrix_order, matrix_order, count) ||
        !multiply(count, sizeof(float), output_size)) {
      return make_result(LowbitMatmulStatus::InvalidArgument,
                         "Output tensor layout overflows size_t.");
    }
    return make_result(LowbitMatmulStatus::Available, "Layouts are valid.");
  }

  LowbitMatmulResult
  make_input_descriptor(MTL::TensorDataType data_type,
                        const LowbitMatmulOperandLayout &layout,
                        MTL::TensorDescriptor *&descriptor) const {
    const auto dimension = static_cast<NS::Integer>(matrix_order);
    const auto stride =
        static_cast<NS::Integer>(layout.data_leading_dimension_elements);
    auto dimensions = extents(dimension, dimension);
    auto strides = extents(1, stride);
    if (!dimensions || !strides) {
      return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                         "Cannot allocate tensor extents.");
    }
    descriptor = MTL::TensorDescriptor::alloc()->init();
    if (!descriptor) {
      return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                         "Cannot allocate an input tensor descriptor.");
    }
    descriptor->setDimensions(dimensions.get());
    descriptor->setStrides(strides.get());
    descriptor->setUsage(MTL::TensorUsageCompute);
    descriptor->setDataType(data_type);

    if (has_scales) {
      auto block_factors = extents(kScaleBlock, 1);
      Owned<MTL::TensorAuxiliaryPlaneDescriptor> scale(
          MTL::TensorAuxiliaryPlaneDescriptor::alloc()->init());
      Owned<MTL::TensorAuxiliaryPlaneDescriptorMap> planes(
          MTL::TensorAuxiliaryPlaneDescriptorMap::alloc()->init());
      if (!block_factors || !scale || !planes) {
        return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                           "Cannot allocate scale-plane descriptors.");
      }
      scale.get()->setDataType(MTL::TensorDataTypeMetalFloat8UE8M0);
      scale.get()->setBlockFactors(block_factors.get());
      planes.get()->setDescriptor(scale.get(), MTL::TensorPlaneTypeScales);
      descriptor->setAuxiliaryPlanes(planes.get());
    }
    return make_result(LowbitMatmulStatus::Available,
                       "Input tensor descriptor is valid.");
  }

  LowbitMatmulResult build_descriptors() {
    LowbitMatmulResult result =
        make_input_descriptor(left_data_type, left, left_descriptor);
    if (result.status != LowbitMatmulStatus::Available)
      return result;
    result = make_input_descriptor(right_data_type, right, right_descriptor);
    if (result.status != LowbitMatmulStatus::Available)
      return result;

    const auto dimension = static_cast<NS::Integer>(matrix_order);
    auto dimensions = extents(dimension, dimension);
    auto strides = extents(1, dimension);
    if (!dimensions || !strides) {
      return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                         "Cannot allocate output tensor extents.");
    }
    output_descriptor = MTL::TensorDescriptor::alloc()->init();
    if (!output_descriptor) {
      return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                         "Cannot allocate an output tensor descriptor.");
    }
    output_descriptor->setDimensions(dimensions.get());
    output_descriptor->setStrides(strides.get());
    output_descriptor->setUsage(MTL::TensorUsageCompute);
    output_descriptor->setDataType(MTL::TensorDataTypeFloat32);
    return make_result(LowbitMatmulStatus::Available,
                       "Tensor descriptors are valid.");
  }

  Owned<MTL::Tensor> input_tensor(MTL::Buffer *data, MTL::Buffer *scales,
                                  MTL::TensorDescriptor *descriptor,
                                  LowbitMatmulResult &failure) const {
    NS::Error *error = nullptr;
    MTL::Tensor *tensor = nullptr;
    if (has_scales) {
      Owned<MTL::TensorBufferAttachments> attachments(
          MTL::TensorBufferAttachments::alloc()->init());
      if (!attachments) {
        failure = make_result(LowbitMatmulStatus::ResourceCreationFailed,
                              "Cannot allocate tensor attachments.");
        return {};
      }
      attachments.get()->setBuffer(data, 0, MTL::TensorPlaneTypeData);
      attachments.get()->setBuffer(scales, 0, MTL::TensorPlaneTypeScales);
      tensor = device->newTensor(descriptor, attachments.get(), &error);
    } else {
      tensor = data->newTensor(descriptor, 0, &error);
    }
    if (!tensor) {
      failure =
          make_result(LowbitMatmulStatus::TensorCreationFailed,
                      std::string("Metal rejected the input tensor: ") +
                          error_text(error) + ". No fallback was selected.");
    }
    return Owned<MTL::Tensor>(tensor);
  }

  LowbitMatmulResult probe_tensors() const {
    Owned<MTL::Buffer> left_data(
        device->newBuffer(left.data_bytes, MTL::ResourceStorageModeShared));
    Owned<MTL::Buffer> right_data(
        device->newBuffer(right.data_bytes, MTL::ResourceStorageModeShared));
    Owned<MTL::Buffer> left_scales;
    Owned<MTL::Buffer> right_scales;
    if (has_scales) {
      left_scales = Owned<MTL::Buffer>(
          device->newBuffer(left.scale_bytes, MTL::ResourceStorageModeShared));
      right_scales = Owned<MTL::Buffer>(
          device->newBuffer(right.scale_bytes, MTL::ResourceStorageModeShared));
    }
    Owned<MTL::Buffer> output(
        device->newBuffer(output_size, MTL::ResourceStorageModeShared));
    if (!left_data || !right_data || !output ||
        (has_scales && (!left_scales || !right_scales))) {
      return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                         "Cannot allocate runtime tensor-probe buffers.");
    }
    auto probe = make_result(LowbitMatmulStatus::Available, "");
    auto left_tensor = input_tensor(left_data.get(), left_scales.get(),
                                    left_descriptor, probe);
    if (!left_tensor)
      return probe;
    auto right_tensor = input_tensor(right_data.get(), right_scales.get(),
                                     right_descriptor, probe);
    if (!right_tensor)
      return probe;
    NS::Error *error = nullptr;
    Owned<MTL::Tensor> destination(
        output.get()->newTensor(output_descriptor, 0, &error));
    if (!destination) {
      return make_result(LowbitMatmulStatus::TensorCreationFailed,
                         std::string("Metal rejected the output tensor: ") +
                             error_text(error));
    }
    return make_result(LowbitMatmulStatus::Available,
                       "Runtime tensor creation succeeded.");
  }

  LowbitMatmulResult build_pipeline() {
    auto *name =
        NS::String::string(function_name.c_str(), NS::UTF8StringEncoding);
    Owned<MTL::Function> function(library->newFunction(name));
    if (!function) {
      return make_result(LowbitMatmulStatus::UnsupportedPipeline,
                         std::string("Library has no ") + function_name +
                             " function. No fallback was selected.");
    }
    NS::Error *error = nullptr;
    pipeline = device->newComputePipelineState(function.get(), &error);
    if (!pipeline) {
      return make_result(LowbitMatmulStatus::UnsupportedPipeline,
                         std::string("Device rejected ") + function_name +
                             ": " + error_text(error) +
                             ". No fallback was selected.");
    }
    const std::size_t width = pipeline->threadExecutionWidth();
    if (!width ||
        width > std::numeric_limits<std::size_t>::max() / kSimdgroups ||
        width * kSimdgroups > pipeline->maxTotalThreadsPerThreadgroup()) {
      pipeline->release();
      pipeline = nullptr;
      return make_result(LowbitMatmulStatus::UnsupportedPipeline,
                         "Pipeline cannot run four cooperative SIMD-groups.");
    }
    threads = width * kSimdgroups;
    return make_result(LowbitMatmulStatus::Available,
                       "Native MPP pipeline is available.");
  }

  LowbitMatmulResult validate(MTL4::CommandBuffer *command,
                              const LowbitMatmulBuffers &buffers) const {
    if (!command || !buffers.left || !buffers.right || !buffers.output ||
        !buffers.residency_set) {
      return make_result(
          LowbitMatmulStatus::InvalidArgument,
          "Command, data planes, output, and residency set are required.");
    }
    if ((has_scales && (!buffers.left_scales || !buffers.right_scales)) ||
        (!has_scales && (buffers.left_scales || buffers.right_scales))) {
      return make_result(
          LowbitMatmulStatus::InvalidArgument,
          "Scale planes must be supplied exactly for MX formats.");
    }
    if (command->device() != device || buffers.left->device() != device ||
        buffers.right->device() != device ||
        buffers.output->device() != device ||
        buffers.residency_set->device() != device ||
        (has_scales && (buffers.left_scales->device() != device ||
                        buffers.right_scales->device() != device))) {
      return make_result(LowbitMatmulStatus::InvalidArgument,
                         "All resources must belong to the product device.");
    }
    if (buffers.output == buffers.left || buffers.output == buffers.right ||
        (has_scales && (buffers.output == buffers.left_scales ||
                        buffers.output == buffers.right_scales))) {
      return make_result(LowbitMatmulStatus::InvalidArgument,
                         "Output must not alias an input plane.");
    }
    if (buffers.left->length() < left.data_bytes ||
        buffers.right->length() < right.data_bytes ||
        buffers.output->length() < output_size ||
        (has_scales && (buffers.left_scales->length() < left.scale_bytes ||
                        buffers.right_scales->length() < right.scale_bytes))) {
      return make_result(LowbitMatmulStatus::InvalidArgument,
                         "A buffer is smaller than its reported layout.");
    }
    if (!buffers.residency_set->containsAllocation(buffers.left) ||
        !buffers.residency_set->containsAllocation(buffers.right) ||
        !buffers.residency_set->containsAllocation(buffers.output) ||
        (has_scales &&
         (!buffers.residency_set->containsAllocation(buffers.left_scales) ||
          !buffers.residency_set->containsAllocation(buffers.right_scales)))) {
      return make_result(LowbitMatmulStatus::InvalidArgument,
                         "Residency set does not contain every tensor plane.");
    }
    return make_result(LowbitMatmulStatus::Available, "Buffers are valid.");
  }

  MTL::Device *device = nullptr;
  MTL::Library *library = nullptr;
  std::size_t matrix_order = 0;
  LowbitDataType input_type = LowbitDataType::Float16;
  LowbitMatmulTile input_tile = LowbitMatmulTile::Rows64Columns32;
  std::size_t tile_rows = 0;
  std::size_t tile_columns = 0;
  const char *tile_suffix = nullptr;
  std::size_t left_bits = 0;
  std::size_t right_bits = 0;
  MTL::TensorDataType left_data_type = MTL::TensorDataTypeNone;
  MTL::TensorDataType right_data_type = MTL::TensorDataTypeNone;
  std::string function_name;
  bool has_scales = false;
  LowbitMatmulOperandLayout left;
  LowbitMatmulOperandLayout right;
  std::size_t output_size = 0;
  MTL::TensorDescriptor *left_descriptor = nullptr;
  MTL::TensorDescriptor *right_descriptor = nullptr;
  MTL::TensorDescriptor *output_descriptor = nullptr;
  MTL::ComputePipelineState *pipeline = nullptr;
  std::size_t threads = 0;
  LowbitMatmulResult state;
};

LowbitMatmulEncoderResources::~LowbitMatmulEncoderResources() {
  for (NS::Object *object : objects_) {
    object->release();
  }
}

void LowbitMatmulEncoderResources::retain(NS::Object *object) {
  object->retain();
  try {
    objects_.push_back(object);
  } catch (...) {
    object->release();
    throw;
  }
}

LowbitMatmul::LowbitMatmul(MTL::Device *device, MTL::Library *library,
                           std::size_t order, LowbitDataType data_type,
                           LowbitMatmulTile tile)
    : impl_(std::make_unique<Impl>(device, library, order, data_type, tile)) {}
LowbitMatmul::LowbitMatmul(MTL::Device *device, MTL::Library *library,
                           std::size_t order, LowbitMatmulSignature signature,
                           LowbitMatmulTile tile)
    : impl_(std::make_unique<Impl>(device, library, order, signature, tile)) {}
LowbitMatmul::~LowbitMatmul() = default;
std::size_t LowbitMatmul::order() const noexcept { return impl_->matrix_order; }
LowbitDataType LowbitMatmul::data_type() const noexcept {
  return impl_->input_type;
}
LowbitMatmulTile LowbitMatmul::tile() const noexcept {
  return impl_->input_tile;
}
std::size_t LowbitMatmul::tile_rows() const noexcept {
  return impl_->tile_rows;
}
std::size_t LowbitMatmul::tile_columns() const noexcept {
  return impl_->tile_columns;
}
LowbitAccumulationType LowbitMatmul::accumulation_type() const noexcept {
  return LowbitAccumulationType::Float32;
}
const LowbitMatmulOperandLayout &LowbitMatmul::left_layout() const noexcept {
  return impl_->left;
}
const LowbitMatmulOperandLayout &LowbitMatmul::right_layout() const noexcept {
  return impl_->right;
}
std::size_t LowbitMatmul::output_bytes() const noexcept {
  return impl_->output_size;
}
LowbitMatmulResult LowbitMatmul::availability() const { return impl_->state; }

LowbitMatmulResult
LowbitMatmul::encode(MTL4::ComputeCommandEncoder *encoder,
                     const LowbitMatmulBuffers &buffers,
                     LowbitMatmulEncoderResources &resources) const {
  if (impl_->state.status != LowbitMatmulStatus::Available)
    return impl_->state;
  MTL4::CommandBuffer *command =
      encoder == nullptr ? nullptr : encoder->commandBuffer();
  auto validation = impl_->validate(command, buffers);
  if (validation.status != LowbitMatmulStatus::Available)
    return validation;

  auto creation = make_result(LowbitMatmulStatus::Available, "");
  auto left = impl_->input_tensor(buffers.left, buffers.left_scales,
                                  impl_->left_descriptor, creation);
  if (!left)
    return creation;
  auto right = impl_->input_tensor(buffers.right, buffers.right_scales,
                                   impl_->right_descriptor, creation);
  if (!right)
    return creation;
  NS::Error *error = nullptr;
  Owned<MTL::Tensor> output(
      buffers.output->newTensor(impl_->output_descriptor, 0, &error));
  if (!output) {
    return make_result(LowbitMatmulStatus::TensorCreationFailed,
                       std::string("Metal rejected the output tensor: ") +
                           error_text(error));
  }

  Owned<MTL4::ArgumentTableDescriptor> table_descriptor(
      MTL4::ArgumentTableDescriptor::alloc()->init());
  if (!table_descriptor) {
    return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                       "Cannot allocate an argument-table descriptor.");
  }
  table_descriptor.get()->setMaxBufferBindCount(kBindingCount);
  table_descriptor.get()->setInitializeBindings(true);
  error = nullptr;
  Owned<MTL4::ArgumentTable> table(
      impl_->device->newArgumentTable(table_descriptor.get(), &error));
  if (!table) {
    return make_result(LowbitMatmulStatus::ResourceCreationFailed,
                       std::string("Cannot create argument table: ") +
                           error_text(error));
  }
  table.get()->setResource(left.get()->gpuResourceID(), 0);
  table.get()->setResource(right.get()->gpuResourceID(), 1);
  table.get()->setResource(output.get()->gpuResourceID(), 2);
  resources.retain(left.get());
  resources.retain(right.get());
  resources.retain(output.get());
  resources.retain(table.get());

  encoder->setComputePipelineState(impl_->pipeline);
  encoder->setArgumentTable(table.get());
  const MTL::Size groups = MTL::Size::Make(
      (impl_->matrix_order + impl_->tile_columns - 1) / impl_->tile_columns,
      (impl_->matrix_order + impl_->tile_rows - 1) / impl_->tile_rows, 1);
  encoder->dispatchThreadgroups(groups, MTL::Size::Make(impl_->threads, 1, 1));
  return make_result(LowbitMatmulStatus::Encoded,
                     "Encoded native MPP matmul with FP32 output.");
}

LowbitMatmulResult
LowbitMatmul::encode(MTL4::CommandBuffer *command,
                     const LowbitMatmulBuffers &buffers) const {
  if (impl_->state.status != LowbitMatmulStatus::Available)
    return impl_->state;
  auto validation = impl_->validate(command, buffers);
  if (validation.status != LowbitMatmulStatus::Available)
    return validation;
  command->useResidencySet(buffers.residency_set);
  MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
  if (encoder == nullptr) {
    return make_result(LowbitMatmulStatus::EncoderCreationFailed,
                       "Cannot create an MTL4 compute encoder.");
  }
  LowbitMatmulEncoderResources resources;
  LowbitMatmulResult result = encode(encoder, buffers, resources);
  encoder->endEncoding();
  return result;
}

} // namespace msqrt
