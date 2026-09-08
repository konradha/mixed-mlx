#include "tensor_layout.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace msqrt {

void validate_mx_tensor_layout(MTL::Buffer *values, MTL::Buffer *scales,
                               std::size_t elements, bool fp4) {
  if (elements == 0 || elements % 32 != 0 ||
      elements >
          static_cast<std::size_t>(std::numeric_limits<NS::Integer>::max())) {
    throw std::runtime_error(
        "An MX tensor dimension must be a positive multiple of 32.");
  }

  const NS::Integer dimension_values[] = {
      static_cast<NS::Integer>(elements),
  };
  const NS::Integer stride_values[] = {1};
  const NS::Integer block_values[] = {32};

  MTL::TensorExtents *dimensions =
      MTL::TensorExtents::alloc()->init(1, dimension_values);
  MTL::TensorExtents *strides =
      MTL::TensorExtents::alloc()->init(1, stride_values);
  MTL::TensorExtents *block_factors =
      MTL::TensorExtents::alloc()->init(1, block_values);
  MTL::TensorAuxiliaryPlaneDescriptor *scale_descriptor =
      MTL::TensorAuxiliaryPlaneDescriptor::alloc()->init();
  scale_descriptor->setDataType(MTL::TensorDataTypeMetalFloat8UE8M0);
  scale_descriptor->setBlockFactors(block_factors);

  MTL::TensorAuxiliaryPlaneDescriptorMap *plane_map =
      MTL::TensorAuxiliaryPlaneDescriptorMap::alloc()->init();
  plane_map->setDescriptor(scale_descriptor, MTL::TensorPlaneTypeScales);

  MTL::TensorDescriptor *descriptor = MTL::TensorDescriptor::alloc()->init();
  descriptor->setDimensions(dimensions);
  descriptor->setStrides(strides);
  descriptor->setDataType(fp4 ? MTL::TensorDataTypeMetalFloat4E2M1
                              : MTL::TensorDataTypeMetalFloat8E4M3);
  descriptor->setUsage(MTL::TensorUsageCompute);
  descriptor->setAuxiliaryPlanes(plane_map);

  MTL::TensorBufferAttachments *attachments =
      MTL::TensorBufferAttachments::alloc()->init();
  attachments->setBuffer(values, 0, MTL::TensorPlaneTypeData);
  attachments->setBuffer(scales, 0, MTL::TensorPlaneTypeScales);
  NS::Error *error = nullptr;
  MTL::Tensor *tensor =
      values->device()->newTensor(descriptor, attachments, &error);
  std::string error_text;
  if (tensor == nullptr) {
    error_text = error == nullptr ? "unknown error"
                                  : error->localizedDescription()->utf8String();
  }
  if (error != nullptr) {
    error->release();
  }
  if (tensor != nullptr) {
    tensor->release();
  }
  attachments->release();
  descriptor->release();
  plane_map->release();
  scale_descriptor->release();
  block_factors->release();
  strides->release();
  dimensions->release();

  if (!error_text.empty()) {
    throw std::runtime_error("Metal rejected the MX tensor layout: " +
                             error_text);
  }
}

} // namespace msqrt
