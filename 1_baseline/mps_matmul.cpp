#include "mps_matmul.hpp"

#include <objc/message.h>
#include <objc/runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace msqrt {
namespace {

using Object = void *;
constexpr std::uint32_t mps_float32 = UINT32_C(0x10000020);

template <typename Result, typename... Args>
Result message(Object object, const char *selector, Args... args) {
  using Function = Result (*)(Object, SEL, Args...);
  const auto function = reinterpret_cast<Function>(objc_msgSend);
  return function(object, sel_registerName(selector), args...);
}

Object object(const void *value) { return const_cast<void *>(value); }

Object class_object(const char *name) {
  return reinterpret_cast<Object>(objc_getClass(name));
}

Object make_matrix(Object buffer, Object descriptor) {
  Object allocated = message<Object>(class_object("MPSMatrix"), "alloc");
  return message<Object>(allocated, "initWithBuffer:descriptor:", buffer,
                         descriptor);
}

} // namespace

class MpsMatmul::Impl {
public:
  Object descriptor = nullptr;
  Object kernel = nullptr;
};

MpsMatmul::MpsMatmul(MTL::Buffer *buffer, std::size_t order)
    : impl_(std::make_unique<Impl>()) {
  if (order == 0 ||
      order > std::numeric_limits<NS::UInteger>::max() / sizeof(float)) {
    throw std::runtime_error("Matrix order exceeds MPS range.");
  }
  Object descriptor = message<Object>(
      class_object("MPSMatrixDescriptor"),
      "matrixDescriptorWithRows:columns:rowBytes:dataType:", order, order,
      order * sizeof(float), mps_float32);
  impl_->descriptor = message<Object>(descriptor, "retain");

  Object allocated =
      message<Object>(class_object("MPSMatrixMultiplication"), "alloc");
  impl_->kernel = message<Object>(
      allocated, "initWithDevice:resultRows:resultColumns:interiorColumns:",
      object(buffer->device()), order, order, order);
  if (impl_->descriptor == nullptr || impl_->kernel == nullptr) {
    throw std::runtime_error("Cannot create the MPS matrix multiply kernel.");
  }
}

MpsMatmul::~MpsMatmul() {
  message<void>(impl_->kernel, "release");
  message<void>(impl_->descriptor, "release");
}

void MpsMatmul::encode(MTL::CommandBuffer *command, MTL::Buffer *left,
                       MTL::Buffer *right, MTL::Buffer *output) const {
  Object left_matrix = make_matrix(object(right), impl_->descriptor);
  Object right_matrix = make_matrix(object(left), impl_->descriptor);
  Object output_matrix = make_matrix(object(output), impl_->descriptor);
  message<void>(impl_->kernel,
                "encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:",
                object(command), left_matrix, right_matrix, output_matrix);
  message<void>(output_matrix, "release");
  message<void>(right_matrix, "release");
  message<void>(left_matrix, "release");
}

} // namespace msqrt
