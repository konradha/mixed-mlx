#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal.hpp"

#include <Foundation/Foundation.hpp>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace msqrt {
namespace {

std::string error_text(NS::Error *error) {
  return error == nullptr ? "Unknown Metal error."
                          : error->localizedDescription()->utf8String();
}

} // namespace

class Metal::Impl {
public:
  explicit Impl(const std::string &path) {
    device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
      throw std::runtime_error("No Metal device is available.");
    }
    queue = device->newCommandQueue();
    if (queue == nullptr) {
      throw std::runtime_error("Cannot create the Metal command queue.");
    }
    NS::String *library_path =
        NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    NS::Error *error = nullptr;
    library = device->newLibrary(library_path, &error);
    if (library == nullptr) {
      throw std::runtime_error("Cannot load Metal library " + path + ": " +
                               error_text(error));
    }
  }

  ~Impl() {
    for (auto &[name, value] : pipelines) {
      (void)name;
      value->release();
    }
    library->release();
    queue->release();
    device->release();
  }

  MTL::ComputePipelineState *get_pipeline(const std::string &name) {
    const auto found = pipelines.find(name);
    if (found != pipelines.end()) {
      return found->second;
    }
    NS::String *function_name =
        NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    MTL::Function *function = library->newFunction(function_name);
    if (function == nullptr) {
      throw std::runtime_error("Cannot find Metal kernel: " + name);
    }
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *result =
        device->newComputePipelineState(function, &error);
    function->release();
    if (result == nullptr) {
      throw std::runtime_error("Cannot create Metal pipeline " + name + ": " +
                               error_text(error));
    }
    pipelines.emplace(name, result);
    return result;
  }

  MTL::Device *device = nullptr;
  MTL::CommandQueue *queue = nullptr;
  MTL::Library *library = nullptr;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines;
};

Metal::Metal(const std::string &library_path)
    : impl_(std::make_unique<Impl>(library_path)) {}

Metal::~Metal() = default;

MTL::Buffer *Metal::buffer(std::size_t bytes) const {
  MTL::Buffer *result =
      impl_->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
  if (result == nullptr) {
    throw std::runtime_error("Cannot allocate a shared Metal buffer.");
  }
  return result;
}

MTL::ComputePipelineState *Metal::pipeline(const std::string &name) {
  return impl_->get_pipeline(name);
}

double Metal::submit(
    const std::function<void(MTL::ComputeCommandEncoder *)> &encode) const {
  MTL::CommandBuffer *command = impl_->queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
  encode(encoder);
  encoder->endEncoding();
  command->commit();
  command->waitUntilCompleted();
  if (command->status() == MTL::CommandBufferStatusError) {
    throw std::runtime_error("Metal command failed: " +
                             error_text(command->error()));
  }
  return command->GPUEndTime() - command->GPUStartTime();
}
const char *Metal::device_name() const {
  return impl_->device->name()->utf8String();
}

void Metal::dispatch1d(MTL::ComputeCommandEncoder *encoder,
                       MTL::ComputePipelineState *pipeline,
                       std::size_t elements, std::size_t threads) {
  encoder->setComputePipelineState(pipeline);
  const std::size_t width = std::min(elements, threads);
  encoder->dispatchThreads(MTL::Size::Make(elements, 1, 1),
                           MTL::Size::Make(width, 1, 1));
}

void Metal::dispatch2d(MTL::ComputeCommandEncoder *encoder,
                       MTL::ComputePipelineState *pipeline, std::size_t columns,
                       std::size_t rows) {
  encoder->setComputePipelineState(pipeline);
  encoder->dispatchThreads(MTL::Size::Make(columns, rows, 1),
                           MTL::Size::Make(16, 16, 1));
}

void release_buffers(std::vector<MTL::Buffer *> &buffers) {
  for (MTL::Buffer *buffer : buffers) {
    buffer->release();
  }
  buffers.clear();
}

} // namespace msqrt
