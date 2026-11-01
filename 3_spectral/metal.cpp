#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal.hpp"

#include <Foundation/Foundation.hpp>

#include <algorithm>
#include <condition_variable>

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace msqrt {
namespace {

std::string error_text(NS::Error *error) {
  return error == nullptr ? "Unknown Metal error."
                          : error->localizedDescription()->utf8String();
}

void encode_mtl4_dispatch(MTL4::ComputeCommandEncoder *encoder,
                          MTL::ComputePipelineState *pipeline,
                          MTL::ResidencySet *residency,
                          std::initializer_list<MTL::Buffer *> buffers,
                          MTL4EncoderResources &resources, MTL::Size grid,
                          MTL::Size threads) {
  MTL4::CommandBuffer *command =
      encoder == nullptr ? nullptr : encoder->commandBuffer();
  if (command == nullptr || pipeline == nullptr || residency == nullptr) {
    throw std::runtime_error(
        "MTL4 dispatch requires an encoder, pipeline, and residency set.");
  }
  MTL4::ArgumentTableDescriptor *descriptor =
      MTL4::ArgumentTableDescriptor::alloc()->init();
  if (descriptor == nullptr) {
    throw std::runtime_error("Cannot allocate an MTL4 argument descriptor.");
  }
  descriptor->setMaxBufferBindCount(buffers.size());
  descriptor->setInitializeBindings(true);
  NS::Error *error = nullptr;
  MTL4::ArgumentTable *table =
      command->device()->newArgumentTable(descriptor, &error);
  descriptor->release();
  if (table == nullptr) {
    throw std::runtime_error("Cannot create an MTL4 argument table: " +
                             error_text(error));
  }
  NS::UInteger index = 0;
  for (MTL::Buffer *buffer : buffers) {
    if (buffer == nullptr || !residency->containsAllocation(buffer)) {
      table->release();
      throw std::runtime_error("Every MTL4 buffer binding must be resident.");
    }
    table->setAddress(buffer->gpuAddress(), index++);
  }
  resources.adopt(table);
  encoder->setComputePipelineState(pipeline);
  encoder->setArgumentTable(table);
  encoder->dispatchThreads(grid, threads);
}

void dispatch_mtl4(MTL4::CommandBuffer *command,
                   MTL::ComputePipelineState *pipeline,
                   MTL::ResidencySet *residency,
                   std::initializer_list<MTL::Buffer *> buffers, MTL::Size grid,
                   MTL::Size threads) {
  if (command == nullptr || residency == nullptr) {
    throw std::runtime_error(
        "MTL4 dispatch requires a command and residency set.");
  }
  command->useResidencySet(residency);
  MTL4::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
  if (encoder == nullptr) {
    throw std::runtime_error("Cannot create an MTL4 compute encoder.");
  }
  MTL4EncoderResources resources;
  encode_mtl4_dispatch(encoder, pipeline, residency, buffers, resources, grid,
                       threads);
  encoder->endEncoding();
}

}  // namespace

MTL4EncoderResources::~MTL4EncoderResources() {
  for (MTL4::ArgumentTable *table : tables_) {
    table->release();
  }
}

void MTL4EncoderResources::adopt(MTL4::ArgumentTable *table) {
  try {
    tables_.push_back(table);
  } catch (...) {
    table->release();
    throw;
  }
}

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
    queue4 = device->newMTL4CommandQueue();
    allocator = device->newCommandAllocator();
    if (queue4 == nullptr || allocator == nullptr) {
      throw std::runtime_error("Cannot create the MTL4 command runtime.");
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
    allocator->release();
    queue4->release();
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
  MTL4::CommandQueue *queue4 = nullptr;
  MTL4::CommandAllocator *allocator = nullptr;
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
  return submit_commands([&](MTL::CommandBuffer *command) {
    MTL::ComputeCommandEncoder *encoder = command->computeCommandEncoder();
    if (encoder == nullptr) {
      throw std::runtime_error("Cannot create a Metal compute encoder.");
    }
    encode(encoder);
    encoder->endEncoding();
  });
}

double Metal::submit_commands(
    const std::function<void(MTL::CommandBuffer *)> &encode) const {
  MTL::CommandBuffer *command = impl_->queue->commandBuffer();
  if (command == nullptr) {
    throw std::runtime_error("Cannot create a Metal command buffer.");
  }
  encode(command);
  command->commit();
  command->waitUntilCompleted();
  if (command->status() == MTL::CommandBufferStatusError) {
    throw std::runtime_error("Metal command failed: " +
                             error_text(command->error()));
  }
  return command->GPUEndTime() - command->GPUStartTime();
}

double Metal::submit_mtl4(
    const std::function<void(MTL4::CommandBuffer *)> &encode) const {
  MTL4::CommandBuffer *command = impl_->device->newCommandBuffer();
  MTL4::CommitOptions *options = MTL4::CommitOptions::alloc()->init();
  if (command == nullptr || options == nullptr) {
    if (options != nullptr)
      options->release();
    if (command != nullptr)
      command->release();
    throw std::runtime_error("Cannot allocate MTL4 command resources.");
  }

  impl_->allocator->reset();
  command->beginCommandBuffer(impl_->allocator);
  try {
    encode(command);
    command->endCommandBuffer();
  } catch (...) {
    options->release();
    command->release();
    throw;
  }

  std::mutex mutex;
  std::condition_variable condition;
  bool completed = false;
  double seconds = 0.0;
  std::string failure;
  options->addFeedbackHandler(
      MTL4::CommitFeedbackHandlerFunction([&](MTL4::CommitFeedback *feedback) {
        std::lock_guard<std::mutex> lock(mutex);
        if (feedback->error() != nullptr) {
          failure = error_text(feedback->error());
        } else {
          seconds = feedback->GPUEndTime() - feedback->GPUStartTime();
        }
        completed = true;
        condition.notify_one();
      }));
  const MTL4::CommandBuffer *commands[] = {command};
  impl_->queue4->commit(commands, 1, options);
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return completed; });
  }
  options->release();
  command->release();
  if (!failure.empty()) {
    throw std::runtime_error("MTL4 command failed: " + failure);
  }
  return seconds;
}

MTL::ResidencySet *Metal::residency_set(
    const std::vector<MTL::Buffer *> &buffers) const {
  MTL::ResidencySetDescriptor *descriptor =
      MTL::ResidencySetDescriptor::alloc()->init();
  if (descriptor == nullptr) {
    throw std::runtime_error("Cannot allocate a residency-set descriptor.");
  }
  descriptor->setInitialCapacity(buffers.size());
  NS::Error *error = nullptr;
  MTL::ResidencySet *result =
      impl_->device->newResidencySet(descriptor, &error);
  descriptor->release();
  if (result == nullptr) {
    throw std::runtime_error("Cannot create a residency set: " +
                             error_text(error));
  }
  for (MTL::Buffer *buffer : buffers) {
    if (buffer == nullptr || buffer->device() != impl_->device) {
      result->release();
      throw std::runtime_error(
          "Residency buffers must be non-null and belong to this device.");
    }
    result->addAllocation(buffer);
  }
  result->commit();
  return result;
}

MTL::Device *Metal::device() const {
  return impl_->device;
}

MTL::Library *Metal::library() const {
  return impl_->library;
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

void Metal::dispatch1d(MTL4::CommandBuffer *command,
                       MTL::ComputePipelineState *pipeline,
                       MTL::ResidencySet *residency,
                       std::initializer_list<MTL::Buffer *> buffers,
                       std::size_t elements, std::size_t threads) {
  dispatch_mtl4(command, pipeline, residency, buffers,
                MTL::Size::Make(elements, 1, 1),
                MTL::Size::Make(std::min(elements, threads), 1, 1));
}

void Metal::dispatch2d(MTL4::CommandBuffer *command,
                       MTL::ComputePipelineState *pipeline,
                       MTL::ResidencySet *residency,
                       std::initializer_list<MTL::Buffer *> buffers,
                       std::size_t columns, std::size_t rows) {
  dispatch_mtl4(command, pipeline, residency, buffers,
                MTL::Size::Make(columns, rows, 1), MTL::Size::Make(16, 16, 1));
}

void Metal::dispatch1d(MTL4::ComputeCommandEncoder *encoder,
                       MTL::ComputePipelineState *pipeline,
                       MTL::ResidencySet *residency,
                       std::initializer_list<MTL::Buffer *> buffers,
                       MTL4EncoderResources &resources, std::size_t elements,
                       std::size_t threads) {
  encode_mtl4_dispatch(encoder, pipeline, residency, buffers, resources,
                       MTL::Size::Make(elements, 1, 1),
                       MTL::Size::Make(std::min(elements, threads), 1, 1));
}

void Metal::dispatch2d(MTL4::ComputeCommandEncoder *encoder,
                       MTL::ComputePipelineState *pipeline,
                       MTL::ResidencySet *residency,
                       std::initializer_list<MTL::Buffer *> buffers,
                       MTL4EncoderResources &resources, std::size_t columns,
                       std::size_t rows) {
  encode_mtl4_dispatch(encoder, pipeline, residency, buffers, resources,
                       MTL::Size::Make(columns, rows, 1),
                       MTL::Size::Make(16, 16, 1));
}

void release_buffers(std::vector<MTL::Buffer *> &buffers) {
  for (MTL::Buffer *buffer : buffers) {
    buffer->release();
  }
  buffers.clear();
}

}  // namespace msqrt
