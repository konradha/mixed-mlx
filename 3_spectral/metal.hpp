#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace msqrt {

class MTL4EncoderResources {
public:
  MTL4EncoderResources() = default;
  ~MTL4EncoderResources();
  MTL4EncoderResources(const MTL4EncoderResources &) = delete;
  MTL4EncoderResources &operator=(const MTL4EncoderResources &) = delete;

  void adopt(MTL4::ArgumentTable *table);

private:
  std::vector<MTL4::ArgumentTable *> tables_;
};

class Metal {
public:
  explicit Metal(const std::string &library_path);
  ~Metal();
  Metal(const Metal &) = delete;
  Metal &operator=(const Metal &) = delete;

  MTL::Buffer *buffer(std::size_t bytes) const;
  MTL::ComputePipelineState *pipeline(const std::string &name);
  double
  submit(const std::function<void(MTL::ComputeCommandEncoder *)> &encode) const;
  double submit_commands(
      const std::function<void(MTL::CommandBuffer *)> &encode) const;
  double
  submit_mtl4(const std::function<void(MTL4::CommandBuffer *)> &encode) const;
  MTL::ResidencySet *
  residency_set(const std::vector<MTL::Buffer *> &buffers) const;
  MTL::Device *device() const;
  MTL::Library *library() const;
  const char *device_name() const;

  static void dispatch1d(MTL::ComputeCommandEncoder *encoder,
                         MTL::ComputePipelineState *pipeline,
                         std::size_t elements, std::size_t threads = 256);
  static void dispatch2d(MTL::ComputeCommandEncoder *encoder,
                         MTL::ComputePipelineState *pipeline,
                         std::size_t columns, std::size_t rows);
  static void dispatch1d(MTL4::CommandBuffer *command,
                         MTL::ComputePipelineState *pipeline,
                         MTL::ResidencySet *residency,
                         std::initializer_list<MTL::Buffer *> buffers,
                         std::size_t elements, std::size_t threads = 256);
  static void dispatch2d(MTL4::CommandBuffer *command,
                         MTL::ComputePipelineState *pipeline,
                         MTL::ResidencySet *residency,
                         std::initializer_list<MTL::Buffer *> buffers,
                         std::size_t columns, std::size_t rows);
  static void dispatch1d(MTL4::ComputeCommandEncoder *encoder,
                         MTL::ComputePipelineState *pipeline,
                         MTL::ResidencySet *residency,
                         std::initializer_list<MTL::Buffer *> buffers,
                         MTL4EncoderResources &resources, std::size_t elements,
                         std::size_t threads = 256);
  static void dispatch2d(MTL4::ComputeCommandEncoder *encoder,
                         MTL::ComputePipelineState *pipeline,
                         MTL::ResidencySet *residency,
                         std::initializer_list<MTL::Buffer *> buffers,
                         MTL4EncoderResources &resources, std::size_t columns,
                         std::size_t rows);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

void release_buffers(std::vector<MTL::Buffer *> &buffers);

} // namespace msqrt
