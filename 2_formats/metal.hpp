#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace msqrt {

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
  const char *device_name() const;

  static void dispatch1d(MTL::ComputeCommandEncoder *encoder,
                         MTL::ComputePipelineState *pipeline,
                         std::size_t elements, std::size_t threads = 256);
  static void dispatch2d(MTL::ComputeCommandEncoder *encoder,
                         MTL::ComputePipelineState *pipeline,
                         std::size_t columns, std::size_t rows);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

void release_buffers(std::vector<MTL::Buffer *> &buffers);

} // namespace msqrt
