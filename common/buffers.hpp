#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>
#include <vector>

namespace msqrt::common {

template <typename Metal>
class BufferGuard {
 public:
  explicit BufferGuard(Metal &metal) : metal_(metal) {}
  ~BufferGuard() {
    for (MTL::Buffer *buffer : buffers_) {
      buffer->release();
    }
  }
  BufferGuard(const BufferGuard &) = delete;
  BufferGuard &operator=(const BufferGuard &) = delete;

  MTL::Buffer *make(std::size_t bytes) {
    MTL::Buffer *buffer = metal_.buffer(bytes);
    buffers_.push_back(buffer);
    return buffer;
  }

  const std::vector<MTL::Buffer *> &buffers() const { return buffers_; }

 private:
  Metal &metal_;
  std::vector<MTL::Buffer *> buffers_;
};

}  // namespace msqrt::common
