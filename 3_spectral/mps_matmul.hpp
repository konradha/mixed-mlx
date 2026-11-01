#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>
#include <memory>

namespace msqrt {

class MpsMatmul {
 public:
  explicit MpsMatmul(MTL::Buffer *buffer, std::size_t order);
  ~MpsMatmul();
  MpsMatmul(const MpsMatmul &) = delete;
  MpsMatmul &operator=(const MpsMatmul &) = delete;

  void encode(MTL::CommandBuffer *command, MTL::Buffer *left,
              MTL::Buffer *right, MTL::Buffer *output) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace msqrt
