#pragma once

#include <Metal/Metal.hpp>

#include <cstddef>

namespace msqrt {

void validate_mx_tensor_layout(MTL::Buffer *values, MTL::Buffer *scales,
                               std::size_t elements, bool fp4);

}  // namespace msqrt
