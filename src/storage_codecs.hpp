#pragma once

#include "precision.hpp"

#include <cstddef>
#include <vector>

namespace MTL {
class Buffer;
}

namespace mixed_sqrt {

struct StorageBuffers {
  MTL::Buffer *values = nullptr;
  MTL::Buffer *scales = nullptr;
};

void checkLowPrecisionCodecs();
void copyToStorage(const PrecisionSpec &precision,
                   const std::vector<float> &source, MTL::Buffer *destination,
                   MTL::Buffer *scale_destination);
float storageValueAt(const PrecisionSpec &precision, MTL::Buffer *source,
                     MTL::Buffer *scale_source, size_t index);
size_t spectrumMetadataBytes(const PrecisionSpec &precision, size_t elements);
void copySpectrumToStorage(const PrecisionSpec &precision,
                           const std::vector<float> &source,
                           MTL::Buffer *destination,
                           MTL::Buffer *metadata_destination);
float spectrumValueAt(const PrecisionSpec &precision, MTL::Buffer *source,
                      MTL::Buffer *metadata_source, size_t index);
void unpackFromStorage(const PrecisionSpec &precision, MTL::Buffer *source,
                       MTL::Buffer *scale_source,
                       std::vector<double> &destination);
void copyResidualToStorage(const PrecisionSpec &precision,
                           const std::vector<double> &source,
                           MTL::Buffer *destination,
                           MTL::Buffer *scale_destination);
void addCorrectionFromStorage(const PrecisionSpec &precision,
                              MTL::Buffer *source, MTL::Buffer *scale_source,
                              std::vector<double> &destination);
double orthogonalityError(const PrecisionSpec &precision,
                          const StorageBuffers &basis, size_t order);

} // namespace mixed_sqrt
