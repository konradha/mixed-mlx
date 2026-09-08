#include "lowbit_matmul.hpp"
#include "metal.hpp"
#include "mps_matmul.hpp"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
template <class T> class Owned {
  T *p_ = nullptr;

public:
  Owned() = default;
  explicit Owned(T *p) : p_(p) {}
  ~Owned() {
    if (p_)
      p_->release();
  }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;
  Owned(Owned &&o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
  Owned &operator=(Owned &&o) noexcept {
    if (this != &o) {
      if (p_)
        p_->release();
      p_ = std::exchange(o.p_, nullptr);
    }
    return *this;
  }
  T *get() const { return p_; }
  explicit operator bool() const { return p_; }
};
std::string err(NS::Error *e) {
  return e && e->localizedDescription()
             ? e->localizedDescription()->utf8String()
             : "unknown Metal error";
}
struct Opt {
  size_t n = 128, warm = 50, eval = 100;
  std::string out = "artifacts/products.csv";
};
Opt parse(int ac, char **av) {
  Opt o;
  for (int i = 1; i < ac; ++i) {
    std::string a = av[i];
    auto v = [&](const char *n) {
      if (++i >= ac)
        throw std::runtime_error(std::string("Missing value for ") + n + '.');
      return std::string(av[i]);
    };
    if (a == "--order")
      o.n = std::stoull(v("--order"));
    else if (a == "--warmups")
      o.warm = std::stoull(v("--warmups"));
    else if (a == "--evaluations")
      o.eval = std::stoull(v("--evaluations"));
    else if (a == "--output")
      o.out = v("--output");
    else if (a == "--quick") {
      o.n = 128;
      o.warm = 1;
      o.eval = 3;
    } else
      throw std::runtime_error("Unknown option: " + a);
  }
  if (!o.n || !o.eval || o.n > UINT32_MAX)
    throw std::runtime_error("Invalid order or evaluation count.");
  return o;
}
struct Row {
  std::string backend, format, tile, reason, order, evaluation, conversion,
      kernel, paired_mps, paired_speedup, arithmetic_error, forward_error;
  bool supported;
};
std::string num(double x, bool sci = false) {
  std::ostringstream s;
  s << (sci ? std::scientific : std::fixed) << std::setprecision(6) << x;
  return s.str();
}
Row unsupported(std::string backend, std::string format, std::string tile,
                std::string reason, std::size_t order) {
  return {std::move(backend),
          std::move(format),
          std::move(tile),
          std::move(reason),
          std::to_string(order),
          "",
          "",
          "",
          "",
          "",
          "",
          "",
          false};
}

class Runtime {
  MTL::Device *d_ = nullptr;
  MTL::CommandQueue *q_ = nullptr;
  MTL4::CommandQueue *q4_ = nullptr;
  MTL::Library *l_ = nullptr;
  std::vector<std::pair<std::string, MTL::ComputePipelineState *>> ps_;

public:
  Runtime() {
    d_ = MTL::CreateSystemDefaultDevice();
    if (!d_)
      throw std::runtime_error("No Metal device is available.");
    q_ = d_->newCommandQueue();
    q4_ = d_->newMTL4CommandQueue();
    NS::Error *e = nullptr;
    l_ = d_->newLibrary(NS::String::string("build/3_spectral/products.metallib",
                                           NS::UTF8StringEncoding),
                        &e);
    if (!q_ || !q4_ || !l_)
      throw std::runtime_error(l_ ? "Cannot create Metal queues." : err(e));
  }
  ~Runtime() {
    for (auto &p : ps_)
      p.second->release();
    if (l_)
      l_->release();
    if (q4_)
      q4_->release();
    if (q_)
      q_->release();
    if (d_)
      d_->release();
  }
  MTL::Device *device() { return d_; }
  MTL::Library *library() { return l_; }
  const char *name() { return d_->name()->utf8String(); }
  Owned<MTL::Buffer> buffer(size_t n) {
    Owned<MTL::Buffer> b(d_->newBuffer(n, MTL::ResourceStorageModeShared));
    if (!b)
      throw std::runtime_error("Cannot allocate Metal buffer.");
    std::memset(b.get()->contents(), 0, n);
    return b;
  }
  MTL::ComputePipelineState *pipeline(const std::string &n) {
    for (auto &p : ps_)
      if (p.first == n)
        return p.second;
    Owned<MTL::Function> f(
        l_->newFunction(NS::String::string(n.c_str(), NS::UTF8StringEncoding)));
    if (!f)
      throw std::runtime_error("Library has no " + n + " function.");
    NS::Error *e = nullptr;
    auto *p = d_->newComputePipelineState(f.get(), &e);
    if (!p)
      throw std::runtime_error(err(e));
    ps_.push_back({n, p});
    return p;
  }
  double legacy(const std::function<void(MTL::ComputeCommandEncoder *)> &fn,
                std::string &failure) {
    auto *c = q_->commandBuffer();
    auto *e = c->computeCommandEncoder();
    if (!e) {
      failure = "Cannot create Metal encoder.";
      return 0;
    }
    fn(e);
    e->endEncoding();
    c->commit();
    c->waitUntilCompleted();
    if (c->status() == MTL::CommandBufferStatusError) {
      failure = err(c->error());
      return 0;
    }
    return 1e3 * (c->GPUEndTime() - c->GPUStartTime());
  }
  double mps(const std::function<void(MTL::CommandBuffer *)> &fn,
             std::string &failure) {
    auto *c = q_->commandBuffer();
    fn(c);
    c->commit();
    c->waitUntilCompleted();
    if (c->status() == MTL::CommandBufferStatusError) {
      failure = err(c->error());
      return 0;
    }
    return 1e3 * (c->GPUEndTime() - c->GPUStartTime());
  }
  double mtl4(
      const std::function<msqrt::LowbitMatmulResult(MTL4::CommandBuffer *)> &fn,
      std::string &failure) {
    Owned<MTL4::CommandAllocator> a(d_->newCommandAllocator());
    Owned<MTL4::CommandBuffer> c(d_->newCommandBuffer());
    Owned<MTL4::CommitOptions> o(MTL4::CommitOptions::alloc()->init());
    if (!a || !c || !o) {
      failure = "Cannot allocate MTL4 command resources.";
      return 0;
    }
    c.get()->beginCommandBuffer(a.get());
    auto r = fn(c.get());
    c.get()->endCommandBuffer();
    if (!r.encoded()) {
      failure = r.message;
      return 0;
    }
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    double ms = 0;
    o.get()->addFeedbackHandler(
        MTL4::CommitFeedbackHandlerFunction([&](MTL4::CommitFeedback *f) {
          std::lock_guard<std::mutex> g(m);
          if (f->error())
            failure = err(f->error());
          else
            ms = 1e3 * (f->GPUEndTime() - f->GPUStartTime());
          done = true;
          cv.notify_one();
        }));
    const MTL4::CommandBuffer *cs[] = {c.get()};
    q4_->commit(cs, 1, o.get());
    std::unique_lock<std::mutex> g(m);
    cv.wait(g, [&] { return done; });
    return ms;
  }
};
struct Fmt {
  msqrt::LowbitDataType type;
  const char *name, *enc, *dec;
  size_t bits;
  bool mx;
};
const Fmt FP16{msqrt::LowbitDataType::Float16,
               "FP16",
               "encode_matrix_fp16",
               "decode_fp16",
               16,
               false};
const Fmt E4{msqrt::LowbitDataType::Float8E4M3,
             "E4M3",
             "encode_matrix_fp8_e4m3",
             "decode_fp8_e4m3",
             8,
             false};
const Fmt E5{msqrt::LowbitDataType::Float8E5M2,
             "E5M2",
             "encode_matrix_fp8_e5m2",
             "decode_fp8_e5m2",
             8,
             false};
const Fmt MX8{msqrt::LowbitDataType::MxFloat8E4M3,
              "MXFP8",
              "encode_matrix_mxfp8_e4m3",
              "decode_mxfp8_e4m3",
              8,
              true};
const Fmt MX4{msqrt::LowbitDataType::MxFloat4E2M1,
              "MXFP4",
              "encode_matrix_mxfp4_e2m1",
              "decode_mxfp4_e2m1",
              4,
              true};
struct Params {
  uint32_t elements, blocks;
};
struct Rep {
  Owned<MTL::Buffer> a, b, as, bs;
  std::vector<float> qa, qb;
  double conversion = 0;
};
std::vector<float> logical(const float *p, size_t n,
                           const msqrt::LowbitMatmulOperandLayout &l) {
  std::vector<float> x(n * n);
  for (size_t c = 0; c < n; ++c)
    for (size_t r = 0; r < n; ++r) {
      size_t i = l.storage == msqrt::LowbitMatrixStorage::ColumnMajor
                     ? r + c * l.data_leading_dimension_elements
                     : c + r * l.data_leading_dimension_elements;
      x[r + c * n] = p[i];
    }
  return x;
}
void encode_matrix(MTL::ComputeCommandEncoder *encoder,
                   MTL::ComputePipelineState *pipeline, MTL::Buffer *source,
                   MTL::Buffer *encoded, MTL::Buffer *scales, size_t n,
                   const msqrt::LowbitMatmulOperandLayout &layout,
                   const Fmt &format) {
  const msqrt::MatrixCodecParams params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(layout.data_leading_dimension_elements),
      static_cast<uint32_t>(layout.scale_leading_dimension_bytes),
      static_cast<uint32_t>(layout.storage ==
                            msqrt::LowbitMatrixStorage::TransposedColumnMajor)};
  encoder->setBuffer(source, 0, 0);
  encoder->setBuffer(encoded, 0, 1);
  if (format.mx)
    encoder->setBuffer(scales, 0, 2);
  encoder->setBytes(&params, sizeof(params), format.mx ? 3 : 2);
  const size_t threads = format.mx
                             ? n * n
                             : n * layout.data_leading_dimension_elements /
                                   (format.bits == 16 ? 1 : 4);
  msqrt::Metal::dispatch1d(encoder, pipeline, threads, format.mx ? 32 : 256);
}

void decode_reference(MTL::ComputeCommandEncoder *encoder,
                      MTL::ComputePipelineState *pipeline, MTL::Buffer *encoded,
                      MTL::Buffer *decoded, MTL::Buffer *scales, size_t n,
                      const msqrt::LowbitMatmulOperandLayout &layout,
                      const Fmt &format) {
  const size_t elements =
      format.mx ? n : n * layout.data_leading_dimension_elements;
  const Params params{static_cast<uint32_t>(elements),
                      static_cast<uint32_t>((elements + 31) / 32)};
  const size_t width = format.bits == 16 ? 1 : format.bits == 8 ? 4 : 8;
  encoder->setBytes(&params, sizeof(params), format.mx ? 3 : 2);
  if (format.mx) {
    const size_t data_row =
        layout.data_leading_dimension_elements * format.bits / 8;
    const size_t float_row =
        layout.data_leading_dimension_elements * sizeof(float);
    for (size_t row = 0; row < n; ++row) {
      encoder->setBuffer(encoded, row * data_row, 0);
      encoder->setBuffer(scales, row * layout.scale_leading_dimension_bytes, 1);
      encoder->setBuffer(decoded, row * float_row, 2);
      msqrt::Metal::dispatch1d(encoder, pipeline,
                               (elements + width - 1) / width);
    }
  } else {
    encoder->setBuffer(encoded, 0, 0);
    encoder->setBuffer(decoded, 0, 1);
    msqrt::Metal::dispatch1d(encoder, pipeline, (elements + width - 1) / width);
  }
}

Rep prepare(Runtime &rt, const Fmt &format, size_t n, MTL::Buffer *a,
            MTL::Buffer *b, const msqrt::LowbitMatmulOperandLayout &al,
            const msqrt::LowbitMatmulOperandLayout &bl) {
  const size_t ae = al.data_leading_dimension_elements * n;
  const size_t be = bl.data_leading_dimension_elements * n;
  if (ae > UINT32_MAX || be > UINT32_MAX)
    throw std::runtime_error("Codec element count exceeds uint32_t.");
  const auto start = Clock::now();
  Rep result;
  result.a = rt.buffer(al.data_bytes);
  result.b = rt.buffer(bl.data_bytes);
  if (format.mx) {
    result.as = rt.buffer(al.scale_bytes);
    result.bs = rt.buffer(bl.scale_bytes);
  }
  std::string failure;
  auto *encode_pipeline = rt.pipeline(format.enc);
  rt.legacy(
      [&](auto *encoder) {
        encode_matrix(encoder, encode_pipeline, a, result.a.get(),
                      result.as.get(), n, al, format);
        encode_matrix(encoder, encode_pipeline, b, result.b.get(),
                      result.bs.get(), n, bl, format);
      },
      failure);
  if (!failure.empty())
    throw std::runtime_error(failure);
  result.conversion =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();

  auto da = rt.buffer(ae * sizeof(float));
  auto db = rt.buffer(be * sizeof(float));
  auto *decode_pipeline = rt.pipeline(format.dec);
  rt.legacy(
      [&](auto *encoder) {
        decode_reference(encoder, decode_pipeline, result.a.get(), da.get(),
                         result.as.get(), n, al, format);
        decode_reference(encoder, decode_pipeline, result.b.get(), db.get(),
                         result.bs.get(), n, bl, format);
      },
      failure);
  if (!failure.empty())
    throw std::runtime_error(failure);
  result.qa = logical(static_cast<const float *>(da.get()->contents()), n, al);
  result.qb = logical(static_cast<const float *>(db.get()->contents()), n, bl);
  return result;
}
std::vector<float> product(const std::vector<float> &a,
                           const std::vector<float> &b, size_t n) {
  std::vector<float> c(n * n);
  for (size_t j = 0; j < n; ++j)
    for (size_t i = 0; i < n; ++i) {
      double s = 0;
      for (size_t k = 0; k < n; ++k)
        s += (double)a[i + k * n] * b[k + j * n];
      c[i + j * n] = (float)s;
    }
  return c;
}
double relative_error(const float *x, const std::vector<float> &reference) {
  double difference = 0;
  double scale = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    double delta = x[i] - reference[i];
    difference += delta * delta;
    scale += static_cast<double>(reference[i]) * reference[i];
  }
  return std::sqrt(difference /
                   std::max(scale, std::numeric_limits<double>::min()));
}
Owned<MTL::ResidencySet> residency(Runtime &rt,
                                   const std::vector<MTL::Buffer *> &bs) {
  Owned<MTL::ResidencySetDescriptor> d(
      MTL::ResidencySetDescriptor::alloc()->init());
  d.get()->setInitialCapacity(bs.size());
  NS::Error *e = nullptr;
  Owned<MTL::ResidencySet> r(rt.device()->newResidencySet(d.get(), &e));
  if (!r)
    throw std::runtime_error(err(e));
  for (auto *b : bs)
    r.get()->addAllocation(b);
  r.get()->commit();
  return r;
}
const char *mpp_tile(msqrt::LowbitMatmulTile tile) {
  switch (tile) {
  case msqrt::LowbitMatmulTile::Rows32Columns32:
    return "32x32";
  case msqrt::LowbitMatmulTile::Rows64Columns32:
    return "64x32";
  case msqrt::LowbitMatmulTile::Rows64Columns64:
    return "64x64";
  }
  throw std::logic_error("Unknown MPP tile configuration.");
}

struct BenchmarkCase {
  std::string backend, format, tile;
  double conversion_ms;
  Owned<MTL::Buffer> output;
  Owned<MTL::ResidencySet> residency;
  std::shared_ptr<const std::vector<float>> arithmetic_reference;
  std::shared_ptr<const std::vector<float>> forward_reference;
  std::function<double(std::string &)> execute;
};

struct Errors {
  double arithmetic;
  double forward;
};

struct Sample {
  double kernel_ms;
  double arithmetic_error;
  double forward_error;
};

double execute_one(BenchmarkCase &benchmark) {
  std::string failure;
  double kernel_ms = benchmark.execute(failure);
  if (!failure.empty())
    throw std::runtime_error(
        benchmark.backend + " " + benchmark.format +
        (benchmark.tile.empty() ? "" : " " + benchmark.tile) + ": " + failure);
  if (!(kernel_ms > 0) || !std::isfinite(kernel_ms))
    throw std::runtime_error(
        benchmark.backend + " " + benchmark.format +
        (benchmark.tile.empty() ? "" : " " + benchmark.tile) +
        " produced an invalid timing.");
  return kernel_ms;
}

Errors check_output(const BenchmarkCase &benchmark) {
  const auto *output =
      static_cast<const float *>(benchmark.output.get()->contents());
  double arithmetic = relative_error(output, *benchmark.arithmetic_reference);
  double forward = relative_error(output, *benchmark.forward_reference);
  if (!std::isfinite(arithmetic) || !std::isfinite(forward))
    throw std::runtime_error(
        benchmark.backend + " " + benchmark.format +
        (benchmark.tile.empty() ? "" : " " + benchmark.tile) +
        " produced an invalid output.");
  return {arithmetic, forward};
}

std::string csv(std::string s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos)
    return s;
  std::string result = "\"";
  for (char c : s) {
    result += c;
    if (c == '"')
      result += '"';
  }
  return result + '"';
}

void output(const Opt &o, const std::vector<Row> &rows) {
  std::filesystem::path path(o.out);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file)
    throw std::runtime_error("Cannot open CSV: " + o.out);
  file << "backend,format,tile,supported,reason,order,evaluation,"
          "conversion_ms,kernel_ms,paired_mps_ms,paired_speedup,"
          "arithmetic_error,forward_error\n";
  for (const auto &row : rows)
    file << csv(row.backend) << ',' << csv(row.format) << ',' << csv(row.tile)
         << ',' << (row.supported ? "true" : "false") << ',' << csv(row.reason)
         << ',' << row.order << ',' << row.evaluation << ',' << row.conversion
         << ',' << row.kernel << ',' << row.paired_mps << ','
         << row.paired_speedup << ',' << row.arithmetic_error << ','
         << row.forward_error << '\n';

  std::cout << "backend  format  tile   supported  conversion_ms  kernel_ms  "
               "paired_speedup  arithmetic_error  forward_error  reason\n";
  for (size_t i = 0; i < rows.size();) {
    size_t j = i + 1;
    while (j < rows.size() && rows[j].backend == rows[i].backend &&
           rows[j].format == rows[i].format && rows[j].tile == rows[i].tile)
      ++j;
    const auto &row = rows[i];
    std::string kernel, speedup, arithmetic, forward;
    if (row.supported) {
      std::vector<double> kernels, speedups;
      for (size_t z = i; z < j; ++z) {
        kernels.push_back(std::stod(rows[z].kernel));
        speedups.push_back(std::stod(rows[z].paired_speedup));
      }
      std::sort(kernels.begin(), kernels.end());
      std::sort(speedups.begin(), speedups.end());
      kernel = num(kernels[kernels.size() / 2]);
      speedup = num(speedups[speedups.size() / 2]);
      arithmetic = row.arithmetic_error;
      forward = row.forward_error;
    }
    std::cout << std::setw(7) << row.backend << "  " << std::setw(6)
              << row.format << "  " << std::setw(5) << row.tile << "  "
              << std::setw(9) << (row.supported ? "yes" : "no") << "  "
              << std::setw(13) << row.conversion << "  " << std::setw(9)
              << kernel << "  " << std::setw(14) << speedup << "  "
              << std::setw(16) << arithmetic << "  " << std::setw(13) << forward
              << "  " << row.reason << '\n';
    i = j;
  }
  std::cout
      << "arithmetic_error = ||C - Q(A)Q(B)||_F / ||Q(A)Q(B)||_F\n"
      << "forward_error = ||C - A_fp32 B_fp32||_F / ||A_fp32 B_fp32||_F\n";
}
} // namespace

int main(int ac, char **av) {
  auto *pool = NS::AutoreleasePool::alloc()->init();
  int rc = 0;
  try {
    Opt o = parse(ac, av);
    Runtime rt;
    std::vector<float> a(o.n * o.n), b(o.n * o.n);
    for (size_t c = 0; c < o.n; ++c)
      for (size_t r = 0; r < o.n; ++r) {
        size_t i = r + c * o.n;
        a[i] = .25f * std::sin(.03125f * (i + 1)) +
               .125f * std::cos(.0625f * (r + 3));
        b[i] = .25f * std::cos(.02734375f * (i + 5)) -
               .125f * std::sin(.046875f * (c + 1));
      }
    auto fa = rt.buffer(a.size() * 4), fb = rt.buffer(b.size() * 4);
    std::memcpy(fa.get()->contents(), a.data(), a.size() * 4);
    std::memcpy(fb.get()->contents(), b.data(), b.size() * 4);

    constexpr msqrt::LowbitMatmulTile tiles[] = {
        msqrt::LowbitMatmulTile::Rows32Columns32,
        msqrt::LowbitMatmulTile::Rows64Columns32,
        msqrt::LowbitMatmulTile::Rows64Columns64,
    };
    auto candidates = [&](msqrt::LowbitDataType type) {
      std::vector<std::unique_ptr<msqrt::LowbitMatmul>> result;
      result.reserve(3);
      for (auto tile : tiles)
        result.push_back(std::make_unique<msqrt::LowbitMatmul>(
            rt.device(), rt.library(), o.n, type, tile));
      return result;
    };
    auto m16 = candidates(FP16.type);
    auto m4 = candidates(E4.type);
    auto m5 = candidates(E5.type);
    auto mx8 = candidates(MX8.type);
    auto mx4 = candidates(MX4.type);

    auto forward_reference =
        std::make_shared<const std::vector<float>>(product(a, b, o.n));
    std::vector<BenchmarkCase> cases;
    std::vector<Row> unsupported_rows;

    const size_t mps_index = cases.size();

    auto mps_output = rt.buffer(o.n * o.n * sizeof(float));
    auto *mps_output_buffer = mps_output.get();
    auto mps = std::make_shared<msqrt::MpsMatmul>(mps_output_buffer, o.n);
    cases.push_back(
        {"MPS",
         "FP32",
         "",
         0,
         std::move(mps_output),
         {},
         forward_reference,
         forward_reference,
         [&rt, mps, left = fa.get(), right = fb.get(),
          mps_output_buffer](std::string &failure) {
           return rt.mps(
               [mps, left, right, mps_output_buffer](auto *command_buffer) {
                 mps->encode(command_buffer, left, right, mps_output_buffer);
               },
               failure);
         }});

    auto any_available = [](const auto &matmuls) {
      return std::any_of(matmuls.begin(), matmuls.end(),
                         [](const auto &matmul) {
                           return matmul->availability().status ==
                                  msqrt::LowbitMatmulStatus::Available;
                         });
    };
    auto prepare_rep = [&](const Fmt &format, auto &matmuls, bool needed,
                           std::string &failure) -> std::shared_ptr<Rep> {
      if (!needed)
        return {};
      auto layout =
          std::find_if(matmuls.begin(), matmuls.end(), [](const auto &matmul) {
            return matmul->availability().status ==
                   msqrt::LowbitMatmulStatus::Available;
          });
      if (layout == matmuls.end())
        layout = matmuls.begin();
      try {
        return std::make_shared<Rep>(prepare(rt, format, o.n, fa.get(),
                                             fb.get(), (*layout)->left_layout(),
                                             (*layout)->right_layout()));
      } catch (const std::exception &e) {
        failure = e.what();
        return {};
      }
    };
    auto add_mpp = [&](const Fmt &format,
                       std::unique_ptr<msqrt::LowbitMatmul> &matmul,
                       const std::shared_ptr<Rep> &rep,
                       const std::shared_ptr<const std::vector<float>>
                           &arithmetic_reference,
                       const std::string &preparation_failure) {
      auto availability = matmul->availability();
      std::string tile = mpp_tile(matmul->tile());
      if (availability.status != msqrt::LowbitMatmulStatus::Available) {
        unsupported_rows.push_back(
            unsupported("MPP", format.name, tile, availability.message, o.n));
        return;
      }
      if (!rep) {
        unsupported_rows.push_back(
            unsupported("MPP", format.name, tile, preparation_failure, o.n));
        return;
      }

      auto output_buffer = rt.buffer(matmul->output_bytes());
      std::vector<MTL::Buffer *> resident_buffers{rep->a.get(), rep->b.get(),
                                                  output_buffer.get()};
      if (format.mx) {
        resident_buffers.push_back(rep->as.get());
        resident_buffers.push_back(rep->bs.get());
      }
      auto resident_set = residency(rt, resident_buffers);
      auto *output = output_buffer.get();
      auto *residency_set = resident_set.get();
      auto *operation = matmul.get();
      cases.push_back(
          {"MPP", format.name, std::move(tile), rep->conversion,
           std::move(output_buffer), std::move(resident_set),
           arithmetic_reference, forward_reference,
           [&rt, operation, rep, output, residency_set](std::string &failure) {
             msqrt::LowbitMatmulBuffers buffers{rep->a.get(),  rep->b.get(),
                                                rep->as.get(), rep->bs.get(),
                                                output,        residency_set};
             return rt.mtl4(
                 [operation, &buffers](auto *command_buffer) {
                   return operation->encode(command_buffer, buffers);
                 },
                 failure);
           }});
    };
    auto add_format = [&](const Fmt &format, auto &matmuls) {
      std::string preparation_failure;
      auto rep = prepare_rep(format, matmuls, any_available(matmuls),
                             preparation_failure);
      std::shared_ptr<const std::vector<float>> arithmetic_reference;
      if (rep)
        arithmetic_reference = std::make_shared<const std::vector<float>>(
            product(rep->qa, rep->qb, o.n));
      for (auto &matmul : matmuls)
        add_mpp(format, matmul, rep, arithmetic_reference, preparation_failure);
    };

    add_format(FP16, m16);

    add_format(E4, m4);
    add_format(E5, m5);
    add_format(MX8, mx8);
    add_format(MX4, mx4);

    for (size_t round = 0; round < o.warm; ++round) {
      for (size_t offset = 0; offset < cases.size(); ++offset)
        execute_one(cases[(round + offset) % cases.size()]);
      for (const auto &benchmark : cases)
        check_output(benchmark);
    }

    std::vector<std::vector<Sample>> samples(cases.size());
    for (auto &backend_samples : samples)
      backend_samples.reserve(o.eval);
    std::vector<double> round_kernel_ms(cases.size());
    for (size_t evaluation = 0; evaluation < o.eval; ++evaluation) {
      for (size_t offset = 0; offset < cases.size(); ++offset) {
        size_t index = (o.warm + evaluation + offset) % cases.size();
        round_kernel_ms[index] = execute_one(cases[index]);
      }
      for (size_t index = 0; index < cases.size(); ++index) {
        auto errors = check_output(cases[index]);
        samples[index].push_back(
            {round_kernel_ms[index], errors.arithmetic, errors.forward});
      }
    }

    std::vector<Row> rows;
    rows.reserve(cases.size() * o.eval + unsupported_rows.size());
    for (size_t index = 0; index < cases.size(); ++index)
      for (size_t evaluation = 0; evaluation < o.eval; ++evaluation) {
        const auto &benchmark = cases[index];
        const auto &sample = samples[index][evaluation];
        double paired_mps_ms = samples[mps_index][evaluation].kernel_ms;
        rows.push_back({benchmark.backend, benchmark.format, benchmark.tile, "",
                        std::to_string(o.n), std::to_string(evaluation),
                        num(benchmark.conversion_ms), num(sample.kernel_ms),
                        num(paired_mps_ms),
                        num(paired_mps_ms / sample.kernel_ms),
                        num(sample.arithmetic_error, true),
                        num(sample.forward_error, true), true});
      }
    rows.insert(rows.end(), std::make_move_iterator(unsupported_rows.begin()),
                std::make_move_iterator(unsupported_rows.end()));

    std::cout << "Device: " << rt.name() << "\nOrder: " << o.n
              << ", warmups: " << o.warm << ", evaluations: " << o.eval << '\n';
    output(o, rows);
    std::cout << "CSV: " << o.out << '\n';
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    rc = 1;
  }
  pool->release();
  return rc;
}
