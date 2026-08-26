// Host-side CUDA utilities shared by the tests and the benchmark harness:
// checked-call macros, checked kernel-launch synchronization, and an owning
// device allocation.

#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace gpu {

namespace detail {

inline void cudaFail(cudaError_t status, const char *expr, const char *file,
                     int line) {
  std::fprintf(stderr, "%s:%d: CUDA error in `%s`: %s (%s)\n", file, line, expr,
               cudaGetErrorName(status), cudaGetErrorString(status));
  std::abort();
}

inline void cudaCheckImpl(cudaError_t status, const char *expr,
                          const char *file, int line) {
  if (status != cudaSuccess) {
    cudaFail(status, expr, file, line);
  }
}

} // namespace detail

#define GPU_CUDA_CHECK(expr)                                                   \
  ::gpu::detail::cudaCheckImpl((expr), #expr, __FILE__, __LINE__)

// Use immediately after every <<<>>> launch.
#define GPU_CUDA_CHECK_KERNEL()                                                \
  do {                                                                         \
    ::gpu::detail::cudaCheckImpl(cudaGetLastError(), "kernel launch",          \
                                 __FILE__, __LINE__);                          \
    ::gpu::detail::cudaCheckImpl(cudaDeviceSynchronize(),                      \
                                 "kernel execution", __FILE__, __LINE__);      \
  } while (0)

// Check once before running tests or benchmarks that can exercise spinlocks.
inline void requireVoltaOrNewer(FILE *stream, bool detailedDeviceLine) {
  int deviceCount = 0;
  GPU_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::fprintf(stderr, "No CUDA device found.\n");
    std::exit(1);
  }

  int device = 0;
  GPU_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  GPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  if (detailedDeviceLine) {
    int clockRateKHz = 0;
    GPU_CUDA_CHECK(
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device));
    std::fprintf(stream,
                 "device: %s, compute %d.%d, %d SMs, %.1f GB, "
                 "L2 %d KB, clock %.0f MHz\n",
                 properties.name, properties.major, properties.minor,
                 properties.multiProcessorCount,
                 static_cast<double>(properties.totalGlobalMem) / (1 << 30),
                 properties.l2CacheSize / 1024,
                 static_cast<double>(clockRateKHz) / 1000.0);
  } else {
    std::fprintf(stream, "device: %s (compute %d.%d, %d SMs)\n",
                 properties.name, properties.major, properties.minor,
                 properties.multiProcessorCount);
  }

  if (properties.major < 7) {
    std::fprintf(stderr,
                 "This device is compute %d.%d. The lock-based variants need "
                 "independent thread scheduling (compute 7.0, Volta): below "
                 "that, the threads of a warp share a program counter and a "
                 "lane holding a spinlock cannot reach the release while its "
                 "peers spin. The kernels would deadlock, not run slowly.\n",
                 properties.major, properties.minor);
    std::exit(1);
  }
}

// Small owning device allocation used by tests and benchmarks.
template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : m_data(other.m_data), m_count(other.m_count) {
    other.m_data = nullptr;
    other.m_count = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      release();
      m_data = other.m_data;
      m_count = other.m_count;
      other.m_data = nullptr;
      other.m_count = 0;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    release();
    if (count == 0) {
      return;
    }
    GPU_CUDA_CHECK(cudaMalloc(&m_data, count * sizeof(T)));
    m_count = count;
  }

  void release() {
    if (m_data != nullptr) {
      GPU_CUDA_CHECK(cudaFree(m_data));
      m_data = nullptr;
      m_count = 0;
    }
  }

  void zero() {
    if (m_count != 0) {
      GPU_CUDA_CHECK(cudaMemset(m_data, 0, m_count * sizeof(T)));
    }
  }

  void copyFromHost(const T *source, std::size_t count) {
    GPU_CUDA_CHECK(cudaMemcpy(m_data, source, count * sizeof(T),
                              cudaMemcpyHostToDevice));
  }

  void copyToHost(T *destination, std::size_t count) const {
    GPU_CUDA_CHECK(cudaMemcpy(destination, m_data, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
  }

  T *get() const { return m_data; }
  std::size_t count() const { return m_count; }
  std::size_t bytes() const { return m_count * sizeof(T); }

private:
  T *m_data = nullptr;
  std::size_t m_count = 0;
};

} // namespace gpu
