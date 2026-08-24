// Host-side CUDA plumbing shared by the tests and the benchmark harness:
// a checked-call macro, a checked kernel-launch macro, and an owning device
// allocation.
//
// The macros abort rather than propagate. A CUDA error in this project is
// always a bug in the harness or an exhausted device, never a condition to
// recover from, and a silently ignored cudaMalloc failure would produce a
// complete-looking sweep of numbers measured on a null pointer.
//
// GPU_CUDA_CHECK_KERNEL exists separately because kernel launches do not
// return an error code. Two calls are needed: cudaGetLastError() catches an
// invalid launch configuration, and cudaDeviceSynchronize() catches a fault
// raised while the kernel ran. Checking only the first is the usual way an
// out-of-bounds write goes unnoticed until it corrupts an unrelated result.

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

// Owning device allocation. Deliberately not a general-purpose container: it
// has exactly the operations the harness uses, so there is no copy-assignment
// path that could silently share a device pointer between two owners.
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
