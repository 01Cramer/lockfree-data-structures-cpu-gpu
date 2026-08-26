// Device spinlock used by the GPU lock-based queue variants.
//
// CUDA kernels have no futex/OS blocking primitive, so a GPU mutex would still
// be a spinlock. Failed acquisitions use __nanosleep() with exponential
// backoff, following NVIDIA's documented mutex example.

#pragma once

#include <cuda_runtime.h>

#include "gpu/shared/gpu_atomics.cuh"

namespace gpu {

namespace detail {

// One backoff step.
__device__ inline void spinlockBackoff(unsigned int nanoseconds) {
#if __CUDA_ARCH__ >= 700
  __nanosleep(nanoseconds);
#else
  // Unreachable: gpu_atomics.cuh static_asserts the CC 7.0 floor.
  (void)nanoseconds;
#endif
}

} // namespace detail

// Test-and-test-and-set, the same shape as cpu::Spinlock.
class Spinlock {
public:
  // No constructor: instances live in device memory allocated by cudaMalloc
  // and are set up by an initialization kernel (or by cudaMemset to zero,
  // which is the same state).
  __device__ void initialize() {
    DeviceAtomicRef<unsigned int> flag(m_locked);
    flag.store(0u, cuda::memory_order_relaxed);
  }

  __device__ bool tryLock() {
    DeviceAtomicRef<unsigned int> flag(m_locked);
    if (flag.load(cuda::memory_order_relaxed) != 0u) {
      return false;
    }
    return flag.exchange(1u, cuda::memory_order_acquire) == 0u;
  }

  __device__ void lock() {
    unsigned int backoffNanos = kInitialBackoffNanos;
    for (int spins = 0; !tryLock(); ++spins) {
      if (spins == kSpinBudget) {
        spins = 0;
        detail::spinlockBackoff(backoffNanos);
        backoffNanos = backoffNanos < kMaxBackoffNanos ? backoffNanos * 2u
                                                       : kMaxBackoffNanos;
      }
    }
  }

  __device__ void unlock() {
    DeviceAtomicRef<unsigned int> flag(m_locked);
    flag.store(0u, cuda::memory_order_release);
  }

  static constexpr int kSpinBudget = 8;
  static constexpr unsigned int kInitialBackoffNanos = 32u;
  static constexpr unsigned int kMaxBackoffNanos = 1024u;

private:
  unsigned int m_locked;
};

// Scope guard for device critical sections.
class LockGuard {
public:
  __device__ explicit LockGuard(Spinlock &lock) : m_lock(lock) {
    m_lock.lock();
  }

  __device__ ~LockGuard() { m_lock.unlock(); }

  LockGuard(const LockGuard &) = delete;
  LockGuard &operator=(const LockGuard &) = delete;

private:
  Spinlock &m_lock;
};

} // namespace gpu
