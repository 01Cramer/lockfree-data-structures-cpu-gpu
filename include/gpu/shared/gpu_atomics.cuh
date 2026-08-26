// Atomic aliases and layout constants shared by the GPU data structures.

#pragma once

#if !__has_include(<cuda/atomic>)
#error "The GPU targets need libcu++ (<cuda/atomic>), shipped with the CUDA \
Toolkit since 10.2. Check that the CUDA include directory is on the path."
#endif

#include <cuda/atomic>

namespace gpu {

// Compute capability 7.0+ is required. The lock-based variants rely on
// independent thread scheduling, and gpu_spinlock.cuh uses __nanosleep().
#ifdef __CUDA_ARCH__
static_assert(__CUDA_ARCH__ >= 700,
              "gpu:: requires compute capability 7.0 or newer (independent "
              "thread scheduling). Build with -DCMAKE_CUDA_ARCHITECTURES=70 "
              "or higher.");
#endif

// All shared state in a global-memory data structure is device-scoped.
template <typename T>
using DeviceAtomicRef = cuda::atomic_ref<T, cuda::thread_scope_device>;

// Alignment used to keep independently contended synchronization words from
// accidentally sharing one NVIDIA L2 cache line.
inline constexpr int kDeviceCacheLineBytes = 128;

} // namespace gpu
