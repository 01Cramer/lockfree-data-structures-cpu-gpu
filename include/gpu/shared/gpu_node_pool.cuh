// GPU node pool used by the queue variants.
//
// Nodes are allocated once before a run and are never reclaimed. Each thread
// gets a private slice and advances a local cursor, so allocation does not add
// a shared atomic operation to the measured queue path.
//
// Nodes are addressed by 32-bit indices into one flat device array. If a slice
// is exhausted, the allocator sets a sticky overflow flag that the host checks
// after the kernel.

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/shared/gpu_atomics.cuh"
#include "gpu/shared/gpu_cuda_utils.cuh"

namespace gpu {

// A node is addressed by its position in the pool array. -1 is the null link.
using NodeIndex = int;

inline constexpr NodeIndex kNullIndex = -1;
inline constexpr NodeIndex kSentinelIndex = 0;

// Stored as plain data so the pool can be zero-filled from the host. Queue
// variants wrap fields in DeviceAtomicRef when an operation needs atomic access.
template <typename T> struct Node {
  T value;
  NodeIndex next;
};

// Device-side handle to the pool.
//
// Array layout:
//   [0]                                        the sentinel
//   [1, reserved)                              prefill arena, drawn from by a
//                                              single thread before the run
//   [reserved + t*nodesPerThread, +n)          thread t's private slice
template <typename T> struct PoolView {
  Node<T> *nodes;
  int reserved;
  int nodesPerThread;
  int numThreads;
  // set by the first thread to exhaust its slice.
  int *overflow;
};

// Thread-private bump allocator.
template <typename T> class NodeAllocator {
public:
  __device__ NodeAllocator(const PoolView<T> &pool, int base, int capacity)
      : m_overflow(pool.overflow), m_base(base), m_capacity(capacity),
        m_cursor(0) {}

  __device__ NodeIndex take() {
    if (m_cursor >= m_capacity) {
      DeviceAtomicRef<int> flag(*m_overflow);
      flag.store(1, cuda::memory_order_relaxed);
      return kNullIndex;
    }
    return m_base + m_cursor++;
  }

private:
  int *m_overflow;
  int m_base;
  int m_capacity;
  int m_cursor;
};

template <typename T>
__device__ inline NodeAllocator<T> threadAllocator(const PoolView<T> &pool,
                                                   int threadId) {
  return NodeAllocator<T>(pool, pool.reserved + threadId * pool.nodesPerThread,
                          pool.nodesPerThread);
}

// Allocator over the prefill arena.
template <typename T>
__device__ inline NodeAllocator<T> prefillAllocator(const PoolView<T> &pool) {
  return NodeAllocator<T>(pool, 1, pool.reserved - 1);
}

// Host-side owner.
template <typename T> class NodePool {
public:
  NodePool(int prefillCapacity, int nodesPerThread, int numThreads)
      : m_reserved(1 + prefillCapacity), m_nodesPerThread(nodesPerThread),
        m_numThreads(numThreads) {
    const std::size_t total = static_cast<std::size_t>(m_reserved) +
                              static_cast<std::size_t>(nodesPerThread) *
                                  static_cast<std::size_t>(numThreads);
    if (total > static_cast<std::size_t>(kMaxNodes)) {
      failTooLarge(total);
    }
    m_nodes.allocate(total);
    m_nodes.zero();
    m_overflow.allocate(1);
    m_overflow.zero();
  }

  NodePool(const NodePool &) = delete;
  NodePool &operator=(const NodePool &) = delete;

  PoolView<T> view() const {
    PoolView<T> v;
    v.nodes = m_nodes.get();
    v.reserved = m_reserved;
    v.nodesPerThread = m_nodesPerThread;
    v.numThreads = m_numThreads;
    v.overflow = m_overflow.get();
    return v;
  }

  std::size_t bytes() const { return m_nodes.bytes(); }

  bool overflowed() const {
    int flag = 0;
    m_overflow.copyToHost(&flag, 1);
    return flag != 0;
  }

  // Call after every kernel that allocates.
  void failIfOverflowed(const char *context) const {
    if (!overflowed()) {
      return;
    }
    std::fprintf(stderr,
                 "gpu::NodePool: a thread slice was exhausted during %s "
                 "(nodesPerThread %d, numThreads %d, prefill arena %d). The "
                 "operation budget exceeds nodesPerThread; any measurement "
                 "from this run would be invalid.\n",
                 context, m_nodesPerThread, m_numThreads, m_reserved - 1);
    std::abort();
  }

private:
  // NodeIndex is a signed 32-bit int, so the node array cannot exceed INT_MAX.
  static constexpr long long kMaxNodes = 2147483647LL;

  [[noreturn]] static void failTooLarge(std::size_t total) {
    std::fprintf(stderr,
                 "gpu::NodePool: %zu nodes requested, over the %lld addressable "
                 "by a 32-bit NodeIndex. Lower opsPerThread or the block "
                 "count.\n",
                 total, kMaxNodes);
    std::abort();
  }

  DeviceBuffer<Node<T>> m_nodes;
  DeviceBuffer<int> m_overflow;
  int m_reserved;
  int m_nodesPerThread;
  int m_numThreads;
};

} // namespace gpu
