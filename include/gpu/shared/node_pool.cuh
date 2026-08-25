// Design rationale
//
// The device counterpart of cpu/shared/node_pool.hpp, and it inherits that
// header's central decision unchanged: storage is allocated once before the
// run, handed out by a bump cursor, and never reclaimed. Safe memory
// reclamation for lock-free structures is a research problem of its own; the
// question here is the cost of the synchronization mechanism, and a reclaimer
// in the timed region would be measured as if it were part of that cost.
//
// Not reusing nodes has a second consequence on the device that it does not
// have on the host: it removes ABA outright. A node index handed to one thread
// is never handed to another, so a CAS that observes the same index twice has
// genuinely observed the same node. Michael & Scott's counted pointer -- the
// <pointer, tag> pair that doubles the width of every CAS -- is therefore dead
// weight here and is not implemented. Misra & Chaudhuri (ICPADS 2012) make the
// same argument for the same reason.
//
// Three things differ from the CPU pool.
//
//   Nodes are addressed by a 32-bit index into one array, not by pointer. That
//   halves the node (an int payload plus an int link is 8 bytes, against 16
//   for a payload and a 64-bit pointer), which matters because the whole pool
//   is sized as O(total operations) and must fit in device memory. It also
//   keeps a 64-bit tagged CAS available should a future variant need one.
//
//   The cursor is a local variable, not a member. A thread's slice is derived
//   from its own id, so nothing about allocation is shared: the cursor lives
//   in a register, there is no store to global memory per allocation, and
//   there is nothing to false-share. This is simpler than the CPU pool, which
//   has to keep cache-line-aligned per-thread slice descriptors precisely
//   because it cannot put the cursor in the caller's frame. The cost is that
//   the allocator must be passed into enqueue() rather than reached through
//   the structure, which is why the device enqueue signatures take a
//   NodeAllocator& where the host ones take a threadId.
//
//   Exhaustion cannot abort. There is no std::abort() in a kernel, and
//   __trap() destroys the context, taking with it the diagnostic buffers the
//   host would need to explain what happened. So an exhausted slice sets a
//   sticky device flag and returns kNullIndex; the operation reports failure
//   and is not counted, and the host calls NodePool::failIfOverflowed() after
//   the kernel to turn the flag into the same fatal, explicit message the CPU
//   pool prints. Checking that flag is mandatory: an over-budget run silently
//   degenerates into a queue that stops growing, which reads as a plausible
//   throughput number rather than as an error.

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/shared/atomics.cuh"
#include "gpu/shared/cuda_error.cuh"

namespace gpu {

// A node is addressed by its position in the pool array. -1 is the null link;
// it is not a valid index, so it cannot collide with a real node.
using NodeIndex = int;

inline constexpr NodeIndex kNullIndex = -1;

// The sentinel every queue variant is built around lives at index 0 of the
// pool. It is a fixed location rather than an allocation so that a queue can
// be initialized without an allocator, and so that all three variants place
// their sentinel identically.
inline constexpr NodeIndex kSentinelIndex = 0;

// `next` is written by one thread and read by another in every variant, so it
// is only ever touched through DeviceAtomicRef, never directly. It is stored
// as a plain int rather than a cuda::atomic member so that Node stays
// trivially copyable and the pool can be memset or filled from the host.
template <typename T> struct Node {
  T value;
  NodeIndex next;
};

// The device-side handle to the pool. Trivially copyable and passed to kernels
// by value: it is three words, so this is cheaper than an indirection through
// global memory on every allocation.
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
  int totalNodes;
  // Sticky, set by the first thread to exhaust its slice. Read by the host
  // after the kernel; see NodePool::failIfOverflowed().
  int *overflow;
  // Debug-only corruption witness. When a queue observes an invalid node index
  // before dereferencing it, it records {flag, where, value, threadId}.
  int *badIndex;
};

// A thread's private bump allocator. Constructed at the top of a kernel and
// left in registers for its lifetime; it holds no reference to anything
// another thread can observe.
template <typename T> class NodeAllocator {
public:
  __device__ NodeAllocator(const PoolView<T> &pool, int base, int capacity)
      : m_overflow(pool.overflow), m_base(base), m_capacity(capacity),
        m_cursor(0) {}

  // Returns kNullIndex once the slice is spent. The caller must treat that as
  // a failed operation and must not count it.
  __device__ NodeIndex take() {
    if (m_cursor >= m_capacity) {
      // Sticky flag, not a counter: the host only needs to know that it
      // happened, and one relaxed store per exhausted thread is cheaper than
      // an accumulation on a path that is already a run-invalidating error.
      DeviceAtomicRef<int> flag(*m_overflow);
      flag.store(1, cuda::memory_order_relaxed);
      return kNullIndex;
    }
    return m_base + m_cursor++;
  }

  __device__ int used() const { return m_cursor; }
  __device__ int capacity() const { return m_capacity; }

private:
  // Only the overflow flag is held: the allocator hands out indices, and the
  // caller already has the node array. Everything else is a register.
  int *m_overflow;
  int m_base;
  int m_capacity;
  int m_cursor;
};

// Allocator over thread `threadId`'s private slice. threadId is the caller's
// logical numbering, matching the CPU pool's contract; the benchmark driver
// numbers its threads 0..N-1 from the global thread index.
template <typename T>
__device__ inline NodeAllocator<T> threadAllocator(const PoolView<T> &pool,
                                                   int threadId) {
  return NodeAllocator<T>(pool, pool.reserved + threadId * pool.nodesPerThread,
                          pool.nodesPerThread);
}

// Allocator over the prefill arena. Used by exactly one thread, outside the
// timed region, to build the queue's starting contents.
template <typename T>
__device__ inline NodeAllocator<T> prefillAllocator(const PoolView<T> &pool) {
  return NodeAllocator<T>(pool, 1, pool.reserved - 1);
}

// Host-side owner. Allocates the node array and the overflow flag, and hands
// out the device view.
template <typename T> class NodePool {
public:
  // prefillCapacity : nodes reserved for the pre-run fill, over and above the
  //                   sentinel.
  // nodesPerThread  : capacity of each thread's private slice.
  // numThreads      : number of logical thread ids that will allocate.
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
    m_badIndex.allocate(4);
    m_badIndex.zero();
  }

  NodePool(const NodePool &) = delete;
  NodePool &operator=(const NodePool &) = delete;

  PoolView<T> view() const {
    PoolView<T> v;
    v.nodes = m_nodes.get();
    v.reserved = m_reserved;
    v.nodesPerThread = m_nodesPerThread;
    v.numThreads = m_numThreads;
    v.totalNodes = static_cast<int>(m_nodes.count());
    v.overflow = m_overflow.get();
    v.badIndex = m_badIndex.get();
    return v;
  }

  std::size_t nodeCount() const { return m_nodes.count(); }
  std::size_t bytes() const { return m_nodes.bytes(); }

  bool overflowed() const {
    int flag = 0;
    m_overflow.copyToHost(&flag, 1);
    return flag != 0;
  }

  // Call after every kernel that allocates. See the header comment: an
  // unchecked overflow turns an over-budget configuration into a plausible
  // number rather than an error.
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

  void failIfCorrupted(const char *context) const {
    int witness[4] = {};
    m_badIndex.copyToHost(witness, 4);
    if (witness[0] == 0) {
      return;
    }
    std::fprintf(stderr,
                 "gpu::NodePool: invalid node index observed during %s "
                 "(where %d, value %d, threadId %d, totalNodes %zu). This "
                 "indicates queue index corruption before the guarded "
                 "dereference.\n",
                 context, witness[1], witness[2], witness[3],
                 m_nodes.count());
    std::abort();
  }

  // Reset the flag between runs that share one pool. Not used by the
  // benchmark, which builds a fresh pool per configuration, but the tests
  // reuse a pool across scenarios.
  void clearOverflow() { m_overflow.zero(); }
  void clearCorruption() { m_badIndex.zero(); }

private:
  // NodeIndex is a signed 32-bit int, so the array cannot exceed INT_MAX
  // entries. Caught here rather than by an index wrapping negative at run
  // time, which would be read as kNullIndex and terminate a list early.
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
  DeviceBuffer<int> m_badIndex;
  int m_reserved;
  int m_nodesPerThread;
  int m_numThreads;
};

} // namespace gpu
