#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <vector>

// Design rationale
//
// The CPU data structures in this project exist to benchmark synchronization
// mechanisms (mutexes vs. compare-and-swap) in isolation. Safe memory
// reclamation for lock-free data structures -- hazard pointers, epoch-based
// reclamation, RCU -- is a research problem in its own right, and is out of
// scope here; including it would conflate the cost of reclamation with the
// cost of the synchronization primitive being studied.
//
// This pool therefore provides a uniform, intentionally simple memory model
// shared by every structure: storage is allocated once at construction,
// handed out as needed, and freed in ~NodePool. Nothing is reclaimed during
// execution. As a result, the benchmarked operations (push / enqueue /
// insert) contain no allocator or reclamation work, and the timed region
// reflects only the synchronization mechanism.

namespace cpu {

namespace detail {

// Assigns each OS thread a stable, dense index in [0, N), lazily on first call.
// One process-wide counter, so a given thread keeps the same index across every
// NodePool instance for its entire lifetime.
inline std::size_t threadIndex() {
  static std::atomic<std::size_t> next{0};
  static thread_local std::size_t id =
      next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

} // namespace detail

// Pre-allocated, per-thread-sliced storage of NodeT.
//
// - Each thread draws from its own slice via a non-atomic cursor; no
//   coordination, no allocator calls on the hot path.
// - Slices are cache-line aligned and each owns its own heap allocation,
//   so threads cannot false-share node memory or cursor metadata.
// - One-shot semantics: acquired nodes are never returned. All storage is
//   freed in ~NodePool.
template <typename NodeT> class NodePool {
public:
  // nodesPerThread : capacity of each thread's slice (upper bound on
  //                  push/enqueue/insert calls per thread for this pool).
  // numThreads     : must be >= the total number of threads that will touch
  //                  any pool in this process. cpu::detail::threadIndex() is
  //                  global, so IDs are assigned across all pools combined.
  NodePool(std::size_t nodesPerThread, std::size_t numThreads)
      : m_slices(numThreads) {
    for (auto &slice : m_slices) {
      // (size_type) ctor + move-assign: requires NodeT to be DefaultInsertable
      // only. resize() would require MoveInsertable, which fails when NodeT
      // holds a std::atomic.
      slice.storage = std::vector<NodeT>(nodesPerThread);
    }
  }

  NodePool(const NodePool &) = delete;
  NodePool &operator=(const NodePool &) = delete;
  NodePool(NodePool &&) = delete;
  NodePool &operator=(NodePool &&) = delete;

  NodeT *acquire() {
    const std::size_t tid = detail::threadIndex();
    assert(tid < m_slices.size() && "numThreads too small for active threads");
    Slice &slice = m_slices[tid];
    assert(slice.cursor < slice.storage.size() && "node pool exhausted");
    return &slice.storage[slice.cursor++];
  }

private:
  struct alignas(64) Slice {
    std::vector<NodeT> storage;
    std::size_t cursor = 0;
  };

  std::vector<Slice> m_slices;
};

} // namespace cpu
