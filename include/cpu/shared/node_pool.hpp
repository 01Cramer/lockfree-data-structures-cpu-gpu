// Design rationale
//
// The CPU data structures in this project exist to benchmark synchronization
// algorithms (lock-based vs. lock-free) in isolation. Safe memory
// reclamation for lock-free data structures -- hazard pointers, epoch-based
// reclamation, RCU -- is a research problem in its own right, and is out of
// scope here; including it would conflate the cost of reclamation with the
// cost of the synchronization primitive being studied.
//
// This pool therefore provides a uniform, intentionally simple memory model
// shared by every structure: storage is allocated once at construction,
// handed out as needed, and freed in NodePool destructor. Nothing is reclaimed
// during execution. As a result, the benchmarked operations (push / enqueue /
// insert) contain no allocator or reclamation work, and the timed region
// reflects only the synchronization mechanism.

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cpu/shared/cache.hpp"

namespace cpu {

// - Pre-allocated, per-thread-sliced storage of NodeT.
// - Each thread draws from its own slice via a non-atomic cursor; no
//   synchronization, no allocator calls on the hot path.
// - Slice descriptors are cache-line aligned and each owns its own heap
//   allocation, so threads cannot false-share cursor metadata.
// - One-shot semantics: nodes taken from the pool are never returned. All
//   storage is freed in NodePool destructor.
// - Pool exhaustion aborts the process (in benchmarks it should never happen).
//
// Thread-to-slice mapping is explicit: the caller passes a logical thread id
// in [0, numThreads) to takeNode(). The benchmark driver owns this numbering
// (it assigns 0..N-1 to its workers and re-numbers from 0 every run).
template <typename NodeT> class NodePool {
public:
  // getNodesPerThread : capacity of each thread's slice.
  // numThreads     : number of logical thread ids that will call takeNode();
  NodePool(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_slices(numThreads) {
    for (auto &slice : m_slices) {
      slice.storage = std::vector<NodeT>(getNodesPerThread);
      slice.capacity = getNodesPerThread;
    }
  }

  NodePool(const NodePool &) = delete;
  NodePool &operator=(const NodePool &) = delete;
  NodePool(NodePool &&) = delete;
  NodePool &operator=(NodePool &&) = delete;

  NodeT *takeNode(std::size_t threadId) {
    if (threadId >= m_slices.size()) [[unlikely]] {
      failBadThreadId(threadId, m_slices.size());
    }

    Slice &slice = m_slices[threadId];
    if (slice.cursor >= slice.capacity) [[unlikely]] {
      failExhausted(threadId, slice.capacity);
    }
    return &slice.storage[slice.cursor++];
  }

private:
  struct alignas(cacheLineSize) Slice {
    std::size_t cursor = 0;
    std::size_t capacity = 0;
    std::vector<NodeT> storage;
  };

  std::vector<Slice> m_slices;

  [[noreturn]] static void failExhausted(std::size_t threadId,
                                         std::size_t capacity) {
    std::fprintf(stderr,
                 "cpu::NodePool: slice %zu exhausted (capacity %zu). The "
                 "operation budget exceeds getNodesPerThread; any measurement "
                 "from this run would be invalid.\n",
                 threadId, capacity);
    std::abort();
  }

  [[noreturn]] static void failBadThreadId(std::size_t threadId,
                                           std::size_t numThreads) {
    std::fprintf(stderr,
                 "cpu::NodePool: threadId %zu out of range (numThreads %zu).\n",
                 threadId, numThreads);
    std::abort();
  }
};

} // namespace cpu
