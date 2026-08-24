// Design rationale
//
// Storage for the request-combining queue (CCLQ). A separate pool from
// gpu/shared/node_pool.cuh rather than a template parameter of it, because
// CCLQ's node is a different thing: it holds 32 payloads and a claimed-range
// cursor, and it is addressed by *two* levels of index.
//
// The two levels are the part of Zhang, Deng & Mu's design that is easiest to
// misread, so, explicitly:
//
//   position -> node.  `nextPos[p]` is the index of the batch node linked at
//   queue position p, or -1 if position p is not linked yet. Positions are
//   monotonically increasing integers; m_head and m_tail in the queue are
//   positions, not node indices. The live range is [head + 1, tail], which is
//   the paper's "the range of data is between Global_head+1 and Global_tail".
//   nextPos[tail + 1] plays exactly the role of tail->next in Michael & Scott,
//   which is why the paper calls it nextpos_list.
//
//   node -> payload.  Node n owns data[n*32 .. n*32+31], startPos[n] and
//   endPos[n]. Items [startPos[n], endPos[n]) are the ones still unclaimed;
//   endPos is written once at enqueue and never changes, while startPos is the
//   word consumers CAS to claim a range.
//
// Structure of arrays, not array of structures, and that is the paper's choice
// (Fig. 1 (a), lines 6-7: "we use three separate arrays ... to be more
// efficient for GPU execution"). It is also the right choice for its own
// reasons: the 32 lanes of a warp write data[base + lane] in the same
// instruction, which is one fully coalesced transaction, whereas an interleaved
// struct layout would scatter it.
//
// Positions are never recycled, and neither are nodes. Same decision as the
// other variants, for the same reason -- no reclaimer in the timed region -- and
// with the same consequence: the total operation count must be known before
// launch, because the position array is sized from it. This is not a
// convenience of the benchmark. Zhang's `nextpos_list` has the identical
// constraint, which is why the paper fixes 1000 operations per thread.
//
// One asymmetry worth knowing before reading pool sizes: a batch node is
// allocated per warp *enqueue call*, not per item. So the node budget scales
// with producer warps, not producer threads -- a factor of 32 fewer nodes than
// variants 1-3 need, each 33 words instead of 2. At activeLanes = 1 a full
// 32-slot node carries one item, so the arrays are then 32x larger than the
// data in them; that waste is the price of holding the node layout fixed across
// the activeLanes sweep, which is what makes the sweep interpretable.

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "gpu/shared/atomics.cuh"
#include "gpu/shared/cuda_error.cuh"
#include "gpu/shared/warp_scan.cuh"

namespace gpu {

// Index into the per-node arrays.
using BatchNodeIndex = int;

// Index into nextPos. A queue position, not a node.
using QueuePosition = int;

inline constexpr BatchNodeIndex kNullBatchNode = -1;

// Items per batch node. 32 by the paper (`datatype data_list[32]`), and it must
// equal the warp size: the combining step gives one slot to each lane.
inline constexpr int kBatchCapacity = kWarpSize;

// Device-side handle. Trivially copyable; carried by value inside the queue
// object, which is read-only after initialization.
template <typename T> struct BatchPoolView {
  T *data;                  // nodeCount * kBatchCapacity
  int *startPos;            // nodeCount
  int *endPos;              // nodeCount
  BatchNodeIndex *nextPos;  // positionCapacity, -1 = position not linked
  int nodeCount;
  int positionCapacity;
  int reserved;             // nodes [1, reserved) are the prefill arena
  int nodesPerWarp;
  int *overflow;            // sticky; read by the host after the kernel
};

// A warp's private bump allocator over batch nodes.
//
// Every lane of the warp holds one and every lane calls take(), so all 32
// cursors advance together and return the same index. That is only true while
// enqueue is called warp-uniformly, which the harness guarantees; the queue
// nevertheless broadcasts the combiner's value rather than trusting it, so a
// divergent call site would produce a wrong-but-consistent index instead of 32
// different ones.
template <typename T> class BatchAllocator {
public:
  __device__ BatchAllocator(const BatchPoolView<T> &pool, int base,
                            int capacity)
      : m_overflow(pool.overflow), m_base(base), m_capacity(capacity),
        m_cursor(0) {}

  __device__ BatchNodeIndex take() {
    if (m_cursor >= m_capacity) {
      DeviceAtomicRef<int> flag(*m_overflow);
      flag.store(1, cuda::memory_order_relaxed);
      return kNullBatchNode;
    }
    return m_base + m_cursor++;
  }

private:
  int *m_overflow;
  int m_base;
  int m_capacity;
  int m_cursor;
};

// Allocator over warp `warpId`'s private slice.
template <typename T>
__device__ inline BatchAllocator<T>
warpBatchAllocator(const BatchPoolView<T> &pool, int warpId) {
  return BatchAllocator<T>(pool, pool.reserved + warpId * pool.nodesPerWarp,
                           pool.nodesPerWarp);
}

// Allocator over the prefill arena, used by one warp before the timed region.
template <typename T>
__device__ inline BatchAllocator<T>
prefillBatchAllocator(const BatchPoolView<T> &pool) {
  return BatchAllocator<T>(pool, 1, pool.reserved - 1);
}

// Host-side owner.
template <typename T> class BatchPool {
public:
  // prefillNodes : batch nodes reserved for the pre-run fill.
  // nodesPerWarp : batch nodes each warp may allocate, i.e. its enqueue budget.
  // numWarps     : warps that will allocate.
  BatchPool(int prefillNodes, int nodesPerWarp, int numWarps)
      : m_reserved(1 + prefillNodes), m_nodesPerWarp(nodesPerWarp),
        m_numWarps(numWarps) {
    const long long nodes = static_cast<long long>(m_reserved) +
                            static_cast<long long>(nodesPerWarp) *
                                static_cast<long long>(numWarps);
    // Two spare positions: one because position 0 is the dummy that head and
    // tail start on and is never linked, and one so that a consumer's read of
    // nextPos[head + 1] is in bounds even when head has reached the last
    // linked position.
    const long long positions = nodes + 2;
    if (nodes > kMaxIndex || positions > kMaxIndex ||
        nodes * kBatchCapacity > kMaxIndex) {
      failTooLarge(nodes, positions);
    }

    m_nodeCount = static_cast<int>(nodes);
    m_positionCapacity = static_cast<int>(positions);

    m_data.allocate(static_cast<std::size_t>(m_nodeCount) * kBatchCapacity);
    m_startPos.allocate(static_cast<std::size_t>(m_nodeCount));
    m_endPos.allocate(static_cast<std::size_t>(m_nodeCount));
    m_nextPos.allocate(static_cast<std::size_t>(m_positionCapacity));
    m_overflow.allocate(1);

    m_data.zero();
    m_startPos.zero();
    m_endPos.zero();
    m_overflow.zero();
    // Every position starts unlinked. -1 in two's complement is all bits set,
    // so a byte-wise fill produces it; this is the one array whose initial
    // value is not zero and getting it wrong would make the queue appear to
    // start full of node 0.
    GPU_CUDA_CHECK(cudaMemset(m_nextPos.get(), 0xFF, m_nextPos.bytes()));
  }

  BatchPool(const BatchPool &) = delete;
  BatchPool &operator=(const BatchPool &) = delete;

  BatchPoolView<T> view() const {
    BatchPoolView<T> v;
    v.data = m_data.get();
    v.startPos = m_startPos.get();
    v.endPos = m_endPos.get();
    v.nextPos = m_nextPos.get();
    v.nodeCount = m_nodeCount;
    v.positionCapacity = m_positionCapacity;
    v.reserved = m_reserved;
    v.nodesPerWarp = m_nodesPerWarp;
    v.overflow = m_overflow.get();
    return v;
  }

  std::size_t bytes() const {
    return m_data.bytes() + m_startPos.bytes() + m_endPos.bytes() +
           m_nextPos.bytes();
  }

  int nodeCount() const { return m_nodeCount; }
  int positionCapacity() const { return m_positionCapacity; }

  bool overflowed() const {
    int flag = 0;
    m_overflow.copyToHost(&flag, 1);
    return flag != 0;
  }

  // Call after every kernel that enqueues. The flag covers both an exhausted
  // node slice and an exhausted position array, and either one turns further
  // enqueues into silent no-ops.
  void failIfOverflowed(const char *context) const {
    if (!overflowed()) {
      return;
    }
    std::fprintf(stderr,
                 "gpu::BatchPool: exhausted during %s (nodesPerWarp %d, "
                 "numWarps %d, prefill arena %d, positions %d). The operation "
                 "budget exceeds the pool; any measurement from this run would "
                 "be invalid.\n",
                 context, m_nodesPerWarp, m_numWarps, m_reserved - 1,
                 m_positionCapacity);
    std::abort();
  }

  void clearOverflow() { m_overflow.zero(); }

private:
  static constexpr long long kMaxIndex = 2147483647LL;

  [[noreturn]] static void failTooLarge(long long nodes, long long positions) {
    std::fprintf(stderr,
                 "gpu::BatchPool: %lld nodes / %lld positions requested; a "
                 "32-bit index addresses at most %lld, and the payload array "
                 "is 32x the node count. Lower the operation budget or the "
                 "warp count.\n",
                 nodes, positions, kMaxIndex);
    std::abort();
  }

  DeviceBuffer<T> m_data;
  DeviceBuffer<int> m_startPos;
  DeviceBuffer<int> m_endPos;
  DeviceBuffer<BatchNodeIndex> m_nextPos;
  DeviceBuffer<int> m_overflow;
  int m_reserved;
  int m_nodesPerWarp;
  int m_numWarps;
  int m_nodeCount = 0;
  int m_positionCapacity = 0;
};

} // namespace gpu
