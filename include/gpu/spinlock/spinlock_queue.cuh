#pragma once

#include "gpu/shared/atomics.cuh"
#include "gpu/shared/node_pool.cuh"
#include "gpu/shared/spinlock.cuh"

namespace gpu {

namespace spinlock {

// Variant 1: FIFO queue behind a single spinlock.
//
// This is Michael & Scott's *blocking* algorithm (1996, §3) in its single-lock
// form, the same one Cederman, Chatterjee & Tsigas evaluate on a GPU in
// Euro-Par 2012. The device port of cpu/spinlock/spinlock_queue.hpp: same
// structure, same sentinel, only the lock and the node addressing differ.
//
// A sentinel node keeps m_head and m_tail non-null, so neither enqueue nor
// dequeue branches on emptiness and all three variants have the same shape.
// The sentinel is pool node 0 rather than a member, because the queue itself
// lives in device memory as a flat object with no constructor.
//
// Memory ordering: every access to a node's `next` happens inside the critical
// section, so the lock's acquire/release pair provides all the ordering there
// is and the links are plain, non-atomic loads and stores. That is exactly
// what makes this the cheapest-per-operation and least-scalable variant, and
// it is the honest baseline for "what does the synchronization cost": nothing
// in the data path is paying for concurrency, the lock is paying for all of
// it.
template <typename T> class alignas(kDeviceCacheLineBytes) Queue {
public:
  // Called by exactly one thread before any operation. Not a constructor:
  // the object is raw device memory.
  __device__ void initialize(const PoolView<T> &pool) {
    m_nodes = pool.nodes;
    m_nodes[kSentinelIndex].next = kNullIndex;
    m_head = kSentinelIndex;
    m_tail = kSentinelIndex;
    m_lock.initialize();
  }

  // Returns false only when the caller's pool slice is exhausted, which is a
  // run-invalidating error the host detects through NodePool::overflow. A
  // false is never a legitimate "queue full": the queue is unbounded.
  __device__ bool enqueue(const T &value, NodeAllocator<T> &allocator) {
    const NodeIndex newNode = allocator.take();
    if (newNode == kNullIndex) {
      return false;
    }
    m_nodes[newNode].value = value;
    m_nodes[newNode].next = kNullIndex;

    const LockGuard guard(m_lock);
    m_nodes[m_tail].next = newNode;
    m_tail = newNode;
    return true;
  }

  // Returns false when the queue is empty. The value is written through `out`
  // rather than returned in an optional; there is no std::optional in device
  // code, and since no cross-platform ratio is quoted the difference in
  // signature from the CPU side carries no comparison risk.
  __device__ bool dequeue(T &out) {
    const LockGuard guard(m_lock);
    const NodeIndex next = m_nodes[m_head].next;
    if (next == kNullIndex) {
      return false;
    }
    out = m_nodes[next].value;
    m_head = next;
    return true;
  }

private:
  // Read-only after initialize(), so it shares no line with anything written.
  Node<T> *m_nodes;

  // One synchronization point, so head, tail and the lock word are one group
  // on one line: they are only ever touched by the thread holding the lock,
  // and separating them would cost that thread extra lines for no reduction in
  // contention. See atomics.cuh on why the grouping is fixed rather than
  // swept.
  alignas(kDeviceCacheLineBytes) Spinlock m_lock;
  NodeIndex m_head;
  NodeIndex m_tail;
};

} // namespace spinlock

} // namespace gpu
