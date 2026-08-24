#pragma once

#include "gpu/shared/atomics.cuh"
#include "gpu/shared/node_pool.cuh"
#include "gpu/shared/spinlock.cuh"

namespace gpu {

namespace spinlock {

// Variant 2: Michael & Scott's two-lock FIFO queue (1996, §4), with the
// mutexes replaced by gpu::Spinlock. The device port of
// cpu/spinlock/spinlock_queue_two_lock.hpp.
//
// Separate head and tail locks let one enqueue and one dequeue proceed
// concurrently, so the sentinel is doing real work here rather than just
// removing a branch: it guarantees the two ends never refer to the same node
// while the queue is non-empty, which is what makes the two critical sections
// independent.
//
// Memory ordering: this is the only place the two variants differ in more than
// the number of lock words. `next` is written under the tail lock and read
// under the head lock, and those two locks order nothing with respect to each
// other, so the link is a release store paired with an acquire load. Without
// it a dequeuer could observe the link before the payload the enqueuer wrote
// just above it -- the same hole that Zhang et al.'s published CCLQ code has,
// where the shared arrays are `volatile` and nothing is fenced at all.
// `volatile` suppresses caching; it does not order.
template <typename T> class alignas(kDeviceCacheLineBytes) QueueTwoLock {
public:
  __device__ void initialize(const PoolView<T> &pool) {
    m_nodes = pool.nodes;
    DeviceAtomicRef<NodeIndex> sentinelNext(m_nodes[kSentinelIndex].next);
    sentinelNext.store(kNullIndex, cuda::memory_order_relaxed);
    m_head = kSentinelIndex;
    m_tail = kSentinelIndex;
    m_headLock.initialize();
    m_tailLock.initialize();
  }

  __device__ bool enqueue(const T &value, NodeAllocator<T> &allocator) {
    const NodeIndex newNode = allocator.take();
    if (newNode == kNullIndex) {
      return false;
    }
    m_nodes[newNode].value = value;
    DeviceAtomicRef<NodeIndex> newNext(m_nodes[newNode].next);
    newNext.store(kNullIndex, cuda::memory_order_relaxed);

    const LockGuard guard(m_tailLock);
    // Release: publishes both the payload written above and the terminating
    // null link, to whichever thread acquires this link under the head lock.
    DeviceAtomicRef<NodeIndex> tailNext(m_nodes[m_tail].next);
    tailNext.store(newNode, cuda::memory_order_release);
    m_tail = newNode;
    return true;
  }

  __device__ bool dequeue(T &out) {
    const LockGuard guard(m_headLock);
    // Acquire: pairs with the release above, so the payload read below is
    // guaranteed to be the value the enqueuer stored.
    DeviceAtomicRef<NodeIndex> headNext(m_nodes[m_head].next);
    const NodeIndex next = headNext.load(cuda::memory_order_acquire);
    if (next == kNullIndex) {
      return false;
    }
    out = m_nodes[next].value;
    m_head = next;
    return true;
  }

private:
  Node<T> *m_nodes;

  // Two independent synchronization points, one per line. This is the whole
  // point of the variant: putting them on the same line would serialize the
  // two critical sections at the L2 and reproduce single-lock behaviour with
  // twice the code.
  alignas(kDeviceCacheLineBytes) Spinlock m_headLock;
  NodeIndex m_head;

  alignas(kDeviceCacheLineBytes) Spinlock m_tailLock;
  NodeIndex m_tail;
};

} // namespace spinlock

} // namespace gpu
