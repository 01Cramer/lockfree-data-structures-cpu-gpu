#pragma once

#include "gpu/shared/gpu_atomics.cuh"
#include "gpu/shared/gpu_node_pool.cuh"
#include "gpu/shared/gpu_spinlock.cuh"

namespace gpu {

namespace spinlock {

// FIFO queue with separate spinlocks for the head and tail.
// The device port of cpu/spinlock/spinlock_queue_two_lock.hpp:
// same structure, same sentinel, only the locks and node addressing differ.
//
// Separate head and tail locks let one enqueue and one dequeue proceed
// concurrently. The sentinel node keeps m_head and m_tail non-null and lets
// the two ends move independently while the queue is non-empty.
//
// Memory ordering: a node's 'next' is written while holding the tail lock and
// read while holding the head lock. Those locks do not order each other, so
// the link is written with release and read with acquire.
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

  // Independent synchronization points for the two ends. Each lock is aligned
  // so its atomic traffic is not coupled to the other lock by layout.
  alignas(kDeviceCacheLineBytes) Spinlock m_headLock;
  NodeIndex m_head;

  alignas(kDeviceCacheLineBytes) Spinlock m_tailLock;
  NodeIndex m_tail;
};

} // namespace spinlock

} // namespace gpu
