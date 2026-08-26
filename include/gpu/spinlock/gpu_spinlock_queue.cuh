#pragma once

#include "gpu/shared/gpu_atomics.cuh"
#include "gpu/shared/gpu_node_pool.cuh"
#include "gpu/shared/gpu_spinlock.cuh"

namespace gpu {

namespace spinlock {

// FIFO queue behind a single spinlock.
// The device port of cpu/spinlock/spinlock_queue.hpp: 
// same structure, same sentinel, only the lock and the node addressing differ.
//
// A sentinel node keeps m_head and m_tail non-null, so neither enqueue nor
// dequeue branches on emptiness and all three variants have the same shape.
// The sentinel is pool node 0 rather than a member, because the queue itself
// lives in device memory as a flat object with no constructor.
//
// Memory ordering: every access to a node's 'next' happens inside the critical
// section, so the lock's acquire/release pair provides all the ordering there
// is and the links are plain, non-atomic loads and stores.

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
  // run-invalidating error the host detects through NodePool::overflow.
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

  // Returns false when the queue is empty.
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
  Node<T> *m_nodes;

  // Single synchronization point for the whole queue. The lock is aligned so its
  // atomic traffic is not coupled to preceding fields by layout.
  alignas(kDeviceCacheLineBytes) Spinlock m_lock;
  NodeIndex m_head;
  NodeIndex m_tail;
};

} // namespace spinlock

} // namespace gpu
