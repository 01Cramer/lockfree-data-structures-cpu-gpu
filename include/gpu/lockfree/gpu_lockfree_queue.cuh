#pragma once

#include "gpu/shared/gpu_atomics.cuh"
#include "gpu/shared/gpu_node_pool.cuh"

namespace gpu {

namespace lockfree {

// Lock-free Michael-Scott FIFO queue, using node indices instead of pointers.
// Nodes are never reused, so the ABA tag from the original algorithm is not
// needed for this benchmark.
//
// Memory ordering: the CAS that links a new node uses release. Threads that
// later follow that link, or read head/tail values that may point at that node,
// use acquire before dereferencing the node.
//
// The tail-help branch is part of the algorithm: if a thread sees that tail is
// behind, it advances tail before retrying its own operation.
template <typename T> class alignas(kDeviceCacheLineBytes) Queue {
public:
  __device__ void initialize(const PoolView<T> &pool) {
    m_nodes = pool.nodes;
    nextRef(kSentinelIndex).store(kNullIndex, cuda::memory_order_relaxed);
    headRef().store(kSentinelIndex, cuda::memory_order_relaxed);
    tailRef().store(kSentinelIndex, cuda::memory_order_relaxed);
  }

  __device__ bool enqueue(const T &value, NodeAllocator<T> &allocator) {
    const NodeIndex newNode = allocator.take();
    if (newNode == kNullIndex) {
      return false;
    }
    m_nodes[newNode].value = value;
    nextRef(newNode).store(kNullIndex, cuda::memory_order_relaxed);

    NodeIndex tail = kNullIndex;

    while (true) {
      tail = tailRef().load(cuda::memory_order_acquire);
      // Acquire before following a node published by another enqueuer.
      const NodeIndex tailNext = nextRef(tail).load(cuda::memory_order_acquire);

      // Snapshot check from Michael-Scott: retry if tail changed meanwhile.
      if (tail != tailRef().load(cuda::memory_order_acquire)) {
        continue;
      }

      if (tailNext == kNullIndex) {
        NodeIndex expected = kNullIndex;
        // Release publishes the initialized node.
        if (nextRef(tail).compare_exchange_strong(expected, newNode,
                                                  cuda::memory_order_release,
                                                  cuda::memory_order_relaxed)) {
          break;
        }
      } else {
        // Tail lags a completed link; help it forward.
        NodeIndex expected = tail;
        tailRef().compare_exchange_strong(expected, tailNext,
                                          cuda::memory_order_release,
                                          cuda::memory_order_relaxed);
      }
    }

    // Best-effort tail update. Failure means another thread already helped.
    NodeIndex expected = tail;
    tailRef().compare_exchange_strong(expected, newNode,
                                      cuda::memory_order_release,
                                      cuda::memory_order_relaxed);
    return true;
  }

  __device__ bool dequeue(T &out) {
    while (true) {
      const NodeIndex head = headRef().load(cuda::memory_order_acquire);
      const NodeIndex tail = tailRef().load(cuda::memory_order_acquire);
      // Acquire before reading the dequeued node's payload.
      const NodeIndex headNext = nextRef(head).load(cuda::memory_order_acquire);

      if (head != headRef().load(cuda::memory_order_acquire)) {
        continue;
      }

      if (head == tail) {
        if (headNext == kNullIndex) {
          // Head and tail both point at the sentinel/last node with no next.
          return false;
        }
        // Tail is behind; help it forward.
        NodeIndex expected = tail;
        tailRef().compare_exchange_strong(expected, headNext,
                                          cuda::memory_order_release,
                                          cuda::memory_order_relaxed);
      } else {
        // Read before the CAS. Several threads may read the same candidate
        // node, but only one will advance head and return the value.
        const T value = m_nodes[headNext].value;
        NodeIndex expected = head;
        if (headRef().compare_exchange_strong(expected, headNext,
                                              cuda::memory_order_release,
                                              cuda::memory_order_relaxed)) {
          out = value;
          return true;
        }
      }
    }
  }

private:
  __device__ DeviceAtomicRef<NodeIndex> nextRef(NodeIndex node) {
    return DeviceAtomicRef<NodeIndex>(m_nodes[node].next);
  }

  __device__ DeviceAtomicRef<NodeIndex> headRef() {
    return DeviceAtomicRef<NodeIndex>(m_head);
  }

  __device__ DeviceAtomicRef<NodeIndex> tailRef() {
    return DeviceAtomicRef<NodeIndex>(m_tail);
  }

  Node<T> *m_nodes;

  // Head and tail are independently contended, so keep them separated by
  // layout.
  alignas(kDeviceCacheLineBytes) NodeIndex m_head;
  alignas(kDeviceCacheLineBytes) NodeIndex m_tail;
};

} // namespace lockfree

} // namespace gpu
