#pragma once

#include "gpu/shared/atomics.cuh"
#include "gpu/shared/node_pool.cuh"

namespace gpu {

namespace lockfree {

// Variant 3: the lock-free Michael & Scott queue (1996, §3), in node indices.
// The device port of cpu/lockfree/lockfree_queue.hpp -- structurally the same
// algorithm, with three deliberate differences that are all consequences of
// decisions recorded elsewhere.
//
//   No counted pointer. M&S carry a monotonically increasing tag alongside
//   every pointer and CAS the pair, purely to detect ABA. Nodes here are never
//   reused (see node_pool.cuh), so an index that compares equal is the same
//   node, and there is no A-B-A to detect. The tag is dropped, and with it the
//   need for a 64-bit CAS. Misra & Chaudhuri (ICPADS 2012) justify the same
//   omission the same way.
//
//   The linking CAS carries release. This is the ordering that publishes the
//   whole node -- its payload and its initialized null successor -- to whichever
//   thread later acquires the link. Dequeuers need that before reading the
//   payload; enqueuers need it too before helping the tail forward and then
//   treating the linked node as a tail whose `next` field is initialized.
//
//   The tail-help branch is kept. When the observed tail already has a
//   successor, this thread advances the tail on the lagging thread's behalf
//   and retries. That branch is not an optimization: it is what makes the
//   queue lock-free, because it lets any thread complete another's half-done
//   enqueue. CCLQ as printed drops it and re-reads an unchanged tail instead,
//   which means a thread can spin until some other thread happens to advance
//   the tail -- forfeiting the lock-freedom the paper claims. Restoring it is
//   not a deviation from M&S; dropping it was.
//
// A note on compare_exchange_strong. There are no spurious CAS failures on the
// device: atom.cas either observes the expected value or does not. weak and
// strong therefore generate the same code, and strong is used because it makes
// the retry structure explicit at the call site.
template <typename T> class alignas(kDeviceCacheLineBytes) Queue {
public:
  __device__ void initialize(const PoolView<T> &pool) {
    m_nodes = pool.nodes;
    m_nodeCount = pool.totalNodes;
    m_badIndex = pool.badIndex;
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
      if (!validIndex(tail)) {
        recordBadIndex(1, tail);
        return false;
      }
      // Acquire pairs with the release CAS that linked `tailNext`, so if we
      // help the tail forward to that node then its payload and null successor
      // are visible before any later iteration reads nextRef(tailNext).
      const NodeIndex tailNext = nextRef(tail).load(cuda::memory_order_acquire);
      if (tailNext != kNullIndex && !validIndex(tailNext)) {
        recordBadIndex(2, tailNext);
        return false;
      }

      // M&S's snapshot check: confirm the tail did not move between the two
      // reads above. Retained for faithfulness and because it discards a CAS
      // attempt that is already known to be doomed, but note that without ABA
      // it is not load-bearing for safety -- if tailNext is null then `tail`
      // is the last node whatever m_tail currently says, and linking there is
      // correct regardless.
      if (tail != tailRef().load(cuda::memory_order_acquire)) {
        continue;
      }

      if (tailNext == kNullIndex) {
        NodeIndex expected = kNullIndex;
        // Release: publishes m_nodes[newNode].value, written above.
        if (nextRef(tail).compare_exchange_strong(expected, newNode,
                                                  cuda::memory_order_release,
                                                  cuda::memory_order_relaxed)) {
          break;
        }
      } else {
        // The tail lags a completed link. Finish the other thread's work
        // before retrying our own; this is the helping step.
        NodeIndex expected = tail;
        tailRef().compare_exchange_strong(expected, tailNext,
                                          cuda::memory_order_release,
                                          cuda::memory_order_relaxed);
      }
    }

    // Swing the tail to the node just linked. A hint only: the CAS is allowed
    // to fail, because another thread's helping step has then already done it,
    // and the queue is correct either way.
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
      if (!validIndex(head)) {
        recordBadIndex(3, head);
        return false;
      }
      // Acquire: pairs with the linking release in enqueue, so the payload
      // read below is the value the enqueuer stored and not whatever the pool
      // was initialized to.
      const NodeIndex headNext = nextRef(head).load(cuda::memory_order_acquire);
      if (headNext != kNullIndex && !validIndex(headNext)) {
        recordBadIndex(4, headNext);
        return false;
      }

      if (head != headRef().load(cuda::memory_order_acquire)) {
        continue;
      }

      if (head == tail) {
        if (headNext == kNullIndex) {
          // Genuinely empty, and linearizable at the load of headNext: head
          // never passes tail, so observing head == tail and no successor
          // means there was an instant at which the queue held nothing.
          return false;
        }
        // Empty-looking only because the tail lags. Help it forward.
        NodeIndex expected = tail;
        tailRef().compare_exchange_strong(expected, headNext,
                                          cuda::memory_order_release,
                                          cuda::memory_order_relaxed);
      } else {
        // Read the payload BEFORE the CAS that claims the node. Several
        // threads may be reading this same node concurrently and all but one
        // will lose; the read must therefore be non-destructive. (On the host
        // side the same line carries a stronger warning, because moving out of
        // the node there would corrupt the winner's result for a non-trivial
        // T. Here T is a scalar and a copy is all there is, but the ordering
        // requirement is identical and worth keeping visible.)
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
  __device__ bool validIndex(NodeIndex node) const {
    return node >= 0 && node < m_nodeCount;
  }

  __device__ void recordBadIndex(int where, NodeIndex value) {
    if (m_badIndex == nullptr) {
      return;
    }
    DeviceAtomicRef<int> flag(m_badIndex[0]);
    int expected = 0;
    if (flag.compare_exchange_strong(expected, 1, cuda::memory_order_relaxed,
                                     cuda::memory_order_relaxed)) {
      m_badIndex[1] = where;
      m_badIndex[2] = value;
      m_badIndex[3] =
          static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    }
  }

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
  int m_nodeCount;
  int *m_badIndex;

  // Two independently contended words, one line each: enqueuers hammer m_tail
  // and dequeuers hammer m_head, and unlike the single-lock variant no thread
  // needs both at once.
  alignas(kDeviceCacheLineBytes) NodeIndex m_head;
  alignas(kDeviceCacheLineBytes) NodeIndex m_tail;
};

} // namespace lockfree

} // namespace gpu
