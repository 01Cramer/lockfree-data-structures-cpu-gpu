// Variant 4: CCLQ, the combined-and-CAS-loop queue of Zhang, Deng & Mu,
// "Toward Concurrent Lock-Free Queues on GPUs", IEICE Trans. Inf. & Syst.
// E97-D(7), 2014. Implemented from the pseudo code in Fig. 1.
//
// What it is, and what it is here to answer
//
// CCLQ is Michael & Scott's lock-free queue with two additions: a node carries
// 32 items instead of one, and the 32 lanes of a warp combine their operations
// into a single one before touching the shared structure. One CAS therefore
// links 32 items, and one CAS claims up to 32.
//
// The paper reports up to 40x over its own reimplementation of MSQ. The
// arithmetic matters when reading that: a batch node holds 32 items, so
// amortization alone accounts for a factor of ~32 before any question of
// synchronization arises. Warp-aggregated atomics are in any case a
// long-established NVIDIA technique, so the combining is not itself novel.
// This variant exists to separate those two effects -- how much of the
// advantage is amortization and how much is the synchronization mechanism --
// against variants 1-3, which hold the algorithm fixed and vary only the lock.
//
// -------------------------------------------------------------------------
// Deviations from the published pseudo code
//
// Every one of these is a correction, and every one is required on Volta or
// later. The design is sound; the printed code is not runnable on the target
// hardware. Zhang's evaluation was on Kepler and Fermi, where the threads of a
// warp share a program counter, and three of the four items below are places
// where the code depends on that.
//
//   1. scanWarp. A shared-memory Hillis-Steele scan with no synchronization
//      between its five steps. Replaced with __shfl_up_sync; see
//      gpu/shared/warp_scan.cuh, which carries the full argument. Effect on
//      results: none in the intended direction. A shuffle is at least as cheap
//      as the shared-memory round trip it replaces, so the corrected version
//      cannot be accused of handicapping CCLQ.
//
//   2. No fences anywhere. The paper marks global_head and global_tail
//      `volatile` and uses no __threadfence(). volatile is a code-generation
//      property: it stops the compiler caching an access in a register. It
//      orders nothing. As printed, nothing separates the writes of a node's 32
//      payloads, its startPos and its endPos from the CAS that links the node
//      into the queue, so a consumer can observe a linked node whose contents
//      have not arrived. Here the linking CAS carries release and the
//      consumer's read of nextPos carries acquire.
//
//      There is a subtlety specific to combining, and it is why a
//      __threadfence() would not have been enough on its own: the payloads are
//      written by all 32 lanes, and the linking CAS is executed by one of them.
//      A fence orders only the accesses of the thread that executes it, so
//      lane 31's fence says nothing about lane 5's store. The __syncwarp()
//      before the combining step is what places the other lanes' stores before
//      lane 31's release in the happens-before order, and the release then
//      carries all of them to any acquiring thread.
//
//   3. Enqueue drops Michael & Scott's tail-help branch. In Fig. 1 (b) the
//      `else CAS(global_tail, tail, tail + 1)` at line 32 sits inside the
//      `if (nextpos == -1)` branch, attached to the failed link CAS. When the
//      observed slot is already filled -- nextpos != -1, meaning some other
//      warp linked a node and has not yet advanced the tail -- the loop body
//      has no statement at all, so the iteration re-reads an unchanged
//      global_tail and spins. It makes progress only when some other thread
//      happens to advance the tail for it.
//
//      That outer else is not an optimization: it is the helping step, and it
//      is what makes the algorithm lock-free. As printed, CCLQ's enqueue
//      forfeits the central claim of the paper it appears in. Section 3's
//      linearizability argument does not cover this; it establishes only that
//      two CASes on one word cannot both succeed, which is a property of CAS
//      rather than of the algorithm. Restored here.
//
//   4. No barrier between the combiner filling the warp's output store and the
//      other lanes reading it. Fig. 1 (c) line 68 has every lane return
//      this_data_list[this_pos], with nothing between it and lane 31's writes
//      at lines 57 and 62. Implicit lockstep again. A __syncwarp() is added.
//
// Two things the pseudo code does not have at all, rather than has wrongly:
//
//   5. Dequeue cannot report failure. If the queue is empty the loop breaks at
//      line 46 with nowlength < num_request, and every lane still returns
//      this_data_list[this_pos] -- reading a slot the combiner never wrote. A
//      `bool dequeue(T&, ...)` is used here instead, and a warp that obtains
//      fewer items than it requested hands them to the lowest-numbered
//      requesters, the rest failing. Prefix order is the right choice rather
//      than an arbitrary one: it is the order the scan already assigned, so it
//      agrees with the order the items would have been enqueued in.
//
//   6. No bounds checks. Neither the node arrays nor the position array is
//      checked, and an overrun on a GPU does not fault: it returns an address
//      past the buffer and corrupts silently. Both are checked here against the
//      pool, and an exhausted pool sets a sticky flag the host turns into a
//      fatal diagnostic.
//
// One deviation that is *not* made: partial-node claiming is implemented in
// full (lines 49-64), including the case where a warp's request spans more than
// one node. It could have been left out -- under the warp-uniform operation
// assignment this experiment uses, every batch is exactly full and every
// request is exactly the same width, so start_pos only ever holds 0 or the
// node's end and the partial path is never taken. It is implemented anyway
// because it is the algorithm the paper describes, because omitting it would
// mean the variant could not be run with asymmetric producer and consumer
// widths at all, and because a whole-batch-only dequeue cannot be made
// conservation-safe without somewhere to park surplus items -- which is exactly
// the job partial claiming does.
//
// -------------------------------------------------------------------------
// Calling contract, which differs from variants 1-3
//
// enqueue and dequeue are warp-collective. All 32 lanes must call them, and
// must call the same one: a lane that has returned from the kernel makes the
// full-mask shuffles and the __syncwarp() undefined behaviour.
//
// That interacts with the activeLanes control. Variants 1-3 implement it by
// retiring lanes (`if (lane >= activeLanes) return;`), which is not available
// here. CCLQ instead expresses non-participation through the `request` flag --
// which is Zhang's own mechanism, described in the paper as threads with
// "request[threadIdx.x] = 0" that "just assist to execute prefix-sum". So the
// control becomes `request = lane < activeLanes`, and the inactive lanes stay
// resident and do useful work in the scan.
//
// This is a real difference in what activeLanes means for this variant, not a
// harness detail, and it should be stated as one: for variants 1-3 the retired
// lanes contribute nothing, while for CCLQ they contribute the combining. It is
// also the reason CCLQ is expected to be far less sensitive to activeLanes
// than the lock-based variants -- which is the point of measuring it.

#pragma once

#include "gpu/shared/atomics.cuh"
#include "gpu/shared/batch_pool.cuh"
#include "gpu/shared/warp_scan.cuh"

namespace gpu {

namespace lockfree {

template <typename T> class alignas(kDeviceCacheLineBytes) BatchQueue {
public:
  // Called by exactly one thread before any operation.
  //
  // head and tail both start at position 0, which is a dummy that is never
  // linked: the live range is [head + 1, tail], so head == tail == 0 is empty.
  // nextPos is already all -1 from the pool's construction.
  __device__ void initialize(const BatchPoolView<T> &pool) {
    m_pool = pool;
    headRef().store(0, cuda::memory_order_relaxed);
    tailRef().store(0, cuda::memory_order_relaxed);
  }

  // Warp-collective. `request` says whether this lane has an item to add;
  // lanes with request == false still participate in the scan.
  //
  // Returns true if THIS lane's item was enqueued. False either because the
  // lane did not request, or because the pool is exhausted -- which is a
  // run-invalidating error the host detects through the pool's overflow flag,
  // never a legitimate "queue full".
  __device__ bool enqueue(const T &value, bool request,
                          BatchAllocator<T> &allocator) {
    // Fig. 1 (b) line 22. Inclusive scan; this lane's slot in the batch is the
    // exclusive prefix, and lane 31 holds the warp total.
    const int inclusive = warpInclusiveScan(request ? 1 : 0);
    const int slot = inclusive - 1;
    const int total = broadcastFromCombiner(inclusive);
    if (total == 0) {
      // No lane in this warp wants to enqueue. Nothing to combine, and no node
      // is consumed -- which matters, because a node spent on an empty batch
      // would still occupy a position.
      return false;
    }

    // Fig. 1 (b) lines 20-21. Every lane advances its own cursor so the warp's
    // allocators stay in step across calls; the combiner's value is then
    // broadcast so that agreement is structural rather than an assumption about
    // uniform control flow.
    BatchNodeIndex node = allocator.take();
    node = broadcastFromCombiner(node);
    if (node == kNullBatchNode) {
      return false;
    }

    // Fig. 1 (b) lines 23-24. One coalesced store across the warp.
    if (request) {
      m_pool.data[node * kBatchCapacity + slot] = value;
    }

    // Deviation 2. Places the payload stores of every lane before the
    // combiner's release below, in the happens-before order. Not replaceable
    // by a fence in the combiner: a fence orders only its own thread's
    // accesses.
    __syncwarp();

    int linked = 1;
    if (isCombiner()) {
      // Fig. 1 (b) line 26. endPos is written once and never changes;
      // startPos is the word consumers will CAS to claim a range.
      startRef(node).store(0, cuda::memory_order_relaxed);
      endRef(node).store(total, cuda::memory_order_relaxed);
      linked = link(node) ? 1 : 0;
    }
    linked = broadcastFromCombiner(linked);

    return request && linked != 0;
  }

  // Warp-collective. `warpStore` must point at kBatchCapacity elements of
  // storage private to this warp -- the paper's this_data_list. Shared memory
  // is the natural home for it; the queue does not care, so the choice stays
  // with the caller.
  //
  // Returns true if THIS lane received an item.
  __device__ bool dequeue(T &out, bool request, T *warpStore) {
    // Fig. 1 (c) line 39.
    const int inclusive = warpInclusiveScan(request ? 1 : 0);
    const int slot = inclusive - 1;
    const int total = broadcastFromCombiner(inclusive);
    if (total == 0) {
      return false;
    }

    // Fig. 1 (c) lines 40-67.
    int obtained = 0;
    if (isCombiner()) {
      obtained = claim(total, warpStore);
    }
    obtained = broadcastFromCombiner(obtained);

    // Deviation 4. Publishes the combiner's writes into warpStore to the lanes
    // that are about to read them.
    __syncwarp();

    // Deviation 5. The paper's line 68 reads unconditionally; a warp that got
    // fewer items than it asked for gives them to the lowest requesters and the
    // rest fail.
    if (request && slot < obtained) {
      out = warpStore[slot];
      return true;
    }
    return false;
  }

private:
  // Fig. 1 (b) lines 27-35. Link a filled node at the tail.
  //
  // Returns false only when the position array is exhausted.
  __device__ bool link(BatchNodeIndex node) {
    QueuePosition tail = 0;
    while (true) {
      tail = tailRef().load(cuda::memory_order_relaxed);
      if (tail + 1 >= m_pool.positionCapacity) {
        // Deviation 6. The paper has no such check, and without it this writes
        // past the end of nextPos without faulting.
        flagOverflow();
        return false;
      }

      const BatchNodeIndex next =
          nextRef(tail + 1).load(cuda::memory_order_relaxed);
      if (next == kNullBatchNode) {
        BatchNodeIndex expected = kNullBatchNode;
        // Release: publishes this node's 32 payloads, startPos and endPos.
        // Reaches the other lanes' stores through the __syncwarp() above.
        if (nextRef(tail + 1)
                .compare_exchange_strong(expected, node,
                                         cuda::memory_order_release,
                                         cuda::memory_order_relaxed)) {
          break;
        }
      }

      // Deviation 3, and the paper's line 32 folded into it. Both the
      // lost-race case (the slot was empty and our CAS failed) and the
      // already-linked case (the slot was full when we read it) are handled by
      // helping the tail forward. The second of those is the branch Fig. 1 (b)
      // does not have, and without it this loop spins on an unchanged tail.
      helpAdvanceTail(tail);
    }

    // Fig. 1 (b) line 35. A hint: it is allowed to fail, because another
    // warp's helping step has then already advanced the tail.
    helpAdvanceTail(tail);
    return true;
  }

  // Fig. 1 (c) lines 42-66. Claim up to `wanted` items into warpStore and
  // return how many were obtained. Executed by the combiner alone.
  __device__ int claim(int wanted, T *warpStore) {
    int obtained = 0;

    while (obtained < wanted) {
      const QueuePosition head = headRef().load(cuda::memory_order_relaxed);
      const QueuePosition tail = tailRef().load(cuda::memory_order_relaxed);
      if (head + 1 >= m_pool.positionCapacity) {
        // Deviation 6 again. Reaching this means every position ever linked has
        // been fully consumed, so reporting empty is the right answer -- but
        // the node budget runs out before the position array does (the pool
        // sizes positions at nodes + 2), so in practice the enqueue side stops
        // first. The check is here because the read below would otherwise be
        // out of bounds, and an out-of-bounds read does not fault.
        break;
      }

      // Acquire: pairs with the release in link(), so this node's payloads,
      // startPos and endPos are all visible below.
      const BatchNodeIndex next =
          nextRef(head + 1).load(cuda::memory_order_acquire);

      if (head == tail) {
        if (next == kNullBatchNode) {
          // Genuinely empty. head and tail are read in that order and both are
          // monotonically increasing with head <= tail always, so head == tail
          // here means there was no live position at the time of the second
          // read.
          break;
        }
        // A node is linked at tail + 1 but its enqueuer has not advanced the
        // tail yet. Help, then retry.
        helpAdvanceTail(tail);
        continue;
      }

      // head != tail, so position head + 1 is within the live range and was
      // linked before the tail reached it. `next` is therefore a real node
      // index and not -1. (Michael & Scott's re-read of the head is not needed
      // here: claiming is governed by the CAS on startPos, not by the head, so
      // a stale head can at worst make this iteration claim from a node that
      // has since been fully consumed -- which the startPos check below
      // catches.)
      const int oldStart = startRef(next).load(cuda::memory_order_relaxed);
      const int end = endRef(next).load(cuda::memory_order_relaxed);

      if (oldStart >= end) {
        // Fig. 1 (c) lines 51-52. Node fully consumed; retire its position.
        QueuePosition expected = head;
        headRef().compare_exchange_strong(expected, head + 1,
                                          cuda::memory_order_relaxed,
                                          cuda::memory_order_relaxed);
        continue;
      }

      // Fig. 1 (c) lines 53-63. The two arms of the paper's if/else -- take
      // exactly what is still wanted, or take all that is left and come back
      // for more -- differ only in the size of the claim, so they are one
      // statement here.
      //
      // (The printed code is ambiguous at this point: the `else` on line 58 is
      // a dangling else that C binds to the CAS on line 55 rather than to the
      // comparison on line 54, which would make a failed CAS fall into the
      // partial arm. The indentation says the comparison was meant, and that
      // is what is implemented.)
      const int available = end - oldStart;
      const int wantedNow = wanted - obtained;
      const int take = wantedNow < available ? wantedNow : available;

      int expected = oldStart;
      // The linearization point of the dequeue. Relaxed is sufficient: the
      // payloads were already made visible by the acquire on nextPos above,
      // and this CAS publishes nothing of its own.
      if (startRef(next).compare_exchange_strong(expected, oldStart + take,
                                                 cuda::memory_order_relaxed,
                                                 cuda::memory_order_relaxed)) {
        for (int i = 0; i < take; ++i) {
          warpStore[obtained] =
              m_pool.data[next * kBatchCapacity + oldStart + i];
          ++obtained;
        }
      }
      // A failed CAS means another warp claimed from this node first; loop and
      // re-read.
    }

    return obtained;
  }

  __device__ void helpAdvanceTail(QueuePosition tail) {
    QueuePosition expected = tail;
    tailRef().compare_exchange_strong(expected, tail + 1,
                                      cuda::memory_order_relaxed,
                                      cuda::memory_order_relaxed);
  }

  __device__ void flagOverflow() {
    DeviceAtomicRef<int> flag(*m_pool.overflow);
    flag.store(1, cuda::memory_order_relaxed);
  }

  __device__ DeviceAtomicRef<BatchNodeIndex> nextRef(QueuePosition position) {
    return DeviceAtomicRef<BatchNodeIndex>(m_pool.nextPos[position]);
  }

  __device__ DeviceAtomicRef<int> startRef(BatchNodeIndex node) {
    return DeviceAtomicRef<int>(m_pool.startPos[node]);
  }

  __device__ DeviceAtomicRef<int> endRef(BatchNodeIndex node) {
    return DeviceAtomicRef<int>(m_pool.endPos[node]);
  }

  __device__ DeviceAtomicRef<QueuePosition> headRef() {
    return DeviceAtomicRef<QueuePosition>(m_head);
  }

  __device__ DeviceAtomicRef<QueuePosition> tailRef() {
    return DeviceAtomicRef<QueuePosition>(m_tail);
  }

  // Read-only after initialize().
  BatchPoolView<T> m_pool;

  // Two independently contended words, one line each: enqueuers hammer the
  // tail and consumers hammer the head. Held fixed across all variants; see
  // atomics.cuh.
  alignas(kDeviceCacheLineBytes) QueuePosition m_head;
  alignas(kDeviceCacheLineBytes) QueuePosition m_tail;
};

} // namespace lockfree

} // namespace gpu
