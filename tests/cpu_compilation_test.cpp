// Compilation smoke test: instantiates every CPU structure with int, under
// every field layout, and exercises each public operation once. Also checks
// that the objects really land on a cache line.

#include "cpu/lockfree/lockfree_list.hpp"
#include "cpu/lockfree/lockfree_queue.hpp"
#include "cpu/lockfree/lockfree_stack.hpp"
#include "cpu/mutex/mutex_list.hpp"
#include "cpu/mutex/mutex_list_hand_over_hand.hpp"
#include "cpu/mutex/mutex_queue.hpp"
#include "cpu/mutex/mutex_queue_two_lock.hpp"
#include "cpu/mutex/mutex_stack.hpp"
#include "cpu/spinlock/spinlock_list.hpp"
#include "cpu/spinlock/spinlock_list_hand_over_hand.hpp"
#include "cpu/spinlock/spinlock_queue.hpp"
#include "cpu/spinlock/spinlock_queue_two_lock.hpp"
#include "cpu/spinlock/spinlock_stack.hpp"

#include "cpu/shared/cache.hpp"

#include <cstdint>
#include <cstdio>

namespace {

using cpu::cacheLineSize;
using cpu::NoPad;
using cpu::PadLockFromData;
using cpu::PadSyncPoints;
using cpu::PadSyncPointsAndLockFromData;

// Is the two-lock queue instantiable under this layout? PadLockFromData is the
// one cell it rejects, so exerciseAll skips it there rather than tripping the
// structure's own static_assert.
inline constexpr bool twoLockQueueSupports(cpu::Layout L) {
  return !(L.padLockFromData && !L.padSyncPoints);
}

// checks that every object given starts on a cache-line boundary.
template <typename... Ss> bool startOnCacheLine(const Ss &...objects) {
  static_assert((... && (alignof(Ss) == cacheLineSize)),
                "structure is not aligned to a cache line");

  return (... &&
          (reinterpret_cast<std::uintptr_t>(&objects) % cacheLineSize == 0));
}

// Instantiate and exercise every structure under one layout.
template <cpu::Layout L> bool exerciseAll() {
  constexpr std::size_t nodesPerThread = 1;
  constexpr std::size_t numThreads = 1;

  cpu::mutex::Stack<int, L> lbStack(nodesPerThread, numThreads);
  cpu::mutex::Queue<int, L> lbQueue(nodesPerThread, numThreads);
  cpu::mutex::List<int, L> lbList(nodesPerThread, numThreads);
  cpu::mutex::HandOverHandList<int, L> lbHohList(nodesPerThread, numThreads);
  cpu::lockfree::Stack<int, L> lfStack(nodesPerThread, numThreads);
  cpu::lockfree::Queue<int, L> lfQueue(nodesPerThread, numThreads);
  cpu::lockfree::List<int, L> lfList(nodesPerThread, numThreads);
  cpu::spinlock::Stack<int, L> slStack(nodesPerThread, numThreads);
  cpu::spinlock::Queue<int, L> slQueue(nodesPerThread, numThreads);
  cpu::spinlock::List<int, L> slList(nodesPerThread, numThreads);
  cpu::spinlock::HandOverHandList<int, L> slHohList(nodesPerThread, numThreads);

  bool aligned =
      startOnCacheLine(lbStack, lbQueue, lbList, lbHohList, lfStack, lfQueue,
                       lfList, slStack, slQueue, slList, slHohList);

  // Single-threaded smoke test: the sole caller is logical thread 0.
  constexpr std::size_t threadId = 0;

  lbStack.push(1, threadId);
  (void)lbStack.pop();

  lbQueue.enqueue(1, threadId);
  (void)lbQueue.dequeue();

  if constexpr (twoLockQueueSupports(L)) {
    cpu::mutex::QueueTwoLock<int, L> lbQueueTwoLock(nodesPerThread, numThreads);
    lbQueueTwoLock.enqueue(1, threadId);
    (void)lbQueueTwoLock.dequeue();

    cpu::spinlock::QueueTwoLock<int, L> slQueueTwoLock(nodesPerThread,
                                                       numThreads);
    slQueueTwoLock.enqueue(1, threadId);
    (void)slQueueTwoLock.dequeue();

    aligned = aligned && startOnCacheLine(lbQueueTwoLock, slQueueTwoLock);
  }

  (void)lbList.insert(1, threadId);
  (void)lbList.find(1);
  (void)lbList.remove(1);

  (void)lbHohList.insert(1, threadId);
  (void)lbHohList.find(1);
  (void)lbHohList.remove(1);

  lfStack.push(1, threadId);
  (void)lfStack.pop();

  lfQueue.enqueue(1, threadId);
  (void)lfQueue.dequeue();

  (void)lfList.insert(1, threadId);
  (void)lfList.find(1);
  (void)lfList.remove(1);

  slStack.push(1, threadId);
  (void)slStack.pop();

  slQueue.enqueue(1, threadId);
  (void)slQueue.dequeue();

  (void)slList.insert(1, threadId);
  (void)slList.find(1);
  (void)slList.remove(1);

  (void)slHohList.insert(1, threadId);
  (void)slHohList.find(1);
  (void)slHohList.remove(1);

  return aligned;
}

} // namespace

int main() {
  bool aligned = exerciseAll<NoPad>();
  aligned = exerciseAll<PadLockFromData>() && aligned;
  aligned = exerciseAll<PadSyncPoints>() && aligned;
  aligned = exerciseAll<PadSyncPointsAndLockFromData>() && aligned;

  if (!aligned) {
    std::fprintf(stderr, "a structure did not start on a cache line\n");
    return 1;
  }

  return 0;
}
