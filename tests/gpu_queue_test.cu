// Correctness tests for the three GPU queue variants.
//
// The oracles and the value encoding live in tests/support/queue_oracles.hpp,
// which carries the argument for why the CPU ones transfer unchanged.
//
// Three scenarios, in increasing strength:
//
//   sequential  one thread, enqueue then dequeue. No concurrency at all. It
//               exists to fail first: a bug here is in the algorithm's plain
//               structure, and finding it under 512 concurrent threads is
//               strictly harder.
//   fill/drain  every thread enqueues its own range, then (after the kernel
//               boundary, which is a grid-wide barrier) every thread drains.
//               Separating the phases means dequeue never has to distinguish
//               "empty" from "a producer has not got there yet", so the drain
//               terminates without any cross-block spin -- which would require
//               the whole grid to be resident and would deadlock at high block
//               counts.
//   mixed       the shape the benchmark actually runs: a pre-filled queue,
//               producer warps and consumer warps live at the same time,
//               warp-uniform operation assignment. Conservation is checked
//               over (prefill + produced) against (consumed + what was left),
//               so nothing has to be assumed about how many dequeues found
//               work.
//
// Run these under compute-sanitizer as well as plain: --tool memcheck catches
// a pool overrun (which does not fault -- it silently returns an index past
// the slice), --tool racecheck catches a missing fence, --tool synccheck
// catches warp-synchronization misuse.

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/lockfree/lockfree_queue.cuh"
#include "gpu/shared/cuda_error.cuh"
#include "gpu/shared/node_pool.cuh"
#include "gpu/spinlock/spinlock_queue.cuh"
#include "gpu/spinlock/spinlock_queue_two_lock.cuh"
#include "support/gpu_test_harness.cuh"
#include "support/queue_oracles.hpp"

using namespace gpu_test;

// A named namespace, not an anonymous one. __global__ functions with internal
// linkage have been a moving target across nvcc versions, and there is nothing
// to gain by finding out which side of it the local toolkit falls on.
namespace queuetest {

using queue_oracle::checkConservation;
using queue_oracle::checkPerProducerFifo;
using queue_oracle::encodingFits;
using queue_oracle::flatten;
using queue_oracle::Key;
using queue_oracle::makeKey;

// A lane participates iff its index within the warp is below activeLanes. This
// is the one line the whole experiment turns on, and it is tested here in the
// same form the benchmark uses so the tests cover the configuration that gets
// measured. Per warp, not per block: with blockDim > 32 a per-block test would
// disable whole warps instead of lanes.
__host__ __device__ inline bool laneParticipates(int threadId,
                                                 int activeLanes) {
  return (threadId % 32) < activeLanes;
}

// --- kernels -------------------------------------------------------------
//
// All of them take the queue by pointer into device memory. The queue objects
// have no constructors: they are raw cudaMalloc'd storage set up by
// initQueueKernel, which is what lets one flat object be shared by every
// block.

template <typename QueueT>
__global__ void initQueueKernel(QueueT *queue, gpu::PoolView<Key> pool) {
  queue->initialize(pool);
}

// Single-threaded, untimed, and deliberately through the real enqueue path:
// the starting contents are then exactly what the algorithm produces, not a
// hand-built chain that might differ from it.
template <typename QueueT>
__global__ void prefillKernel(QueueT *queue, gpu::PoolView<Key> pool,
                              int count, int prefillProducer) {
  gpu::NodeAllocator<Key> allocator = gpu::prefillAllocator(pool);
  for (int i = 0; i < count; ++i) {
    queue->enqueue(makeKey(prefillProducer, i), allocator);
  }
}

template <typename QueueT>
__global__ void fillKernel(QueueT *queue, gpu::PoolView<Key> pool,
                           int opsPerThread, int activeLanes) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (!laneParticipates(threadId, activeLanes)) {
    return;
  }
  gpu::NodeAllocator<Key> allocator = gpu::threadAllocator(pool, threadId);
  for (int op = 0; op < opsPerThread; ++op) {
    queue->enqueue(makeKey(threadId, op), allocator);
  }
}

// Bounded by `capacity` so the output slice cannot overrun; the host repeats
// the kernel until a round drains nothing, which both bounds the memory and
// keeps each thread's records in dequeue order.
template <typename QueueT>
__global__ void drainKernel(QueueT *queue, Key *out, int *counts, int capacity,
                            int activeLanes) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (!laneParticipates(threadId, activeLanes)) {
    return;
  }
  int taken = 0;
  Key value = 0;
  while (taken < capacity && queue->dequeue(value)) {
    out[static_cast<long long>(threadId) * capacity + taken] = value;
    ++taken;
  }
  counts[threadId] = taken;
}

// Warp-uniform operation assignment: a whole warp enqueues or a whole warp
// dequeues, never a mix within one warp. Two reasons, both from the design.
// It removes divergence between two different queue operations, which would
// otherwise be confounded with the activeLanes effect being measured; and it
// puts the operation mix on an axis (which warps) that is independent of the
// lane axis.
template <typename QueueT>
__global__ void mixedKernel(QueueT *queue, gpu::PoolView<Key> pool,
                            int opsPerThread, int activeLanes, Key *out,
                            int *counts) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (!laneParticipates(threadId, activeLanes)) {
    return;
  }
  const int warpId = threadId / 32;
  const bool isProducer = (warpId % 2) == 0;

  gpu::NodeAllocator<Key> allocator = gpu::threadAllocator(pool, threadId);
  int taken = 0;
  for (int op = 0; op < opsPerThread; ++op) {
    if (isProducer) {
      queue->enqueue(makeKey(threadId, op), allocator);
    } else {
      Key value = 0;
      if (queue->dequeue(value)) {
        out[static_cast<long long>(threadId) * opsPerThread + taken] = value;
        ++taken;
      }
    }
  }
  counts[threadId] = taken;
}

// --- host-side scenario plumbing -----------------------------------------

// Everything one run needs: a fresh pool and a fresh queue object. Built per
// scenario rather than reset between them, so no scenario can inherit state
// from the one before it.
template <typename QueueT> struct Fixture {
  Fixture(int prefillCapacity, int nodesPerThread, int threads)
      : pool(prefillCapacity, nodesPerThread, threads), queue(1) {
    initQueueKernel<QueueT><<<1, 1>>>(queue.get(), pool.view());
    GPU_CUDA_CHECK_KERNEL();
  }

  gpu::NodePool<Key> pool;
  gpu::DeviceBuffer<QueueT> queue;
};

// Repeat the bounded drain until a round finds nothing. Returns each thread's
// records in the order that thread dequeued them.
template <typename QueueT>
std::vector<std::vector<Key>> drainAll(QueueT *queue, int blocks,
                                       int activeLanes, int capacity) {
  const int threads = blocks * kWarpSize;
  const std::size_t slots = static_cast<std::size_t>(threads) *
                            static_cast<std::size_t>(capacity);

  gpu::DeviceBuffer<Key> out(slots);
  gpu::DeviceBuffer<int> counts(static_cast<std::size_t>(threads));
  std::vector<Key> hostOut(slots);
  std::vector<int> hostCounts(static_cast<std::size_t>(threads));
  std::vector<std::vector<Key>> records(static_cast<std::size_t>(threads));

  while (true) {
    counts.zero();
    drainKernel<QueueT><<<blocks, kWarpSize>>>(queue, out.get(), counts.get(),
                                               capacity, activeLanes);
    GPU_CUDA_CHECK_KERNEL();
    counts.copyToHost(hostCounts.data(), static_cast<std::size_t>(threads));

    long long roundTotal = 0;
    for (int t = 0; t < threads; ++t) {
      roundTotal += hostCounts[static_cast<std::size_t>(t)];
    }
    if (roundTotal == 0) {
      return records;
    }

    out.copyToHost(hostOut.data(), slots);
    for (int t = 0; t < threads; ++t) {
      const std::size_t base =
          static_cast<std::size_t>(t) * static_cast<std::size_t>(capacity);
      const std::size_t n =
          static_cast<std::size_t>(hostCounts[static_cast<std::size_t>(t)]);
      records[static_cast<std::size_t>(t)].insert(
          records[static_cast<std::size_t>(t)].end(), hostOut.begin() + base,
          hostOut.begin() + base + n);
    }
  }
}

// --- scenarios -----------------------------------------------------------

template <typename QueueT> void queueSequential() {
  constexpr int kItems = 4096;
  if (!encodingFits(1, kItems)) {
    return;
  }

  Fixture<QueueT> fixture(/*prefillCapacity=*/kItems, /*nodesPerThread=*/0,
                          /*threads=*/1);
  prefillKernel<QueueT><<<1, 1>>>(fixture.queue.get(), fixture.pool.view(),
                                  kItems, /*prefillProducer=*/0);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue sequential prefill");

  const std::vector<std::vector<Key>> records =
      drainAll(fixture.queue.get(), /*blocks=*/1, /*activeLanes=*/1, kItems);

  // One producer, one consumer, no concurrency: the order is not merely
  // per-producer monotonic, it is exactly the enqueue order. Checked directly
  // rather than through the weaker oracles.
  const std::vector<Key> observed = flatten(records);
  CHECK_EQ(observed.size(), static_cast<std::size_t>(kItems));
  for (std::size_t i = 0; i < observed.size(); ++i) {
    CHECK_EQ(observed[i], makeKey(0, static_cast<int>(i)));
  }
}

template <typename QueueT> void queueFillDrain() {
  const Config &cfg = gpuConfig();
  const int blocks = cfg.warps;
  const int threads = blocks * kWarpSize;
  if (!encodingFits(threads, cfg.opsPerThread)) {
    return;
  }

  Fixture<QueueT> fixture(/*prefillCapacity=*/0, cfg.opsPerThread, threads);

  fillKernel<QueueT><<<blocks, kWarpSize>>>(
      fixture.queue.get(), fixture.pool.view(), cfg.opsPerThread,
      cfg.activeLanes);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue fill");

  // Slack over the fair share, so a greedy thread does not force many rounds.
  const int capacity = cfg.opsPerThread * 8;
  const std::vector<std::vector<Key>> records =
      drainAll(fixture.queue.get(), blocks, cfg.activeLanes, capacity);

  std::vector<Key> expected;
  for (int t = 0; t < threads; ++t) {
    if (!laneParticipates(t, cfg.activeLanes)) {
      continue;
    }
    for (int op = 0; op < cfg.opsPerThread; ++op) {
      expected.push_back(makeKey(t, op));
    }
  }

  checkConservation(flatten(records), expected);
  checkPerProducerFifo(records, threads);
}

template <typename QueueT> void queueMixed() {
  const Config &cfg = gpuConfig();
  const int blocks = cfg.warps;
  const int threads = blocks * kWarpSize;
  // Producers are the even warps; the prefill borrows the id one past the last
  // real thread, so its items are just another producer to the oracles and
  // their FIFO order is checked too.
  const int prefillProducer = threads;
  if (!encodingFits(threads + 1,
                         std::max(cfg.opsPerThread, cfg.prefill))) {
    return;
  }

  Fixture<QueueT> fixture(cfg.prefill, cfg.opsPerThread, threads);

  prefillKernel<QueueT><<<1, 1>>>(fixture.queue.get(), fixture.pool.view(),
                                  cfg.prefill, prefillProducer);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue prefill");

  const std::size_t slots = static_cast<std::size_t>(threads) *
                            static_cast<std::size_t>(cfg.opsPerThread);
  gpu::DeviceBuffer<Key> out(slots);
  gpu::DeviceBuffer<int> counts(static_cast<std::size_t>(threads));
  counts.zero();

  mixedKernel<QueueT><<<blocks, kWarpSize>>>(
      fixture.queue.get(), fixture.pool.view(), cfg.opsPerThread,
      cfg.activeLanes, out.get(), counts.get());
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue mixed run");

  std::vector<Key> hostOut(slots);
  std::vector<int> hostCounts(static_cast<std::size_t>(threads));
  out.copyToHost(hostOut.data(), slots);
  counts.copyToHost(hostCounts.data(), static_cast<std::size_t>(threads));

  std::vector<std::vector<Key>> records(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    const std::size_t base = static_cast<std::size_t>(t) *
                             static_cast<std::size_t>(cfg.opsPerThread);
    const std::size_t n =
        static_cast<std::size_t>(hostCounts[static_cast<std::size_t>(t)]);
    records[static_cast<std::size_t>(t)].assign(hostOut.begin() + base,
                                                hostOut.begin() + base + n);
  }

  // Whatever the consumers did not take is still in the queue. Draining it
  // afterwards is what makes conservation checkable without knowing in advance
  // how many dequeues found work.
  const std::vector<std::vector<Key>> leftover = drainAll(
      fixture.queue.get(), blocks, cfg.activeLanes, cfg.opsPerThread * 8);

  std::vector<Key> observed = flatten(records);
  const std::vector<Key> tail = flatten(leftover);
  observed.insert(observed.end(), tail.begin(), tail.end());

  std::vector<Key> expected;
  for (int i = 0; i < cfg.prefill; ++i) {
    expected.push_back(makeKey(prefillProducer, i));
  }
  for (int t = 0; t < threads; ++t) {
    const bool isProducer = ((t / 32) % 2) == 0;
    if (!isProducer || !laneParticipates(t, cfg.activeLanes)) {
      continue;
    }
    for (int op = 0; op < cfg.opsPerThread; ++op) {
      expected.push_back(makeKey(t, op));
    }
  }

  checkConservation(observed, expected);
  // The concurrent phase and the leftover drain are separate observation
  // sequences and are checked separately: the drain is not a continuation of
  // any one consumer's view.
  checkPerProducerFifo(records, threads + 1);
  checkPerProducerFifo(leftover, threads + 1);
}

} // namespace queuetest

using namespace queuetest;

using SpinlockQueue = gpu::spinlock::Queue<Key>;
using SpinlockQueueTwoLock = gpu::spinlock::QueueTwoLock<Key>;
using LockfreeQueue = gpu::lockfree::Queue<Key>;

GPU_TEST(queue_sequential_spinlock) { queueSequential<SpinlockQueue>(); }
GPU_TEST(queue_sequential_spinlock_two_lock) {
  queueSequential<SpinlockQueueTwoLock>();
}
GPU_TEST(queue_sequential_lockfree) { queueSequential<LockfreeQueue>(); }

GPU_TEST(queue_fill_drain_spinlock) { queueFillDrain<SpinlockQueue>(); }
GPU_TEST(queue_fill_drain_spinlock_two_lock) {
  queueFillDrain<SpinlockQueueTwoLock>();
}
GPU_TEST(queue_fill_drain_lockfree) { queueFillDrain<LockfreeQueue>(); }

GPU_TEST(queue_mixed_spinlock) { queueMixed<SpinlockQueue>(); }
GPU_TEST(queue_mixed_spinlock_two_lock) {
  queueMixed<SpinlockQueueTwoLock>();
}
GPU_TEST(queue_mixed_lockfree) { queueMixed<LockfreeQueue>(); }

int main(int argc, char **argv) { return gpu_test::runAll(argc, argv); }
