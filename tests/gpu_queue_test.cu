// Correctness tests for the three GPU queue variants.
//
// Scenarios: sequential enqueue/dequeue, concurrent fill then drain, and the
// mixed operation shape used by the benchmark. The oracles check item
// conservation and per-producer FIFO order.

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/lockfree/gpu_lockfree_queue.cuh"
#include "gpu/shared/gpu_cuda_utils.cuh"
#include "gpu/shared/gpu_node_pool.cuh"
#include "gpu/spinlock/gpu_spinlock_queue.cuh"
#include "gpu/spinlock/gpu_spinlock_queue_two_lock.cuh"
#include "support/gpu_queue_workload.cuh"
#include "support/gpu_test_harness.cuh"
#include "support/gpu_queue_oracles.hpp"

using namespace gpu_test;

// Named namespace keeps kernel linkage simple across nvcc versions.
namespace queuetest {

using queue_oracle::checkConservation;
using queue_oracle::checkPerProducerFifo;
using queue_oracle::encodingFits;
using queue_oracle::flatten;
using queue_oracle::Key;
using queue_oracle::makeKey;

// --- kernels -------------------------------------------------------------
//
// Queues live in raw device memory and are initialized by a kernel.

// Single-threaded prefill through the real enqueue path.
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
                           int opsPerThread) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  gpu::NodeAllocator<Key> allocator = gpu::threadAllocator(pool, threadId);
  for (int op = 0; op < opsPerThread; ++op) {
    queue->enqueue(makeKey(threadId, op), allocator);
  }
}

// Bounded drain; the host repeats until a round removes nothing.
template <typename QueueT>
__global__ void drainKernel(QueueT *queue, Key *out, int *counts, int capacity,
                            int totalThreads) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= totalThreads) {
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

template <typename QueueT>
__global__ void dequeueOneKernel(QueueT *queue, Key *out, int *success) {
  Key value = 0;
  if (queue->dequeue(value)) {
    *out = value;
    *success = 1;
  } else {
    *success = 0;
  }
}

template <typename QueueT>
__global__ void mixedKernel(QueueT *queue, gpu::PoolView<Key> pool,
                            int opsPerThread, int enqueueOps,
                            int dequeueOps, Key *out, int *counts) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  gpu::NodeAllocator<Key> allocator = gpu::threadAllocator(pool, threadId);
  gpubench::QueueWorkload workload(threadId, enqueueOps, dequeueOps);
  int produced = 0;
  int taken = 0;
  for (int op = 0; op < opsPerThread; ++op) {
    if (workload.next() == gpubench::QueueOp::Enqueue) {
      queue->enqueue(makeKey(threadId, produced), allocator);
      ++produced;
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

__global__ void workloadCountKernel(int enqueueOps, int dequeueOps,
                                    int *observedEnqueues,
                                    int *observedDequeues) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  gpubench::QueueWorkload workload(threadId, enqueueOps, dequeueOps);

  int enqueues = 0;
  int dequeues = 0;
  for (int op = 0; op < enqueueOps + dequeueOps; ++op) {
    if (workload.next() == gpubench::QueueOp::Enqueue) {
      ++enqueues;
    } else {
      ++dequeues;
    }
  }

  observedEnqueues[threadId] = enqueues;
  observedDequeues[threadId] = dequeues;
}

// --- host-side scenario plumbing -----------------------------------------

// Fresh pool and queue for one scenario.
template <typename QueueT> struct Fixture {
  Fixture(int prefillCapacity, int nodesPerThread, int threads)
      : pool(prefillCapacity, nodesPerThread, threads), queue(1) {
    gpubench::initQueueKernel<QueueT><<<1, 1>>>(queue.get(), pool.view());
    GPU_CUDA_CHECK_KERNEL();
  }

  gpu::NodePool<Key> pool;
  gpu::DeviceBuffer<QueueT> queue;
};

template <typename QueueT> bool dequeueOne(QueueT *queue, Key &value) {
  gpu::DeviceBuffer<Key> out(1);
  gpu::DeviceBuffer<int> success(1);
  out.zero();
  success.zero();

  dequeueOneKernel<QueueT><<<1, 1>>>(queue, out.get(), success.get());
  GPU_CUDA_CHECK_KERNEL();

  int hostSuccess = 0;
  success.copyToHost(&hostSuccess, 1);
  if (hostSuccess == 0) {
    return false;
  }
  out.copyToHost(&value, 1);
  return true;
}

// Drain all remaining items while preserving each thread's observation order.
template <typename QueueT>
std::vector<std::vector<Key>> drainAll(QueueT *queue, int blocks,
                                       int blockDim, int capacity) {
  const int threads = blocks * blockDim;
  const std::size_t slots = static_cast<std::size_t>(threads) *
                            static_cast<std::size_t>(capacity);

  gpu::DeviceBuffer<Key> out(slots);
  gpu::DeviceBuffer<int> counts(static_cast<std::size_t>(threads));
  std::vector<Key> hostOut(slots);
  std::vector<int> hostCounts(static_cast<std::size_t>(threads));
  std::vector<std::vector<Key>> records(static_cast<std::size_t>(threads));

  while (true) {
    counts.zero();
    drainKernel<QueueT><<<blocks, blockDim>>>(queue, out.get(), counts.get(),
                                              capacity, threads);
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

  Key value = 0;
  CHECK(!dequeueOne(fixture.queue.get(), value));

  prefillKernel<QueueT><<<1, 1>>>(fixture.queue.get(), fixture.pool.view(),
                                  kItems, /*prefillProducer=*/0);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue sequential prefill");

  const std::vector<std::vector<Key>> records =
      drainAll(fixture.queue.get(), /*blocks=*/1, /*blockDim=*/1, kItems);

  // No concurrency: check exact FIFO order directly.
  const std::vector<Key> observed = flatten(records);
  CHECK_EQ(observed.size(), static_cast<std::size_t>(kItems));
  for (std::size_t i = 0; i < observed.size(); ++i) {
    CHECK_EQ(observed[i], makeKey(0, static_cast<int>(i)));
  }

  CHECK(!dequeueOne(fixture.queue.get(), value));
}

void queueWorkloadCountsExact() {
  constexpr int kBlocks = 2;
  constexpr int kBlockDim = 32;
  constexpr int kThreads = kBlocks * kBlockDim;
  constexpr int kEnqueueOps = 10000;
  constexpr int kDequeueOps = 8000;

  gpu::DeviceBuffer<int> observedEnqueues(kThreads);
  gpu::DeviceBuffer<int> observedDequeues(kThreads);
  observedEnqueues.zero();
  observedDequeues.zero();

  workloadCountKernel<<<kBlocks, kBlockDim>>>(
      kEnqueueOps, kDequeueOps, observedEnqueues.get(),
      observedDequeues.get());
  GPU_CUDA_CHECK_KERNEL();

  std::vector<int> hostEnqueues(kThreads);
  std::vector<int> hostDequeues(kThreads);
  observedEnqueues.copyToHost(hostEnqueues.data(), kThreads);
  observedDequeues.copyToHost(hostDequeues.data(), kThreads);

  for (int t = 0; t < kThreads; ++t) {
    CHECK_EQ(hostEnqueues[static_cast<std::size_t>(t)], kEnqueueOps);
    CHECK_EQ(hostDequeues[static_cast<std::size_t>(t)], kDequeueOps);
  }
}

template <typename QueueT> void queueFillDrain() {
  const Config &cfg = gpuConfig();
  const int blocks = cfg.blocks;
  const int blockDim = cfg.blockDim;
  const int threads = static_cast<int>(gpubench::totalThreads(cfg));
  if (!encodingFits(threads, cfg.opsPerThread)) {
    return;
  }

  Fixture<QueueT> fixture(/*prefillCapacity=*/0, cfg.opsPerThread, threads);

  fillKernel<QueueT><<<blocks, blockDim>>>(
      fixture.queue.get(), fixture.pool.view(), cfg.opsPerThread);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue fill");

  // Extra space reduces the number of drain rounds.
  const int capacity = cfg.opsPerThread * 8;
  const std::vector<std::vector<Key>> records =
      drainAll(fixture.queue.get(), blocks, blockDim, capacity);

  std::vector<Key> expected;
  for (int t = 0; t < threads; ++t) {
    for (int op = 0; op < cfg.opsPerThread; ++op) {
      expected.push_back(makeKey(t, op));
    }
  }

  checkConservation(flatten(records), expected);
  checkPerProducerFifo(records, threads);
}

template <typename QueueT> void queueMixed() {
  const Config &cfg = gpuConfig();
  const int blocks = cfg.blocks;
  const int blockDim = cfg.blockDim;
  const int threads = static_cast<int>(gpubench::totalThreads(cfg));
  const int enqueueOps = gpubench::enqueueOpsPerThread(cfg);
  const int dequeueOps = gpubench::dequeueOpsPerThread(cfg);
  const int prefill = static_cast<int>(gpubench::prefillFor(cfg));
  // Prefill uses a synthetic producer id so FIFO is checked for it too.
  const int prefillProducer = threads;
  if (!encodingFits(threads + 1, std::max(cfg.opsPerThread, prefill))) {
    return;
  }

  Fixture<QueueT> fixture(prefill, enqueueOps, threads);

  prefillKernel<QueueT><<<1, 1>>>(fixture.queue.get(), fixture.pool.view(),
                                  prefill, prefillProducer);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("gpu queue prefill");

  const std::size_t slots = static_cast<std::size_t>(threads) *
                            static_cast<std::size_t>(cfg.opsPerThread);
  gpu::DeviceBuffer<Key> out(slots);
  gpu::DeviceBuffer<int> counts(static_cast<std::size_t>(threads));
  counts.zero();

  mixedKernel<QueueT><<<blocks, blockDim>>>(
      fixture.queue.get(), fixture.pool.view(), cfg.opsPerThread, enqueueOps,
      dequeueOps, out.get(), counts.get());
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

  // Drain leftovers so conservation can be checked exactly.
  const std::vector<std::vector<Key>> leftover = drainAll(
      fixture.queue.get(), blocks, blockDim, cfg.opsPerThread * 8);

  std::vector<Key> observed = flatten(records);
  const std::vector<Key> tail = flatten(leftover);
  observed.insert(observed.end(), tail.begin(), tail.end());

  std::vector<Key> expected;
  for (int i = 0; i < prefill; ++i) {
    expected.push_back(makeKey(prefillProducer, i));
  }
  for (int t = 0; t < threads; ++t) {
    for (int op = 0; op < enqueueOps; ++op) {
      expected.push_back(makeKey(t, op));
    }
  }

  checkConservation(observed, expected);
  // Concurrent records and leftover-drain records are separate observations.
  checkPerProducerFifo(records, threads + 1);
  checkPerProducerFifo(leftover, threads + 1);
}

} // namespace queuetest

using namespace queuetest;

using SpinlockQueue = gpu::spinlock::Queue<Key>;
using SpinlockQueueTwoLock = gpu::spinlock::QueueTwoLock<Key>;
using LockfreeQueue = gpu::lockfree::Queue<Key>;

GPU_TEST(queue_workload_counts_exact) { queueWorkloadCountsExact(); }

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
