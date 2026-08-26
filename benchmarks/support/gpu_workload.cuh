// Common GPU queue workload used by all three variants.
//
// Each participating lane runs the same queue workload shape as the CPU queue
// benchmark: mixPct is the enqueue percentage, the remaining operations are
// dequeues, and the queue is prefilled with one item for every planned dequeue.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/shared/gpu_cuda_utils.cuh"
#include "gpu/shared/gpu_node_pool.cuh"
#include "support/gpu_energy.cuh"
#include "support/gpu_queue_workload.cuh"

namespace gpubench {

using Key = std::uint64_t;

// One point in the sweep.
struct Config {
  std::string variant;
  int blocks;
  int blockDim;
  int opsPerThread;
  int mixPct;
  int nodesPerThread;
};

// What one repetition produced.
struct Result {
  double milliseconds = 0.0;
  long long enqueueSuccess = 0;
  long long enqueueAttempts = 0;
  long long dequeueSuccess = 0;
  long long dequeueAttempts = 0;
  std::size_t poolBytes = 0;
  // Board energy and validity flags for the timed kernel.
  EnergySample energy;
  bool energyWindowOk = false;
};

// Kernel-side parameters, passed by value.
struct Params {
  int opsPerThread;
  int enqueueOps;
  int dequeueOps;
};

// Single-threaded prefill outside the timed region.
template <typename QueueT>
__global__ void prefillKernel(QueueT *queue, gpu::PoolView<Key> pool,
                              int count) {
  gpu::NodeAllocator<Key> allocator = gpu::prefillAllocator(pool);
  for (int i = 0; i < count; ++i) {
    queue->enqueue(static_cast<Key>(i), allocator);
  }
}

template <typename QueueT>
__global__ void benchKernel(QueueT *queue, gpu::PoolView<Key> pool,
                            Params params,
                            unsigned long long *enqueueSuccesses,
                            unsigned long long *dequeueSuccesses,
                            Key *sink) {
  const int threadId =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  gpu::NodeAllocator<Key> allocator = gpu::threadAllocator(pool, threadId);
  QueueWorkload workload(threadId, params.enqueueOps, params.dequeueOps);
  unsigned long long enqueueOk = 0;
  unsigned long long dequeueOk = 0;
  int produced = 0;
  Key observed = 0;

  for (int op = 0; op < params.opsPerThread; ++op) {
    if (workload.next() == QueueOp::Enqueue) {
      if (queue->enqueue(static_cast<Key>(produced), allocator)) {
        ++enqueueOk;
      }
      ++produced;
    } else {
      Key value = 0;
      if (queue->dequeue(value)) {
        ++dequeueOk;
        observed ^= value;
      }
    }
  }

  enqueueSuccesses[threadId] = enqueueOk;
  dequeueSuccesses[threadId] = dequeueOk;
  // Keeps dequeued values live.
  if (observed == 0) {
    sink[threadId] = observed;
  }
}

inline int nodesPerThreadFor(const Config &cfg) {
  return cfg.nodesPerThread > 0
             ? cfg.nodesPerThread
             : enqueueOpsPerThread(cfg) + dequeueOpsPerThread(cfg);
}

// Pool memory budget for one benchmark repetition.
inline std::size_t poolBudgetBytes() {
  const char *override = std::getenv("GPU_BENCH_MAX_POOL_MB");
  if (override != nullptr && override[0] != '\0') {
    return static_cast<std::size_t>(std::atoll(override)) * 1024 * 1024;
  }
  std::size_t free = 0;
  std::size_t total = 0;
  GPU_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  // Leave room for queue state, counters and driver allocations.
  return free / 2;
}

inline int blocksFor(const Config &cfg) {
  return cfg.blocks;
}

// One repetition with a fresh pool and queue.
template <typename QueueT>
Result runRepetition(const Config &cfg, const EnergyMeter &meter) {
  const int threads = static_cast<int>(totalThreads(cfg));
  const int blocks = blocksFor(cfg);
  const long long participating = totalThreads(cfg);
  const int nodesPerThread = nodesPerThreadFor(cfg);
  const long long prefill = prefillFor(cfg);

  if (prefill > std::numeric_limits<int>::max()) {
    std::fprintf(stderr,
                 "gpu bench: this configuration needs %lld prefill nodes, "
                 "which exceeds the 32-bit device index range. Lower "
                 "kOpsPerThread, kBlocks, or kBlockDims.\n",
                 prefill);
    std::abort();
  }

  const std::size_t poolBytes =
      (static_cast<std::size_t>(prefill) + 1 +
       static_cast<std::size_t>(nodesPerThread) *
           static_cast<std::size_t>(participating)) *
      sizeof(gpu::Node<Key>);
  const std::size_t budget = poolBudgetBytes();
  if (poolBytes > budget) {
    std::fprintf(stderr,
                 "gpu bench: this configuration needs %zu MB of pool memory "
                 "(%d nodes x %lld threads + %d prefill), over the %zu MB "
                 "budget. Lower kOpsPerThread, kBlocks, or kBlockDims; or raise "
                 "GPU_BENCH_MAX_POOL_MB.\n",
                 poolBytes / (1024 * 1024), nodesPerThread, participating,
                 static_cast<int>(prefill), budget / (1024 * 1024));
    std::abort();
  }

  gpu::NodePool<Key> pool(static_cast<int>(prefill), nodesPerThread,
                          static_cast<int>(participating));
  gpu::DeviceBuffer<QueueT> queue(1);
  gpu::DeviceBuffer<unsigned long long> enqueueSuccesses(
      static_cast<std::size_t>(participating));
  gpu::DeviceBuffer<unsigned long long> dequeueSuccesses(
      static_cast<std::size_t>(participating));
  gpu::DeviceBuffer<Key> sink(static_cast<std::size_t>(participating));
  enqueueSuccesses.zero();
  dequeueSuccesses.zero();
  sink.zero();

  gpubench::initQueueKernel<QueueT><<<1, 1>>>(queue.get(), pool.view());
  GPU_CUDA_CHECK_KERNEL();

  prefillKernel<QueueT><<<1, 1>>>(queue.get(), pool.view(),
                                  static_cast<int>(prefill));
  GPU_CUDA_CHECK_KERNEL();
  pool.failIfOverflowed("prefill");

  Params params;
  params.opsPerThread = cfg.opsPerThread;
  params.enqueueOps = enqueueOpsPerThread(cfg);
  params.dequeueOps = dequeueOpsPerThread(cfg);

  // CUDA events measure the kernel interval, excluding host launch overhead.
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  GPU_CUDA_CHECK(cudaEventCreate(&start));
  GPU_CUDA_CHECK(cudaEventCreate(&stop));

  // Start energy sample after prefill and before the timed kernel.
  const double joulesBefore = meter.joulesNow();

  GPU_CUDA_CHECK(cudaEventRecord(start));
  benchKernel<QueueT><<<blocks, cfg.blockDim>>>(
      queue.get(), pool.view(), params, enqueueSuccesses.get(),
      dequeueSuccesses.get(), sink.get());
  GPU_CUDA_CHECK(cudaEventRecord(stop));

  // Power-sampling fallback: sample while the kernel is in flight.
  const double wattsDuring = meter.wattsNow();

  GPU_CUDA_CHECK_KERNEL();

  // End energy sample after the kernel has finished.
  const double joulesAfter = meter.joulesNow();

  float milliseconds = 0.0f;
  GPU_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  GPU_CUDA_CHECK(cudaEventDestroy(start));
  GPU_CUDA_CHECK(cudaEventDestroy(stop));

  // Pool overflow invalidates the repetition.
  pool.failIfOverflowed("timed region");

  std::vector<unsigned long long> hostEnqueueSuccesses(
      static_cast<std::size_t>(participating));
  std::vector<unsigned long long> hostDequeueSuccesses(
      static_cast<std::size_t>(participating));
  enqueueSuccesses.copyToHost(hostEnqueueSuccesses.data(),
                              static_cast<std::size_t>(participating));
  dequeueSuccesses.copyToHost(hostDequeueSuccesses.data(),
                              static_cast<std::size_t>(participating));

  Result result;
  result.milliseconds = static_cast<double>(milliseconds);
  result.poolBytes = pool.bytes();
  const double seconds = result.milliseconds / 1000.0;
  result.energy = meter.between(joulesBefore, joulesAfter, wattsDuring, seconds);
  result.energyWindowOk = energyWindowOk(seconds);
  for (long long t = 0; t < participating; ++t) {
    result.enqueueSuccess += static_cast<long long>(
        hostEnqueueSuccesses[static_cast<std::size_t>(t)]);
    result.dequeueSuccess += static_cast<long long>(
        hostDequeueSuccesses[static_cast<std::size_t>(t)]);
  }

  result.enqueueAttempts = participating * enqueueOpsPerThread(cfg);
  result.dequeueAttempts = participating * dequeueOpsPerThread(cfg);

  return result;
}

} // namespace gpubench
