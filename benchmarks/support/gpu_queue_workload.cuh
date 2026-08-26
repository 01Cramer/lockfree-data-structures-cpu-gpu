// Device-side queue operation stream shared by the GPU benchmark and tests.

#pragma once

#include "gpu/shared/gpu_node_pool.cuh"

namespace gpubench {

enum class QueueOp { Enqueue, Dequeue };

template <typename QueueT, typename KeyT>
__global__ void initQueueKernel(QueueT *queue, gpu::PoolView<KeyT> pool) {
  queue->initialize(pool);
}

template <typename ConfigT> inline int enqueueOpsPerThread(const ConfigT &cfg) {
  return cfg.opsPerThread * cfg.mixPct / 100;
}

template <typename ConfigT> inline int dequeueOpsPerThread(const ConfigT &cfg) {
  return cfg.opsPerThread - enqueueOpsPerThread(cfg);
}

template <typename ConfigT> inline long long totalThreads(const ConfigT &cfg) {
  return static_cast<long long>(cfg.blocks) * cfg.blockDim;
}

template <typename ConfigT> inline long long prefillFor(const ConfigT &cfg) {
  return totalThreads(cfg) * dequeueOpsPerThread(cfg);
}

// Same rolling-count policy as the CPU Workload helper: choose among the
// operations still left in this thread's budget, then decrement that count.
// The RNG is a small per-thread LCG because std::mt19937 is host-only.
struct QueueWorkload {
  static constexpr unsigned int kRngMultiplier = 1664525u;
  static constexpr unsigned int kRngIncrement = 1013904223u;
  static constexpr unsigned int kSeedSalt = 0x9e3779b9u;

  unsigned int state;
  int enqueues;
  int dequeues;

  __device__ QueueWorkload(int threadId, int enqueueOps, int dequeueOps)
      : state(kSeedSalt ^ static_cast<unsigned int>(threadId + 1)),
        enqueues(enqueueOps), dequeues(dequeueOps) {}

  __device__ QueueOp next() {
    state = state * kRngMultiplier + kRngIncrement;
    const int remaining = enqueues + dequeues;
    const int roll =
        static_cast<int>(state % static_cast<unsigned int>(remaining));
    if (roll < enqueues) {
      --enqueues;
      return QueueOp::Enqueue;
    }
    --dequeues;
    return QueueOp::Dequeue;
  }
};

} // namespace gpubench
