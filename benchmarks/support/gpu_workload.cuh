// The GPU workload, kept separate from the structure under test (the template
// parameter) and from the sweep driver (gpu_bench_queue.cu). One kernel body
// serves all three variants, so none can be measured through a different code
// path.
//
// The four parameters
//
//   warps          concurrency. blockDim is 32, so one block is one warp is one
//                  agent. (blockDim is overridable for the second-order
//                  experiment, but only at constant total warps.)
//   activeLanes    participating lanes per warp, 1 or 32. This is the
//                  contribution; see the guard in benchKernel.
//   interOpWork    contention knob: units of thread-private arithmetic between
//                  two queue operations, the only knob that changes how often
//                  operations collide without changing which code path runs.
//                  GPU-only -- the CPU harness has no counterpart, so this is
//                  not one arm of a cross-platform factor.
//   opsPerThread   fixed before launch: nodes are never reclaimed, so the pool
//                  is O(total operations) and cannot be sized against a count
//                  the run discovers.
//
// Operation assignment is warp-uniform -- even warps produce, odd warps consume,
// giving 50/50 interleaved across SMs. Lanes within a warp taking different
// operations would fold that divergence into the measured cost of
// activeLanes = 32, leaving the axis unattributable.
//
// The queue is pre-filled before the timed region, or a dequeue-heavy interval
// mostly measures how fast each variant detects emptiness, which is cheap and
// strongly variant-dependent. The fraction of failed dequeues is reported anyway,
// so too shallow a prefill is visible.
//
// Only successful operations are counted in the throughput numerator; attempts
// are reported alongside.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/shared/cuda_error.cuh"
#include "gpu/shared/node_pool.cuh"
#include "support/gpu_energy.cuh"

namespace gpubench {

// Scalar payload: a 32-bit value keeps a node at 8 bytes. A heavier payload would
// compress the differences being measured by adding a fixed cost to every variant.
using Key = unsigned int;

inline constexpr int kWarpSize = 32;

// One point in the sweep.
struct Config {
  std::string variant;
  int warps = 0;         // total warps in the grid
  int blockDim = 32;     // threads per block
  int activeLanes = 32;  // participating lanes per warp, in [1, 32]
  int opsPerThread = 0;
  int interOpWork = 0;
  int prefill = 0;
  int nodesPerThread = 0; // 0 = derive from opsPerThread
  bool capped = false;    // opsPerThread was lowered by --max-total-ops
};

// What one repetition produced.
struct Result {
  double milliseconds = 0.0;
  long long enqueueSuccess = 0;
  long long enqueueAttempts = 0;
  long long dequeueSuccess = 0;
  long long dequeueAttempts = 0;
  std::size_t poolBytes = 0;
  // Board energy over the timed region. `energy.valid` is false when no meter
  // was available or the counter went backwards; `energyWindowOk` says whether
  // the window was long enough for the figure to mean anything. Both travel
  // with the row rather than being checked here, so a questionable point is
  // visible in the output instead of dropped.
  EnergySample energy;
  bool energyWindowOk = false;
};

// Kernel-side parameters, passed by value.
struct Params {
  int opsPerThread;
  int activeLanes;
  int interOpWork;
};

// Thread-private work between two queue operations. A dependent integer chain, so
// it cannot be vectorized, unrolled or reassociated; `units` is a kernel argument
// so the closed form is unavailable; and the result is consumed by a store the
// compiler cannot prove dead (see benchKernel). All three keep it in the SASS.
__device__ __forceinline__ unsigned int interOpWork(int units,
                                                    unsigned int state) {
  for (int i = 0; i < units; ++i) {
    state = state * 1664525u + 1013904223u;
  }
  return state;
}

template <typename QueueT>
__global__ void initQueueKernel(QueueT *queue, gpu::PoolView<Key> pool) {
  queue->initialize(pool);
}

// Single-threaded and outside the timed region. Uses the real enqueue path so
// the starting state is exactly what the algorithm produces.
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
                            Params params, unsigned long long *successes,
                            unsigned int *sink) {
  // The activeLanes axis. Tested per warp, not per block: with blockDim > 32 a
  // per-block test would switch off entire trailing warps, measuring a smaller
  // grid rather than a narrower warp. Both settings launch the same threads at
  // the same occupancy and register footprint, so activeLanes = 1 is a control
  // isolating inter-warp contention, and the difference from 32 is the intra-warp
  // penalty of the synchronization mechanism.
  const int laneInWarp = static_cast<int>(threadIdx.x) % kWarpSize;
  if (laneInWarp >= params.activeLanes) {
    return;
  }

  const int threadId =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int globalWarp = threadId / kWarpSize;
  const bool isProducer = (globalWarp % 2) == 0;

  gpu::NodeAllocator<Key> allocator = gpu::threadAllocator(pool, threadId);
  unsigned long long succeeded = 0;
  unsigned int state = 0x9e3779b9u ^ static_cast<unsigned int>(threadId);

  for (int op = 0; op < params.opsPerThread; ++op) {
    if (isProducer) {
      if (queue->enqueue(static_cast<Key>(op), allocator)) {
        ++succeeded;
      }
    } else {
      Key value = 0;
      if (queue->dequeue(value)) {
        ++succeeded;
        state ^= value;
      }
    }
    state = interOpWork(params.interOpWork, state);
  }

  successes[threadId] = succeeded;
  // Keeps the interOpWork chain and the dequeued values alive: never true in
  // practice, but the compiler cannot prove it. Do not replace with something
  // like `successes[threadId] += state * 0`, which folds away.
  if (state == 0u) {
    sink[threadId] = state;
  }
}

// Nodes each thread may allocate. Consumer warps allocate nothing, so half the
// pool is unused at the 50/50 mix; that waste buys a slice mapping that is a pure
// function of the thread id rather than of which warps are producers.
inline int nodesPerThreadFor(const Config &cfg) {
  return cfg.nodesPerThread > 0 ? cfg.nodesPerThread : cfg.opsPerThread;
}

// Refuse a configuration whose pool would not fit, rather than failing a
// cudaMalloc partway through a sweep.
inline std::size_t poolBudgetBytes() {
  const char *override = std::getenv("GPU_BENCH_MAX_POOL_MB");
  if (override != nullptr && override[0] != '\0') {
    return static_cast<std::size_t>(std::atoll(override)) * 1024 * 1024;
  }
  std::size_t free = 0;
  std::size_t total = 0;
  GPU_CUDA_CHECK(cudaMemGetInfo(&free, &total));
  // Leaves room for the queue object, the counters and the driver's allocations.
  return free / 2;
}

inline int totalThreads(const Config &cfg) {
  return cfg.warps * kWarpSize;
}

inline int blocksFor(const Config &cfg) {
  return totalThreads(cfg) / cfg.blockDim;
}

// Participating threads by role, so attempt counts can be derived on the host.
inline void participants(const Config &cfg, int &producers, int &consumers) {
  producers = 0;
  consumers = 0;
  for (int warp = 0; warp < cfg.warps; ++warp) {
    if (warp % 2 == 0) {
      producers += cfg.activeLanes;
    } else {
      consumers += cfg.activeLanes;
    }
  }
}

// One repetition, from a freshly built pool and a freshly initialized queue.
// Fresh by necessity: nodes are never reclaimed, so a second repetition on the
// same pool would start with the first one's allocations already spent.
template <typename QueueT>
Result runRepetition(const Config &cfg, const EnergyMeter &meter) {
  const int threads = totalThreads(cfg);
  const int blocks = blocksFor(cfg);
  const int nodesPerThread = nodesPerThreadFor(cfg);

  const std::size_t poolBytes =
      (static_cast<std::size_t>(cfg.prefill) + 1 +
       static_cast<std::size_t>(nodesPerThread) *
           static_cast<std::size_t>(threads)) *
      sizeof(gpu::Node<Key>);
  const std::size_t budget = poolBudgetBytes();
  if (poolBytes > budget) {
    std::fprintf(stderr,
                 "gpu bench: this configuration needs %zu MB of pool memory "
                 "(%d nodes x %d threads + %d prefill), over the %zu MB "
                 "budget. Lower --ops, lower --warps, or raise "
                 "GPU_BENCH_MAX_POOL_MB.\n",
                 poolBytes / (1024 * 1024), nodesPerThread, threads,
                 cfg.prefill, budget / (1024 * 1024));
    std::abort();
  }

  gpu::NodePool<Key> pool(cfg.prefill, nodesPerThread, threads);
  gpu::DeviceBuffer<QueueT> queue(1);
  gpu::DeviceBuffer<unsigned long long> successes(
      static_cast<std::size_t>(threads));
  gpu::DeviceBuffer<unsigned int> sink(static_cast<std::size_t>(threads));
  successes.zero();
  sink.zero();

  initQueueKernel<QueueT><<<1, 1>>>(queue.get(), pool.view());
  GPU_CUDA_CHECK_KERNEL();

  prefillKernel<QueueT><<<1, 1>>>(queue.get(), pool.view(), cfg.prefill);
  GPU_CUDA_CHECK_KERNEL();
  pool.failIfCorrupted("prefill");
  pool.failIfOverflowed("prefill");

  Params params;
  params.opsPerThread = cfg.opsPerThread;
  params.activeLanes = cfg.activeLanes;
  params.interOpWork = cfg.interOpWork;

  // CUDA events, not a host clock: recorded in the stream, so the interval is the
  // kernel's, with no launch latency or synchronization cost folded in.
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  GPU_CUDA_CHECK(cudaEventCreate(&start));
  GPU_CUDA_CHECK(cudaEventCreate(&stop));

  // Safe to read here: the prefill kernel above was synchronized, so the device
  // is idle and nothing between this read and the launch consumes energy the
  // run should not be charged for.
  const double joulesBefore = meter.joulesNow();

  GPU_CUDA_CHECK(cudaEventRecord(start));
  benchKernel<QueueT><<<blocks, cfg.blockDim>>>(
      queue.get(), pool.view(), params, successes.get(), sink.get());
  GPU_CUDA_CHECK(cudaEventRecord(stop));

  // Sampled here, deliberately: the launch is asynchronous, so the kernel is
  // still running at this point and this reads power under load. Only the
  // power-sampling fallback uses it, and taking it after the synchronize below
  // would read the device idling down instead -- which does not merely add
  // noise, it biases that fallback low every time.
  const double wattsDuring = meter.wattsNow();

  GPU_CUDA_CHECK_KERNEL();
  pool.failIfCorrupted("timed region");

  // After the synchronize inside GPU_CUDA_CHECK_KERNEL, so the kernel has
  // finished and all of its energy is in the counter. Reading before the
  // synchronize is the mistake the ml.energy guidance calls out: the CPU races
  // ahead of the GPU and the difference undercounts.
  const double joulesAfter = meter.joulesNow();

  float milliseconds = 0.0f;
  GPU_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  GPU_CUDA_CHECK(cudaEventDestroy(start));
  GPU_CUDA_CHECK(cudaEventDestroy(stop));

  // Mandatory: an exhausted slice turns every later enqueue into a no-op, which
  // reads as a plausible throughput number for a queue that stopped growing.
  pool.failIfOverflowed("timed region");

  std::vector<unsigned long long> hostSuccesses(
      static_cast<std::size_t>(threads));
  successes.copyToHost(hostSuccesses.data(),
                       static_cast<std::size_t>(threads));

  Result result;
  result.milliseconds = static_cast<double>(milliseconds);
  result.poolBytes = pool.bytes();
  const double seconds = result.milliseconds / 1000.0;
  result.energy = meter.between(joulesBefore, joulesAfter, wattsDuring, seconds);
  result.energyWindowOk = energyWindowOk(seconds);
  for (int t = 0; t < threads; ++t) {
    const long long ok =
        static_cast<long long>(hostSuccesses[static_cast<std::size_t>(t)]);
    if ((t / kWarpSize) % 2 == 0) {
      result.enqueueSuccess += ok;
    } else {
      result.dequeueSuccess += ok;
    }
  }

  int producers = 0;
  int consumers = 0;
  participants(cfg, producers, consumers);
  result.enqueueAttempts =
      static_cast<long long>(producers) * cfg.opsPerThread;
  result.dequeueAttempts =
      static_cast<long long>(consumers) * cfg.opsPerThread;

  return result;
}

} // namespace gpubench
