// Progress and mutual-exclusion tests for the GPU spinlock.
//
// The lock-based queue variants rely on 32 lanes of one warp being able to
// contend for one lock while the winner still reaches unlock(). The watchdog
// turns a hang into a clear failure message.

#include <cstdio>

#include <cuda_runtime.h>

#include "gpu/shared/gpu_cuda_utils.cuh"
#include "gpu/shared/gpu_spinlock.cuh"
#include "support/gpu_test_harness.cuh"

using namespace gpu_test;

// Named namespace keeps kernel linkage simple across nvcc versions.
namespace progresstest {

// Long enough for progress, short enough to catch a real hang.
constexpr double kWatchdogSeconds = 60.0;

// Shared state for one run. One cache line's worth, allocated on the device.
struct Shared {
  gpu::Spinlock lock;
  int counter;        // incremented non-atomically inside the critical section
  int witnessOwner;   // thread id of the current holder
  int witnessFailures;
  int giveUps;
};

__global__ void initShared(Shared *shared) {
  shared->lock.initialize();
  shared->counter = 0;
  shared->witnessOwner = -1;
  shared->witnessFailures = 0;
  shared->giveUps = 0;
}

// Shared critical section used by the blocking and bounded lock tests.
__device__ inline void criticalSection(Shared *shared, int threadId) {
  gpu::DeviceAtomicRef<int> counter(shared->counter);
  counter.store(counter.load(cuda::memory_order_relaxed) + 1,
                cuda::memory_order_relaxed);

  gpu::DeviceAtomicRef<int> owner(shared->witnessOwner);
  owner.store(threadId, cuda::memory_order_relaxed);
  // Gives another accidental holder time to overwrite the witness.
  __nanosleep(64);
  if (owner.load(cuda::memory_order_relaxed) != threadId) {
    gpu::DeviceAtomicRef<int> failures(shared->witnessFailures);
    failures.fetch_add(1, cuda::memory_order_relaxed);
  }
}

// Production lock path.
__global__ void blockingLockKernel(Shared *shared, int opsPerThread) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int op = 0; op < opsPerThread; ++op) {
    shared->lock.lock();
    criticalSection(shared, threadId);
    shared->lock.unlock();
  }
}

// Bounded tryLock path.
__global__ void boundedLockKernel(Shared *shared, int opsPerThread,
                                  int attemptBudget) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int op = 0; op < opsPerThread; ++op) {
    bool acquired = false;
    unsigned int backoff = gpu::Spinlock::kInitialBackoffNanos;
    for (int attempt = 0; attempt < attemptBudget; ++attempt) {
      if (shared->lock.tryLock()) {
        acquired = true;
        break;
      }
      const int budget = gpu::Spinlock::kSpinBudget;
      if (attempt % budget == budget - 1) {
        __nanosleep(backoff);
        backoff = backoff < gpu::Spinlock::kMaxBackoffNanos ? backoff * 2u
                                                            : backoff;
      }
    }
    if (!acquired) {
      gpu::DeviceAtomicRef<int> giveUps(shared->giveUps);
      giveUps.fetch_add(1, cuda::memory_order_relaxed);
      continue;
    }
    criticalSection(shared, threadId);
    shared->lock.unlock();
  }
}

Shared readBack(const gpu::DeviceBuffer<Shared> &buffer) {
  Shared host{};
  buffer.copyToHost(&host, 1);
  return host;
}

// Run one blocking-lock configuration under the watchdog.
void runBlocking(int blocks, int opsPerThread, const char *label) {
  gpu::DeviceBuffer<Shared> shared(1);

  initShared<<<1, 1>>>(shared.get());
  GPU_CUDA_CHECK_KERNEL();

  cudaStream_t stream = nullptr;
  GPU_CUDA_CHECK(cudaStreamCreate(&stream));
  blockingLockKernel<<<blocks, kWarpSize, 0, stream>>>(shared.get(),
                                                       opsPerThread);
  GPU_CUDA_CHECK(cudaGetLastError());
  waitWithWatchdog(stream, kWatchdogSeconds, label);
  GPU_CUDA_CHECK(cudaStreamDestroy(stream));

  const Shared result = readBack(shared);
  const int expected = blocks * kWarpSize * opsPerThread;

  // A short count is a lost update, which means two threads were inside the
  // critical section at once.
  CHECK_EQ(result.counter, expected);
  CHECK_EQ(result.witnessFailures, 0);
}

} // namespace progresstest

using namespace progresstest;

// One warp, 32 lanes, one lock.
GPU_TEST(spinlock_progress_single_warp) {
  runBlocking(/*blocks=*/1, gpuConfig().opsPerThread,
              "single-warp spinlock progress (32 lanes, one lock)");
}

// Same lock, many warps.
GPU_TEST(spinlock_progress_many_warps) {
  runBlocking(gpuConfig().warps, gpuConfig().opsPerThread,
              "multi-warp spinlock progress");
}

// Wider mutual-exclusion check.
GPU_TEST(spinlock_mutual_exclusion) {
  runBlocking(gpuConfig().warps, gpuConfig().opsPerThread * 4,
              "spinlock mutual exclusion");
}

// Bounded acquisition should not give up under this workload.
GPU_TEST(spinlock_bounded_acquisition) {
  const int blocks = gpuConfig().warps;
  const int opsPerThread = gpuConfig().opsPerThread;
  // Loose enough that a give-up indicates starvation.
  const int attemptBudget = blocks * kWarpSize * 4096;

  gpu::DeviceBuffer<Shared> shared(1);
  initShared<<<1, 1>>>(shared.get());
  GPU_CUDA_CHECK_KERNEL();

  cudaStream_t stream = nullptr;
  GPU_CUDA_CHECK(cudaStreamCreate(&stream));
  boundedLockKernel<<<blocks, kWarpSize, 0, stream>>>(
      shared.get(), opsPerThread, attemptBudget);
  GPU_CUDA_CHECK(cudaGetLastError());
  waitWithWatchdog(stream, kWatchdogSeconds, "bounded spinlock acquisition");
  GPU_CUDA_CHECK(cudaStreamDestroy(stream));

  const Shared result = readBack(shared);
  CHECK_EQ(result.giveUps, 0);
  CHECK_EQ(result.witnessFailures, 0);
  // Counts the acquisitions that happened, so it stays exact even if a
  // give-up is reported above.
  CHECK_EQ(result.counter, blocks * kWarpSize * opsPerThread - result.giveUps);
}

int main(int argc, char **argv) { return gpu_test::runAll(argc, argv); }
