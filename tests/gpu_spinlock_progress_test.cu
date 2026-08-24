// The day-one test. Nothing else in the GPU phase should be built until this
// passes on the target device.
//
// Why it is separate from the queue tests
//
// Both lock-based variants rest on one assumption: that all 32 lanes of a warp
// can contend for the same spinlock and the winner can reach unlock() while
// its peers are still spinning. That is true only because Volta gave every
// thread its own program counter. On earlier hardware it is false
// unconditionally -- the warp shares a PC, the winner is stuck at the loop the
// losers are executing, and the kernel hangs forever. Cederman, Chatterjee &
// Tsigas ran one operating thread per block for exactly this reason, which is
// also why the activeLanes axis this project sweeps did not exist for them.
//
// If the assumption fails, the entire lock-based half of the design changes.
// That is worth learning on day one from a thirty-line kernel rather than on
// day four from a queue that hangs and could be hanging for any of six
// reasons.
//
// What is checked, in increasing strength
//
//   1. Progress, one warp.  32 lanes, one lock, the production lock(). The
//      purest intra-warp case: if independent thread scheduling is not doing
//      what it claims, this alone never returns. Wrapped in a watchdog so the
//      failure prints a diagnosis instead of hanging the suite.
//   2. Progress, many warps. Adds inter-warp and inter-SM contention on the
//      same lock.
//   3. Mutual exclusion.  A non-atomic read-modify-write inside the critical
//      section (a lost update proves two holders), plus an ownership witness:
//      each holder stamps its thread id, waits, and re-reads it. The counter
//      alone can survive a broken lock by luck; the witness makes a second,
//      independent way to catch it.
//   4. Bounded acquisition. The same workload through tryLock() with an
//      attempt budget, which separates "deadlocked" from "starving": a
//      deadlock times out in 1-2, whereas a thread that is merely never
//      winning shows up here as a give-up with everything else still healthy.

#include <cstdio>

#include <cuda_runtime.h>

#include "gpu/shared/cuda_error.cuh"
#include "gpu/shared/spinlock.cuh"
#include "support/gpu_test_harness.cuh"

using namespace gpu_test;

// A named namespace, not an anonymous one. __global__ functions with internal
// linkage have been a moving target across nvcc versions, and there is nothing
// to gain by finding out which side of it the local toolkit falls on.
namespace progresstest {

// Generous relative to any plausible wait, tight enough that a genuine
// deadlock is reported in under a minute.
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

// The critical section, shared by every kernel below so that all of them are
// testing the same body and differ only in how the lock is taken.
//
// The counter is loaded and stored as two separate relaxed atomic accesses.
// Relaxed, so the compiler cannot keep it in a register across iterations and
// the test measures memory rather than a register; two accesses, so it is
// still a read-modify-write that a second concurrent holder can lose. An
// atomicAdd here would pass with no lock at all.
__device__ inline void criticalSection(Shared *shared, int threadId) {
  gpu::DeviceAtomicRef<int> counter(shared->counter);
  counter.store(counter.load(cuda::memory_order_relaxed) + 1,
                cuda::memory_order_relaxed);

  gpu::DeviceAtomicRef<int> owner(shared->witnessOwner);
  owner.store(threadId, cuda::memory_order_relaxed);
  // Long enough that a second holder would have to be very lucky to slip in
  // and out unseen; short enough not to dominate the run.
  __nanosleep(64);
  if (owner.load(cuda::memory_order_relaxed) != threadId) {
    gpu::DeviceAtomicRef<int> failures(shared->witnessFailures);
    failures.fetch_add(1, cuda::memory_order_relaxed);
  }
}

// Kernels 1-3: the production lock, unbounded. A hang here is the finding.
__global__ void blockingLockKernel(Shared *shared, int opsPerThread) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  for (int op = 0; op < opsPerThread; ++op) {
    shared->lock.lock();
    criticalSection(shared, threadId);
    shared->lock.unlock();
  }
}

// Kernel 4: bounded acquisition. Distinguishes starvation from deadlock.
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

// Run one blocking-lock configuration under the watchdog and check every
// invariant. `label` names the configuration in the timeout message.
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

// 1. One warp, 32 lanes, one lock. The purest statement of the assumption the
//    whole lock-based half depends on.
GPU_TEST(spinlock_progress_single_warp) {
  runBlocking(/*blocks=*/1, gpuConfig().opsPerThread,
              "single-warp spinlock progress (32 lanes, one lock)");
}

// 2. Same lock, many warps across many SMs. Adds the inter-warp contention the
//    literature measured, on top of the intra-warp contention it did not.
GPU_TEST(spinlock_progress_many_warps) {
  runBlocking(gpuConfig().warps, gpuConfig().opsPerThread,
              "multi-warp spinlock progress");
}

// 3. Mutual exclusion at the widest configuration the test config asks for,
//    with more operations per thread so the interleaving has room to go wrong.
GPU_TEST(spinlock_mutual_exclusion) {
  runBlocking(gpuConfig().warps, gpuConfig().opsPerThread * 4,
              "spinlock mutual exclusion");
}

// 4. Bounded acquisition. Everything should still add up, and no thread should
//    exhaust a budget this far above any fair wait; a give-up here with tests
//    1-3 passing means the lock makes progress but starves someone, which is a
//    finding about the backoff, not about the hardware.
GPU_TEST(spinlock_bounded_acquisition) {
  const int blocks = gpuConfig().warps;
  const int opsPerThread = gpuConfig().opsPerThread;
  // Contenders x a very loose per-contender allowance. Reaching this without
  // acquiring is not a slow wait, it is not being served at all.
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
