// Design rationale
//
// The device counterpart of cpu/spinlock/spinlock.hpp, and the only lock this
// project has on the GPU.
//
// There is no mutex variant on the device. A blocking mutex is a futex: the
// waiter tells the operating system scheduler to stop running it and to run it
// again when the word changes. There is no such call in a kernel and no
// scheduler to make it to, so a device "mutex" could only ever be a spinlock
// under another name. The CPU results therefore have a mutex row with no GPU
// counterpart, and that is a property of the hardware rather than an omission.
//
// The same fact removes the CPU half's headline spinlock finding from play.
// There, backoff strength was decisive because the winning spinlock escalates
// to nanosleep(), which hands the core back to the OS scheduler -- the thread
// stops competing with the lock holder for the core it needs. A GPU warp
// occupies an SM issue slot, not a core, and there is nothing to hand it back
// to. __nanosleep() is the nearest available thing: it removes the warp from
// issue consideration for roughly the requested interval, which frees issue
// slots and, more importantly, stops the waiter's atomics from queueing at the
// L2 ahead of the holder's. Weaker than the CPU mechanism, and the difference
// is worth stating rather than glossing.
//
// Backoff shape, and what is held fixed
//
// The spin budget (8) is inherited from the CPU lock. The escalation is not:
// the CPU lock's backoff is a single fixed step, while this one doubles the
// sleep up to a cap. That is NVIDIA's documented pattern for a device spin
// loop, and it matters more here, because at activeLanes = 32 a single lock is
// contended by 32 lanes of the same warp whose failed atomics all issue from
// the same instruction. A fixed minimal backoff leaves them re-issuing in
// lockstep.
//
// These constants are NOT a swept axis. They are identical in Queue and
// QueueTwoLock, so the two lock-based variants remain comparable with each
// other; what they are not is tuned, and a claim that the lock-based variants
// are slow should be read as "slow at this backoff", not "slow at their best".
//
// Independent thread scheduling, and what must not appear below
//
// lock() is safe with all 32 lanes of a warp contending only because Volta and
// later give each thread its own program counter: the lane that wins the
// exchange can run to unlock() while its peers are still spinning. On earlier
// hardware the warp shares a PC, the winner cannot advance past the loop the
// losers are still executing, and the kernel deadlocks -- unconditionally, not
// probabilistically. gpu/shared/atomics.cuh enforces the CC 7.0 floor, and
// tests/gpu_spinlock_progress_test.cu confirms it on the actual device before
// anything is built on top of it.
//
// For the same reason, nothing in the acquire loop may be a warp-collective
// operation. No __syncwarp(), no __ballot_sync(), no shuffle: a lane that
// already holds the lock is diverged and would never arrive, so any of them
// reintroduces exactly the deadlock the hardware removed.
//
// One further constraint, from CUDA's forward-progress rules rather than from
// the warp: the lock word must not have automatic storage duration. CUDA
// guarantees that once one device thread of a block makes progress every
// thread of that block eventually does, but the examples in the execution
// model appendix are explicit that a spin loop on an automatic-storage object
// is allowed to make no progress at all. Every Spinlock in this project is a
// member of a queue that lives in cudaMalloc'd global memory, which satisfies
// the requirement -- but a Spinlock declared as a kernel local, or one placed
// in __shared__ and spun on at device scope, would not, and would be a
// legitimately non-terminating program rather than a bug in the hardware.
//
// The guarantee is also only "eventually", with no bound, which is why
// tests/gpu_spinlock_progress_test.cu exists: the formal model permits
// progress that is far too slow to be usable, and the point of the day-one
// test is to see the real thing on the real device.

#pragma once

#include <cuda_runtime.h>

#include "gpu/shared/atomics.cuh"

namespace gpu {

namespace detail {

// One backoff step. Both the duration and the doubling live in the caller so
// the escalation state stays in a register.
__device__ inline void spinlockBackoff(unsigned int nanoseconds) {
#if __CUDA_ARCH__ >= 700
  __nanosleep(nanoseconds);
#else
  // Unreachable: atomics.cuh static_asserts the CC 7.0 floor. Present so the
  // header still parses if that assertion is ever relaxed for an experiment.
  (void)nanoseconds;
#endif
}

} // namespace detail

// Test-and-test-and-set, the same shape as cpu::Spinlock.
//
// The relaxed load in the loop condition short-circuits the exchange while the
// lock is visibly held. On the CPU that keeps the line in shared state instead
// of bouncing it exclusive; on the device it keeps a failing waiter's traffic
// to a load that the L2 can service without a read-modify-write, which is the
// same saving for a different reason.
//
// The lock word is a plain unsigned int reached through DeviceAtomicRef rather
// than a cuda::atomic member, so a Spinlock is trivially copyable and a
// structure containing one can be zero-initialized from the host.
class Spinlock {
public:
  // No constructor: instances live in device memory allocated by cudaMalloc
  // and are set up by an initialization kernel (or by cudaMemset to zero,
  // which is the same state).
  __device__ void initialize() {
    DeviceAtomicRef<unsigned int> flag(m_locked);
    flag.store(0u, cuda::memory_order_relaxed);
  }

  // Acquire ordering: everything the previous holder did before its release
  // store is visible to this thread once the exchange returns 0.
  __device__ bool tryLock() {
    DeviceAtomicRef<unsigned int> flag(m_locked);
    if (flag.load(cuda::memory_order_relaxed) != 0u) {
      return false;
    }
    return flag.exchange(1u, cuda::memory_order_acquire) == 0u;
  }

  __device__ void lock() {
    unsigned int backoffNanos = kInitialBackoffNanos;
    for (int spins = 0; !tryLock(); ++spins) {
      if (spins == kSpinBudget) {
        spins = 0;
        detail::spinlockBackoff(backoffNanos);
        backoffNanos = backoffNanos < kMaxBackoffNanos ? backoffNanos * 2u
                                                       : kMaxBackoffNanos;
      }
    }
  }

  __device__ void unlock() {
    DeviceAtomicRef<unsigned int> flag(m_locked);
    flag.store(0u, cuda::memory_order_release);
  }

  static constexpr int kSpinBudget = 8;
  static constexpr unsigned int kInitialBackoffNanos = 32u;
  static constexpr unsigned int kMaxBackoffNanos = 1024u;

private:
  unsigned int m_locked;
};

// Scope guard, so the critical sections read the same as the CPU ones, which
// use std::lock_guard. There is no std::lock_guard in device code.
class LockGuard {
public:
  __device__ explicit LockGuard(Spinlock &lock) : m_lock(lock) {
    m_lock.lock();
  }

  __device__ ~LockGuard() { m_lock.unlock(); }

  LockGuard(const LockGuard &) = delete;
  LockGuard &operator=(const LockGuard &) = delete;

private:
  Spinlock &m_lock;
};

} // namespace gpu
