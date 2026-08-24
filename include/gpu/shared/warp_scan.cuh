// Warp-collective primitives for the request-combining queue (CCLQ).
//
// Its own header rather than a detail of cclq_queue.cuh, because the prefix sum
// is where the published algorithm is broken and the correction is worth
// stating once, in isolation, and testing on its own.
//
// What Zhang, Deng & Mu print (IEICE 2014, Fig. 1 (a), lines 9-17)
//
//   __device__ int scanWarp(int *request) {
//     const int tid = threadIdx.x, lane_id = tid & 31;
//     if (lane_id >=  1) request[tid] += request[tid -  1];
//     if (lane_id >=  2) request[tid] += request[tid -  2];
//     if (lane_id >=  4) request[tid] += request[tid -  4];
//     if (lane_id >=  8) request[tid] += request[tid -  8];
//     if (lane_id >= 16) request[tid] += request[tid - 16];
//     return request[tid] - 1;
//   }
//
// A Hillis-Steele inclusive scan over a shared-memory array, with no
// synchronization between the five steps. Every step has lane i reading
// request[i - offset] while lane i + offset writes request[i + offset]; the
// only thing that makes step k's reads see step k-1's writes is the assumption
// that all 32 lanes execute each statement in lockstep. That was true up to
// Kepler, where the warp shares a program counter, and the paper's hardware was
// Kepler and Fermi. On Volta and later each thread has its own program counter,
// the lanes are free to be at different statements, and the code is a data
// race whose result is not merely unordered but arithmetically wrong -- a lane
// can read a neighbour's value from the wrong step and produce a duplicate
// position, at which point two items are written to the same batch slot and one
// is silently lost.
//
// The correction below computes the identical five-step scan with
// __shfl_up_sync, which is a warp-collective instruction: the exchange and the
// synchronization are the same operation, so there is no window to race in. It
// also removes the shared-memory array altogether, and with it the `int
// *request` parameter that the paper's signature carries.
//
// This is the only place the fix belongs. Adding __syncwarp() between the five
// shared-memory steps would also be correct, but it keeps a shared-memory
// buffer per warp for a quantity that never needs to leave a register.

#pragma once

#include <cuda_runtime.h>

#include "gpu/shared/atomics.cuh"

namespace gpu {

inline constexpr int kWarpSize = 32;

// All 32 lanes. Every warp-collective call in the CCLQ path uses the full mask,
// which makes a hard requirement of the calling code: a lane must never retire
// before a collective call. For variants 1-3 the activeLanes control is
// `if (lane >= activeLanes) return;`, and that would be undefined behaviour
// here. CCLQ instead expresses non-participation through the `request` flag,
// which is what Zhang's `request[threadIdx.x] = 0` lanes are for -- they stay
// in the warp and assist the scan. See cclq_queue.cuh.
inline constexpr unsigned int kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ int laneId() {
  return static_cast<int>(threadIdx.x) % kWarpSize;
}

// Inclusive prefix sum across the warp. Lane i receives the sum of lanes
// 0..i, so lane 31 receives the warp total.
//
// The `if (lane >= offset)` guard is not optional: __shfl_up_sync returns the
// caller's own value when the source lane is out of range, so without the
// guard the low lanes would double-count themselves.
__device__ __forceinline__ int warpInclusiveScan(int value) {
#pragma unroll
  for (int offset = 1; offset < kWarpSize; offset <<= 1) {
    const int received = __shfl_up_sync(kFullWarpMask, value, offset);
    if (laneId() >= offset) {
      value += received;
    }
  }
  return value;
}

// Broadcast lane 31's value to the whole warp.
//
// Lane 31 is the combiner throughout CCLQ, following the paper
// (`if (lane_id == 31)`), and it is also the lane that holds the scan total, so
// one lane serves both roles and no extra reduction is needed.
__device__ __forceinline__ int broadcastFromCombiner(int value) {
  return __shfl_sync(kFullWarpMask, value, kWarpSize - 1);
}

inline constexpr int kCombinerLane = kWarpSize - 1;

__device__ __forceinline__ bool isCombiner() {
  return laneId() == kCombinerLane;
}

} // namespace gpu
