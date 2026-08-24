// Correctness tests for variant 4, the CCLQ request-combining queue.
//
// A separate file from gpu_queue_test.cu because the API is warp-collective and
// so the kernels have to be: every lane must call enqueue or dequeue, and
// non-participation is expressed through the `request` flag rather than by
// retiring the lane. The oracles are the same ones, from
// tests/support/queue_oracles.hpp.
//
// Four things are checked, and the order is deliberate.
//
//   1. The prefix sum, on its own, against a host-computed answer. This is the
//      correction to Zhang's scanWarp (see gpu/shared/warp_scan.cuh), it is the
//      foundation of both operations, and a wrong scan does not announce itself
//      as a scan bug -- it announces itself as two lanes writing the same batch
//      slot, i.e. as a lost item several layers up. Cheap to test directly, so
//      it is tested directly.
//
//   2. Sequential, one warp. No inter-warp concurrency; catches anything wrong
//      with the position/node indexing before 512 threads are involved.
//
//   3. Fill/drain and mixed, the same two scenarios as the other variants, so
//      conservation and per-producer FIFO are checked on the same workload
//      shape the benchmark runs.
//
//   4. Asymmetric widths. Producer warps and consumer warps request different
//      numbers of items, which is the ONLY way to reach the partial-node
//      claiming path (Fig. 1 (c) lines 53-63). Under the warp-uniform,
//      equal-width assignment that the benchmark uses, every batch is exactly
//      full and every request is exactly its size, so start_pos only ever holds
//      0 or the node's end and that branch is never taken. It is real code and
//      it is the paper's algorithm, so it gets a test that exercises it in both
//      directions: a request smaller than one node, and a request spanning
//      several.

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/lockfree/cclq_queue.cuh"
#include "gpu/shared/batch_pool.cuh"
#include "gpu/shared/cuda_error.cuh"
#include "gpu/shared/warp_scan.cuh"
#include "support/gpu_test_harness.cuh"
#include "support/queue_oracles.hpp"

using namespace gpu_test;

// Named, not anonymous: __global__ functions with internal linkage have been a
// moving target across nvcc versions.
namespace cclqtest {

using queue_oracle::checkConservation;
using queue_oracle::checkPerProducerFifo;
using queue_oracle::encodingFits;
using queue_oracle::flatten;
using queue_oracle::Key;
using queue_oracle::makeKey;

using Queue = gpu::lockfree::BatchQueue<Key>;

// The warp's output store (Zhang's this_data_list). Sized for the largest block
// this project launches; the tests themselves always use blockDim 32.
constexpr int kMaxBlockThreads = 128;

__device__ inline Key *warpStoreFor(Key *blockStore) {
  return blockStore + (threadIdx.x / 32) * gpu::kBatchCapacity;
}

// Batch nodes a prefill of `count` items consumes. One node per collective
// call, and the prefill fills 32 at a time.
inline int prefillNodesFor(int count) { return (count + 31) / 32 + 1; }

// --- kernels -------------------------------------------------------------

__global__ void initQueueKernel(Queue *queue, gpu::BatchPoolView<Key> pool) {
  queue->initialize(pool);
}

// One warp, outside the timed region, through the real enqueue path.
//
// `done` and `batch` are computed identically by every lane, so the loop is
// warp-uniform and the collective calls inside it are legal.
__global__ void prefillKernel(Queue *queue, gpu::BatchPoolView<Key> pool,
                              int count, int prefillProducer) {
  const int lane = static_cast<int>(threadIdx.x);
  gpu::BatchAllocator<Key> allocator = gpu::prefillBatchAllocator(pool);

  int done = 0;
  while (done < count) {
    const int remaining = count - done;
    const int batch = remaining < 32 ? remaining : 32;
    queue->enqueue(makeKey(prefillProducer, done + lane), lane < batch,
                   allocator);
    done += batch;
  }
}

__global__ void fillKernel(Queue *queue, gpu::BatchPoolView<Key> pool,
                           int opsPerThread, int activeLanes) {
  const int lane = gpu::laneId();
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int warpId = threadId / 32;
  const bool request = lane < activeLanes;

  gpu::BatchAllocator<Key> allocator = gpu::warpBatchAllocator(pool, warpId);
  for (int op = 0; op < opsPerThread; ++op) {
    // Every lane calls, including the non-requesting ones: they carry the scan.
    queue->enqueue(makeKey(threadId, op), request, allocator);
  }
}

// Bounded drain. The exit must be warp-uniform, which __all_sync provides: a
// round in which no lane of the warp obtained anything means the queue is empty
// (or every requester has filled its slice), and every lane sees the same
// answer.
__global__ void drainKernel(Queue *queue, Key *out, int *counts, int capacity,
                            int activeLanes) {
  __shared__ Key blockStore[kMaxBlockThreads];
  Key *warpStore = warpStoreFor(blockStore);

  const int lane = gpu::laneId();
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int taken = 0;

  for (int round = 0; round < capacity; ++round) {
    const bool request = lane < activeLanes && taken < capacity;
    Key value = 0;
    const bool got = queue->dequeue(value, request, warpStore);
    if (got) {
      out[static_cast<long long>(threadId) * capacity + taken] = value;
      ++taken;
    }
    if (__all_sync(gpu::kFullWarpMask, !got)) {
      break;
    }
  }

  counts[threadId] = taken;
}

// Warp-uniform operation assignment: a whole warp enqueues or a whole warp
// dequeues. For CCLQ this is not only the design's choice but a requirement --
// the two operations are separate collectives, so lanes of one warp cannot take
// different ones.
//
// producerLanes and consumerLanes are separate so the asymmetric scenario can
// drive the partial-claiming path; the symmetric scenarios pass the same value
// for both.
__global__ void mixedKernel(Queue *queue, gpu::BatchPoolView<Key> pool,
                            int opsPerThread, int producerLanes,
                            int consumerLanes, Key *out, int *counts) {
  __shared__ Key blockStore[kMaxBlockThreads];
  Key *warpStore = warpStoreFor(blockStore);

  const int lane = gpu::laneId();
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int warpId = threadId / 32;
  const bool isProducer = (warpId % 2) == 0;

  gpu::BatchAllocator<Key> allocator = gpu::warpBatchAllocator(pool, warpId);
  int taken = 0;

  for (int op = 0; op < opsPerThread; ++op) {
    if (isProducer) {
      queue->enqueue(makeKey(threadId, op), lane < producerLanes, allocator);
    } else {
      Key value = 0;
      if (queue->dequeue(value, lane < consumerLanes, warpStore)) {
        out[static_cast<long long>(threadId) * opsPerThread + taken] = value;
        ++taken;
      }
    }
  }

  counts[threadId] = taken;
}

// Item 1: the prefix sum in isolation.
__global__ void scanKernel(const int *values, int *inclusive, int *totals) {
  const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const int scanned = gpu::warpInclusiveScan(values[threadId]);
  inclusive[threadId] = scanned;
  totals[threadId] = gpu::broadcastFromCombiner(scanned);
}

// --- host-side plumbing --------------------------------------------------

struct Fixture {
  Fixture(int prefillNodes, int nodesPerWarp, int warps)
      : pool(prefillNodes, nodesPerWarp, warps), queue(1) {
    initQueueKernel<<<1, 1>>>(queue.get(), pool.view());
    GPU_CUDA_CHECK_KERNEL();
  }

  gpu::BatchPool<Key> pool;
  gpu::DeviceBuffer<Queue> queue;
};

// Repeat the bounded drain until a round finds nothing.
std::vector<std::vector<Key>> drainAll(Queue *queue, int blocks,
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
    drainKernel<<<blocks, kWarpSize>>>(queue, out.get(), counts.get(), capacity,
                                       activeLanes);
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
      std::vector<Key> &slot = records[static_cast<std::size_t>(t)];
      slot.insert(slot.end(), hostOut.begin() + base,
                  hostOut.begin() + base + n);
    }
  }
}

// Read back one mixed run's per-lane records, in dequeue order.
std::vector<std::vector<Key>> readRecords(const gpu::DeviceBuffer<Key> &out,
                                          const gpu::DeviceBuffer<int> &counts,
                                          int threads, int capacity) {
  const std::size_t slots = static_cast<std::size_t>(threads) *
                            static_cast<std::size_t>(capacity);
  std::vector<Key> hostOut(slots);
  std::vector<int> hostCounts(static_cast<std::size_t>(threads));
  out.copyToHost(hostOut.data(), slots);
  counts.copyToHost(hostCounts.data(), static_cast<std::size_t>(threads));

  std::vector<std::vector<Key>> records(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    const std::size_t base =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(capacity);
    const std::size_t n =
        static_cast<std::size_t>(hostCounts[static_cast<std::size_t>(t)]);
    records[static_cast<std::size_t>(t)].assign(hostOut.begin() + base,
                                                hostOut.begin() + base + n);
  }
  return records;
}

// --- scenarios -----------------------------------------------------------

void scanCorrectness() {
  const int blocks = gpuConfig().warps;
  const int threads = blocks * kWarpSize;

  gpu::DeviceBuffer<int> values(static_cast<std::size_t>(threads));
  gpu::DeviceBuffer<int> inclusive(static_cast<std::size_t>(threads));
  gpu::DeviceBuffer<int> totals(static_cast<std::size_t>(threads));

  std::vector<int> hostValues(static_cast<std::size_t>(threads));
  std::vector<int> hostInclusive(static_cast<std::size_t>(threads));
  std::vector<int> hostTotals(static_cast<std::size_t>(threads));

  // Two patterns. The 0/1 pattern is what the queue actually feeds the scan and
  // is what a duplicate slot would come from. The small-integer pattern is
  // there because 0/1 values cannot distinguish a correct scan from one that
  // drops the `if (lane >= offset)` guard on __shfl_up_sync in some cases,
  // whereas distinct magnitudes can.
  for (int pattern = 0; pattern < 2; ++pattern) {
    unsigned int state = 0x12345677u + static_cast<unsigned int>(pattern);
    for (int t = 0; t < threads; ++t) {
      state = state * 1664525u + 1013904223u;
      hostValues[static_cast<std::size_t>(t)] =
          pattern == 0 ? static_cast<int>((state >> 16) & 1u)
                       : static_cast<int>((state >> 16) % 7u);
    }
    values.copyFromHost(hostValues.data(), static_cast<std::size_t>(threads));

    scanKernel<<<blocks, kWarpSize>>>(values.get(), inclusive.get(),
                                      totals.get());
    GPU_CUDA_CHECK_KERNEL();

    inclusive.copyToHost(hostInclusive.data(),
                         static_cast<std::size_t>(threads));
    totals.copyToHost(hostTotals.data(), static_cast<std::size_t>(threads));

    for (int warp = 0; warp < blocks; ++warp) {
      int running = 0;
      for (int lane = 0; lane < kWarpSize; ++lane) {
        const std::size_t index =
            static_cast<std::size_t>(warp) * kWarpSize + lane;
        running += hostValues[index];
        CHECK_EQ(hostInclusive[index], running);
      }
      // Lane 31's inclusive value is the warp total, which is what the queue
      // broadcasts as num_request.
      for (int lane = 0; lane < kWarpSize; ++lane) {
        const std::size_t index =
            static_cast<std::size_t>(warp) * kWarpSize + lane;
        CHECK_EQ(hostTotals[index], running);
      }
    }
  }
}

void cclqSequential() {
  constexpr int kItems = 4096;
  if (!encodingFits(1, kItems)) {
    return;
  }

  Fixture fixture(prefillNodesFor(kItems), /*nodesPerWarp=*/0, /*warps=*/1);
  prefillKernel<<<1, kWarpSize>>>(fixture.queue.get(), fixture.pool.view(),
                                  kItems, /*prefillProducer=*/0);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("cclq sequential prefill");

  // One warp draining, all 32 lanes requesting: a batch of 32 is enqueued and a
  // batch of 32 dequeued, so this is the exactly-matched case and the whole
  // stream comes back in enqueue order.
  const std::vector<std::vector<Key>> records =
      drainAll(fixture.queue.get(), /*blocks=*/1, /*activeLanes=*/32, kItems);

  std::vector<Key> observed = flatten(records);
  CHECK_EQ(observed.size(), static_cast<std::size_t>(kItems));

  // Order within one dequeue call is by lane, and lane L of drain round r holds
  // item 32*r + L, so the per-lane records interleave. Conservation plus
  // per-producer FIFO is the right check here rather than exact sequence.
  std::vector<Key> expected;
  for (int i = 0; i < kItems; ++i) {
    expected.push_back(makeKey(0, i));
  }
  checkConservation(observed, expected);
  checkPerProducerFifo(records, 1);
}

void cclqFillDrain() {
  const Config &cfg = gpuConfig();
  const int blocks = cfg.warps;
  const int threads = blocks * kWarpSize;
  if (!encodingFits(threads, cfg.opsPerThread)) {
    return;
  }

  Fixture fixture(/*prefillNodes=*/0, /*nodesPerWarp=*/cfg.opsPerThread,
                  blocks);

  fillKernel<<<blocks, kWarpSize>>>(fixture.queue.get(), fixture.pool.view(),
                                    cfg.opsPerThread, cfg.activeLanes);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("cclq fill");

  const int capacity = cfg.opsPerThread * 8;
  const std::vector<std::vector<Key>> records =
      drainAll(fixture.queue.get(), blocks, cfg.activeLanes, capacity);

  std::vector<Key> expected;
  for (int t = 0; t < threads; ++t) {
    if (t % kWarpSize >= cfg.activeLanes) {
      continue;
    }
    for (int op = 0; op < cfg.opsPerThread; ++op) {
      expected.push_back(makeKey(t, op));
    }
  }

  checkConservation(flatten(records), expected);
  checkPerProducerFifo(records, threads);
}

// producerLanes / consumerLanes equal for the symmetric case; different to
// drive partial claiming.
void cclqMixed(int producerLanes, int consumerLanes) {
  const Config &cfg = gpuConfig();
  const int blocks = cfg.warps;
  const int threads = blocks * kWarpSize;
  const int prefillProducer = threads;
  if (!encodingFits(threads + 1, std::max(cfg.opsPerThread, cfg.prefill))) {
    return;
  }

  Fixture fixture(prefillNodesFor(cfg.prefill),
                  /*nodesPerWarp=*/cfg.opsPerThread, blocks);

  prefillKernel<<<1, kWarpSize>>>(fixture.queue.get(), fixture.pool.view(),
                                  cfg.prefill, prefillProducer);
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("cclq prefill");

  const std::size_t slots = static_cast<std::size_t>(threads) *
                            static_cast<std::size_t>(cfg.opsPerThread);
  gpu::DeviceBuffer<Key> out(slots);
  gpu::DeviceBuffer<int> counts(static_cast<std::size_t>(threads));
  counts.zero();

  mixedKernel<<<blocks, kWarpSize>>>(fixture.queue.get(), fixture.pool.view(),
                                     cfg.opsPerThread, producerLanes,
                                     consumerLanes, out.get(), counts.get());
  GPU_CUDA_CHECK_KERNEL();
  fixture.pool.failIfOverflowed("cclq mixed run");

  const std::vector<std::vector<Key>> records =
      readRecords(out, counts, threads, cfg.opsPerThread);

  // Whatever the consumers did not take is still in the queue. Draining it
  // afterwards is what makes conservation checkable without knowing in advance
  // how many dequeues found work.
  const std::vector<std::vector<Key>> leftover = drainAll(
      fixture.queue.get(), blocks, /*activeLanes=*/32, cfg.opsPerThread * 8);

  std::vector<Key> observed = flatten(records);
  const std::vector<Key> tail = flatten(leftover);
  observed.insert(observed.end(), tail.begin(), tail.end());

  std::vector<Key> expected;
  for (int i = 0; i < cfg.prefill; ++i) {
    expected.push_back(makeKey(prefillProducer, i));
  }
  for (int t = 0; t < threads; ++t) {
    const bool isProducer = ((t / kWarpSize) % 2) == 0;
    if (!isProducer || t % kWarpSize >= producerLanes) {
      continue;
    }
    for (int op = 0; op < cfg.opsPerThread; ++op) {
      expected.push_back(makeKey(t, op));
    }
  }

  checkConservation(observed, expected);
  // The concurrent phase and the leftover drain are separate observation
  // sequences and are checked separately: the drain is not a continuation of
  // any one consumer lane's view.
  checkPerProducerFifo(records, threads + 1);
  checkPerProducerFifo(leftover, threads + 1);
}

} // namespace cclqtest

using namespace cclqtest;

GPU_TEST(cclq_warp_scan) { scanCorrectness(); }

GPU_TEST(cclq_sequential) { cclqSequential(); }

GPU_TEST(cclq_fill_drain) { cclqFillDrain(); }

GPU_TEST(cclq_mixed_symmetric) {
  cclqMixed(gpuConfig().activeLanes, gpuConfig().activeLanes);
}

// Consumers narrower than producers: a warp asks for 8 and finds a node holding
// 32, so it claims part of the node and leaves start_pos mid-range. The next
// consumer must pick up from where it stopped.
GPU_TEST(cclq_mixed_partial_claim_narrow_consumers) { cclqMixed(32, 8); }

// Consumers wider than producers: a warp asks for 32 and each node holds 8, so
// one request has to span four nodes, retiring three positions on the way.
GPU_TEST(cclq_mixed_partial_claim_wide_consumers) { cclqMixed(8, 32); }

int main(int argc, char **argv) { return gpu_test::runAll(argc, argv); }
