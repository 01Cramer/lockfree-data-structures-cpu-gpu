// Host-side queue correctness oracles used by the GPU tests.
//
// Conservation checks that every produced item is observed exactly once.
// Per-producer FIFO checks that each observer sees a producer's sequence in
// increasing order.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "support/test_harness.hpp"

namespace queue_oracle {

// Test payload: wide enough to encode default stress-test producer ids and
// sequence numbers without reducing the workload.
using Key = std::uint64_t;

// value = producer * kStride + seq, so each item is unique and decodable.
inline constexpr Key kStride = 1ull << 32;

// Kernels encode values; host oracles decode them.
#ifdef __CUDACC__
#define QUEUE_ORACLE_HD __host__ __device__
#else
#define QUEUE_ORACLE_HD
#endif

QUEUE_ORACLE_HD inline Key makeKey(int producer, int seq) {
  return static_cast<Key>(producer) * kStride + static_cast<Key>(seq);
}

inline int producerOf(Key value) { return static_cast<int>(value / kStride); }
inline int seqOf(Key value) { return static_cast<int>(value % kStride); }

// Fail loudly if a test configuration cannot be encoded.
inline bool encodingFits(int producers, int maxSeq) {
  const bool seqFits = static_cast<Key>(maxSeq) < kStride;
  const Key limit = std::numeric_limits<Key>::max();
  bool productFits = static_cast<Key>(producers) <= limit / kStride;
  if (productFits) {
    const Key producerPart = static_cast<Key>(producers) * kStride;
    productFits = producerPart <= limit - static_cast<Key>(maxSeq);
  }
  CHECK(seqFits);
  CHECK(productFits);
  return seqFits && productFits;
}

// Check that observed and expected contain the same values.
inline void checkConservation(std::vector<Key> observed,
                             std::vector<Key> expected) {
  CHECK_EQ(observed.size(), expected.size());
  std::sort(observed.begin(), observed.end());
  std::sort(expected.begin(), expected.end());

  const std::size_t common = std::min(observed.size(), expected.size());
  int reported = 0;
  for (std::size_t i = 0; i < common && reported < 8; ++i) {
    if (observed[i] != expected[i]) {
      std::fprintf(stderr,
                   "  divergence at %zu: observed producer %d seq %d, "
                   "expected producer %d seq %d\n",
                   i, producerOf(observed[i]), seqOf(observed[i]),
                   producerOf(expected[i]), seqOf(expected[i]));
      ++reported;
      CHECK(observed[i] == expected[i]);
    }
  }
}

// Check per-producer FIFO order inside each observation sequence.
inline void checkPerProducerFifo(const std::vector<std::vector<Key>> &records,
                                 int producerCount) {
  for (const std::vector<Key> &seen : records) {
    std::vector<int> lastSeq(static_cast<std::size_t>(producerCount), -1);
    for (Key value : seen) {
      const int producer = producerOf(value);
      const int seq = seqOf(value);
      CHECK(producer >= 0 && producer < producerCount);
      if (producer < 0 || producer >= producerCount) {
        continue;
      }
      CHECK(seq > lastSeq[static_cast<std::size_t>(producer)]);
      lastSeq[static_cast<std::size_t>(producer)] = seq;
    }
  }
}

inline std::vector<Key> flatten(const std::vector<std::vector<Key>> &records) {
  std::vector<Key> all;
  for (const std::vector<Key> &v : records) {
    all.insert(all.end(), v.begin(), v.end());
  }
  return all;
}

} // namespace queue_oracle
