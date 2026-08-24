// The queue correctness oracles, and the value encoding they read.
//
// Host-only C++ with no CUDA in it, shared by every GPU queue test. Extracted
// into its own header when the CCLQ variant arrived, because that variant has a
// warp-collective API and therefore needs its own kernels -- but exactly the
// same oracles. Two copies of a conservation check is one copy too many: a
// divergence between them would be indistinguishable from a divergence between
// the queues.
//
// The oracles are the CPU ones (tests/cpu_stress_test.cpp) in substance. That
// they transfer unchanged is a property of what is being checked: conservation
// and per-producer FIFO are statements about a queue, not about a platform. The
// only difference is that the observations arrive in a buffer copied back from
// the device rather than in a std::vector the consumer thread filled itself.
//
//   Conservation      every enqueued item is dequeued exactly once. Catches a
//                     dropped link, a node claimed twice, a lost tail update,
//                     and -- for the combining variant -- two lanes computing
//                     the same batch slot, which is what the unsynchronized
//                     prefix sum in Zhang's published scanWarp produces.
//   Per-producer FIFO within any single observer's ordered view, one producer's
//                     items appear in increasing seq. Catches reordering that
//                     conservation cannot see: a queue returning the right
//                     multiset in the wrong order passes conservation.
//
// A note on the second oracle and the batching variant. It might look as though
// per-producer FIFO cannot apply to a queue that combines 32 items into one
// node, since items combined within a warp have no defined order relative to
// each other. It does apply, and the reason is worth writing down: each
// producer lane contributes exactly one item per batch, and its successive
// items therefore land in successive batch nodes, which are linked at
// increasing positions. Consumers claim positions in increasing order and, from
// one node, claim slots in increasing order. So a single consumer lane's record
// is monotone in (position, slot) for every producer, which is the encoding
// order. What is genuinely undefined for the batching variant is order
// *between* producers within one batch -- and neither oracle checks that.

#pragma once

#include <algorithm>
#include <cstdio>
#include <vector>

#include "support/test_harness.hpp"

namespace queue_oracle {

// Scalar payload, 32 bits: it keeps a plain node at 8 bytes, which matters
// because the pools are sized as O(total operations).
using Key = unsigned int;

// value = producer * kStride + seq. Makes every enqueued value globally unique
// and lets the host decode which thread produced it, which is what the
// per-producer order check needs. Both factors are checked against the run's
// configuration by encodingFits() rather than assumed.
inline constexpr Key kStride = 1u << 18;

// The encoder is needed on both sides -- kernels produce the values, the host
// decodes them -- so it is annotated for both rather than written twice. The
// decoders below are host-only because only the oracles use them.
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

// Reports through CHECK so a configuration that cannot be encoded fails loudly
// instead of producing a quietly wrong oracle.
inline bool encodingFits(int producers, int maxSeq) {
  const bool seqFits = static_cast<Key>(maxSeq) < kStride;
  const bool productFits =
      static_cast<unsigned long long>(producers) * kStride +
          static_cast<unsigned long long>(maxSeq) <
      0xffffffffull;
  CHECK(seqFits);
  CHECK(productFits);
  return seqFits && productFits;
}

// Nothing lost, nothing duplicated, nothing invented. Sorting both sides and
// comparing reports the first divergence, which is more use than a bare count
// mismatch.
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

// Within one ordered observation sequence, each producer's items must appear in
// strictly increasing seq.
//
// Concatenating one slot's records across drain rounds is sound: the rounds are
// separated by kernel boundaries, so the concatenation still respects real
// time, and any real-time-ordered sequence of observations is a valid witness
// for FIFO.
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
