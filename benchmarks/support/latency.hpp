// The file below is adapted from Fedor Pikus's "LockFree" repository
// (https://github.com/fpikus/LockFree).
//
//   MIT License
//   Copyright (c) 2026 fpikus
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the "Software"),
//   to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

namespace bench {

// Read once: this is tested per operation, so it must not become a getenv call
// inside the loop.
inline bool latencyEnabled() {
  static const bool enabled = [] {
    const char *raw = std::getenv("BENCH_LATENCY");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

// Off by default
inline bool histogramDumpEnabled() {
  static const bool enabled = [] {
    const char *raw = std::getenv("BENCH_HIST_DUMP");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

using Tick = std::uint64_t;

// Opening stamp
inline Tick tickStart() {
  _mm_lfence();
  const Tick stamp = static_cast<Tick>(__rdtsc());
  _mm_lfence();
  return stamp;
}

// Closing stamp
inline Tick tickEnd() {
  unsigned processor = 0;
  const Tick stamp = static_cast<Tick>(__rdtscp(&processor));
  _mm_lfence();
  return stamp;
}

// Cycles to nanoseconds, plus the cost of a stamp pair.
//
// Construct before the timed region: latencyClock() is called from the setup
// path in experiment.hpp because the calibration takes 100 ms.
class LatencyClock {
public:
  LatencyClock()
      : m_cyclesPerNano(calibrateCyclesPerNano()),
        m_overhead(measureOverhead()) {
    if (latencyEnabled()) {
      warnIfTscUnreliable();
    }
  }

  double cyclesPerNano() const { return m_cyclesPerNano; }

  // Median of back-to-back stamp pairs: what a measured interval contains
  // beyond the operation itself. Reported, never subtracted.
  Tick overhead() const { return m_overhead; }

  double nanos(double cycles) const {
    return m_cyclesPerNano > 0.0 ? cycles / m_cyclesPerNano : 0.0;
  }

private:
  // Wall-clock against cycle count. Assumes an invariant TSC, which is what the
  // check below is for; 100 ms is enough to swamp sleep_for scheduling jitter
  // without a perceptible startup delay.
  static double calibrateCyclesPerNano() {
    const auto wallBefore = std::chrono::steady_clock::now();
    const Tick before = tickStart();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const Tick after = tickEnd();
    const auto wallAfter = std::chrono::steady_clock::now();

    const double nanos =
        std::chrono::duration<double, std::nano>(wallAfter - wallBefore)
            .count();
    if (nanos <= 0.0 || after <= before) {
      return 1.0;
    }
    return static_cast<double>(after - before) / nanos;
  }

  static Tick measureOverhead() {
    constexpr std::size_t samples = 1024;
    std::array<Tick, samples> deltas{};
    for (std::size_t i = 0; i < samples; ++i) {
      const Tick before = tickStart();
      const Tick after = tickEnd();
      deltas[i] = after > before ? after - before : 0;
    }
    // Median, not mean: an interrupt during calibration would drag a mean up
    // and then be reported as the cost of every stamp.
    std::sort(deltas.begin(), deltas.end());
    return deltas[samples / 2];
  }

  // The counter must tick at a fixed rate and keep ticking across C-states, or
  // the calibration above means nothing. Warns rather than aborts: it is a
  // quality-of-measurement question, and LatencyHistogram::negative counts the
  // migrations that would follow.
  static void warnIfTscUnreliable() {
#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
    std::FILE *file = std::fopen("/proc/cpuinfo", "r");
    if (file == nullptr) {
      return;
    }
    char line[4096];
    bool sawFlags = false;
    bool constant = false;
    bool nonstop = false;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      if (std::strncmp(line, "flags", 5) != 0) {
        continue;
      }
      sawFlags = true;
      constant = std::strstr(line, "constant_tsc") != nullptr;
      nonstop = std::strstr(line, "nonstop_tsc") != nullptr;
      break;
    }
    std::fclose(file);

    if (sawFlags && !(constant && nonstop)) {
      std::fprintf(stderr,
                   "bench: BENCH_LATENCY=1 but /proc/cpuinfo reports "
                   "constant_tsc=%d nonstop_tsc=%d. The cycle counter may not "
                   "run at a fixed rate, so latency figures are suspect.\n",
                   static_cast<int>(constant), static_cast<int>(nonstop));
    }
#endif
  }

  double m_cyclesPerNano;
  Tick m_overhead;
};

inline LatencyClock &latencyClock() {
  static LatencyClock clock;
  return clock;
}

// --- Log-linear histogram ---------------------------------------------------
inline constexpr std::size_t kHistSubBits = 3;
inline constexpr std::size_t kHistSubs = std::size_t{1} << kHistSubBits;
inline constexpr std::size_t kHistBuckets = 512; // max index 495

// Percentile list
inline constexpr std::array<double, 5> kLatencyPercentiles{50.0, 90.0, 95.0,
                                                           99.0, 99.9};

// Index of the highest set bit. __builtin_clzll is not available on MSVC.
inline unsigned topBit(std::uint64_t value) {
#if defined(_MSC_VER)
  unsigned long index = 0;
  _BitScanReverse64(&index, value);
  return static_cast<unsigned>(index);
#else
  return 63u - static_cast<unsigned>(__builtin_clzll(value));
#endif
}

inline std::size_t bucketIndex(std::uint64_t value) {
  if (value < kHistSubs) {
    return static_cast<std::size_t>(value);
  }
  const unsigned exponent = topBit(value);
  const unsigned sub = static_cast<unsigned>(
      (value >> (exponent - kHistSubBits)) & (kHistSubs - 1));
  return (static_cast<std::size_t>(exponent) - kHistSubBits + 1) * kHistSubs +
         sub;
}

inline std::uint64_t bucketLower(std::size_t bucket) {
  if (bucket < kHistSubs) {
    return bucket;
  }
  const std::size_t exponent = bucket / kHistSubs + kHistSubBits - 1;
  const std::size_t sub = bucket % kHistSubs;
  return static_cast<std::uint64_t>(kHistSubs + sub)
         << (exponent - kHistSubBits);
}

inline std::uint64_t bucketWidth(std::size_t bucket) {
  if (bucket < kHistSubs) {
    return 1;
  }
  const std::size_t exponent = bucket / kHistSubs + kHistSubBits - 1;
  return std::uint64_t{1} << (exponent - kHistSubBits);
}

struct LatencyHistogram {
  std::array<std::uint64_t, kHistBuckets> bins{};
  std::int64_t count = 0;

  // Intervals that ran backwards. On x86 that means the thread migrated between
  // cores whose counters are not synchronized.
  std::int64_t negative = 0;

  double sum = 0.0;
  double sumOfSquares = 0.0;
  Tick min = std::numeric_limits<Tick>::max();
  Tick max = 0;

  // The whole hot path, all of it into this thread's own record.
  void record(Tick before, Tick after) {
    if (after < before) {
      ++negative;
      return;
    }
    const Tick latency = after - before;
    if (latency < min) {
      min = latency;
    }
    if (latency > max) {
      max = latency;
    }
    const double value = static_cast<double>(latency);
    sum += value;
    sumOfSquares += value * value;
    ++count;
    ++bins[bucketIndex(latency)];
  }

  void add(const LatencyHistogram &other) {
    count += other.count;
    negative += other.negative;
    sum += other.sum;
    sumOfSquares += other.sumOfSquares;
    min = std::min(min, other.min);
    max = std::max(max, other.max);
    for (std::size_t bucket = 0; bucket < kHistBuckets; ++bucket) {
      bins[bucket] += other.bins[bucket];
    }
  }

  double meanCycles() const {
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
  }

  double stdevCycles() const {
    if (count <= 1) {
      return 0.0;
    }
    const double n = static_cast<double>(count);
    const double mean = sum / n;
    const double numerator = sumOfSquares - n * mean * mean;
    return std::sqrt(numerator > 0.0 ? numerator / (n - 1.0) : 0.0);
  }

  // The pth percentile in cycles, linearly interpolating inside the bin.
  double percentileCycles(double percent) const {
    if (count == 0) {
      return 0.0;
    }
    const double target = percent * 0.01 * static_cast<double>(count);
    std::uint64_t cumulative = 0;
    for (std::size_t bucket = 0; bucket < kHistBuckets; ++bucket) {
      const std::uint64_t inBucket = bins[bucket];
      if (inBucket == 0) {
        continue;
      }
      const double reached = static_cast<double>(cumulative + inBucket);
      if (reached >= target) {
        const double fraction = (target - static_cast<double>(cumulative)) /
                                static_cast<double>(inBucket);
        return static_cast<double>(bucketLower(bucket)) +
               fraction * static_cast<double>(bucketWidth(bucket));
      }
      cumulative += inBucket;
    }
    return 0.0; // Should be unreachable.
  }

  double percentileNanos(double percent) const {
    return latencyClock().nanos(percentileCycles(percent));
  }
};

// --- Optional full-histogram dump -----------------------------------------
inline void dumpHistogram(const char *name, const LatencyHistogram &histogram) {
  if (histogram.count == 0) {
    return;
  }
  std::uint64_t modal = 0;
  for (const std::uint64_t inBucket : histogram.bins) {
    modal = std::max(modal, inBucket);
  }
  if (modal == 0) {
    return;
  }

  const double perCycle = 1.0 / latencyClock().cyclesPerNano();
  constexpr int barWidth = 40;

  std::fprintf(stderr, "# histogram %s (%lld observations)\n", name,
               static_cast<long long>(histogram.count));
  std::fprintf(stderr,
               "#   lower_ns    upper_ns        count    pct    cum_pct\n");

  std::uint64_t cumulative = 0;
  for (std::size_t bucket = 0; bucket < kHistBuckets; ++bucket) {
    const std::uint64_t inBucket = histogram.bins[bucket];
    if (inBucket == 0) {
      continue;
    }
    cumulative += inBucket;
    const double lower = static_cast<double>(bucketLower(bucket)) * perCycle;
    const double upper =
        static_cast<double>(bucketLower(bucket) + bucketWidth(bucket)) *
        perCycle;
    const double percent = 100.0 * static_cast<double>(inBucket) /
                           static_cast<double>(histogram.count);
    const double cumulativePercent = 100.0 * static_cast<double>(cumulative) /
                                     static_cast<double>(histogram.count);
    const int bars = static_cast<int>(static_cast<double>(barWidth) *
                                      static_cast<double>(inBucket) /
                                      static_cast<double>(modal));

    std::fprintf(stderr, "# %10.2f %11.2f %12llu %6.2f%% %8.2f%%  ", lower,
                 upper, static_cast<unsigned long long>(inBucket), percent,
                 cumulativePercent);
    for (int bar = 0; bar < bars; ++bar) {
      std::fputc('#', stderr);
    }
    std::fputc('\n', stderr);
  }
  std::fflush(stderr);
}

} // namespace bench
