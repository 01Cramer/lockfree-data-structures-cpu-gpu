// Queue: single-lock and two-lock under both mutex and spinlock, plus the
// lock-free Michael-Scott queue. Five variants.

#include <cstdint>
#include <vector>

#include "cpu/lockfree/lockfree_queue.hpp"
#include "cpu/mutex/mutex_queue.hpp"
#include "cpu/mutex/mutex_queue_two_lock.hpp"
#include "cpu/spinlock/spinlock_queue.hpp"
#include "cpu/spinlock/spinlock_queue_two_lock.hpp"

#include "support/variants.hpp"

namespace {
constexpr std::int64_t kOpsPerThread = 500000;
// Percentage of operations that enqueue.
constexpr std::int64_t kMixes[] = {90, 50, 10};
constexpr std::int64_t kCentreMix = 50;

constexpr int kRepetitions = 5;

bench::Config config(std::int64_t mixPct) {
  bench::Config cfg;
  cfg.opsPerThread = bench::scaledBudget(kOpsPerThread);
  cfg.mixPct = mixPct;
  cfg.keyRange = 0;
  return cfg;
}

template <cpu::Layout L>
void registerSingleLockFamily(const bench::Config &cfg,
                              const std::vector<int> &threads,
                              int repetitions) {
  bench::registerVariant<cpu::mutex::Queue, L, bench::QueueOps>(
      "mutex", cfg, threads, repetitions);
  bench::registerVariant<cpu::spinlock::Queue, L, bench::QueueOps>(
      "spinlock", cfg, threads, repetitions);
  bench::registerVariant<cpu::lockfree::Queue, L, bench::QueueOps>(
      "lockfree", cfg, threads, repetitions);
}

template <cpu::Layout L>
void registerTwoLockFamily(const bench::Config &cfg,
                           const std::vector<int> &threads, int repetitions) {
  // Mirrors the static_assert in {mutex,spinlock}_queue_two_lock.hpp: under
  // PadLockFromData an unpadded m_tail lands on m_headMutex's line, so the
  // class refuses to instantiate.
  if constexpr (!(L.padLockFromData && !L.padSyncPoints)) {
    bench::registerVariant<cpu::mutex::QueueTwoLock, L, bench::QueueOps>(
        "mutex_two_lock", cfg, threads, repetitions);
    bench::registerVariant<cpu::spinlock::QueueTwoLock, L, bench::QueueOps>(
        "spinlock_two_lock", cfg, threads, repetitions);
  }
}

template <cpu::Layout L>
void registerFamily(const bench::Config &cfg, const std::vector<int> &threads,
                    int repetitions) {
  registerSingleLockFamily<L>(cfg, threads, repetitions);
  registerTwoLockFamily<L>(cfg, threads, repetitions);
}

void registerAll() {
  const std::vector<int> ladder = bench::threadLadder();
  const int reps = bench::repetitions(kRepetitions);

  // Main grid: default layout, every mix, the whole ladder.
  for (std::int64_t mix : kMixes) {
    registerFamily<cpu::NoPad>(config(mix), ladder, reps);
  }

  // Supplementary: every layout, default mix, the whole ladder.
  registerFamily<cpu::PadLockFromData>(config(kCentreMix), ladder, reps);
  registerFamily<cpu::PadSyncPoints>(config(kCentreMix), ladder, reps);
  registerFamily<cpu::PadSyncPointsAndLockFromData>(config(kCentreMix), ladder,
                                                    reps);
}

} // namespace

int main(int argc, char **argv) {
  return bench::runBenchmarks(argc, argv, &registerAll);
}
