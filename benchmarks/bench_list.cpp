// List (set ADT): coarse and hand-over-hand locking under both mutex and
// spinlock, plus the lock-free Harris list. Five variants.

#include <cstdint>
#include <vector>

#include "cpu/lockfree/lockfree_list.hpp"
#include "cpu/mutex/mutex_list.hpp"
#include "cpu/mutex/mutex_list_hand_over_hand.hpp"
#include "cpu/spinlock/spinlock_list.hpp"
#include "cpu/spinlock/spinlock_list_hand_over_hand.hpp"

#include "support/variants.hpp"

namespace {
constexpr std::int64_t kOpsPerThread = 10000;
constexpr std::int64_t kKeyRange = 2048;

// Percentage of operations that are find; insert and remove split the rest
// evenly, which holds the set at half the key range for the whole run.
constexpr std::int64_t kMixes[] = {90, 50, 10};
constexpr std::int64_t kCentreMix = 50;

constexpr int kRepetitions = 5;

bench::Config config(std::int64_t mixPct) {
  bench::Config cfg;
  cfg.opsPerThread = bench::scaledBudget(kOpsPerThread);
  cfg.mixPct = mixPct;
  cfg.keyRange = kKeyRange;
  return cfg;
}

template <cpu::Layout L>
void registerFamily(const bench::Config &cfg, const std::vector<int> &threads,
                    int repetitions) {
  bench::registerVariant<cpu::mutex::List, L, bench::ListOps>(
      "mutex", cfg, threads, repetitions);
  bench::registerVariant<cpu::spinlock::List, L, bench::ListOps>(
      "spinlock", cfg, threads, repetitions);
  bench::registerVariant<cpu::lockfree::List, L, bench::ListOps>(
      "lockfree", cfg, threads, repetitions);
  bench::registerVariant<cpu::mutex::HandOverHandList, L, bench::ListOps>(
      "mutex_hand_over_hand", cfg, threads, repetitions);
  bench::registerVariant<cpu::spinlock::HandOverHandList, L, bench::ListOps>(
      "spinlock_hand_over_hand", cfg, threads, repetitions);
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
