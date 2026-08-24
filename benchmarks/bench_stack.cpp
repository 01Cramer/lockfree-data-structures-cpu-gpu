// Stack: cpu::mutex::Stack, cpu::spinlock::Stack, cpu::lockfree::Stack

#include <cstdint>
#include <vector>

#include "cpu/lockfree/lockfree_stack.hpp"
#include "cpu/mutex/mutex_stack.hpp"
#include "cpu/spinlock/spinlock_stack.hpp"

#include "support/variants.hpp"

namespace {
constexpr std::int64_t kOpsPerThread = 500000;
// Percentage of operations that push.
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
void registerFamily(const bench::Config &cfg, const std::vector<int> &threads,
                    int repetitions) {
  bench::registerVariant<cpu::mutex::Stack, L, bench::StackOps>(
      "mutex", cfg, threads, repetitions);
  bench::registerVariant<cpu::spinlock::Stack, L, bench::StackOps>(
      "spinlock", cfg, threads, repetitions);
  bench::registerVariant<cpu::lockfree::Stack, L, bench::StackOps>(
      "lockfree", cfg, threads, repetitions);
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
