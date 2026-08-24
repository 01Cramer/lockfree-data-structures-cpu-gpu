#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"

#include "cpu/shared/cache.hpp"
#include "support/experiment.hpp"
#include "support/workload.hpp"

namespace bench {

inline const char *layoutName(cpu::Layout layout) {
  if (!layout.padSyncPoints && !layout.padLockFromData) {
    return "NoPad";
  }
  if (layout.padSyncPoints && !layout.padLockFromData) {
    return "PadSyncPoints";
  }
  if (!layout.padSyncPoints && layout.padLockFromData) {
    return "PadLockFromData";
  }
  return "PadSyncPointsAndLockFromData";
}

// BENCH_QUICK=1 cuts the budget and the repetition count to a smoke test.
inline bool quickMode() {
  const char *enabled = std::getenv("BENCH_QUICK");
  return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

inline std::int64_t scaledBudget(std::int64_t full) {
  constexpr std::int64_t quickOps = 1000;
  return quickMode() ? quickOps : full;
}

inline int repetitions(int full) { return quickMode() ? 2 : full; }

// Powers of two, up to the largest that still fits inside 2H: on the
// measurement host (H = 72) that is 1 2 4 8 16 32 64 128.
inline std::vector<int> threadLadder() {
  const int maximum = 2 * hardwareThreads();
  std::vector<int> counts;
  for (int threads = 1; threads <= maximum; threads *= 2) {
    counts.push_back(threads);
  }
  return counts;
}

inline std::string benchmarkName(const char *adt, const char *impl,
                                 const char *layout, const Config &cfg) {
  return std::string("adt=") + adt + "/impl=" + impl + "/layout=" + layout +
         "/mix=" + std::to_string(cfg.mixPct) +
         "/keys=" + std::to_string(cfg.keyRange) +
         "/ops=" + std::to_string(cfg.opsPerThread);
}

// Tmpl  : the structure class template, e.g. cpu::mutex::Stack
// L     : the Layout
// OpsT  : the workload policy, e.g. bench::StackOps
template <template <typename, cpu::Layout> class Tmpl, cpu::Layout L,
          template <typename> class OpsT>
void registerVariant(const char *impl, const Config &cfg,
                     const std::vector<int> &threadCounts, int repetitions) {
  using Structure = Tmpl<Key, L>;
  using Ops = OpsT<Structure>;

  if constexpr (L.padLockFromData && !Structure::hasLockWord) {
    return;
  }

  auto *registered = benchmark::RegisterBenchmark(
      benchmarkName(Ops::adt, impl, layoutName(L), cfg),
      [cfg](benchmark::State &state) {
        runExperiment<Ops, Structure>(state, cfg);
      });

  // One iteration is one whole experiment, so the statistics come entirely from
  // the repetitions, each of which rebuilds the structure and its pool.
  registered->UseManualTime()->Iterations(1)->Repetitions(repetitions);

  for (int threads : threadCounts) {
    registered->Threads(threads);
  }
}

inline int runBenchmarks(int argc, char **argv, void (*registerAll)()) {
  benchmark::Initialize(&argc, argv);
  if (quickMode()) {
    std::fprintf(stderr, "BENCH_QUICK=1: reduced budget and repetitions. "
                         "Smoke test only -- these numbers are not results.\n");
  }
  registerAll();
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

} // namespace bench
