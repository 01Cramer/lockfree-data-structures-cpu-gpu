// ONE GOOGLE BENCHMARK ITERATION == ONE COMPLETE EXPERIMENT, registered as
// Iterations(1) + Repetitions(k) + UseManualTime(). NodePool never reclaims, so
// the pool must be sized before the run starts, which rules out the framework's
// adaptive iteration count. Each repetition re-enters the body and rebuilds the
// structure and pool.
//
// Google Benchmark guarantees no thread enters the iteration loop until all
// have arrived and none leaves until all are done, which makes the
// construct-on-thread-0-before-the-loop pattern safe.

#pragma once

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "benchmark/benchmark.h"

#include "cpu/shared/cache.hpp"
#include "support/energy_meter.hpp"
#include "support/latency.hpp"
#include "support/perf_counters.hpp"
#include "support/thread_cpu.hpp"
#include "support/topology.hpp"
#include "support/workload.hpp"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace bench {

// Refuse to start a run whose pool would exceed this.
// The pool is allocated before the timed loop and freed after it, so only one
// is ever live.
inline constexpr std::size_t poolBudgetBytes() {
  constexpr std::int64_t megabytes = 8192; // 8 GB
  return static_cast<std::size_t>(megabytes) * 1024 * 1024;
}

[[noreturn]] inline void failPoolBudget(std::size_t requestedBytes,
                                        std::size_t budgetBytes,
                                        std::size_t nodesPerThread,
                                        std::size_t numThreads,
                                        std::size_t nodeBytes) {
  std::fprintf(stderr,
               "bench: this configuration needs %zu MB of pool memory "
               "(%zu nodes x %zu threads x %zu B per node), over the %zu MB "
               "budget. Lower opsPerThread, or raise the budget in "
               "poolBudgetBytes().\n",
               requestedBytes / (1024 * 1024), nodesPerThread, numThreads,
               nodeBytes, budgetBytes / (1024 * 1024));
  std::abort();
}

// Opened once per process
inline cpu::EnergyMeter &energyMeter() {
  static cpu::EnergyMeter meter;
  return meter;
}

// Counter names reach the CSV reporter unescaped, and nested RAPL domain labels
// contain a '/' ("package-0/dram").
inline std::string sanitized(const std::string &name) {
  std::string out = name;
  for (char &c : out) {
    if (c == '/' || c == ' ' || c == ',') {
      c = '_';
    }
  }
  return out;
}

inline int hardwareThreads() {
  const unsigned reported = std::thread::hardware_concurrency();
  return reported == 0 ? 1 : static_cast<int>(reported);
}

[[noreturn]] inline void failPin(std::size_t threadId, const char *reason) {
  std::fprintf(
      stderr,
      "bench: BENCH_PIN=1 but thread %zu could not be pinned (%s). Its "
      "placement would be unknown, so the run aborts rather than "
      "reporting numbers under a thread count that means nothing.\n",
      threadId, reason);
  std::abort();
}

inline bool pinningEnabled() {
  const char *enabled = std::getenv("BENCH_PIN");
  return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

// Pin worker i to logical CPU i, wrapping when there are more threads than
// CPUs. Linux only, deliberately. The measurement host runs Linux;
inline void pinThread(std::size_t threadId) {
  if (!pinningEnabled()) {
    return;
  }
#if defined(__linux__)
  // Identity, then wrap. That is enough on the measurement host because its CPU
  // numbering already is the placement we want - see topology.hpp for the
  // layout, and verifyMeasurementHost() for the one check that it still holds.
  const std::size_t cpu =
      threadId % static_cast<std::size_t>(hardwareThreads());

  if (cpu >= static_cast<std::size_t>(CPU_SETSIZE)) {
    failPin(threadId, "beyond CPU_SETSIZE");
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set) != 0) {
    failPin(threadId, "pthread_setaffinity_np failed");
  }
#else
  failPin(threadId, "pinning is implemented on Linux only");
#endif
}

// Per-thread observations. Cache-line aligned: written inside the timed region,
// so false sharing between neighbouring records would read as contention on the
// structure.
struct alignas(cpu::cacheLineSize) ThreadRecord {
  std::chrono::steady_clock::time_point start{};
  std::chrono::steady_clock::time_point stop{};
  std::int64_t ops = 0;
  std::int64_t ineffective = 0;
  std::int64_t delta = 0;

  // CPU consumed over the interval.
  ThreadCpu cpu{};

  // Hardware counters over an interval.
  PerfCounters perf{};

  // Every operation when latency is enabled.
  LatencyHistogram latency{};
};

// The per-thread records reduced to one set of figures.
struct Aggregate {
  using Clock = std::chrono::steady_clock;

  Clock::time_point earliestStart{Clock::time_point::max()};
  Clock::time_point latestStart{Clock::time_point::min()};
  Clock::time_point latestStop{Clock::time_point::min()};

  double slowestThread = 0.0;
  double fastestThread = std::numeric_limits<double>::max();

  std::int64_t ops = 0;
  std::int64_t ineffective = 0;
  std::int64_t delta = 0;

  double cpuSeconds = 0.0;
  double cpuUserSeconds = 0.0;
  double cpuSystemSeconds = 0.0;
  double mostCpu = 0.0;
  double leastCpu = std::numeric_limits<double>::max();
  std::int64_t voluntary = 0;
  std::int64_t involuntary = 0;

  std::uint64_t perfValues[kPerfSlotCount] = {};
  std::uint64_t perfEnabledNanos = 0;
  std::uint64_t perfRunningNanos = 0;

  // All three are all-or-nothing: a figure missing on one thread makes the
  // total meaningless, not merely incomplete.
  bool haveCpuTime = true;
  bool haveSwitches = true;
  bool havePerf = true;

  LatencyHistogram latency;

  void add(const ThreadRecord &record) {
    earliestStart = std::min(earliestStart, record.start);
    latestStart = std::max(latestStart, record.start);
    latestStop = std::max(latestStop, record.stop);

    const double elapsed =
        std::chrono::duration<double>(record.stop - record.start).count();
    slowestThread = std::max(slowestThread, elapsed);
    fastestThread = std::min(fastestThread, elapsed);

    ops += record.ops;
    ineffective += record.ineffective;
    delta += record.delta;

    cpuSeconds += record.cpu.totalSeconds;
    cpuUserSeconds += record.cpu.userSeconds;
    cpuSystemSeconds += record.cpu.systemSeconds;
    mostCpu = std::max(mostCpu, record.cpu.totalSeconds);
    leastCpu = std::min(leastCpu, record.cpu.totalSeconds);
    voluntary += record.cpu.voluntary;
    involuntary += record.cpu.involuntary;
    haveCpuTime = haveCpuTime && record.cpu.haveTime;
    haveSwitches = haveSwitches && record.cpu.haveSwitches;

    for (int slot = 0; slot < kPerfSlotCount; ++slot) {
      perfValues[slot] += record.perf.values[slot];
    }
    perfEnabledNanos += record.perf.enabledNanos;
    perfRunningNanos += record.perf.runningNanos;
    havePerf = havePerf && record.perf.have;

    latency.add(record.latency);
  }

  // The window spans first start to last stop, so a thread that finishes early
  // is still inside the denominator: throughput is work divided by the wall
  // time the machine was occupied.
  double windowSeconds() const {
    return std::chrono::duration<double>(latestStop - earliestStart).count();
  }

  double startSkewMillis() const {
    return std::chrono::duration<double>(latestStart - earliestStart).count() *
           1e3;
  }
};

// Everything one repetition needs, built and torn down by thread 0.
template <typename Structure> struct Run {
  Run(const Config &configuration, std::size_t nodesPerThread)
      : cfg(configuration), structure(nodesPerThread, configuration.numThreads),
        gate(static_cast<std::ptrdiff_t>(configuration.numThreads)), release(1),
        records(configuration.numThreads), poolNodesPerThread(nodesPerThread),
        energyBefore(energyMeter().makeReading()),
        energyAfter(energyMeter().makeReading()) {}

  Config cfg;
  Structure structure;

  // Three synchronization points:
  //   - Google Benchmark's own entry barrier (implicit, not a member here)
  //   makes
  //     thread 0's construction above visible to everyone.
  //   - 'gate' is the rendezvous between phases: all threads have finished
  //     prefilling, or all have finished their ops.
  //   - 'release' is not a rendezvous - 'gate' just did that. It exists only
  //     because thread 0 reads the package energy counter after that barrier
  //     and before the clock starts, and the others have to wait out that one
  //     read. Hence a count of 1, not numThreads: one releaser, T-1 waiters.
  std::barrier<> gate;
  std::latch release;
  std::vector<ThreadRecord> records;
  std::size_t poolNodesPerThread;
  cpu::EnergyReading energyBefore;
  cpu::EnergyReading energyAfter;
  double windowSeconds = 0.0;
};

// One live run per structure type. Google Benchmark runs instances
// sequentially, so a single slot cannot be contended.
template <typename Structure> inline Run<Structure> *currentRun = nullptr;

template <typename Ops, typename Structure>
void finish(benchmark::State &state, Run<Structure> &run);

[[noreturn]] inline void failStructure(const char *name,
                                       const cpu::Validation &validation,
                                       std::int64_t expected) {
  std::fprintf(
      stderr,
      "bench: %s failed its post-run structural check -- size %lld against an "
      "expected %lld, sorted=%d terminated=%d noMarked=%d. The structure was "
      "corrupted during the run, so its throughput figure is meaningless and "
      "so is every other variant's, which is why this aborts rather than "
      "reports.\n",
      name, static_cast<long long>(validation.count),
      static_cast<long long>(expected), static_cast<int>(validation.sorted),
      static_cast<int>(validation.terminated),
      static_cast<int>(validation.noMarked));
  std::abort();
}

template <typename Ops, typename Structure>
void runExperiment(benchmark::State &state, Config cfg) {
  const std::size_t threadId = static_cast<std::size_t>(state.thread_index());
  cfg.numThreads = static_cast<std::size_t>(state.threads());

  // numThreads above is the only field decided here; opsPerThread arrives fixed
  // from the harness.

  pinThread(threadId);

  if (threadId == 0) {
    const std::size_t nodes = Ops::getNodesPerThread(cfg);
    const std::size_t bytes = nodes * cfg.numThreads * Ops::nodeBytes;
    if (bytes > poolBudgetBytes()) {
      failPoolBudget(bytes, poolBudgetBytes(), nodes, cfg.numThreads,
                     Ops::nodeBytes);
    }

    energyMeter();

    if (pinningEnabled()) {
      verifyMeasurementHost();
    }

    // Forced here because the calibration spins for milliseconds; on first use
    // inside the loop it would land in the measured window.
    latencyClock();

    currentRun<Structure> = new Run<Structure>(cfg, nodes);
  }

  // Google Benchmark's entry barrier publishes thread 0's work above.
  for (auto _ : state) {
    Run<Structure> &run = *currentRun<Structure>;
    ThreadRecord &record = run.records[threadId];
    Workload work = Ops::makeWorkload(threadId, cfg);

    PerfGroup perf;
    perf.open();

    // Untimed, and per-thread from its own pool slice: this is where each
    // thread first-touches its own pages, which fixes their NUMA placement.
    Ops::prefill(run.structure, threadId, cfg);

    // Past this barrier every thread has prefilled. The energy counter is
    // package-wide, so exactly one thread reads it, and the rest wait rather
    // than start the loop before the interval opens.
    run.gate.arrive_and_wait();
    if (threadId == 0) {
      energyMeter().sample(run.energyBefore);
      run.release.count_down();
    } else {
      run.release.wait();
    }

    std::int64_t ineffective = 0;
    std::int64_t delta = 0;

    const bool measureLatency = latencyEnabled();

    const PerfCounters perfBefore = perf.read();
    const ThreadCpu cpuBefore = readThreadCpu();
    record.start = std::chrono::steady_clock::now();
    for (std::int64_t i = 0; i < cfg.opsPerThread; ++i) {
      StepResult result;
      if (measureLatency) {
        const Tick before = tickStart();
        result = Ops::step(run.structure, threadId, work, cfg,
                           static_cast<std::uint64_t>(i));
        record.latency.record(before, tickEnd());
      } else {
        result = Ops::step(run.structure, threadId, work, cfg,
                           static_cast<std::uint64_t>(i));
      }
      delta += result.delta;
      if (!result.effective) {
        ++ineffective;
      }
    }
    benchmark::ClobberMemory();
    record.stop = std::chrono::steady_clock::now();
    record.cpu = readThreadCpu() - cpuBefore;
    record.perf = perf.read() - perfBefore;

    record.ops = cfg.opsPerThread;
    record.ineffective = ineffective;
    record.delta = delta;

    run.gate.arrive_and_wait();
    if (threadId == 0) {
      energyMeter().sample(run.energyAfter);
      finish<Ops>(state, run);
    }
    run.gate.arrive_and_wait();

    // Every thread reports the same figure: Google Benchmark sums manual times
    // across threads and divides by the summed iteration count, so identical
    // per-thread values yield exactly this window.
    state.SetIterationTime(run.windowSeconds);
  }

  if (threadId == 0) {
    delete currentRun<Structure>;
    currentRun<Structure> = nullptr;
  }
}

// Counters. All are emitted by thread 0 alone: Google Benchmark sums counters
// across threads, so a single writer makes the reported value the computed one.
template <typename Ops, typename Structure>
void finish(benchmark::State &state, Run<Structure> &run) {
  const std::size_t threads = run.cfg.numThreads;

  Aggregate totals;
  for (const ThreadRecord &record : run.records) {
    totals.add(record);
  }
  run.windowSeconds = totals.windowSeconds();

  const double total = static_cast<double>(totals.ops);
  const double seconds = run.windowSeconds;

  state.counters["ops_total"] = total;
  state.counters["ops_per_sec"] = seconds > 0.0 ? total / seconds : 0.0;
  state.counters["window_seconds"] = seconds;

  // How unevenly the threads completed their equal budgets. Far from 1.0 means
  // the tail of the window was less contended than its thread count suggests.
  state.counters["thread_time_spread"] =
      totals.fastestThread > 0.0 ? totals.slowestThread / totals.fastestThread
                                 : 0.0;

  // How far apart the threads entered the loop.
  state.counters["start_skew_ms"] = totals.startSkewMillis();

  // Interpretation differs by ADT.
  //   stack / queue: pops that found the structure empty. Any appreciable value
  //     means the prefill was too shallow, and failing is cheap and
  //     variant-dependent.
  //   list: inserts and removes that made no structural change. At steady state
  //     this sits near half the insert+remove share; a departure says the set
  //     drifted off keyRange/2.
  state.counters["ops_ineffective_frac"] =
      total > 0.0 ? static_cast<double>(totals.ineffective) / total : 0.0;

  // Cost, as distinct from speed: ops_per_sec divides work by the time the
  // machine was occupied, ops_per_cpu_sec by the CPU time consumed. A spinlock
  // is expected to win the first while losing the second.
  if (totals.haveCpuTime && totals.cpuSeconds > 0.0) {
    state.counters["ops_per_cpu_sec"] = total / totals.cpuSeconds;

    // Consumed CPU against the CPU the thread count nominally claims. Above the
    // hardware thread count this cannot reach 1.0.
    state.counters["cpu_utilization"] =
        seconds > 0.0
            ? totals.cpuSeconds / (seconds * static_cast<double>(threads))
            : 0.0;

    // Separates unequal scheduler service from unequal work, which
    // thread_time_spread cannot.
    state.counters["thread_cpu_spread"] =
        totals.leastCpu > 0.0 ? totals.mostCpu / totals.leastCpu : 0.0;

    // Mechanism fingerprint: a mutex blocks in the kernel, a spinlock burns
    // user time, a lock-free structure does neither.
    const double split = totals.cpuUserSeconds + totals.cpuSystemSeconds;
    if (split > 0.0) {
      state.counters["cpu_sys_frac"] = totals.cpuSystemSeconds / split;
    }
  }

  // A direct count of how often contention forced a thread off its core.
  if (totals.haveSwitches) {
    state.counters["ctx_voluntary"] = static_cast<double>(totals.voluntary);
    state.counters["ctx_involuntary"] = static_cast<double>(totals.involuntary);
  }

  // Hardware counters. See perf_counters.hpp for the event list
  if (totals.havePerf && (totals.perfValues[kPerfCycles] > 0)) {
    const double cycles = static_cast<double>(totals.perfValues[kPerfCycles]);

    // Cost of one operation in cycles rather than nanoseconds, so it is
    // frequency-independent.
    state.counters["cycles_per_op"] = total > 0.0 ? cycles / total : 0.0;

    state.counters["ipc"] =
        static_cast<double>(totals.perfValues[kPerfInstructions]) / cycles;

    // Fraction of cycles in which no uop executed.
    state.counters["stall_frac"] =
        static_cast<double>(totals.perfValues[kPerfTotalStalls]) / cycles;

    // A narrower cause: the store buffer was full. Every lock release and every
    // successful CAS is a store, so this separates a structure that is stalled
    // on its own writes from one stalled on loads.
    state.counters["sb_stall_frac"] =
        static_cast<double>(totals.perfValues[kPerfStoreBufferStalls]) / cycles;

    if (total > 0.0) {
      state.counters["l1d_miss_per_op"] =
          static_cast<double>(totals.perfValues[kPerfL1dReadMisses]) / total;

      state.counters["branch_misp_per_op"] =
          static_cast<double>(totals.perfValues[kPerfBranchMisses]) / total;
    }

    // 1.0 means the group held the PMU for the whole window and the counts
    // above are exact. Below that the kernel time-sliced the group and every
    // count is an under-count. The event list is sized to the hardware, so the
    // usual cause is the NMI watchdog holding a counter: run
    // `sysctl kernel.nmi_watchdog=0` and measure again rather than scaling
    // these by hand.
    state.counters["perf_running_frac"] =
        totals.perfEnabledNanos > 0
            ? static_cast<double>(totals.perfRunningNanos) /
                  static_cast<double>(totals.perfEnabledNanos)
            : 0.0;
  }

  // Latency distribution, present only when BENCH_LATENCY is set. The only
  // metric here that can test lock-freedom's claim: a progress guarantee is
  // about the worst case, throughput is an average.
  if (totals.latency.count > 0) {
    const LatencyHistogram &latency = totals.latency;
    state.counters["latency_ops"] = static_cast<double>(latency.count);

    for (const double percent : kLatencyPercentiles) {
      const bool whole =
          percent == static_cast<double>(static_cast<int>(percent));
      char name[32];
      std::snprintf(name, sizeof(name), "latency_p%d_ns",
                    static_cast<int>(whole ? percent : percent * 10.0));
      state.counters[name] = latency.percentileNanos(percent);
    }

    state.counters["latency_mean_ns"] =
        latencyClock().nanos(latency.meanCycles());
    state.counters["latency_stdev_ns"] =
        latencyClock().nanos(latency.stdevCycles());

    state.counters["latency_min_ns"] =
        latencyClock().nanos(static_cast<double>(latency.min));
    state.counters["latency_max_ns"] =
        latencyClock().nanos(static_cast<double>(latency.max));

    state.counters["latency_backwards"] = static_cast<double>(latency.negative);

    state.counters["latency_overhead_ns"] =
        latencyClock().nanos(static_cast<double>(latencyClock().overhead()));
    state.counters["latency_cycles_per_ns"] = latencyClock().cyclesPerNano();

    if (histogramDumpEnabled()) {
      dumpHistogram(state.name().c_str(), latency);
    }
  }

  state.counters["pool_nodes_per_thread"] =
      static_cast<double>(run.poolNodesPerThread);

  // The factors echoed as numbers, so plotting need not parse the benchmark
  // name. Thread count is already a first-class JSON field.
  state.counters["ops_per_thread"] = static_cast<double>(run.cfg.opsPerThread);
  state.counters["mix_pct"] = static_cast<double>(run.cfg.mixPct);
  state.counters["key_range"] = static_cast<double>(run.cfg.keyRange);

  state.counters["host_physical_cores"] = static_cast<double>(kPhysicalCores);
  state.counters["host_sockets"] = static_cast<double>(kSockets);

  // Post-run structural check:
  const std::int64_t expectedSize =
      Ops::getPrefillTotal(run.cfg) + totals.delta;
  const cpu::Validation validation = Ops::inspect(run.structure, run.cfg);

  state.counters["check_size"] = static_cast<double>(validation.count);
  state.counters["check_size_expected"] = static_cast<double>(expectedSize);

  if (!validation.terminated || !validation.sorted || !validation.noMarked ||
      validation.count != expectedSize) {
    failStructure(state.name().c_str(), validation, expectedSize);
  }

  cpu::EnergyMeter &meter = energyMeter();
  if (meter.domainCount() == 0) {
    return;
  }

  // A window shorter than the counter refresh interval reads as zero or as one
  // quantum of noise. Reported rather than silently trusted.
  state.counters["energy_window_ok"] =
      seconds * 1e6 >= static_cast<double>(cpu::raplUpdateMicros) ? 1.0 : 0.0;

  // 1.0 is the normal case. 0.0 means a counter went backwards, which at these
  // window lengths is a foreign write to the rw energy_uj file rather than a
  // wrap (see energy_meter.hpp) - so the joules columns for this row are a
  // guess and should be dropped, not averaged in.
  bool decreased = false;
  const std::vector<double> joules =
      meter.joulesBetween(run.energyBefore, run.energyAfter, &decreased);
  state.counters["energy_monotonic"] = decreased ? 0.0 : 1.0;
  for (std::size_t i = 0; i < meter.domainCount(); ++i) {
    // Never summed: core and uncore are subsets of their package and psys
    // overlaps it entirely, so aggregation belongs to the analysis.
    const std::string label = sanitized(meter.domainName(i));
    state.counters["joules_" + label] = joules[i];
    state.counters["nj_per_op_" + label] =
        total > 0.0 ? joules[i] * 1e9 / total : 0.0;
  }
}

} // namespace bench
