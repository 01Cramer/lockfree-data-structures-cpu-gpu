// GPU queue benchmark driver. CSV rows go to stdout; logs go to stderr.
// The whole sweep runs in one process and uses one CUDA context.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/lockfree/gpu_lockfree_queue.cuh"
#include "gpu/shared/gpu_cuda_utils.cuh"
#include "gpu/spinlock/gpu_spinlock_queue.cuh"
#include "gpu/spinlock/gpu_spinlock_queue_two_lock.cuh"
#include "support/gpu_energy.cuh"
#include "support/gpu_workload.cuh"

using gpubench::Config;
using gpubench::EnergyMeter;
using gpubench::Key;
using gpubench::Result;

namespace {

using SpinlockQueue = gpu::spinlock::Queue<Key>;
using SpinlockQueueTwoLock = gpu::spinlock::QueueTwoLock<Key>;
using LockfreeQueue = gpu::lockfree::Queue<Key>;

constexpr const char *kVariants[] = {"spinlock", "spinlock_two_lock",
                                     "lockfree"};
constexpr int kBlocks[] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr int kBlockDims[] = {32, 64, 128, 256, 512};
constexpr int kMixes[] = {50};
constexpr int kOpsPerThread = 5000;
constexpr int kRepetitions = 5;
constexpr int kNodesPerThread = 0; // 0 = derive from the operation mix

struct Metrics {
  double milliseconds = 0.0;
  double enqueueSuccess = 0.0;
  double enqueueAttempts = 0.0;
  double dequeueSuccess = 0.0;
  double dequeueAttempts = 0.0;
  double opsSuccess = 0.0;
  double opsAttempts = 0.0;
  double opsPerSecond = 0.0;
  double dequeueFailFraction = 0.0;
  double poolMb = 0.0;
  double energyJoules = 0.0;
  double energyOk = 0.0;
  double energyWindowOk = 0.0;
  double energyFromCounter = 0.0;
  double powerWatts = 0.0;
  double idleWatts = 0.0;
  double nanoJoulesPerOp = 0.0;
  double marginalNanoJoulesPerOp = 0.0;
};

Result dispatch(const Config &cfg, const EnergyMeter &meter) {
  if (cfg.variant == "spinlock") {
    return gpubench::runRepetition<SpinlockQueue>(cfg, meter);
  }
  if (cfg.variant == "spinlock_two_lock") {
    return gpubench::runRepetition<SpinlockQueueTwoLock>(cfg, meter);
  }
  if (cfg.variant == "lockfree") {
    return gpubench::runRepetition<LockfreeQueue>(cfg, meter);
  }
  std::fprintf(stderr,
               "unknown variant '%s' (expected spinlock, spinlock_two_lock "
               "or lockfree)\n",
               cfg.variant.c_str());
  std::exit(2);
}

void emitHeader() {
  std::printf("variant,blocks,block_dim,threads,participating_threads,"
              "ops_per_thread,mix_pct,"
              "prefill,stat,rep,ms,enq_success,enq_attempts,deq_success,"
              "deq_attempts,ops_success,ops_attempts,ops_per_sec,"
              "deq_fail_frac,pool_mb,"
              "energy_j,energy_ok,energy_window_ok,energy_from_counter,"
              "power_w,idle_power_w,nj_per_op,marginal_nj_per_op\n");
}

Metrics makeMetrics(const Result &result, const EnergyMeter &meter) {
  const long long success = result.enqueueSuccess + result.dequeueSuccess;
  const long long attempts = result.enqueueAttempts + result.dequeueAttempts;
  const double seconds = result.milliseconds / 1000.0;

  Metrics m;
  m.milliseconds = result.milliseconds;
  m.enqueueSuccess = static_cast<double>(result.enqueueSuccess);
  m.enqueueAttempts = static_cast<double>(result.enqueueAttempts);
  m.dequeueSuccess = static_cast<double>(result.dequeueSuccess);
  m.dequeueAttempts = static_cast<double>(result.dequeueAttempts);
  m.opsSuccess = static_cast<double>(success);
  m.opsAttempts = static_cast<double>(attempts);
  m.opsPerSecond =
      seconds > 0.0 ? static_cast<double>(success) / seconds : 0.0;
  m.dequeueFailFraction =
      result.dequeueAttempts > 0
          ? 1.0 - static_cast<double>(result.dequeueSuccess) /
                      static_cast<double>(result.dequeueAttempts)
          : 0.0;
  m.poolMb = static_cast<double>(result.poolBytes) / (1024.0 * 1024.0);

  m.energyJoules = result.energy.joules;
  m.energyOk = result.energy.valid ? 1.0 : 0.0;
  m.energyWindowOk = result.energyWindowOk ? 1.0 : 0.0;
  m.energyFromCounter = result.energy.fromCounter ? 1.0 : 0.0;
  m.powerWatts = result.energy.watts;
  m.idleWatts = meter.idleWatts();
  m.nanoJoulesPerOp =
      (result.energy.valid && success > 0)
          ? result.energy.joules * 1e9 / static_cast<double>(success)
          : 0.0;
  const double marginalJoules =
      result.energy.joules - meter.idleWatts() * seconds;
  m.marginalNanoJoulesPerOp =
      (result.energy.valid && success > 0 && marginalJoules > 0.0)
          ? marginalJoules * 1e9 / static_cast<double>(success)
          : 0.0;
  return m;
}

void emitRow(const Config &cfg, const char *stat, int rep, const Metrics &m) {
  std::printf("%s,%d,%d,%d,%lld,%d,%d,%lld,%s,%d,"
              "%.6f,%.0f,%.0f,%.0f,%.0f,"
              "%.0f,%.0f,%.3f,%.6f,%.2f,"
              "%.6f,%.0f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f\n",
              cfg.variant.c_str(), cfg.blocks, cfg.blockDim,
              static_cast<int>(gpubench::totalThreads(cfg)),
              gpubench::totalThreads(cfg), cfg.opsPerThread, cfg.mixPct,
              gpubench::prefillFor(cfg), stat, rep,
              m.milliseconds, m.enqueueSuccess, m.enqueueAttempts,
              m.dequeueSuccess, m.dequeueAttempts, m.opsSuccess, m.opsAttempts,
              m.opsPerSecond, m.dequeueFailFraction, m.poolMb, m.energyJoules,
              m.energyOk, m.energyWindowOk, m.energyFromCounter, m.powerWatts,
              m.idleWatts, m.nanoJoulesPerOp, m.marginalNanoJoulesPerOp);
  std::fflush(stdout);
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() % 2) == 1) {
    return values[middle];
  }
  return (values[middle - 1] + values[middle]) / 2.0;
}

double meanOf(const std::vector<Metrics> &rows, double Metrics::*field) {
  double sum = 0.0;
  for (const Metrics &row : rows) {
    sum += row.*field;
  }
  return sum / static_cast<double>(rows.size());
}

double medianOf(const std::vector<Metrics> &rows, double Metrics::*field) {
  std::vector<double> values;
  values.reserve(rows.size());
  for (const Metrics &row : rows) {
    values.push_back(row.*field);
  }
  return median(values);
}

void setAggregate(double &target, const std::vector<Metrics> &rows,
                  double Metrics::*field, bool useMedian) {
  target = useMedian ? medianOf(rows, field) : meanOf(rows, field);
}

Metrics aggregateMetrics(const std::vector<Metrics> &rows, bool useMedian) {
  Metrics out;
  setAggregate(out.milliseconds, rows, &Metrics::milliseconds, useMedian);
  setAggregate(out.enqueueSuccess, rows, &Metrics::enqueueSuccess, useMedian);
  setAggregate(out.enqueueAttempts, rows, &Metrics::enqueueAttempts, useMedian);
  setAggregate(out.dequeueSuccess, rows, &Metrics::dequeueSuccess, useMedian);
  setAggregate(out.dequeueAttempts, rows, &Metrics::dequeueAttempts, useMedian);
  setAggregate(out.opsSuccess, rows, &Metrics::opsSuccess, useMedian);
  setAggregate(out.opsAttempts, rows, &Metrics::opsAttempts, useMedian);
  setAggregate(out.opsPerSecond, rows, &Metrics::opsPerSecond, useMedian);
  setAggregate(out.dequeueFailFraction, rows, &Metrics::dequeueFailFraction,
               useMedian);
  setAggregate(out.poolMb, rows, &Metrics::poolMb, useMedian);
  setAggregate(out.energyJoules, rows, &Metrics::energyJoules, useMedian);
  setAggregate(out.energyOk, rows, &Metrics::energyOk, useMedian);
  setAggregate(out.energyWindowOk, rows, &Metrics::energyWindowOk, useMedian);
  setAggregate(out.energyFromCounter, rows, &Metrics::energyFromCounter,
               useMedian);
  setAggregate(out.powerWatts, rows, &Metrics::powerWatts, useMedian);
  setAggregate(out.idleWatts, rows, &Metrics::idleWatts, useMedian);
  setAggregate(out.nanoJoulesPerOp, rows, &Metrics::nanoJoulesPerOp,
               useMedian);
  setAggregate(out.marginalNanoJoulesPerOp, rows,
               &Metrics::marginalNanoJoulesPerOp, useMedian);
  return out;
}

} // namespace

int main() {
  gpu::requireVoltaOrNewer(stderr, true);

  // One meter and one idle baseline for the whole sweep.
  const EnergyMeter meter;
  std::fprintf(stderr, "energy: %s\n", meter.status());

  emitHeader();

  for (const char *variant : kVariants) {
    for (int mix : kMixes) {
      for (int blocks : kBlocks) {
        for (int blockDim : kBlockDims) {
          Config cfg;
          cfg.variant = variant;
          cfg.blocks = blocks;
          cfg.blockDim = blockDim;
          cfg.opsPerThread = kOpsPerThread;
          cfg.mixPct = mix;
          cfg.nodesPerThread = kNodesPerThread;

          std::vector<Metrics> rows;
          rows.reserve(kRepetitions);
          for (int rep = 0; rep < kRepetitions; ++rep) {
            Metrics metrics = makeMetrics(dispatch(cfg, meter), meter);
            rows.push_back(metrics);
            emitRow(cfg, "rep", rep, metrics);
          }
          emitRow(cfg, "mean", -1, aggregateMetrics(rows, false));
          emitRow(cfg, "median", -1, aggregateMetrics(rows, true));

          std::fprintf(stderr,
                       "done %s blocks=%d block_dim=%d mix=%d ops=%d\n",
                       variant, blocks, blockDim, mix, cfg.opsPerThread);
        }
      }
    }
  }

  return 0;
}
