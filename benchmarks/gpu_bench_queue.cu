// The GPU measurement harness. One CSV row per repetition on stdout, everything
// else on stderr, so `gpu_bench_queue ... > run.csv` is directly readable.
//
// The whole sweep runs in one process so that it shares one CUDA context: no
// point in the curve pays a context-creation cost its neighbours did not, and the
// device clock stays settled across neighbouring configurations.
//
// Example, the primary result:
//
//   gpu_bench_queue --variant spinlock,spinlock_two_lock,lockfree \
//                   --warps 1,2,4,8,16,32,64,128,256 \
//                   --lanes 1,32 --work 0,32,256 \
//                   --ops 1000 --prefill 1048576 --reps 5 > run.csv
//
// The second-order experiment. Block size is only interpretable at CONSTANT total
// warps, since block count already controls total warps and blockDim only changes
// how warps are packed onto SMs:
//
//   for d in 32 64 128; do
//     gpu_bench_queue --variant lockfree --warps 2048 --block-dim $d ... ;
//   done

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/lockfree/lockfree_queue.cuh"
#include "gpu/shared/cuda_error.cuh"
#include "gpu/spinlock/spinlock_queue.cuh"
#include "gpu/spinlock/spinlock_queue_two_lock.cuh"
#include "support/gpu_energy.cuh"
#include "support/gpu_workload.cuh"

using gpubench::Config;
using gpubench::EnergyMeter;
using gpubench::Key;
using gpubench::kWarpSize;
using gpubench::Result;

namespace {

using SpinlockQueue = gpu::spinlock::Queue<Key>;
using SpinlockQueueTwoLock = gpu::spinlock::QueueTwoLock<Key>;
using LockfreeQueue = gpu::lockfree::Queue<Key>;

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

std::vector<std::string> splitList(const char *raw) {
  std::vector<std::string> parts;
  std::string current;
  for (const char *p = raw; *p != '\0'; ++p) {
    if (*p == ',') {
      if (!current.empty()) {
        parts.push_back(current);
      }
      current.clear();
    } else {
      current.push_back(*p);
    }
  }
  if (!current.empty()) {
    parts.push_back(current);
  }
  return parts;
}

std::vector<int> splitInts(const char *raw) {
  std::vector<int> values;
  for (const std::string &part : splitList(raw)) {
    values.push_back(static_cast<int>(std::strtol(part.c_str(), nullptr, 10)));
  }
  return values;
}

struct Options {
  std::vector<std::string> variants{"spinlock", "spinlock_two_lock",
                                    "lockfree"};
  // Starts at 2: the mix is carried on warp parity, so a single warp would be a
  // producer-only run measuring a different workload.
  std::vector<int> warps{2, 4, 8, 16, 32, 64, 128, 256};
  std::vector<int> lanes{1, 32};
  std::vector<int> work{0};
  int opsPerThread = 1000; // Zhang et al.'s figure
  int prefill = 1 << 20;
  int reps = 5;
  int warmup = 1;
  int blockDim = 32;
  int nodesPerThread = 0;
  // Ceiling on counted operations per configuration, since the lock-based
  // variants at 32 lanes and a high warp count can run overnight. Reported in
  // the ops_capped column rather than applied silently: a capped point has a
  // shorter timed window than its neighbours.
  long long maxTotalOps = 0; // 0 = uncapped
};

void usage() {
  std::fprintf(
      stderr,
      "usage: gpu_bench_queue [options]   (CSV on stdout, log on stderr)\n"
      "  --variant LIST      spinlock,spinlock_two_lock,lockfree\n"
      "  --warps LIST        total warps in the grid\n"
      "  --lanes LIST        active lanes per warp, 1..32\n"
      "  --work LIST         units of private work between operations\n"
      "  --ops N             operations per participating thread\n"
      "  --prefill N         items in the queue before the timed region\n"
      "  --reps N            recorded repetitions per configuration\n"
      "  --warmup N          discarded repetitions per configuration\n"
      "  --block-dim N       threads per block; only interpretable at\n"
      "                      constant total warps\n"
      "  --nodes-per-thread N  override the pool slice size\n"
      "  --max-total-ops N   cap counted ops per configuration (0 = off)\n"
      "env GPU_BENCH_MAX_POOL_MB overrides the pool memory budget.\n");
}

void failUsage(const char *message) {
  std::fprintf(stderr, "%s\n", message);
  usage();
  std::exit(2);
}

Options parse(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const char *flag = argv[i];
    const bool hasValue = i + 1 < argc;
    if (std::strcmp(flag, "--variant") == 0 && hasValue) {
      options.variants = splitList(argv[++i]);
    } else if (std::strcmp(flag, "--warps") == 0 && hasValue) {
      options.warps = splitInts(argv[++i]);
    } else if (std::strcmp(flag, "--lanes") == 0 && hasValue) {
      options.lanes = splitInts(argv[++i]);
    } else if (std::strcmp(flag, "--work") == 0 && hasValue) {
      options.work = splitInts(argv[++i]);
    } else if (std::strcmp(flag, "--ops") == 0 && hasValue) {
      options.opsPerThread =
          static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(flag, "--prefill") == 0 && hasValue) {
      options.prefill = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(flag, "--reps") == 0 && hasValue) {
      options.reps = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(flag, "--warmup") == 0 && hasValue) {
      options.warmup = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(flag, "--block-dim") == 0 && hasValue) {
      options.blockDim = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(flag, "--nodes-per-thread") == 0 && hasValue) {
      options.nodesPerThread =
          static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(flag, "--max-total-ops") == 0 && hasValue) {
      options.maxTotalOps = std::strtoll(argv[++i], nullptr, 10);
    } else {
      usage();
      std::exit(2);
    }
  }
  return options;
}

// Independent thread scheduling is a correctness precondition for the
// lock-based variants; below compute 7.0 they deadlock rather than run slowly.
void requireVoltaOrNewer() {
  int device = 0;
  GPU_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  GPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int clockRateKHz = 0;
  GPU_CUDA_CHECK(
      cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device));

  std::fprintf(stderr,
               "device: %s, compute %d.%d, %d SMs, %.1f GB, "
               "L2 %d KB, clock %.0f MHz\n",
               properties.name, properties.major, properties.minor,
               properties.multiProcessorCount,
               static_cast<double>(properties.totalGlobalMem) / (1 << 30),
               properties.l2CacheSize / 1024,
               static_cast<double>(clockRateKHz) / 1000.0);

  if (properties.major < 7) {
    std::fprintf(stderr,
                 "compute %d.%d is below the 7.0 floor this project requires "
                 "(independent thread scheduling). The lock-based variants "
                 "would deadlock at activeLanes > 1.\n",
                 properties.major, properties.minor);
    std::exit(1);
  }
}

void emitHeader() {
  std::printf("variant,warps,block_dim,threads,active_lanes,"
              "participating_threads,ops_per_thread,ops_capped,inter_op_work,"
              "prefill,rep,ms,enq_success,enq_attempts,deq_success,"
              "deq_attempts,ops_success,ops_attempts,ops_per_sec,"
              "deq_fail_frac,pool_mb,"
              "energy_j,energy_ok,energy_window_ok,energy_from_counter,"
              "power_w,idle_power_w,nj_per_op,marginal_nj_per_op\n");
}

void emitRow(const Config &cfg, int rep, const Result &result,
             const EnergyMeter &meter) {
  int producers = 0;
  int consumers = 0;
  gpubench::participants(cfg, producers, consumers);

  const long long success = result.enqueueSuccess + result.dequeueSuccess;
  const long long attempts = result.enqueueAttempts + result.dequeueAttempts;
  const double seconds = result.milliseconds / 1000.0;

  // Successful operations only: a failed dequeue is much cheaper than a real one
  // and its cost differs by variant, so counting it would inflate whichever
  // variant detects emptiness fastest.
  const double opsPerSecond = seconds > 0.0
                                  ? static_cast<double>(success) / seconds
                                  : 0.0;
  // If this is not near zero, the queue ran dry and the run partly measured
  // failure detection.
  const double dequeueFailFraction =
      result.dequeueAttempts > 0
          ? 1.0 - static_cast<double>(result.dequeueSuccess) /
                      static_cast<double>(result.dequeueAttempts)
          : 0.0;

  // Energy per successful operation, total and with the board's idle draw
  // removed. The marginal figure is the one to reason from: an idle GPU draws a
  // large fraction of its loaded power, so total energy divided by operations
  // is close to a restatement of throughput -- which is exactly what the CPU
  // half of this study measured, at a correlation of 0.992.
  const double nanoJoulesPerOp =
      (result.energy.valid && success > 0)
          ? result.energy.joules * 1e9 / static_cast<double>(success)
          : 0.0;
  const double marginalJoules =
      result.energy.joules - meter.idleWatts() * seconds;
  const double marginalNanoJoulesPerOp =
      (result.energy.valid && success > 0 && marginalJoules > 0.0)
          ? marginalJoules * 1e9 / static_cast<double>(success)
          : 0.0;

  std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%lld,%lld,%lld,%lld,"
              "%lld,%lld,%.3f,%.6f,%.2f,"
              "%.6f,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
              cfg.variant.c_str(), cfg.warps, cfg.blockDim,
              gpubench::totalThreads(cfg), cfg.activeLanes,
              producers + consumers, cfg.opsPerThread, cfg.capped ? 1 : 0,
              cfg.interOpWork, cfg.prefill, rep, result.milliseconds,
              result.enqueueSuccess, result.enqueueAttempts,
              result.dequeueSuccess, result.dequeueAttempts, success, attempts,
              opsPerSecond, dequeueFailFraction,
              static_cast<double>(result.poolBytes) / (1024.0 * 1024.0),
              result.energy.joules, result.energy.valid ? 1 : 0,
              result.energyWindowOk ? 1 : 0,
              result.energy.fromCounter ? 1 : 0, result.energy.watts,
              meter.idleWatts(), nanoJoulesPerOp, marginalNanoJoulesPerOp);
  std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
  const Options options = parse(argc, argv);
  requireVoltaOrNewer();

  // One meter for the whole sweep: NVML initialisation is not free, and the
  // idle baseline is a property of the device rather than of a configuration.
  const EnergyMeter meter;
  std::fprintf(stderr, "energy: %s\n", meter.status());

  if (options.variants.empty() || options.warps.empty() ||
      options.lanes.empty() || options.work.empty()) {
    failUsage("list-valued options must not be empty");
  }
  if (options.opsPerThread <= 0) {
    failUsage("--ops must be positive");
  }
  if (options.prefill < 0) {
    failUsage("--prefill must be non-negative");
  }
  if (options.reps < 0 || options.warmup < 0) {
    failUsage("--reps and --warmup must be non-negative");
  }
  if (options.nodesPerThread < 0) {
    failUsage("--nodes-per-thread must be non-negative");
  }
  if (options.maxTotalOps < 0) {
    failUsage("--max-total-ops must be non-negative");
  }
  if (options.blockDim % kWarpSize != 0 || options.blockDim <= 0) {
    failUsage("--block-dim must be a positive multiple of 32");
  }
  for (int lanes : options.lanes) {
    if (lanes < 1 || lanes > kWarpSize) {
      failUsage("--lanes values must be in the range 1..32");
    }
  }
  for (int warps : options.warps) {
    if (warps <= 0) {
      failUsage("--warps values must be positive");
    }
  }
  emitHeader();

  for (const std::string &variant : options.variants) {
    for (int work : options.work) {
      for (int lanes : options.lanes) {
        for (int warps : options.warps) {
          Config cfg;
          cfg.variant = variant;
          cfg.warps = warps;
          cfg.blockDim = options.blockDim;
          cfg.activeLanes = lanes;
          cfg.opsPerThread = options.opsPerThread;
          cfg.interOpWork = work;
          cfg.prefill = options.prefill;
          cfg.nodesPerThread = options.nodesPerThread;

          if (warps * kWarpSize % options.blockDim != 0) {
            std::fprintf(stderr,
                         "skipping warps=%d: %d threads is not a whole number "
                         "of blocks at blockDim %d\n",
                         warps, warps * kWarpSize, options.blockDim);
            continue;
          }
          if (warps < 2) {
            std::fprintf(stderr,
                         "skipping warps=1: the 50/50 mix needs at least one "
                         "producer warp and one consumer warp\n");
            continue;
          }

          int producers = 0;
          int consumers = 0;
          gpubench::participants(cfg, producers, consumers);
          const long long participating = producers + consumers;
          if (options.maxTotalOps > 0 &&
              participating * cfg.opsPerThread > options.maxTotalOps) {
            const long long capped = options.maxTotalOps / participating;
            cfg.opsPerThread = static_cast<int>(capped > 0 ? capped : 1);
            cfg.capped = true;
          }

          for (int rep = 0; rep < options.warmup; ++rep) {
            (void)dispatch(cfg, meter);
          }
          for (int rep = 0; rep < options.reps; ++rep) {
            emitRow(cfg, rep, dispatch(cfg, meter), meter);
          }

          std::fprintf(stderr, "done %s warps=%d lanes=%d work=%d ops=%d%s\n",
                       variant.c_str(), warps, lanes, work, cfg.opsPerThread,
                       cfg.capped ? " (capped)" : "");
        }
      }
    }
  }

  return 0;
}
