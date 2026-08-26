// Small CUDA test harness layer over tests/support/test_harness.hpp.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <cuda_runtime.h>

#include "gpu/shared/gpu_cuda_utils.cuh"
#include "support/test_harness.hpp"

namespace gpu_test {

// The registry, assertions and exit code all come from the host harness.
namespace shared_test = cpu_test;
using cpu_test::config;

// GPU_TEST(name) { ... } - same contract as CPU_TEST
#define GPU_TEST(name) CPU_TEST(name)

// Run-wide GPU test configuration.
struct Config {
  int blocks;
  int blockDim;
  int opsPerThread;
  int mixPct;
};

namespace detail {

inline int readEnvInt(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return static_cast<int>(std::strtol(raw, nullptr, 10));
}

inline void normalizeConfig(Config &c) {
  if (c.blocks < 1) {
    c.blocks = 1;
  }
  if (c.opsPerThread < 1) {
    c.opsPerThread = 1;
  }
  if (c.blockDim < 1) {
    c.blockDim = 1;
  }
  if (c.blockDim > 1024) {
    c.blockDim = 1024;
  }
  if (c.mixPct < 1) {
    c.mixPct = 1;
  }
  if (c.mixPct > 99) {
    c.mixPct = 99;
  }
}

inline Config makeDefaultConfig() {
  Config c;
  c.blocks = readEnvInt("GPU_TEST_BLOCKS", 8);
  c.blockDim = readEnvInt("GPU_TEST_BLOCK_DIM", 128);
  c.opsPerThread = readEnvInt("GPU_TEST_OPS", 5000);
  c.mixPct = readEnvInt("GPU_TEST_MIX", 50);
  normalizeConfig(c);
  return c;
}

inline Config &mutableGpuConfig() {
  static Config c = makeDefaultConfig();
  return c;
}

} // namespace detail

inline const Config &gpuConfig() { return detail::mutableGpuConfig(); }

// Report a likely deadlock and leave without running destructors.
[[noreturn]] inline void reportDeadlock(const char *what, double seconds) {
  std::fflush(stdout);
  std::fprintf(stderr,
               "\nTIMEOUT after %.1f s waiting for %s.\n"
               "This is the signature of a warp-level deadlock: a lane holds "
               "the spinlock and cannot reach unlock() because its peers in "
               "the same warp are still spinning. On compute 7.0+ that should "
               "be impossible; if it happens, do not build the lock-based "
               "queues on this lock, and re-check that the binary was really "
               "compiled for sm_70 or newer (a lower -arch silently "
               "reintroduces the shared program counter).\n",
               seconds, what);
  std::fflush(stderr);
  std::_Exit(1);
}

inline void waitWithWatchdog(cudaStream_t stream, double seconds,
                             const char *what) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(seconds);
  while (true) {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaSuccess) {
      return;
    }
    if (status != cudaErrorNotReady) {
      gpu::detail::cudaFail(status, what, __FILE__, __LINE__);
    }
    if (std::chrono::steady_clock::now() > deadline) {
      reportDeadlock(what, seconds);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

// Parse GPU overrides, check the device, then run the shared test registry.
inline int runAll(int argc, char **argv) {
  Config &c = detail::mutableGpuConfig();
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
      c.blocks = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--block-dim") == 0 && i + 1 < argc) {
      c.blockDim = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--ops") == 0 && i + 1 < argc) {
      c.opsPerThread = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--mix") == 0 && i + 1 < argc) {
      c.mixPct = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    }
  }

  detail::normalizeConfig(c);

  gpu::requireVoltaOrNewer(stdout, false);
  std::printf("gpu config: blocks=%d block_dim=%d ops=%d mix=%d\n", c.blocks,
              c.blockDim, c.opsPerThread, c.mixPct);

  return shared_test::runAll(argc, argv);
}

} // namespace gpu_test
