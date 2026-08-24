// Device-side test harness. Deliberately thin: everything that is not
// CUDA-specific is reused from tests/support/test_harness.hpp, which is
// ordinary host C++ and compiles unchanged under nvcc. CHECK, CHECK_EQ, the
// self-registering test list and the exit-code contract are therefore
// identical on both halves of the project, and so is the reproduce line
// printed on failure.
//
// What this header adds is the three things a device test needs and a host
// test does not:
//
//   - a compute-capability gate, so a run on pre-Volta hardware reports why it
//     cannot proceed instead of hanging in a spinlock;
//   - a GPU-shaped configuration (warps, operations per thread, active lanes)
//     in place of the host one (threads);
//   - a watchdog, because the failure mode this project's day-one test is
//     looking for is a deadlock, and a deadlocked kernel produces no output at
//     all. Without a watchdog "the lock does not make progress with 32 lanes"
//     and "the test runner is still going" look the same from outside.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <cuda_runtime.h>

#include "gpu/shared/cuda_error.cuh"
#include "support/test_harness.hpp"

namespace gpu_test {

// The registry, assertions and exit code all come from the host harness.
using cpu_test::config;

// GPU_TEST(name) { ... } -- same contract as CPU_TEST, renamed so a device
// test file does not read as a host one.
#define GPU_TEST(name) CPU_TEST(name)

// Run-wide GPU configuration. Separate from cpu_test::Config because the
// quantity that matters is warps, not threads: blockDim is fixed at 32 for
// everything (see the design), so one block is one warp is one agent.
struct Config {
  int warps;         // = blocks, at blockDim 32
  int opsPerThread;  // per participating thread
  int activeLanes;   // lanes per warp that participate, in [1, 32]
  int prefill;       // items in the queue before the concurrent phase
};

namespace detail {

inline int readEnvInt(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return static_cast<int>(std::strtol(raw, nullptr, 10));
}

inline Config makeDefaultConfig() {
  Config c;
  // Even, so the warp-uniform producer/consumer split is balanced; small, so
  // the whole suite runs in seconds under compute-sanitizer, which is where
  // these tests earn their keep.
  c.warps = readEnvInt("GPU_TEST_WARPS", 16);
  c.opsPerThread = readEnvInt("GPU_TEST_OPS", 64);
  c.activeLanes = readEnvInt("GPU_TEST_LANES", 32);
  c.prefill = readEnvInt("GPU_TEST_PREFILL", 4096);
  if (c.warps < 2) {
    c.warps = 2;
  }
  if (c.warps % 2 != 0) {
    ++c.warps;
  }
  if (c.activeLanes < 1) {
    c.activeLanes = 1;
  }
  if (c.activeLanes > 32) {
    c.activeLanes = 32;
  }
  return c;
}

inline Config &mutableGpuConfig() {
  static Config c = makeDefaultConfig();
  return c;
}

} // namespace detail

inline const Config &gpuConfig() { return detail::mutableGpuConfig(); }

inline constexpr int kWarpSize = 32;

// Independent thread scheduling is a correctness precondition for the
// lock-based variants, not a performance preference. Checked once, loudly.
inline void requireVoltaOrNewer() {
  int deviceCount = 0;
  GPU_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::fprintf(stderr, "No CUDA device found.\n");
    std::exit(1);
  }

  int device = 0;
  GPU_CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  GPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  std::printf("device: %s (compute %d.%d, %d SMs)\n", properties.name,
              properties.major, properties.minor,
              properties.multiProcessorCount);

  if (properties.major < 7) {
    std::fprintf(stderr,
                 "This device is compute %d.%d. The lock-based variants need "
                 "independent thread scheduling (compute 7.0, Volta): below "
                 "that, the threads of a warp share a program counter and a "
                 "lane holding a spinlock cannot reach the release while its "
                 "peers spin. The kernels would deadlock, not run slowly.\n",
                 properties.major, properties.minor);
    std::exit(1);
  }
}

// Wait for a stream with a deadline. On timeout there is no way to cancel the
// kernel, so the process reports and leaves immediately -- _Exit rather than
// exit, because running static destructors while a kernel is live is its own
// source of confusing errors.
//
// `seconds` should be generously above the expected runtime. The distinction
// being drawn is deadlock versus progress, not fast versus slow.
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

// Parse the GPU overrides, check the device, then hand off to the host
// harness's runner for the registry walk and the exit code.
inline int runAll(int argc, char **argv) {
  Config &c = detail::mutableGpuConfig();
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--warps") == 0 && i + 1 < argc) {
      c.warps = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--ops") == 0 && i + 1 < argc) {
      c.opsPerThread = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--lanes") == 0 && i + 1 < argc) {
      c.activeLanes = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--prefill") == 0 && i + 1 < argc) {
      c.prefill = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
    }
  }

  requireVoltaOrNewer();
  std::printf("gpu config: warps=%d ops=%d lanes=%d prefill=%d\n", c.warps,
              c.opsPerThread, c.activeLanes, c.prefill);

  return cpu_test::runAll(argc, argv);
}

} // namespace gpu_test
