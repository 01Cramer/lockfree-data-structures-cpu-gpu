// NVML energy measurement for the GPU benchmark.
//
// Preferred path: read the accumulated board-energy counter before and after
// the timed kernel. Fallback path: sample instantaneous power and multiply by
// the kernel time. The CSV records which path was used.
//
// Energy is board-level, not kernel-level. The benchmark therefore also records
// an idle-power baseline so analysis can use marginal energy per operation.
// Very short kernels are flagged because NVML counters update too coarsely for
// sub-50 ms windows to be reliable.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include <cuda_runtime.h>

#if defined(GPU_ENABLE_NVML)
#include <nvml.h>
#endif

namespace gpubench {

// Minimum kernel window for trusting the energy counter.
inline constexpr double kMinEnergyWindowSeconds = 0.050;

// One reading pair, plus what is needed to judge whether it means anything.
struct EnergySample {
  bool valid = false;        // the meter produced a figure at all
  bool fromCounter = false;  // true = accumulated counter, false = power x time
  double joules = 0.0;
  double watts = 0.0;        // average over the window
};

#if defined(GPU_ENABLE_NVML)

namespace detail {

inline bool nvmlOk(nvmlReturn_t status, const char *what) {
  if (status == NVML_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "nvml: %s failed: %s\n", what, nvmlErrorString(status));
  return false;
}

} // namespace detail

// Opens NVML, binds to the GPU that CUDA is using, and calibrates the idle
// draw. One instance for the whole sweep: NVML initialisation is not free and
// the idle baseline is a property of the device, not of a configuration.
class EnergyMeter {
public:
  EnergyMeter() { open(); }

  ~EnergyMeter() {
    if (m_initialized) {
      nvmlShutdown();
    }
  }

  EnergyMeter(const EnergyMeter &) = delete;
  EnergyMeter &operator=(const EnergyMeter &) = delete;

  double idleWatts() const { return m_idleWatts; }
  const char *status() const { return m_status; }

  // Raw accumulated counter read, in joules.
  double joulesNow() const {
    if (!m_hasCounter) {
      return 0.0;
    }
    unsigned long long milliJoules = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(m_device, &milliJoules) !=
        NVML_SUCCESS) {
      return 0.0;
    }
    return static_cast<double>(milliJoules) / 1000.0;
  }

  double wattsNow() const {
    if (!m_available) {
      return 0.0;
    }
    unsigned int milliWatts = 0;
    if (nvmlDeviceGetPowerUsage(m_device, &milliWatts) != NVML_SUCCESS) {
      return 0.0;
    }
    return static_cast<double>(milliWatts) / 1000.0;
  }

  // Energy over one timed window.
  EnergySample between(double joulesBefore, double joulesAfter,
                       double wattsDuring, double seconds) const {
    EnergySample sample;
    if (!m_available || seconds <= 0.0) {
      return sample;
    }
    if (m_hasCounter) {
      const double delta = joulesAfter - joulesBefore;
      // Counter wrap/reload: mark the row invalid instead of guessing.
      if (delta < 0.0) {
        return sample;
      }
      sample.valid = true;
      sample.fromCounter = true;
      sample.joules = delta;
      sample.watts = delta / seconds;
      return sample;
    }
    sample.valid = true;
    sample.fromCounter = false;
    sample.watts = wattsDuring;
    sample.joules = wattsDuring * seconds;
    return sample;
  }

private:
  void open() {
    if (!detail::nvmlOk(nvmlInit_v2(), "nvmlInit")) {
      std::snprintf(m_status, sizeof(m_status), "nvmlInit failed");
      return;
    }
    m_initialized = true;

    // Bind NVML to the same physical GPU CUDA is using.
    int cudaDevice = 0;
    if (cudaGetDevice(&cudaDevice) != cudaSuccess) {
      std::snprintf(m_status, sizeof(m_status), "cudaGetDevice failed");
      return;
    }
    char busId[64] = {};
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), cudaDevice) !=
        cudaSuccess) {
      std::snprintf(m_status, sizeof(m_status), "cudaDeviceGetPCIBusId failed");
      return;
    }
    if (!detail::nvmlOk(nvmlDeviceGetHandleByPciBusId_v2(busId, &m_device),
                        "nvmlDeviceGetHandleByPciBusId")) {
      std::snprintf(m_status, sizeof(m_status), "no NVML handle for %s", busId);
      return;
    }

    unsigned int milliWatts = 0;
    if (nvmlDeviceGetPowerUsage(m_device, &milliWatts) != NVML_SUCCESS) {
      std::snprintf(m_status, sizeof(m_status), "power reading unsupported");
      return;
    }
    m_available = true;

    unsigned long long milliJoules = 0;
    m_hasCounter = nvmlDeviceGetTotalEnergyConsumption(
                       m_device, &milliJoules) == NVML_SUCCESS;

    calibrateIdle();
    std::snprintf(m_status, sizeof(m_status), "%s, idle %.1f W",
                  m_hasCounter ? "energy counter" : "power sampling",
                  m_idleWatts);
  }

  // Average idle draw, sampled once at startup.
  void calibrateIdle() {
    cudaDeviceSynchronize();
    double total = 0.0;
    int taken = 0;
    for (int i = 0; i < kIdleSamples; ++i) {
      const double watts = wattsNow();
      if (watts > 0.0) {
        total += watts;
        ++taken;
      }
      // Sleep so the sampling loop itself does not keep the host busy.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    m_idleWatts = taken > 0 ? total / static_cast<double>(taken) : 0.0;
  }

  static constexpr int kIdleSamples = 20; // ~1 s

  nvmlDevice_t m_device{};
  bool m_initialized = false;
  bool m_available = false;
  bool m_hasCounter = false;
  double m_idleWatts = 0.0;
  char m_status[128] = "not opened";
};

#else // !GPU_ENABLE_NVML

// Same interface when NVML support is compiled out.
class EnergyMeter {
public:
  // Required for const default-initialization of an empty class.
  EnergyMeter() {}

  double idleWatts() const { return 0.0; }
  const char *status() const {
    return "built without NVML (configure with -DENABLE_NVML=ON)";
  }
  double joulesNow() const { return 0.0; }
  double wattsNow() const { return 0.0; }
  EnergySample between(double, double, double, double) const {
    return EnergySample{};
  }
};

#endif

// Whether the energy window was long enough to trust.
inline bool energyWindowOk(double seconds) {
  return seconds >= kMinEnergyWindowSeconds;
}

} // namespace gpubench
