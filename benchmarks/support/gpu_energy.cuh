// Device energy measurement, the GPU counterpart of benchmarks/support/energy_meter.hpp.
//
// The CPU half reads an accumulating microjoule counter out of the powercap
// sysfs (Intel RAPL). NVML exposes the same shape of instrument for the GPU:
// nvmlDeviceGetTotalEnergyConsumption returns total millijoules consumed by the
// board since the driver was last reloaded, and the energy of a run is the
// difference across it. Same instrument shape, so the same three questions
// apply and are answered the same way.
//
//   1. What does it cover? The whole board -- SMs, memory, fans, everything on
//      the card -- not the kernel. There is no per-kernel or per-SM breakdown,
//      so a figure derived from it is attributable to the benchmark only to the
//      extent that nothing else was using the device. That is the same caveat
//      the CPU package counter carries.
//
//   2. Does it advance fast enough to see the window? No, not always, and this
//      is the trap. NVML's underlying sampling is on the order of tens of
//      milliseconds; a kernel that runs for 5 ms may fall entirely between two
//      updates and report zero energy, or land across one and report a whole
//      sample's worth. Neither is a measurement. `windowOk()` below is the
//      guard, and the fix is to raise --ops until the kernel window clears the
//      threshold. It is recorded per row rather than enforced, so a short run
//      is identifiable in the output instead of silently wrong.
//
//   3. How much of it is the algorithm? Not all of it. A powered-on idle GPU
//      draws a substantial fraction of its loaded draw, exactly as the CPU
//      package did (63 W of 94 W there). Total energy divided by operations
//      would therefore rank the variants in nearly the order throughput already
//      did and carry almost no independent information -- which is precisely
//      what the CPU results showed, at a correlation of 0.992. So the idle draw
//      is calibrated once at startup and reported, and the analysis subtracts
//      it to get the marginal energy the work is responsible for.
//
// Availability. nvmlDeviceGetTotalEnergyConsumption needs Volta or newer and a
// reasonably recent driver. Where it is missing the meter degrades to sampling
// instantaneous power, which is worse -- it assumes the draw was constant over
// the window -- but is better than dropping the column, and which one produced
// a row is recorded in the output rather than inferred.
//
// Relation to the published guidance
//
// The three practices the ml.energy group recommends for GPU energy measurement
// ("Measuring GPU Energy: Best Practices", ml.energy blog) are each satisfied
// here, and it is worth naming them because the thesis should cite the
// methodology rather than appear to have invented it:
//
//   1. Measure, do not estimate from TDP. Nothing here is derived from a spec
//      sheet; every joule comes from the device.
//
//   2. Prefer the energy API over polling the power API. On Volta and later
//      nvmlDeviceGetTotalEnergyConsumption gives an accumulated counter, so a
//      region costs two reads and a subtraction with no polling thread -- and
//      no polling thread means no core spinning at full utilisation, which
//      would itself consume energy and perturb the host side of a measurement.
//      The power path is the documented fallback for older hardware, and that
//      is exactly what it is used for here.
//
//   3. Synchronise the CPU and the GPU before the closing read. A CUDA launch
//      is asynchronous, so a reading taken when the host reaches the next
//      statement undercounts whatever is still in flight. gpu_workload.cuh
//      takes the closing read after cudaDeviceSynchronize, and takes the
//      power-fallback sample before it, while the kernel is still running.
//
// Their reference implementation, ZeusMonitor, is a Python library. It is not
// used here for one structural reason rather than any disagreement: this sweep
// is a single C++ process holding one CUDA context for every configuration, so
// that no point in a curve pays an initialisation cost its neighbours did not.
// Driving the kernels from Python to reach the library would give that up, and
// wrapping the whole binary in it would measure the entire sweep as one window
// instead of one window per configuration -- which is the granularity the
// results need.

#pragma once

#include <cstdio>
#include <cstring>

// nanosleep is POSIX, not standard C, so <time.h> rather than <ctime>: the
// latter is only required to declare the standard subset.
#include <time.h>

#include <cuda_runtime.h>

#if defined(GPU_ENABLE_NVML)
#include <nvml.h>
#endif

namespace gpubench {

// Below this the NVML counter has probably not updated inside the window, and
// the reading is quantisation rather than measurement. Chosen an order of
// magnitude above NVML's typical few-millisecond update so that a row which
// passes is not marginal.
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

  bool available() const { return m_available; }
  bool hasCounter() const { return m_hasCounter; }
  double idleWatts() const { return m_idleWatts; }
  const char *status() const { return m_status; }

  // Raw counter read, in joules. Meaningless on its own; take the difference
  // across the region of interest.
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

  // Energy over a window, given the two counter reads that bracket it and how
  // long it lasted. Falls back to power x time where no counter exists.
  EnergySample between(double joulesBefore, double joulesAfter,
                       double wattsDuring, double seconds) const {
    EnergySample sample;
    if (!m_available || seconds <= 0.0) {
      return sample;
    }
    if (m_hasCounter) {
      const double delta = joulesAfter - joulesBefore;
      // A decrease means the driver was reloaded mid-sweep, or the counter
      // wrapped. Either way the difference is not the run's energy, so the row
      // is marked invalid rather than corrected -- the same decision the CPU
      // meter makes about a RAPL counter that goes backwards.
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

    // Bound by PCI bus id rather than by index. NVML's enumeration order is not
    // guaranteed to match CUDA's, so nvmlDeviceGetHandleByIndex(0) can bind to
    // a different board than the one the kernels run on -- which would produce
    // a plausible energy column measured from the wrong GPU.
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

  // Mean draw with nothing running, so the analysis can charge the algorithm
  // only for what it added. Sampled rather than read once: the figure drifts
  // with board temperature and a single sample lands wherever it lands.
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
      // Sleeping rather than spinning: the point is to leave the device alone
      // while sampling, and a busy host thread can keep it out of its lowest
      // power state.
      struct timespec interval = {0, 50 * 1000 * 1000}; // 50 ms
      nanosleep(&interval, nullptr);
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

// Compiled out. Same shape, so nothing at the call sites is conditional; the
// harness reports that energy was unavailable and every energy column is empty.
class EnergyMeter {
public:
  // User-provided rather than `= default`: the harness declares the meter as a
  // const object, and const default-initialization of a class with no members
  // and no user-provided constructor is ill-formed.
  EnergyMeter() {}

  bool available() const { return false; }
  bool hasCounter() const { return false; }
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

// Whether a window was long enough for the reading across it to be a
// measurement rather than quantisation.
inline bool energyWindowOk(double seconds) {
  return seconds >= kMinEnergyWindowSeconds;
}

} // namespace gpubench
