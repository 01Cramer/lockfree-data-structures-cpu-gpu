// Standalone check that RAPL energy measurement works on this machine.
// Build with -DENABLE_RAPL=ON on the Linux server and run as root:
//   sudo ./energy_probe
// Without the knob it still builds, reports that energy is compiled out, and
// exits successfully.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "support/energy_meter.hpp"

namespace {

constexpr double kWindowSeconds = 2.0;

// Occupies every hardware thread for the window.
double measureBusyLoad(double seconds) {
  const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> threads;
  std::vector<double> spin(workers, 0.0);
  threads.reserve(workers);

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);

  for (unsigned i = 0; i < workers; ++i) {
    threads.emplace_back([&, i] {
      double accumulator = 1.0;
      while (std::chrono::steady_clock::now() < deadline) {
        for (int k = 0; k < 4096; ++k) {
          accumulator = accumulator * 1.000001 + 1.0;
        }
      }
      spin[i] = accumulator;
    });
  }
  for (std::thread &thread : threads) {
    thread.join();
  }

  double total = 0.0;
  for (double value : spin) {
    total += value;
  }
  // Discarded by callers; pthread_create and the deadline protect the loop.
  return total;
}

struct Window {
  std::vector<double> joules;
  double seconds = 0.0;
};

template <typename Load>
Window measure(const cpu::EnergyMeter &meter, Load load) {
  cpu::EnergyReading before = meter.makeReading();
  cpu::EnergyReading after = meter.makeReading();

  const auto start = std::chrono::steady_clock::now();
  meter.sample(before);
  load();
  meter.sample(after);
  const auto stop = std::chrono::steady_clock::now();

  Window window;
  window.joules = meter.joulesBetween(before, after);
  window.seconds = std::chrono::duration<double>(stop - start).count();
  return window;
}

} // namespace

int main() {
  cpu::EnergyMeter meter;

  if (!meter.available()) {
    std::printf("Energy measurement is compiled out "
                "(configure with -DENABLE_RAPL=ON on Linux).\n"
                "The disabled path builds and links correctly.\n");
    return 0;
  }

  std::printf("RAPL domains found: %zu\n\n", meter.domainCount());
  std::printf("%-22s %18s %14s\n", "domain", "wrap point (J)", "wraps after");
  for (std::size_t i = 0; i < meter.domainCount(); ++i) {
    const double rangeJoules =
        static_cast<double>(meter.domainRangeMicroJoules(i)) * 1e-6;
    // Longest safe window at a sustained 200 W; beyond it the counter can wrap
    // twice, which is uncorrectable.
    std::printf("%-22s %18.1f %11.1f min\n", meter.domainName(i).c_str(),
                rangeJoules, rangeJoules / 200.0 / 60.0);
  }

  std::printf("\nMeasuring a %.0f s idle window...\n", kWindowSeconds);
  const Window idle = measure(meter, [] {
    std::this_thread::sleep_for(std::chrono::duration<double>(kWindowSeconds));
  });

  std::printf("Measuring a %.0f s busy window on %u hardware threads...\n\n",
              kWindowSeconds,
              std::max(1u, std::thread::hardware_concurrency()));
  const Window busy = measure(meter, [] { measureBusyLoad(kWindowSeconds); });

  std::printf("%-22s %10s %10s %10s %10s\n", "domain", "idle J", "busy J",
              "idle W", "busy W");
  bool responded = false;
  for (std::size_t i = 0; i < meter.domainCount(); ++i) {
    const double idleWatts = idle.joules[i] / idle.seconds;
    const double busyWatts = busy.joules[i] / busy.seconds;
    std::printf("%-22s %10.2f %10.2f %10.2f %10.2f\n",
                meter.domainName(i).c_str(), idle.joules[i], busy.joules[i],
                idleWatts, busyWatts);
    if (busyWatts > idleWatts * 1.2) {
      responded = true;
    }
  }

  if (!responded) {
    std::printf("\nFAIL: no domain drew measurably more power under load, so "
                "the counters are readable but appear not to track.\n"
                "Rule out one thing before believing that: the idle window is "
                "only idle if the HOST is. RAPL is per-socket, so on a shared "
                "machine a co-tenant's load lands in the idle figure -- and if "
                "it has already pushed the package near its limit, this test's "
                "own load cannot raise it another 20%%. Check the idle watts "
                "above against the package TDP, and re-run when the box is "
                "quiet.\n");
    return 1;
  }

  std::printf("\nOK: counters track load. Remember that these are per-socket "
              "totals for the whole machine, so quiesce the host before a "
              "measurement run.\n");
  return 0;
}
