#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace bench {

// The measurement host:
//
// Two Intel Xeon Gold 6140, 18C/36T each. From lscpu -e, the kernel numbers the
// logical CPUs cores-first:
//
//   CPU  0-17  socket 0, cores 0-17
//   CPU 18-35  socket 1, cores 18-35
//   CPU 36-53  socket 0, siblings of  0-17
//   CPU 54-71  socket 1, siblings of 18-35
inline constexpr int kSockets = 2;
inline constexpr int kPhysicalCores = 36;
inline constexpr int kLogicalCpus = 72;
inline constexpr int kCoresPerSocket = kPhysicalCores / kSockets;

// One file distinguishes the enumeration we rely on from the one that would
// break.
inline bool siblingsComeAfterCores() {
  std::FILE *file = std::fopen(
      "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
  if (file == nullptr) {
    return false;
  }
  char line[64] = {0};
  const char *read = std::fgets(line, sizeof(line), file);
  std::fclose(file);
  if (read == nullptr) {
    return false;
  }
  return std::strncmp(line, "0,36", 4) == 0;
}

// Called once by thread 0, and only under BENCH_PIN=1: an unpinned run makes no
// claim about which CPU a worker sits on, so it has nothing to check.
inline void verifyMeasurementHost() {
  const unsigned cpus = std::thread::hardware_concurrency();
  const bool siblingsLast = siblingsComeAfterCores();
  if ((static_cast<int>(cpus) == kLogicalCpus) && siblingsLast) {
    return;
  }
  std::fprintf(stderr,
               "bench: BENCH_PIN=1 but this is not the measurement host "
               "(expected %d logical CPUs with siblings numbered after cores; "
               "found %u, cpu0 siblings %s). pinThread() maps worker i to CPU "
               "i %% %d, which only describes a placement on that layout. Run "
               "unpinned here, or update topology.hpp for the new machine and "
               "re-check it against lscpu -e.\n",
               kLogicalCpus, cpus,
               siblingsLast ? "as expected" : "not as expected", kLogicalCpus);
  std::abort();
}

} // namespace bench
