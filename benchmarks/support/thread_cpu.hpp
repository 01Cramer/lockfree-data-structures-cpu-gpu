// Per-thread CPU consumption, read once before and once after the timed region
// (both reads outside it, so the syscall never enters a throughput figure).

#pragma once

#include <cstdint>

#if defined(__linux__)
#include <sys/resource.h>
#include <time.h>
#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD 1
#endif
#endif

namespace bench {

// A reading, not a duration. Two of these are subtracted to get the interval.
struct ThreadCpu {
  // CPU time this thread spent executing on a core, user + kernel
  // Excludes blocked and descheduled time
  double totalSeconds = 0.0;
  // time in user mode
  double userSeconds = 0.0;
  // time in kernel mode
  double systemSeconds = 0.0;
  // times the thread was descheduled because it blocked
  std::int64_t voluntary = 0;
  // times it was descheduled while still runnable — preemption
  std::int64_t involuntary = 0;

  // Whether the host supplied each group. Zero must not read as "not measured".
  bool haveTime = false;
  bool haveSwitches = false;
};

inline ThreadCpu readThreadCpu() {
  ThreadCpu reading;

#if defined(__linux__)
  timespec cpuTime{};
  if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpuTime) == 0) {
    reading.totalSeconds = static_cast<double>(cpuTime.tv_sec) +
                           static_cast<double>(cpuTime.tv_nsec) * 1e-9;
    reading.haveTime = true;
  }

  // For the split and the switch counts only; its own total is discarded as
  // tick-accumulated, unlike the clock above.
  rusage usage{};
  if (::getrusage(RUSAGE_THREAD, &usage) == 0) {
    reading.userSeconds = static_cast<double>(usage.ru_utime.tv_sec) +
                          static_cast<double>(usage.ru_utime.tv_usec) * 1e-6;
    reading.systemSeconds = static_cast<double>(usage.ru_stime.tv_sec) +
                            static_cast<double>(usage.ru_stime.tv_usec) * 1e-6;
    reading.voluntary = static_cast<std::int64_t>(usage.ru_nvcsw);
    reading.involuntary = static_cast<std::int64_t>(usage.ru_nivcsw);
    reading.haveSwitches = true;
  }
#endif

  return reading;
}

// after - before, field by field; availability is the AND of both endpoints.
inline ThreadCpu operator-(const ThreadCpu &after, const ThreadCpu &before) {
  ThreadCpu difference;
  difference.totalSeconds = after.totalSeconds - before.totalSeconds;
  difference.userSeconds = after.userSeconds - before.userSeconds;
  difference.systemSeconds = after.systemSeconds - before.systemSeconds;
  difference.voluntary = after.voluntary - before.voluntary;
  difference.involuntary = after.involuntary - before.involuntary;
  difference.haveTime = after.haveTime && before.haveTime;
  difference.haveSwitches = after.haveSwitches && before.haveSwitches;
  return difference;
}

} // namespace bench
