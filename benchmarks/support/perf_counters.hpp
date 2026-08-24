// Hardware performance counters over the timed region, read the same way as
// thread_cpu.hpp: once before, once after, subtract.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <asm/unistd.h>
#include <atomic>
#include <cerrno>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace bench {

enum PerfSlot {
  kPerfCycles = 0,
  kPerfInstructions = 1,
  kPerfL1dReadMisses = 2,
  kPerfBranchMisses = 3,
  kPerfStoreBufferStalls = 4,
  kPerfTotalStalls = 5,
  kPerfSlotCount = 6,
};

// A reading, not a duration. Two of these are subtracted to get the interval.
struct PerfCounters {
  std::uint64_t values[kPerfSlotCount] = {};

  // Nanoseconds the group was scheduled, and of those, how many it actually
  // spent on the PMU. Equal means no multiplexing and the counts are exact.
  std::uint64_t enabledNanos = 0;
  std::uint64_t runningNanos = 0;

  // Whether the host supplied the group at all. Zero must not read as
  // "not measured".
  bool have = false;
};

inline PerfCounters operator-(const PerfCounters &after,
                              const PerfCounters &before) {
  PerfCounters difference;
  for (int slot = 0; slot < kPerfSlotCount; ++slot) {
    difference.values[slot] = after.values[slot] - before.values[slot];
  }
  difference.enabledNanos = after.enabledNanos - before.enabledNanos;
  difference.runningNanos = after.runningNanos - before.runningNanos;
  difference.have = after.have && before.have;
  return difference;
}

inline bool perfEnabled() {
  const char *raw = std::getenv("BENCH_PERF");
  return (raw != nullptr) && (std::strcmp(raw, "1") == 0);
}

#if defined(__linux__)

// PERF_TYPE_RAW encodings for the two events with no portable name. Bits 7:0
// select the event, 15:8 the umask, 31:24 the counter mask. Skylake and
// Skylake-SP specific -- the same bargain topology.hpp makes, and wrong numbers
// here would be counted silently, so both are spelled out against the SDM:
//
//   RESOURCE_STALLS.SB           event 0xa2, umask 0x08
//     Cycles stalled because the store buffer was full. Releasing a lock is a
//     store and so is every successful CAS, so this is where a release-heavy
//     inner loop shows up.
//   CYCLE_ACTIVITY.STALLS_TOTAL  event 0xa3, umask 0x04, cmask 0x04
//     Cycles in which no uop executed, whatever the cause. General backend
//     stalls. The cmask is not optional: CYCLE_ACTIVITY counts cycles meeting a
//     threshold, and without it the event does not mean what its name says.
inline constexpr std::uint64_t kRawResourceStallsSb = 0x000008a2ULL;
inline constexpr std::uint64_t kRawCycleActivityStallsTotal = 0x040004a3ULL;

struct PerfEventSpec {
  std::uint32_t type;
  std::uint64_t config;
  const char *name;
};

// Order matches PerfSlot, and the group read returns values in exactly this
// order, so the two must not drift apart.
inline const PerfEventSpec *perfEventSpecs() {
  static const PerfEventSpec specs[kPerfSlotCount] = {
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
      // The kernel maps this triple to the right raw event per microarchitecture
      // (MEM_LOAD_RETIRED.L1_MISS on Skylake), which is what `perf` itself does
      // for the L1-dcache-load-misses alias.
      {PERF_TYPE_HW_CACHE,
       PERF_COUNT_HW_CACHE_L1D |
           (static_cast<std::uint64_t>(PERF_COUNT_HW_CACHE_OP_READ) << 8) |
           (static_cast<std::uint64_t>(PERF_COUNT_HW_CACHE_RESULT_MISS) << 16),
       "L1-dcache-load-misses"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, "branch-misses"},
      {PERF_TYPE_RAW, kRawResourceStallsSb, "RESOURCE_STALLS.SB"},
      {PERF_TYPE_RAW, kRawCycleActivityStallsTotal,
       "CYCLE_ACTIVITY.STALLS_TOTAL"},
  };
  return specs;
}

inline long perfEventOpen(perf_event_attr *attr, pid_t pid, int cpu,
                          int groupFd, unsigned long flags) {
  return ::syscall(__NR_perf_event_open, attr, pid, cpu, groupFd, flags);
}

// Once per process, however many threads hit it. All three likely causes are
// named because none of them is guessable from errno alone, and a sweep that
// silently dropped these columns would only be noticed after it finished.
inline void warnPerfUnavailable(const char *eventName, int error) {
  static std::atomic<bool> warned{false};
  if (warned.exchange(true)) {
    return;
  }
  std::fprintf(
      stderr,
      "bench: BENCH_PERF=1 but perf_event_open failed for %s (errno %d: %s). "
      "The hardware-counter columns will be absent; throughput and energy are "
      "unaffected. Usual causes, in order of likelihood:\n"
      "  EACCES  -- sysctl kernel.perf_event_paranoid=2 (these events are user "
      "mode only, so 2 is enough; some distributions ship 4)\n"
      "  EMFILE  -- six descriptors per thread, so 768 at the top of the "
      "ladder: ulimit -n 4096\n"
      "  EINVAL  -- a raw event encoding this CPU does not implement; the two "
      "in perf_counters.hpp are Skylake-specific\n",
      eventName, error, std::strerror(error));
}

// One counter group on the calling thread. Not copyable: it owns descriptors.
class PerfGroup {
public:
  PerfGroup() = default;
  ~PerfGroup() { closeAll(); }
  PerfGroup(const PerfGroup &) = delete;
  PerfGroup &operator=(const PerfGroup &) = delete;

  // False leaves every read() empty, which the have flag then propagates into
  // the aggregate. Deliberately not fatal: a missing hardware counter costs an
  // analysis column, unlike a failed pinning or a corrupted structure, which
  // cost the run its meaning.
  bool open() {
    if (!perfEnabled()) {
      return false;
    }

    const PerfEventSpec *specs = perfEventSpecs();
    for (int slot = 0; slot < kPerfSlotCount; ++slot) {
      perf_event_attr attr{};
      attr.size = sizeof(attr);
      attr.type = specs[slot].type;
      attr.config = specs[slot].config;

      // On the leader only. A group read goes through the leader's descriptor
      // and its read_format governs the whole reply, so setting this on a
      // follower would describe a read that never happens.
      if (slot == kPerfCycles) {
        attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                           PERF_FORMAT_TOTAL_TIME_RUNNING;
      }

      // User mode only, for two reasons. It scopes these counters to the
      // algorithm rather than to the kernel it calls -- a mutex's futex path
      // and a spinlock's nanosleep would otherwise inject backend stalls that
      // say nothing about the data structure, and kernel cost is already
      // measured by cpu_sys_frac. It also keeps the group legal at
      // perf_event_paranoid=2, which is the common default.
      //
      // Consequence to remember when reading the CSV: cycles here is user
      // cycles, so it does not reconcile with cpu.totalSeconds times the clock
      // rate on any variant that blocks.
      attr.exclude_kernel = 1;
      attr.exclude_hv = 1;

      // Each worker opens its own group, and none of them forks.
      attr.inherit = 0;

      // Leader disabled, followers not: a follower does not start until the
      // leader is enabled, so the whole group starts on one ioctl.
      attr.disabled = (slot == kPerfCycles) ? 1 : 0;

      const int groupFd = (slot == kPerfCycles) ? -1 : m_fds[kPerfCycles];

      // pid 0 is the calling thread; cpu -1 follows it wherever it is
      // scheduled, which is required since BENCH_PIN is not always on.
      const long fd = perfEventOpen(&attr, 0, -1, groupFd, PERF_FLAG_FD_CLOEXEC);
      if (fd < 0) {
        warnPerfUnavailable(specs[slot].name, errno);
        closeAll();
        return false;
      }
      m_fds[slot] = static_cast<int>(fd);
    }

    if ((::ioctl(m_fds[kPerfCycles], PERF_EVENT_IOC_RESET,
                 PERF_IOC_FLAG_GROUP) != 0) ||
        (::ioctl(m_fds[kPerfCycles], PERF_EVENT_IOC_ENABLE,
                 PERF_IOC_FLAG_GROUP) != 0)) {
      warnPerfUnavailable(specs[kPerfCycles].name, errno);
      closeAll();
      return false;
    }

    m_open = true;
    return true;
  }

  PerfCounters read() const {
    PerfCounters reading;
    if (!m_open) {
      return reading;
    }

    // PERF_FORMAT_GROUP lays the read out as: event count, time_enabled,
    // time_running, then one value per event in the order they were opened.
    // PERF_FORMAT_ID is not set, so each entry is a single u64.
    std::uint64_t buffer[3 + kPerfSlotCount] = {};
    const ssize_t got = ::read(m_fds[kPerfCycles], buffer, sizeof(buffer));
    if (got != static_cast<ssize_t>(sizeof(buffer))) {
      return reading;
    }
    if (buffer[0] != static_cast<std::uint64_t>(kPerfSlotCount)) {
      return reading;
    }

    reading.enabledNanos = buffer[1];
    reading.runningNanos = buffer[2];
    for (int slot = 0; slot < kPerfSlotCount; ++slot) {
      reading.values[slot] = buffer[3 + slot];
    }
    reading.have = true;
    return reading;
  }

private:
  // Followers first, leader last: closing the leader dissolves the group.
  void closeAll() {
    for (int slot = kPerfSlotCount - 1; slot >= 0; --slot) {
      if (m_fds[slot] >= 0) {
        ::close(m_fds[slot]);
        m_fds[slot] = -1;
      }
    }
    m_open = false;
  }

  int m_fds[kPerfSlotCount] = {-1, -1, -1, -1, -1, -1};
  bool m_open = false;
};

#else

// Non-Linux: every reading stays unavailable, so the columns are absent rather
// than zero. Same shape as thread_cpu.hpp's empty reading.
class PerfGroup {
public:
  bool open() { return false; }
  PerfCounters read() const { return PerfCounters{}; }
};

#endif

} // namespace bench
