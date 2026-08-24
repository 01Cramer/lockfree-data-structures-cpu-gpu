// What the benchmark harness actually sees when it asks how many CPUs there are.
//
// hardwareThreads() in benchmarks/support/experiment.hpp is load-bearing twice
// over: it bounds the top of the thread ladder (the largest power of two inside
// 2 x H, so 128 on the measurement host) and it is the modulus of the pinning
// map. Two properties of it need to be facts about the measurement host rather
// than assumptions:
//
//   1. Does it count SMT siblings? (Expected: yes -- 72, not 36.)
//   2. Does it respect a restricted CPU set (taskset, cgroup)? If not,
//      BENCH_PIN=1 cannot be combined with taskset, because the map will target
//      CPUs outside the allowed set.
//
// Build and run:
//   g++ -std=c++20 -O0 -o cpu_topology_probe scripts/cpu_topology_probe.cpp
//   ./cpu_topology_probe
//   taskset -c 0-3 ./cpu_topology_probe      # the differential test
//
// Reading the result:
//   hardware_concurrency == affinity mask count under taskset  -> it respects it
//   hardware_concurrency == online count under taskset         -> it ignores it
//
// Cross-check the topology itself with:
//   lscpu | egrep 'CPU\(s\)|Thread|Core|Socket|NUMA'
//   lscpu -e            # which CPU numbers are siblings of which core
//
// The second command is the one that decides whether the identity map in
// pinThread means "distinct cores" or "SMT pairs" on this host.

#include <cstdio>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

int main() {
  std::printf("std::thread::hardware_concurrency() = %u\n",
              std::thread::hardware_concurrency());

#if defined(__linux__)
  std::printf("sysconf(_SC_NPROCESSORS_ONLN)        = %ld  (online)\n",
              ::sysconf(_SC_NPROCESSORS_ONLN));
  std::printf("sysconf(_SC_NPROCESSORS_CONF)        = %ld  (configured)\n",
              ::sysconf(_SC_NPROCESSORS_CONF));

  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (::sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    std::printf("sched_getaffinity failed\n");
    return 1;
  }

  std::printf("CPU_COUNT(sched_getaffinity)         = %d  (allowed to this "
              "process)\nallowed CPUs:",
              CPU_COUNT(&allowed));
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      std::printf(" %d", cpu);
    }
  }
  std::printf("\n");
#else
  std::printf("(Linux-only figures skipped on this platform)\n");
#endif

  return 0;
}
