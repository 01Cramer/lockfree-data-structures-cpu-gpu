// Minimal in-repo test harness. Provides:
//   - CHECK / CHECK_EQ assertion macros (thread-safe; set a global failure
//   flag)
//   - a self-registering test registry + runAll() that returns non-zero on any
//     failure, so CTest reads correctness straight off the exit code
//   - StartGate, a spin barrier so worker threads begin together and interleave
//   - Config parsed once from env/CLI (iterations, threads, RNG seed). The seed
//     is printed up front and on failure so any run is reproducible.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string_view>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// TSan detection (clang: __has_feature; gcc: __SANITIZE_THREAD__).
#if defined(__SANITIZE_THREAD__)
#define CPU_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define CPU_TEST_TSAN 1
#endif
#endif
#ifndef CPU_TEST_TSAN
#define CPU_TEST_TSAN 0
#endif

namespace cpu_test {

namespace detail {

inline std::atomic<long> &failureCount() {
  static std::atomic<long> count{0};
  return count;
}

// Serialize diagnostic output so messages from concurrent workers stay legible.
inline std::mutex &outMutex() {
  static std::mutex m;
  return m;
}

inline void fail(const char *file, int line, const std::string &msg) {
  failureCount().fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(outMutex());
  std::cerr << "  FAIL " << file << ":" << line << "  " << msg << "\n";
}

template <typename A, typename B>
inline void failEq(const char *file, int line, const char *exprA,
                   const char *exprB, const A &va, const B &vb) {
  failureCount().fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(outMutex());
  std::cerr << "  FAIL " << file << ":" << line << "  " << exprA
            << " == " << exprB << "  (" << va << " vs " << vb << ")\n";
}

} // namespace detail

// CHECK(cond): record a failure (and keep going) if cond is false.
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ::cpu_test::detail::fail(__FILE__, __LINE__, #cond);                     \
    }                                                                          \
  } while (0)

// CHECK_EQ(a, b): like CHECK(a == b) but prints both values. Requires the
// operands to be streamable to std::ostream (use CHECK for anything else).
#define CHECK_EQ(a, b)                                                         \
  do {                                                                         \
    auto cpuTestValA = (a);                                                    \
    auto cpuTestValB = (b);                                                    \
    if (!(cpuTestValA == cpuTestValB)) {                                       \
      ::cpu_test::detail::failEq(__FILE__, __LINE__, #a, #b, cpuTestValA,      \
                                 cpuTestValB);                                 \
    }                                                                          \
  } while (0)

// Run-wide configuration, parsed once from the environment and argv.
struct Config {
  unsigned long iterations;
  unsigned threads; // total worker threads a scenario should spawn
  unsigned seed;
};

namespace detail {

inline unsigned long readEnvUL(const char *name, unsigned long fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return std::strtoul(raw, nullptr, 10);
}

inline Config makeDefaultConfig() {
  const unsigned hw = std::thread::hardware_concurrency()
                          ? std::thread::hardware_concurrency()
                          : 4u;
  Config c;
#if CPU_TEST_TSAN
  c.iterations = detail::readEnvUL("CPU_TEST_ITERS", 5);
  c.threads = static_cast<unsigned>(detail::readEnvUL("CPU_TEST_THREADS", hw));
#else
  c.iterations = detail::readEnvUL("CPU_TEST_ITERS", 200);
  // Oversubscribe (2x cores) to widen the interleaving the scheduler produces.
  c.threads =
      static_cast<unsigned>(detail::readEnvUL("CPU_TEST_THREADS", 2u * hw));
#endif
  if (c.threads < 2) {
    c.threads = 2;
  }
  c.seed =
      static_cast<unsigned>(detail::readEnvUL("CPU_TEST_SEED", 0x9e3779b9u));
  return c;
}

inline Config &mutableConfig() {
  static Config c = makeDefaultConfig();
  return c;
}

} // namespace detail

inline const Config &config() { return detail::mutableConfig(); }

// A one-shot spin gate: workers call wait() right after construction, main
// calls open() once every worker exists, so they start as simultaneously as
// the scheduler allows (maximizes contention vs. a staggered thread launch).
class StartGate {
public:
  void wait() const {
    while (!m_go.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
  void open() { m_go.store(true, std::memory_order_release); }

private:
  std::atomic<bool> m_go{false};
};

// Self-registering test case.
using TestFn = void (*)();

struct TestCase {
  const char *name;
  TestFn fn;
};

namespace detail {
inline std::vector<TestCase> &registry() {
  static std::vector<TestCase> tests;
  return tests;
}
struct Registrar {
  Registrar(const char *name, TestFn fn) { registry().push_back({name, fn}); }
};
} // namespace detail

// CPU_TEST(name) { ... } defines and registers a test function.
#define CPU_TEST(name)                                                         \
  static void name();                                                          \
  static ::cpu_test::detail::Registrar cpuTestReg_##name(#name, &name);        \
  static void name()

// Parse --iters/--threads/--seed overrides, run every registered test, print a
// summary, and return non-zero if any CHECK failed.
inline int runAll(int argc, char **argv) {
  Config &c = detail::mutableConfig();
  const char *filter = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      c.iterations = std::strtoul(argv[++i], nullptr, 10);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      c.threads = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      c.seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
      filter = argv[++i];
    }
  }

  std::cout << "config: iterations=" << c.iterations << " threads=" << c.threads
            << " seed=" << c.seed << (CPU_TEST_TSAN ? " [tsan]" : "") << "\n";

  for (const TestCase &test : detail::registry()) {
    if (filter != nullptr &&
        std::string_view(test.name).find(filter) == std::string_view::npos) {
      continue;
    }
    const long before = detail::failureCount().load(std::memory_order_relaxed);
    std::cout << "[ RUN  ] " << test.name << "\n";
    test.fn();
    const long after = detail::failureCount().load(std::memory_order_relaxed);
    if (after == before) {
      std::cout << "[ PASS ] " << test.name << "\n";
    } else {
      std::cout << "[ FAIL ] " << test.name << " (" << (after - before)
                << " check(s) failed)\n";
    }
  }

  const long total = detail::failureCount().load(std::memory_order_relaxed);
  if (total == 0) {
    std::cout << "ALL TESTS PASSED\n";
    return 0;
  }
  std::cerr << total << " CHECK(s) FAILED. Reproduce with --seed " << c.seed
            << " --threads " << c.threads << " --iters " << c.iterations
            << "\n";
  return 1;
}

} // namespace cpu_test
