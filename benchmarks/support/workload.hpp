// What work each benchmark performs, kept separate from how it is measured
// (experiment.hpp) and which structures it runs on (variants.hpp).
//
// Two rules the mixes follow:
//   - No pure mixes. A 100/0 push/pop workload
//     never exercises the other operation's contention.
//   - Nothing starts empty. A structure that runs dry stops measuring the
//     algorithm and starts measuring failed operations, which are cheaper and
//     variant-dependent.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"

#include "cpu/shared/validation.hpp"

namespace bench {

using Key = std::uint64_t;

struct StepResult {
  int delta = 0;

  // whether the operation found work at all.
  bool effective = true;
};

// Decoded from benchmark::State.
struct Config {
  std::int64_t opsPerThread = 0;
  std::size_t numThreads = 0;

  std::int64_t mixPct = 0;

  // List only: keys drawn uniformly from [0, keyRange).
  std::int64_t keyRange = 0;
};

class Rng {
public:
  explicit Rng(std::int64_t seed) {
    std::seed_seq sequence{seed};
    m_engine.seed(sequence);
  }

  // Uniform in [0, bound)
  std::int64_t boundedRoll(std::int64_t bound) {
    return std::uniform_int_distribution<std::int64_t>(0, bound - 1)(m_engine);
  }

  template <typename Iterator> void shuffle(Iterator first, Iterator last) {
    std::shuffle(first, last, m_engine);
  }

private:
  std::mt19937_64 m_engine;
};

// One thread's workload stream: which operation each step performs
// and which key it touches.
class Workload {
public:
  enum class Op { Add, Remove, Find };

  // The seed is the thread id so the workload they replay is identical across
  // runs. The three counts must sum to the thread's operation budget.
  Workload(std::size_t threadId, std::int64_t finds, std::int64_t adds,
           std::int64_t removes)
      : m_rng(static_cast<std::int64_t>(threadId) + 1), m_finds(finds),
        m_adds(adds), m_removes(removes) {}

  Op next() {
    const std::int64_t opsLeft = m_finds + m_adds + m_removes;
    const std::int64_t roll = m_rng.boundedRoll(opsLeft);

    if (roll < m_finds) {
      m_finds--;
      return Op::Find;
    } else if (roll < m_finds + m_adds) {
      m_adds--;
      return Op::Add;
    } else {
      m_removes--;
      return Op::Remove;
    }
  }

  Key key(std::int64_t range) {
    return static_cast<Key>(m_rng.boundedRoll(range));
  }

private:
  Rng m_rng;
  std::int64_t m_finds;
  std::int64_t m_adds;
  std::int64_t m_removes;
};

// --------------------------------------------------------------------------
// Stack and queue - the "bags"
// --------------------------------------------------------------------------
//
// "Bag" means stack or queue throughout this file: a container with one add and
// one remove operation and no notion of a key. The two share the same workload
// shape, so they share the prefill and node-budget policy below and differ only
// in which methods they call. The list is a set and has its own policy further
// down.

namespace detail {

inline constexpr std::int64_t bagAddOpsCount(const Config &cfg) {
  return cfg.opsPerThread * cfg.mixPct / 100;
}

inline constexpr std::int64_t bagRemoveOpsCount(const Config &cfg) {
  return cfg.opsPerThread - bagAddOpsCount(cfg);
}

// The structure's starting depth, per thread: one element for every remove the
// thread will attempt.
inline constexpr std::int64_t bagPrefillPerThread(const Config &cfg) {
  return bagRemoveOpsCount(cfg);
}

// Pool capacity: one node per prefill element, one per push.
inline constexpr std::int64_t bagNodesPerThread(const Config &cfg) {
  return bagPrefillPerThread(cfg) + bagAddOpsCount(cfg);
}

// Counts what a bag holds by emptying it.
template <typename Structure>
cpu::Validation drainBag(Structure &s, const Config &cfg) {
  const std::int64_t threads = static_cast<std::int64_t>(cfg.numThreads);
  const std::int64_t budget = bagNodesPerThread(cfg) * threads + 1;
  cpu::Validation validation;
  while (validation.count < budget) {
    if (!s.pop().has_value()) {
      return validation;
    }
    ++validation.count;
  }
  validation.terminated = false;
  return validation;
}

} // namespace detail

template <typename Structure> struct StackOps {
  static constexpr const char *adt = "stack";
  static constexpr std::size_t nodeBytes = Structure::nodeBytes;

  static constexpr std::size_t getNodesPerThread(const Config &cfg) {
    return static_cast<std::size_t>(detail::bagNodesPerThread(cfg));
  }

  static constexpr std::int64_t getPrefillTotal(const Config &cfg) {
    return detail::bagPrefillPerThread(cfg) *
           static_cast<std::int64_t>(cfg.numThreads);
  }

  static Workload makeWorkload(std::size_t threadId, const Config &cfg) {
    return Workload(threadId, 0 /* no find op */, detail::bagAddOpsCount(cfg),
                    detail::bagRemoveOpsCount(cfg));
  }

  static void prefill(Structure &s, std::size_t threadId, const Config &cfg) {
    const std::int64_t count = detail::bagPrefillPerThread(cfg);
    for (std::int64_t i = 0; i < count; ++i) {
      s.push(static_cast<Key>(i), threadId);
    }
  }

  static StepResult step(Structure &s, std::size_t threadId, Workload &work,
                         const Config &, std::uint64_t counter) {
    if (work.next() == Workload::Op::Add) {
      s.push(static_cast<Key>(counter), threadId);
      return {+1, true};
    }
    std::optional<Key> value = s.pop();
    benchmark::DoNotOptimize(value);
    return value.has_value() ? StepResult{-1, true} : StepResult{0, false};
  }

  static cpu::Validation inspect(Structure &s, const Config &cfg) {
    return detail::drainBag(s, cfg);
  }
};

template <typename Structure> struct QueueOps {
  static constexpr const char *adt = "queue";
  static constexpr std::size_t nodeBytes = Structure::nodeBytes;

  static constexpr std::size_t getNodesPerThread(const Config &cfg) {
    return static_cast<std::size_t>(detail::bagNodesPerThread(cfg));
  }

  static constexpr std::int64_t getPrefillTotal(const Config &cfg) {
    return detail::bagPrefillPerThread(cfg) *
           static_cast<std::int64_t>(cfg.numThreads);
  }

  static Workload makeWorkload(std::size_t threadId, const Config &cfg) {
    return Workload(threadId, 0 /* no find op */, detail::bagAddOpsCount(cfg),
                    detail::bagRemoveOpsCount(cfg));
  }

  static void prefill(Structure &s, std::size_t threadId, const Config &cfg) {
    const std::int64_t count = detail::bagPrefillPerThread(cfg);
    for (std::int64_t i = 0; i < count; ++i) {
      s.enqueue(static_cast<Key>(i), threadId);
    }
  }

  static StepResult step(Structure &s, std::size_t threadId, Workload &work,
                         const Config &, std::uint64_t counter) {
    if (work.next() == Workload::Op::Add) {
      s.enqueue(static_cast<Key>(counter), threadId);
      return {+1, true};
    }
    std::optional<Key> value = s.dequeue();
    benchmark::DoNotOptimize(value);
    return value.has_value() ? StepResult{-1, true} : StepResult{0, false};
  }

  // Adapts dequeue() to pop() so the same drain helper serves both bags.
  static cpu::Validation inspect(Structure &s, const Config &cfg) {
    struct PopAdapter {
      Structure &inner;
      std::optional<Key> pop() { return inner.dequeue(); }
    } adapter{s};
    return detail::drainBag(adapter, cfg);
  }
};

// --------------------------------------------------------------------------
// List (set ADT)
// --------------------------------------------------------------------------
//
// mixPct is the find percentage; insert and remove split the remainder evenly,
// which keeps the set size stationary around keyRange/2. Prefilling to exactly
// half the key range is the standard steady state: an insert and a remove each
// succeed about half the time, so neither degenerates into an early return.

template <typename Structure> struct ListOps {
  static constexpr const char *adt = "list";
  static constexpr std::size_t nodeBytes = Structure::nodeBytes;

  // Keeps the prefill shuffle off the thread's operation stream, which is
  // seeded from the thread id alone.
  static constexpr std::int64_t kPrefillSeedOffset = 1 << 20;

  static constexpr std::int64_t findCount(const Config &cfg) {
    return cfg.opsPerThread * cfg.mixPct / 100;
  }

  static constexpr std::int64_t insertCount(const Config &cfg) {
    return (cfg.opsPerThread - findCount(cfg)) / 2;
  }

  static constexpr std::int64_t removeCount(const Config &cfg) {
    return cfg.opsPerThread - findCount(cfg) - insertCount(cfg);
  }

  static Workload makeWorkload(std::size_t threadId, const Config &cfg) {
    return Workload(threadId, findCount(cfg), insertCount(cfg),
                    removeCount(cfg));
  }

  // Partitioned by thread, so every thread first-touches its own pool pages.
  static constexpr std::int64_t prefillPerThread(const Config &cfg) {
    const std::int64_t half = cfg.keyRange / 2;
    const std::int64_t threads = static_cast<std::int64_t>(cfg.numThreads);
    return (half + threads - 1) / threads;
  }

  // Threads partition the even keys, so every prefill insert succeeds.
  static constexpr std::int64_t getPrefillTotal(const Config &cfg) {
    return cfg.keyRange / 2;
  }

  // insert() draws a node from the pool BEFORE checking for a duplicate, so the
  // budget is attempted inserts, not successful ones.
  static constexpr std::size_t getNodesPerThread(const Config &cfg) {
    return static_cast<std::size_t>(prefillPerThread(cfg) + insertCount(cfg));
  }

  // Inserted in random order, which is what makes the later traversals a real
  // pointer chase.
  static void prefill(Structure &s, std::size_t threadId, const Config &cfg) {
    const std::int64_t half = cfg.keyRange / 2;
    const std::int64_t stride = static_cast<std::int64_t>(cfg.numThreads);

    std::vector<Key> keys;
    keys.reserve(static_cast<std::size_t>(prefillPerThread(cfg)));
    for (std::int64_t i = static_cast<std::int64_t>(threadId); i < half;
         i += stride) {
      keys.push_back(static_cast<Key>(i * 2));
    }

    // A stream of its own: seeding this the way Workload does would correlate
    // the prefill order with the operation order on the same thread.
    Rng rng(static_cast<std::int64_t>(threadId) + 1 + kPrefillSeedOffset);
    rng.shuffle(keys.begin(), keys.end());

    for (const Key key : keys) {
      s.insert(key, threadId);
    }
  }

  static StepResult step(Structure &s, std::size_t threadId, Workload &work,
                         const Config &cfg, std::uint64_t) {
    const Key key = work.key(cfg.keyRange);

    switch (work.next()) {
    case Workload::Op::Find: {
      const bool found = s.find(key);
      benchmark::DoNotOptimize(found);
      return {0, true};
    }
    case Workload::Op::Add: {
      const bool inserted = s.insert(key, threadId);
      benchmark::DoNotOptimize(inserted);
      return inserted ? StepResult{+1, true} : StepResult{0, false};
    }
    default: {
      const bool removed = s.remove(key);
      benchmark::DoNotOptimize(removed);
      return removed ? StepResult{-1, true} : StepResult{0, false};
    }
    }
  }

  // The budget bounds the walk so a cycle is reported rather than hung on. It
  // is the pool: the chain cannot legitimately hold more nodes than were ever
  // allocated, so a walk that runs past it has found a cycle or a node
  // reachable twice. The key range is the wrong bound -- the lock-free walk
  // also counts marked nodes still physically linked, so a correct chain can be
  // longer than the set it represents, and a slow tail of unlinks would be
  // reported as corruption.
  static cpu::Validation inspect(Structure &s, const Config &cfg) {
    return s.validate(getNodesPerThread(cfg) * cfg.numThreads);
  }
};

} // namespace bench
