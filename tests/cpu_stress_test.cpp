// Multithreaded correctness stress tests.
//
// Oracles (see the plan):
//   - Stack / Queue : multiset conservation (popped == pushed, no loss/dup).
//   - Stack (extra) : per-producer LIFO.
//   - Queue (extra) : per-producer FIFO.
//   - List          : disjoint-key deterministic membership (strong oracle),
//                     and shared-key contention (invariant: successful
//                     inserts - successful removes == final size),
//                     each cross-checked against a structural walk.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include "support/baseline_types.hpp"
#include "support/test_harness.hpp"

using namespace cpu_test;

namespace {

// Value encoding for the conservation tests: value = producerId * kStride + seq
// makes every pushed value globally unique and lets a consumer decode which
// producer emitted it (for the per-producer order checks). kItemsPerProducer
// must stay below kStride.
constexpr int kStride = 1 << 20;
constexpr int kItemsPerProducer = 400;

bool checkEncodingFits(unsigned producers) {
  const bool seqFits = kItemsPerProducer < kStride;
  const bool productFits =
      static_cast<std::int64_t>(producers) * kStride + kItemsPerProducer <
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  CHECK(seqFits);
  CHECK(productFits);
  return seqFits && productFits;
}

// Split the configured worker budget into producers and consumers.
struct Split {
  unsigned producers;
  unsigned consumers;
};

Split splitThreads() {
  const unsigned n = config().threads;
  Split s;
  s.producers = n / 2 > 0 ? n / 2 : 1;
  s.consumers = n - s.producers > 0 ? n - s.producers : 1;
  return s;
}

template <typename Fn> void runThreads(unsigned count, Fn fn) {
  StartGate gate;
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (unsigned i = 0; i < count; ++i) {
    threads.emplace_back([&, i] {
      gate.wait();
      fn(i);
    });
  }
  gate.open();
  for (std::thread &t : threads) {
    t.join();
  }
}

// One producer's whole run: its own decodable range of values.
template <typename PushOp> void produce(PushOp push, unsigned producer) {
  for (int s = 0; s < kItemsPerProducer; ++s) {
    push(static_cast<int>(producer) * kStride + s,
         static_cast<std::size_t>(producer));
  }
}

// One consumer's whole run: pop into 'out' until every pushed item has been
// accounted for. An empty container is not the stop condition -- a producer may
// simply not have got there yet -- so 'remaining' is.
template <typename PopOp>
void drain(PopOp pop, std::atomic<long> &remaining, std::vector<int> &out) {
  while (true) {
    std::optional<int> value = pop();
    if (value.has_value()) {
      out.push_back(*value);
      remaining.fetch_sub(1, std::memory_order_relaxed);
    } else if (remaining.load(std::memory_order_relaxed) == 0) {
      return;
    } else {
      std::this_thread::yield();
    }
  }
}

std::vector<int> flatten(const std::vector<std::vector<int>> &perConsumer) {
  std::vector<int> all;
  for (const std::vector<int> &v : perConsumer) {
    all.insert(all.end(), v.begin(), v.end());
  }
  return all;
}

// --- oracles ---

// Verify a drained container returned exactly the pushed multiset: every
// (producer, seq) pair in range appears exactly once.
void checkConservation(std::vector<int> popped, unsigned producers) {
  const std::size_t total =
      static_cast<std::size_t>(producers) * kItemsPerProducer;
  CHECK_EQ(popped.size(), total);
  if (popped.size() != total) {
    return;
  }
  std::sort(popped.begin(), popped.end());
  std::size_t idx = 0;
  for (unsigned p = 0; p < producers; ++p) {
    for (int s = 0; s < kItemsPerProducer; ++s, ++idx) {
      CHECK_EQ(popped[idx], static_cast<int>(p) * kStride + s);
    }
  }
}

// Within one observer's view, each producer's items must appear in strictly
// monotonic seq order: increasing for a FIFO queue, decreasing for a stack
// drained after all pushes finished.
void checkPerProducerOrder(const std::vector<int> &seen, unsigned producers,
                           bool increasing) {
  std::vector<int> lastSeq(producers, increasing ? -1 : kItemsPerProducer);
  for (int value : seen) {
    const unsigned p = static_cast<unsigned>(value / kStride);
    const int seq = value % kStride;
    CHECK(p < producers);
    if (p >= producers) {
      continue;
    }
    if (increasing) {
      CHECK(seq > lastSeq[p]);
    } else {
      CHECK(seq < lastSeq[p]);
    }
    lastSeq[p] = seq;
  }
}

// cross-check a list against with a structural walk.
// find() is order-blind -- it answers correctly on a list that has lost its
// sort order, grown a cycle, or stranded a run of logically-deleted nodes.
// expected is the caller's independently-derived element count.
template <typename ListT>
void checkStructure(ListT &list, std::int64_t expected, std::size_t maxSteps) {
  const cpu::Validation v = list.validate(maxSteps);
  CHECK(v.terminated); // cycle, or a next that ran off into null
  CHECK(v.sorted);     // lost ordering, or a duplicate key
  CHECK(v.noMarked);   // lock-free: a logically-deleted node still reachable
  if (v.terminated) {
    // Only meaningful if the walk actually finished; a truncated walk
    // undercounts by construction and the mismatch would be noise.
    CHECK_EQ(v.count, expected);
  }
}

// --- scenarios ---

// One interleaved MPMC run, generic over push/pop closures (works for any stack
// or queue). Producers push their unique ranges while consumers drain. Returns
// what each consumer saw, so the caller picks which oracles to apply.
template <typename PushOp, typename PopOp>
std::vector<std::vector<int>> runMpmc(PushOp push, PopOp pop, Split split) {
  std::atomic<long> remaining{static_cast<long>(split.producers) *
                              kItemsPerProducer};
  std::vector<std::vector<int>> perConsumer(split.consumers);

  runThreads(split.producers + split.consumers, [&](unsigned i) {
    if (i < split.producers) {
      produce(push, i);
    } else {
      drain(pop, remaining, perConsumer[i - split.producers]);
    }
  });

  return perConsumer;
}

// Conservation only. A stack's ordering invariant does not survive interleaved
// push/pop, so it is checked separately by stackLifo.
template <typename StackT> void stackConservation() {
  const Split split = splitThreads();
  if (!checkEncodingFits(split.producers)) {
    return;
  }
  for (unsigned long it = 0; it < config().iterations; ++it) {
    StackT stack(kItemsPerProducer, split.producers);
    const std::vector<std::vector<int>> perConsumer =
        runMpmc([&](int v, std::size_t t) { stack.push(v, t); },
                [&] { return stack.pop(); }, split);
    checkConservation(flatten(perConsumer), split.producers);
  }
}

template <typename StackT> void stackLifo() {
  const Split split = splitThreads();
  if (!checkEncodingFits(split.producers)) {
    return;
  }

  for (unsigned long it = 0; it < config().iterations; ++it) {
    StackT stack(kItemsPerProducer, split.producers);
    runThreads(split.producers, [&](unsigned p) {
      produce([&](int v, std::size_t t) { stack.push(v, t); }, p);
    });

    std::atomic<long> remaining{static_cast<long>(split.producers) *
                                kItemsPerProducer};
    std::vector<std::vector<int>> perConsumer(split.consumers);
    runThreads(split.consumers, [&](unsigned c) {
      drain([&] { return stack.pop(); }, remaining, perConsumer[c]);
    });

    // Nothing lost or duplicated, and each consumer saw every producer's items
    // in strictly decreasing seq.
    checkConservation(flatten(perConsumer), split.producers);
    for (const std::vector<int> &seen : perConsumer) {
      checkPerProducerOrder(seen, split.producers, /*increasing=*/false);
    }
  }
}

// Conservation AND per-producer FIFO, from one interleaved run.
template <typename QueueT> void queueFifo() {
  const Split split = splitThreads();
  if (!checkEncodingFits(split.producers)) {
    return;
  }

  for (unsigned long it = 0; it < config().iterations; ++it) {
    QueueT queue(kItemsPerProducer, split.producers);
    const std::vector<std::vector<int>> perConsumer =
        runMpmc([&](int v, std::size_t t) { queue.enqueue(v, t); },
                [&] { return queue.dequeue(); }, split);

    checkConservation(flatten(perConsumer), split.producers);
    for (const std::vector<int> &seen : perConsumer) {
      checkPerProducerOrder(seen, split.producers, /*increasing=*/true);
    }
  }
}

// Disjoint-key set: thread t owns keys { t, t+N, t+2N, ... }. No two threads
// touch the same key, so each thread keeps an exact local model and asserts
// every op's return value; final membership is checked against the union.
template <typename ListT> void listDisjoint() {
  const unsigned n = config().threads;
  constexpr int keysPerThread = 32;
  constexpr int opsPerThread = 3000;

  for (unsigned long it = 0; it < config().iterations; ++it) {
    ListT list(opsPerThread, n); // every op could be an insert
    std::vector<std::vector<char>> present(n,
                                           std::vector<char>(keysPerThread, 0));

    runThreads(n, [&](unsigned t) {
      std::mt19937 rng(config().seed + t * 7919u + it * 104729u);
      std::uniform_int_distribution<int> keyDist(0, keysPerThread - 1);
      std::uniform_int_distribution<int> opDist(0, 2);
      std::vector<char> &mine = present[t];
      for (int op = 0; op < opsPerThread; ++op) {
        const int i = keyDist(rng);
        const int key = static_cast<int>(t) + i * static_cast<int>(n);
        switch (opDist(rng)) {
        case 0: {
          const bool wasAbsent = mine[i] == 0;
          CHECK_EQ(list.insert(key, static_cast<std::size_t>(t)), wasAbsent);
          mine[i] = 1; // present after an insert attempt either way
          break;
        }
        case 1: {
          const bool wasPresent = mine[i] != 0;
          CHECK_EQ(list.remove(key), wasPresent);
          mine[i] = 0;
          break;
        }
        default:
          CHECK_EQ(list.find(key), mine[i] != 0);
          break;
        }
      }
    });

    std::int64_t expected = 0;
    for (unsigned t = 0; t < n; ++t) {
      for (int i = 0; i < keysPerThread; ++i) {
        const int key = static_cast<int>(t) + i * static_cast<int>(n);
        CHECK_EQ(list.find(key), present[t][i] != 0);
        expected += present[t][i] != 0 ? 1 : 0;
      }
    }

    checkStructure(list, expected,
                   static_cast<std::size_t>(opsPerThread) * n + 2);
  }
}

// Shared-key contention: all threads hammer a small key domain. Outcome is
// nondeterministic, so we check the global conservation invariant only:
// (successful inserts) - (successful removes) == final membership count.
template <typename ListT> void listSharedKeys() {
  const unsigned n = config().threads;
  constexpr int domain = 16;
  constexpr int opsPerThread = 3000;

  for (unsigned long it = 0; it < config().iterations; ++it) {
    ListT list(opsPerThread, n);
    std::atomic<long> inserts{0};
    std::atomic<long> removes{0};

    runThreads(n, [&](unsigned t) {
      std::mt19937 rng(config().seed + t * 40503u + it * 2246822519u);
      std::uniform_int_distribution<int> keyDist(0, domain - 1);
      std::uniform_int_distribution<int> opDist(0, 2);
      for (int op = 0; op < opsPerThread; ++op) {
        const int key = keyDist(rng);
        switch (opDist(rng)) {
        case 0:
          if (list.insert(key, static_cast<std::size_t>(t))) {
            inserts.fetch_add(1, std::memory_order_relaxed);
          }
          break;
        case 1:
          if (list.remove(key)) {
            removes.fetch_add(1, std::memory_order_relaxed);
          }
          break;
        default:
          (void)list.find(key);
          break;
        }
      }
    });

    std::int64_t finalSize = 0;
    for (int key = 0; key < domain; ++key) {
      if (list.find(key)) {
        ++finalSize;
      }
    }
    CHECK_EQ(inserts.load() - removes.load(), finalSize);

    // No phantom keys outside the domain we ever inserted.
    for (int key = domain; key < domain + 8; ++key) {
      CHECK(!list.find(key));
    }

    checkStructure(list, inserts.load() - removes.load(),
                   static_cast<std::size_t>(opsPerThread) * n + 2);
  }
}

} // namespace

CPU_TEST(stack_conservation_mutex) {
  stackConservation<baseline::mutex::Stack<int>>();
}
CPU_TEST(stack_conservation_spinlock) {
  stackConservation<baseline::spinlock::Stack<int>>();
}
CPU_TEST(stack_conservation_lockfree) {
  stackConservation<baseline::lockfree::Stack<int>>();
}

CPU_TEST(stack_lifo_mutex) { stackLifo<baseline::mutex::Stack<int>>(); }
CPU_TEST(stack_lifo_spinlock) { stackLifo<baseline::spinlock::Stack<int>>(); }
CPU_TEST(stack_lifo_lockfree) { stackLifo<baseline::lockfree::Stack<int>>(); }

CPU_TEST(queue_fifo_mutex) { queueFifo<baseline::mutex::Queue<int>>(); }
CPU_TEST(queue_fifo_mutex_two_lock) {
  queueFifo<baseline::mutex::QueueTwoLock<int>>();
}
CPU_TEST(queue_fifo_spinlock) { queueFifo<baseline::spinlock::Queue<int>>(); }
CPU_TEST(queue_fifo_spinlock_two_lock) {
  queueFifo<baseline::spinlock::QueueTwoLock<int>>();
}
CPU_TEST(queue_fifo_lockfree) { queueFifo<baseline::lockfree::Queue<int>>(); }

CPU_TEST(list_disjoint_mutex) { listDisjoint<baseline::mutex::List<int>>(); }
CPU_TEST(list_disjoint_mutex_hoh) {
  listDisjoint<baseline::mutex::HandOverHandList<int>>();
}
CPU_TEST(list_disjoint_spinlock) {
  listDisjoint<baseline::spinlock::List<int>>();
}
CPU_TEST(list_disjoint_spinlock_hoh) {
  listDisjoint<baseline::spinlock::HandOverHandList<int>>();
}
CPU_TEST(list_disjoint_lockfree) {
  listDisjoint<baseline::lockfree::List<int>>();
}

CPU_TEST(list_shared_mutex) { listSharedKeys<baseline::mutex::List<int>>(); }
CPU_TEST(list_shared_mutex_hoh) {
  listSharedKeys<baseline::mutex::HandOverHandList<int>>();
}
CPU_TEST(list_shared_spinlock) {
  listSharedKeys<baseline::spinlock::List<int>>();
}
CPU_TEST(list_shared_spinlock_hoh) {
  listSharedKeys<baseline::spinlock::HandOverHandList<int>>();
}
CPU_TEST(list_shared_lockfree) {
  listSharedKeys<baseline::lockfree::List<int>>();
}

int main(int argc, char **argv) { return cpu_test::runAll(argc, argv); }
