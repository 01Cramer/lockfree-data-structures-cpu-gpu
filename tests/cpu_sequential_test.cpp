// Single-threaded functional tests. These are the fast logic gate: they
// pin down each structure's sequential contract (LIFO / FIFO / set semantics)
// before the concurrent stress tests exercise the synchronization.

#include <optional>
#include <random>
#include <set>
#include <vector>

#include "support/baseline_types.hpp"
#include "support/test_harness.hpp"

using namespace cpu_test;

namespace {

constexpr std::size_t kThreadId = 0;

// -------- Stack: LIFO + empty behaviour --------
template <typename StackT> void stackSpec() {
  constexpr int n = 64;
  StackT stack(n, 1);

  // Empty pop yields nullopt.
  CHECK(!stack.pop().has_value());

  for (int i = 0; i < n; ++i) {
    stack.push(i, kThreadId);
  }
  // LIFO: values come back in reverse order.
  for (int i = n - 1; i >= 0; --i) {
    std::optional<int> value = stack.pop();
    CHECK(value.has_value());
    if (value.has_value()) {
      CHECK_EQ(*value, i);
    }
  }
  CHECK(!stack.pop().has_value());
}

// -------- Queue: FIFO + empty behaviour --------
template <typename QueueT> void queueSpec() {
  constexpr int n = 64;
  // n enqueues below, plus 3 more in the interleaved section; the one-shot pool
  // never reclaims, so size for the total number of enqueue calls.
  QueueT queue(n + 3, 1);

  // Empty dequeue yields nullopt.
  CHECK(!queue.dequeue().has_value());

  for (int i = 0; i < n; ++i) {
    queue.enqueue(i, kThreadId);
  }
  // FIFO: values come back in insertion order.
  for (int i = 0; i < n; ++i) {
    std::optional<int> value = queue.dequeue();
    CHECK(value.has_value());
    if (value.has_value()) {
      CHECK_EQ(*value, i);
    }
  }
  CHECK(!queue.dequeue().has_value());

  // Interleaved enqueue/dequeue keeps FIFO across the wrap.
  queue.enqueue(100, kThreadId);
  queue.enqueue(101, kThreadId);
  CHECK_EQ(*queue.dequeue(), 100);
  queue.enqueue(102, kThreadId);
  CHECK_EQ(*queue.dequeue(), 101);
  CHECK_EQ(*queue.dequeue(), 102);
  CHECK(!queue.dequeue().has_value());
}

// -------- List: set semantics --------
template <typename ListT> void listSpec() {

  constexpr int slots = 16;
  ListT list(slots, 1);

  CHECK(!list.find(5)); // absent in empty list
  CHECK(!list.remove(5));

  CHECK(list.insert(5, kThreadId)); // first insert succeeds
  CHECK(list.find(5));
  CHECK(!list.insert(5, kThreadId)); // duplicate rejected
  CHECK(list.find(5));

  CHECK(list.insert(3, kThreadId));
  CHECK(list.insert(7, kThreadId));
  CHECK(list.find(3));
  CHECK(list.find(7));

  CHECK(list.remove(5)); // remove present
  CHECK(!list.find(5));
  CHECK(!list.remove(5)); // remove-again fails
  CHECK(list.find(3));    // neighbours intact
  CHECK(list.find(7));
}

} // namespace

CPU_TEST(stack_mutex) { stackSpec<baseline::mutex::Stack<int>>(); }
CPU_TEST(stack_spinlock) { stackSpec<baseline::spinlock::Stack<int>>(); }
CPU_TEST(stack_lockfree) { stackSpec<baseline::lockfree::Stack<int>>(); }

CPU_TEST(queue_mutex) { queueSpec<baseline::mutex::Queue<int>>(); }
CPU_TEST(queue_mutex_two_lock) {
  queueSpec<baseline::mutex::QueueTwoLock<int>>();
}
CPU_TEST(queue_spinlock) { queueSpec<baseline::spinlock::Queue<int>>(); }
CPU_TEST(queue_spinlock_two_lock) {
  queueSpec<baseline::spinlock::QueueTwoLock<int>>();
}
CPU_TEST(queue_lockfree) { queueSpec<baseline::lockfree::Queue<int>>(); }

CPU_TEST(list_mutex) { listSpec<baseline::mutex::List<int>>(); }
CPU_TEST(list_mutex_hand_over_hand) {
  listSpec<baseline::mutex::HandOverHandList<int>>();
}
CPU_TEST(list_spinlock) { listSpec<baseline::spinlock::List<int>>(); }
CPU_TEST(list_spinlock_hand_over_hand) {
  listSpec<baseline::spinlock::HandOverHandList<int>>();
}
CPU_TEST(list_lockfree) { listSpec<baseline::lockfree::List<int>>(); }

int main(int argc, char **argv) { return cpu_test::runAll(argc, argv); }
