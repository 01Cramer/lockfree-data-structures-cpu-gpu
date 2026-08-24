#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"
#include "cpu/spinlock/spinlock.hpp"

namespace cpu {

namespace spinlock {

// Two-lock FIFO queue (Michael & Scott, 1996) with the mutexes replaced by
// cpu::Spinlock -- the busy-wait synchronization category. Identical to
// mutex::QueueTwoLock otherwise: separate head and tail locks let enqueue
// and dequeue run in parallel, and a dummy sentinel keeps the two ends
// logically independent.

template <typename T, Layout L> class alignas(cacheLineSize) QueueTwoLock {
  static_assert(!(L.padLockFromData && !L.padSyncPoints),
                "PadLockFromData is not expressible for a two-lock queue: "
                "m_tail would land on m_headMutex's line.");

private:
  struct Node {
    T value{};
    std::atomic<Node *> next{nullptr};
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = true;

  QueueTwoLock(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads), m_head(&m_dummy), m_tail(&m_dummy) {
  }

  QueueTwoLock(const QueueTwoLock &) = delete;
  QueueTwoLock &operator=(const QueueTwoLock &) = delete;
  QueueTwoLock(QueueTwoLock &&) = delete;
  QueueTwoLock &operator=(QueueTwoLock &&) = delete;

  void enqueue(const T &value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = value;
    newNode->next.store(nullptr, std::memory_order_relaxed);

    const std::lock_guard<Spinlock> lock(m_tailLock);
    m_tail->next.store(newNode, std::memory_order_release);
    m_tail = newNode;
  }

  void enqueue(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = std::move(value);
    newNode->next.store(nullptr, std::memory_order_relaxed);

    const std::lock_guard<Spinlock> lock(m_tailLock);
    m_tail->next.store(newNode, std::memory_order_release);
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<Spinlock> lock(m_headLock);
    // Acquire: pairs with the enqueue release so next->value is visible.
    Node *next = m_head->next.load(std::memory_order_acquire);
    if (next == nullptr) {
      return std::nullopt;
    }
    std::optional<T> result(std::move(next->value));
    m_head = next;
    return result;
  }

private:
  NodePool<Node> m_pool;
  Node m_dummy;
  alignas(syncAlign(L)) Node *m_head;
  alignas(lockAlign(L)) Spinlock m_headLock;
  alignas(syncAlign(L)) Node *m_tail;
  alignas(lockAlign(L)) Spinlock m_tailLock;
};

} // namespace spinlock

} // namespace cpu
