#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"

namespace cpu {

namespace mutex {

// Two-lock FIFO queue (Michael & Scott, 1996) -- the lock-based half of
// the same paper as the lock-free MS queue. Separate head and tail
// mutexes let enqueue and dequeue run in parallel. A dummy sentinel
// keeps head and tail logically independent even when the queue is
// empty, so the two locks never contend on the same node -- except for
// the next pointer when head == tail, which is therefore an atomic.
// See node_pool.hpp for the memory-model rationale and cache.hpp for the
// Layout parameter.

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

    const std::lock_guard<std::mutex> lock(m_tailMutex);
    m_tail->next.store(newNode, std::memory_order_release);
    m_tail = newNode;
  }

  void enqueue(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = std::move(value);
    newNode->next.store(nullptr, std::memory_order_relaxed);

    const std::lock_guard<std::mutex> lock(m_tailMutex);
    m_tail->next.store(newNode, std::memory_order_release);
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<std::mutex> lock(m_headMutex);
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
  alignas(lockAlign(L)) std::mutex m_headMutex;
  alignas(syncAlign(L)) Node *m_tail;
  alignas(lockAlign(L)) std::mutex m_tailMutex;
};

} // namespace mutex

} // namespace cpu
