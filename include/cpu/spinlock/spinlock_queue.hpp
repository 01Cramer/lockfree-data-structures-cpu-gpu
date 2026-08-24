#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"
#include "cpu/spinlock/spinlock.hpp"

namespace cpu {

namespace spinlock {

// Spinlock-guarded FIFO queue (single lock). Identical to mutex::Queue
// except the mutex is replaced by cpu::Spinlock -- the busy-wait
// synchronization category.
//
// A dummy sentinel (a non-pool Node member) keeps m_head and m_tail
// non-nullable, so enqueue and dequeue never branch on emptiness, matching the
// structural shape of the other queue variants.
// See node_pool.hpp for the memory-model rationale and spinlock.hpp for the
// lock design, and cache.hpp for the Layout parameter.

template <typename T, Layout L> class alignas(cacheLineSize) Queue {
private:
  struct Node {
    T value{};
    Node *next = nullptr;
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = true;

  Queue(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads), m_head(&m_dummy), m_tail(&m_dummy) {
  }

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue &&) = delete;

  void enqueue(const T &value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = value;
    newNode->next = nullptr;

    const std::lock_guard<Spinlock> lock(m_queueLock);
    m_tail->next = newNode;
    m_tail = newNode;
  }

  void enqueue(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = std::move(value);
    newNode->next = nullptr;

    const std::lock_guard<Spinlock> lock(m_queueLock);
    m_tail->next = newNode;
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<Spinlock> lock(m_queueLock);
    Node *next = m_head->next;
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
  Node *m_tail;
  alignas(lockAlign(L)) Spinlock m_queueLock;
};

} // namespace spinlock

} // namespace cpu
