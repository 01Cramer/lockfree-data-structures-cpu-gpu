#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockbased {

// Two-lock FIFO queue (Michael & Scott, 1996) -- the lock-based half of
// the same paper as the lock-free MS queue. Separate head and tail
// mutexes let enqueue and dequeue run in parallel. A dummy sentinel
// keeps head and tail logically independent even when the queue is
// empty, so the two locks never contend on the same node -- except for
// the next pointer when head == tail, which is therefore an atomic.
// See node_pool.hpp for the memory-model rationale.

template <typename T> class QueueTwoLock {
private:
  struct Node {
    T value{};
    std::atomic<Node *> next{nullptr};
  };

public:
  QueueTwoLock(std::size_t nodesPerThread, std::size_t numThreads)
      : m_pool(nodesPerThread, numThreads),
        m_head(&m_dummy),
        m_tail(&m_dummy) {}

  QueueTwoLock(const QueueTwoLock &) = delete;
  QueueTwoLock &operator=(const QueueTwoLock &) = delete;
  QueueTwoLock(QueueTwoLock &&) = delete;
  QueueTwoLock &operator=(QueueTwoLock &&) = delete;

  void enqueue(const T &value) {
    Node *newNode = m_pool.acquire();
    newNode->value = value;

    const std::lock_guard<std::mutex> lock(m_tailMutex);
    m_tail->next.store(newNode);
    m_tail = newNode;
  }

  void enqueue(T &&value) {
    Node *newNode = m_pool.acquire();
    newNode->value = std::move(value);

    const std::lock_guard<std::mutex> lock(m_tailMutex);
    m_tail->next.store(newNode);
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<std::mutex> lock(m_headMutex);
    Node *next = m_head->next.load();
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
  Node *m_head;
  Node *m_tail;
  std::mutex m_headMutex;
  std::mutex m_tailMutex;
};

} // namespace lockbased

} // namespace cpu
