#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockbased {

// Lock-based FIFO queue guarded by a single mutex. Nodes are drawn from
// a NodePool outside the critical section; the mutex protects only the
// linked-list update.
//
// A dummy sentinel (a non-pool Node member) keeps m_head and m_tail
// non-nullable, so enqueue and dequeue never branch on emptiness. This
// matches the structural shape of QueueTwoLock and lockfree::Queue,
// isolating the locking strategy as the only meaningful difference
// between the three queue variants.
// See node_pool.hpp for the memory-model rationale.

template <typename T> class Queue {
private:
  struct Node {
    T value{};
    Node *next = nullptr;
  };

public:
  Queue(std::size_t nodesPerThread, std::size_t numThreads)
      : m_pool(nodesPerThread, numThreads),
        m_head(&m_dummy),
        m_tail(&m_dummy) {}

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue &&) = delete;

  void enqueue(const T &value) {
    Node *newNode = m_pool.acquire();
    newNode->value = value;

    const std::lock_guard<std::mutex> lock(m_queueMutex);
    m_tail->next = newNode;
    m_tail = newNode;
  }

  void enqueue(T &&value) {
    Node *newNode = m_pool.acquire();
    newNode->value = std::move(value);

    const std::lock_guard<std::mutex> lock(m_queueMutex);
    m_tail->next = newNode;
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<std::mutex> lock(m_queueMutex);
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
  Node *m_head;
  Node *m_tail;
  std::mutex m_queueMutex;
};

} // namespace lockbased

} // namespace cpu
