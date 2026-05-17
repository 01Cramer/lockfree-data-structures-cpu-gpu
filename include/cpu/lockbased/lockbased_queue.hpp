#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockbased {

// Lock-based FIFO queue guarded by a single mutex. Nodes are drawn from a
// NodePool outside the critical section; the mutex protects only the
// linked-list update.
// See node_pool.hpp for the memory-model rationale.

template <typename T> class Queue {
private:
  struct Node {
    T value{};
    Node *next = nullptr;
  };

public:
  Queue(std::size_t nodesPerThread, std::size_t numThreads)
      : m_pool(nodesPerThread, numThreads) {}

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue &&) = delete;

  void enqueue(const T &value) {
    Node *newNode = m_pool.acquire();
    newNode->value = value;

    const std::lock_guard<std::mutex> lock(m_queueMutex);
    if (m_head == nullptr) {
      m_head = newNode;
    } else {
      m_tail->next = newNode;
    }
    m_tail = newNode;
  }

  void enqueue(T &&value) {
    Node *newNode = m_pool.acquire();
    newNode->value = std::move(value);

    const std::lock_guard<std::mutex> lock(m_queueMutex);
    if (m_head == nullptr) {
      m_head = newNode;
    } else {
      m_tail->next = newNode;
    }
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<std::mutex> lock(m_queueMutex);
    if (m_head == nullptr) {
      return std::nullopt;
    }
    Node *dequeuedNode = m_head;
    m_head = dequeuedNode->next;
    if (m_head == nullptr) {
      m_tail = nullptr;
    }

    return std::optional<T>(std::move(dequeuedNode->value));
  }

private:
  NodePool<Node> m_pool;
  Node *m_head = nullptr;
  Node *m_tail = nullptr;
  std::mutex m_queueMutex;
};

} // namespace lockbased

} // namespace cpu
