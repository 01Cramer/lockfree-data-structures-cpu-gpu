#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockbased {

// Lock-based stack guarded by a single mutex. Nodes are drawn from a
// NodePool outside the critical section; the mutex protects only the
// linked-list update.
//
// No fine-grained lock-based variant exists in this project: a stack
// has a single synchronization point (the top) intrinsic to LIFO. The
// elimination stack (Hendler/Shavit/Yerushalmi, 2004) reduces contention
// via paired push/pop cancellation but is a separate algorithm, not a
// different locking granularity.
// See node_pool.hpp for the memory-model rationale.

template <typename T> class Stack {
private:
  struct Node {
    T value{};
    Node *next = nullptr;
  };

public:
  Stack(std::size_t nodesPerThread, std::size_t numThreads)
      : m_pool(nodesPerThread, numThreads) {}

  Stack(const Stack &) = delete;
  Stack &operator=(const Stack &) = delete;
  Stack(Stack &&) = delete;
  Stack &operator=(Stack &&) = delete;

  void push(const T &value) {
    Node *newNode = m_pool.acquire();
    newNode->value = value;

    const std::lock_guard<std::mutex> lock(m_topMutex);
    newNode->next = m_top;
    m_top = newNode;
  }

  void push(T &&value) {
    Node *newNode = m_pool.acquire();
    newNode->value = std::move(value);

    const std::lock_guard<std::mutex> lock(m_topMutex);
    newNode->next = m_top;
    m_top = newNode;
  }

  std::optional<T> pop() {
    const std::lock_guard<std::mutex> lock(m_topMutex);
    if (m_top == nullptr) {
      return std::nullopt;
    }
    Node *poppedNode = m_top;
    m_top = poppedNode->next;

    return std::optional<T>(std::move(poppedNode->value));
  }

private:
  NodePool<Node> m_pool;
  Node *m_top = nullptr;
  std::mutex m_topMutex;
};

} // namespace lockbased

} // namespace cpu
