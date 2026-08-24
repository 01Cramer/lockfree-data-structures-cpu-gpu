#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"

namespace cpu {

namespace mutex {

// Lock-based stack guarded by a single mutex. Nodes are drawn from a
// NodePool outside the critical section; the mutex protects only the
// linked-list update.
//
// No fine-grained lock-based variant exists: a stack
// has a single synchronization point (the top) intrinsic to LIFO.
//
// See node_pool.hpp for the memory-model rationale and cache.hpp for the
// Layout parameter (one synchronization point: m_top guarded by m_topMutex).

template <typename T, Layout L> class alignas(cacheLineSize) Stack {
private:
  struct Node {
    T value{};
    Node *next = nullptr;
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = true;

  Stack(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads) {}

  Stack(const Stack &) = delete;
  Stack &operator=(const Stack &) = delete;
  Stack(Stack &&) = delete;
  Stack &operator=(Stack &&) = delete;

  void push(const T &value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = value;

    const std::lock_guard<std::mutex> lock(m_topMutex);
    newNode->next = m_top;
    m_top = newNode;
  }

  void push(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
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
  alignas(syncAlign(L)) Node *m_top = nullptr;
  alignas(lockAlign(L)) std::mutex m_topMutex;
};

} // namespace mutex

} // namespace cpu
