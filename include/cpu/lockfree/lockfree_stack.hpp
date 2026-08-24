#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"

namespace cpu {

namespace lockfree {

// Lock-free Treiber stack (Treiber, 1986).
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
  static constexpr bool hasLockWord = false;

  Stack(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads) {}

  Stack(const Stack &) = delete;
  Stack &operator=(const Stack &) = delete;
  Stack(Stack &&) = delete;
  Stack &operator=(Stack &&) = delete;

  void push(const T &value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = value;

    Node *oldTop = m_top.load(std::memory_order_relaxed);
    do {
      newNode->next = oldTop;
    } while (!m_top.compare_exchange_weak(
        oldTop, newNode, std::memory_order_release, std::memory_order_relaxed));
  }

  void push(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = std::move(value);

    Node *oldTop = m_top.load(std::memory_order_relaxed);
    do {
      newNode->next = oldTop;
    } while (!m_top.compare_exchange_weak(
        oldTop, newNode, std::memory_order_release, std::memory_order_relaxed));
  }

  std::optional<T> pop() {
    Node *oldTop = m_top.load(std::memory_order_acquire);
    Node *newTop;

    do {
      if (oldTop == nullptr) {
        return std::nullopt;
      }
      newTop = oldTop->next;
    } while (!m_top.compare_exchange_weak(
        oldTop, newTop, std::memory_order_acquire, std::memory_order_acquire));

    // Safe to move: only the thread that won the CAS reaches this line for
    // any given oldTop, so no two threads ever move from the same node.
    return std::optional<T>(std::move(oldTop->value));
  }

private:
  NodePool<Node> m_pool;
  alignas(syncAlign(L)) std::atomic<Node *> m_top = nullptr;
};

} // namespace lockfree

} // namespace cpu
