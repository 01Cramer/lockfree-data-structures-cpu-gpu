#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockfree {

// Lock-free Treiber stack (Treiber, 1986). A single atomic top pointer is
// updated via compare-and-swap. Nodes are drawn from a NodePool; push()
// performs the acquire outside the CAS loop.
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

    Node *oldTop = m_top.load(std::memory_order_relaxed);
    do {
      newNode->next = oldTop;
    } while (!m_top.compare_exchange_weak(
        oldTop, newNode, std::memory_order_release, std::memory_order_relaxed));
  }

  void push(T &&value) {
    Node *newNode = m_pool.acquire();
    newNode->value = std::move(value);

    Node *oldTop = m_top.load(std::memory_order_relaxed);
    do {
      newNode->next = oldTop;
    } while (!m_top.compare_exchange_weak(
        oldTop, newNode, std::memory_order_release, std::memory_order_relaxed));
  }

  std::optional<T> pop() {
    Node *oldTop = m_top.load(std::memory_order_relaxed);
    Node *newTop;

    do {
      if (oldTop == nullptr) {
        return std::nullopt;
      }
      newTop = oldTop->next;
    } while (!m_top.compare_exchange_weak(
        oldTop, newTop, std::memory_order_acquire, std::memory_order_relaxed));

    // Safe to move: only the thread that won the CAS reaches this line for
    // any given oldTop, so no two threads ever move from the same node.
    return std::optional<T>(std::move(oldTop->value));
  }

private:
  NodePool<Node> m_pool;
  std::atomic<Node *> m_top = nullptr;
};

} // namespace lockfree

} // namespace cpu
