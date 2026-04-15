#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace cpu {

namespace lockfree {

// Lock-free Treiber stack (Treiber, 1986) using a single atomic pointer
// and compare-and-swap (CAS) for synchronization.
// The structure owns all allocated nodes and uses deferred memory
// reclamation. Nodes are not freed during execution to avoid the need
// for safe memory reclamation (e.g., hazard pointers), ensuring
// consistency with lock-based implementations and isolating
// synchronization costs for fair benchmarking.

template <typename T> class Stack {
private:
  struct Node {
    Node(const T &val) : value(val) {}
    Node(T &&val) : value(std::move(val)) {}

    T value;
    Node *next = nullptr;
  };

public:
  // Preallocates storage to reduce allocator noise during benchmarking
  Stack() { m_allNodes.reserve(defaultNodesSize); }
  ~Stack() { deferredMemoryReclamation(); }

  Stack(const Stack &) = delete;
  Stack &operator=(const Stack &) = delete;
  Stack(Stack &&) = delete;
  Stack &operator=(Stack &&) = delete;

  void push(const T &value) {
    Node *newNode = new Node(value);
    Node *oldTop = m_top.load(std::memory_order_relaxed);

    do {
      newNode->next = oldTop;
    } while (!m_top.compare_exchange_weak(
        oldTop, newNode, std::memory_order_release, std::memory_order_relaxed));

    const std::lock_guard<std::mutex> lock(m_nodesMutex);
    m_allNodes.push_back(newNode);
  }

  void push(T &&value) {
    Node *newNode = new Node(std::move(value));
    Node *oldTop = m_top.load(std::memory_order_relaxed);

    do {
      newNode->next = oldTop;
    } while (!m_top.compare_exchange_weak(
        oldTop, newNode, std::memory_order_release, std::memory_order_relaxed));

    const std::lock_guard<std::mutex> lock(m_nodesMutex);
    m_allNodes.push_back(newNode);
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

    return std::optional<T>(std::move(oldTop->value));
  }

  // Deferred memory reclamation
  // Frees all allocated nodes after concurrent execution completes
  void deferredMemoryReclamation() {
    for (const auto &node : m_allNodes) {
      delete node;
    }
    m_allNodes.clear();
  }

private:
  std::atomic<Node *> m_top = nullptr;

  // Track allocated nodes for deferred reclamation
  std::vector<Node *> m_allNodes;
  std::mutex m_nodesMutex;

  // Initial reservation size to reduce reallocations during benchmarks
  static constexpr size_t defaultNodesSize = 100000;
};

} // namespace lockfree

} // namespace cpu