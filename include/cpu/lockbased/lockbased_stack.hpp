#pragma once

#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace cpu {

namespace lockbased {

// Lock-based stack using a single mutex for synchronization.
// The structure owns all allocated nodes and uses deferred memory reclamation.
// Memory is not freed during execution to ensure consistency with lock-free
// implementations, where safe reclamation is more complex.
// This design isolates synchronization costs and enables fair benchmarking.

template <typename T> class Stack {
private:
  struct Node {
    Node(const T &val) : value(val) {}
    Node(T &&val) : value(std::move(val)) {}

    T value = {};
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
    const std::lock_guard<std::mutex> lock(m_topMutex);
    Node *newNode = new Node(value);
    newNode->next = m_top;
    m_top = newNode;

    m_allNodes.push_back(newNode);
  }

  void push(T &&value) {
    const std::lock_guard<std::mutex> lock(m_topMutex);
    Node *newNode = new Node(std::move(value));
    newNode->next = m_top;
    m_top = newNode;

    m_allNodes.push_back(newNode);
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

  // Deferred memory reclamation
  // Frees all allocated nodes after concurrent execution completes
  void deferredMemoryReclamation() {
    for (const auto &node : m_allNodes) {
      delete node;
    }
    m_allNodes.clear();
  }

private:
  Node *m_top = nullptr;

  std::mutex m_topMutex;

  // Track allocated nodes for deferred reclamation
  std::vector<Node *> m_allNodes;

  // Initial reservation size to reduce reallocations during benchmarks
  static constexpr size_t defaultNodesSize = 100000;
};

} // namespace lockbased

} // namespace cpu