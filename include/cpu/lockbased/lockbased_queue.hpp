#pragma once

#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace cpu {

namespace lockbased {

// Lock-based queue using a single mutex for synchronization.
// The structure owns all allocated nodes and uses deferred memory reclamation.
// Memory is not freed during execution to ensure consistency with lock-free
// implementations, where safe reclamation is more complex.
// This design isolates synchronization costs and enables fair benchmarking.

template <typename T> class Queue {
private:
  struct Node {
    Node(const T &val) : value(val) {}
    Node(T &&val) : value(std::move(val)) {}

    T value = {};
    Node *next = nullptr;
  };

public:
  // Preallocates storage to reduce allocator noise during benchmarking
  Queue() { m_allNodes.reserve(defaultNodesSize); }
  ~Queue() { deferredMemoryReclamation(); }

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue &&) = delete;

  void enqueue(const T &value) {
    const std::lock_guard<std::mutex> lock(m_queueMutex);
    Node *newNode = new Node(value);
    if (m_head == nullptr) {
      m_head = newNode;
    } else {
      m_tail->next = newNode;
    }
    m_tail = newNode;

    m_allNodes.push_back(newNode);
  }

  void enqueue(T &&value) {
    const std::lock_guard<std::mutex> lock(m_queueMutex);
    Node *newNode = new Node(std::move(value));
    if (m_head == nullptr) {
      m_head = newNode;
    } else {
      m_tail->next = newNode;
    }
    m_tail = newNode;

    m_allNodes.push_back(newNode);
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

  // Deferred memory reclamation
  // Frees all allocated nodes after concurrent execution completes
  void deferredMemoryReclamation() {
    for (const auto &node : m_allNodes) {
      delete node;
    }
    m_allNodes.clear();
  }

private:
  Node *m_head = nullptr;
  Node *m_tail = nullptr;

  std::mutex m_queueMutex;
  
  // Track allocated nodes for deferred reclamation
  std::vector<Node *> m_allNodes;

  // Initial reservation size to reduce reallocations during benchmarks
  static constexpr size_t defaultNodesSize = 100000;
};

} // namespace lockbased

} // namespace cpu