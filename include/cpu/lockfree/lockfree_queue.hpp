#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace cpu {

namespace lockfree {

// Lock-free Michael–Scott queue (Michael & Scott, 1996) using atomic head and
// tail pointers and compare-and-swap (CAS) for synchronization. The queue is
// implemented as a singly linked list with a sentinel (dummy) node, enabling
// non-blocking enqueue and dequeue operations.
// The structure owns all allocated nodes and uses deferred memory reclamation.
// Nodes are not freed during execution to avoid the need for safe memory
// reclamation techniques (e.g., hazard pointers), ensuring consistency with
// lock-based implementations and isolating synchronization costs for fair
// benchmarking.

template <typename T> class Queue {
private:
  struct Node {
    Node(const T &val) : value(val) {}
    Node(T &&val) : value(std::move(val)) {}

    T value;
    std::atomic<Node *> next = nullptr;
  };

public:
  Queue() {
    Node *dummyNode = new Node(T{});
    m_head.store(dummyNode);
    m_tail.store(dummyNode);

    // Preallocates storage to reduce allocator noise during benchmarking
    m_allNodes.reserve(defaultNodesSize);

    m_allNodes.push_back(dummyNode);
  }

  ~Queue() { deferredMemoryReclamation(); }

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue &&) = delete;

  void enqueue(const T &value) {
    Node *newNode = new Node(value);
    Node *tail = nullptr;
    Node *tailNext = nullptr;

    while (true) {
      tail = m_tail.load();
      tailNext = tail->next.load();

      if (tail == m_tail.load()) {
        if (tailNext == nullptr) {
          // Try to link new node at the end
          if (tail->next.compare_exchange_weak(tailNext, newNode)) {
            break;
          }
        } else {
          // Tail is behind, help advance it
          m_tail.compare_exchange_weak(tail, tailNext);
        }
      }
    }

    // Try to swing tail to the inserted node
    m_tail.compare_exchange_strong(tail, newNode);

    std::lock_guard<std::mutex> lock(m_nodesMutex);
    m_allNodes.push_back(newNode);
  }

  void enqueue(T &&value) {
    Node *newNode = new Node(std::move(value));
    Node *tail = nullptr;
    Node *tailNext = nullptr;

    while (true) {
      tail = m_tail.load();
      tailNext = tail->next.load();

      if (tail == m_tail.load()) {
        if (tailNext == nullptr) {
          // Try to link new node at the end
          if (tail->next.compare_exchange_weak(tailNext, newNode)) {
            break;
          }
        } else {
          // Tail is behind, help advance it
          m_tail.compare_exchange_weak(tail, tailNext);
        }
      }
    }

    // Try to swing tail to the inserted node
    m_tail.compare_exchange_strong(tail, newNode);

    std::lock_guard<std::mutex> lock(m_nodesMutex);
    m_allNodes.push_back(newNode);
  }

  std::optional<T> dequeue() {
    Node *head = nullptr;
    Node *tail = nullptr;
    Node *headNext = nullptr;
    std::optional<T> returnValue = std::nullopt;
    while (true) {
      head = m_head.load();
      tail = m_tail.load();
      headNext = head->next.load();
      if (head == m_head.load()) {
        if (head == tail) {
          if (headNext == nullptr) {
            return std::nullopt;
          }
          m_tail.compare_exchange_weak(tail, headNext);
        } else {
          returnValue = headNext->value;
          if (m_head.compare_exchange_weak(head, headNext)) {
            break;
          }
        }
      }
    }

    return returnValue;
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
  std::atomic<Node *> m_head;
  std::atomic<Node *> m_tail;

  // Track allocated nodes for deferred reclamation
  std::vector<Node *> m_allNodes;
  std::mutex m_nodesMutex;

  // Initial reservation size to reduce reallocations during benchmarks
  static constexpr size_t defaultNodesSize = 100000 + 1; // + 1 for dummy node
};

} // namespace lockfree

} // namespace cpu