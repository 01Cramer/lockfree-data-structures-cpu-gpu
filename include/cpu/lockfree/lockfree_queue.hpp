#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockfree {

// Lock-free Michael-Scott queue (Michael & Scott, 1996). A singly linked
// list with a dummy sentinel; head and tail are atomic pointers updated via
// compare-and-swap. Enqueue helps advance tail when it lags behind the last
// inserted node. Data nodes are drawn from a NodePool; the dummy sentinel
// is a non-pool member of the queue.
// See node_pool.hpp for the memory-model rationale.

template <typename T> class Queue {
private:
  struct Node {
    T value{};
    std::atomic<Node *> next{nullptr};
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
  }

  void enqueue(T &&value) {
    Node *newNode = m_pool.acquire();
    newNode->value = std::move(value);

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
          // Read by copy, not move: this happens BEFORE the CAS that claims
          // headNext, so multiple threads may read the same value concurrently.
          // Moving would let losers of the CAS leave headNext->value in a
          // moved-from state, corrupting the winner's result for non-trivial T.
          returnValue = headNext->value;
          if (m_head.compare_exchange_weak(head, headNext)) {
            break;
          }
        }
      }
    }

    return returnValue;
  }

private:
  NodePool<Node> m_pool;
  Node m_dummy;
  std::atomic<Node *> m_head;
  std::atomic<Node *> m_tail;
};

} // namespace lockfree

} // namespace cpu
