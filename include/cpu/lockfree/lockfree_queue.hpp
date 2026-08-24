#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"

namespace cpu {

namespace lockfree {

// Lock-free Michael-Scott queue (Michael & Scott, 1996).
// A singly linked list with a dummy sentinel; The dummy sentinel
// is a non-pool member of the queue.
// Data nodes are drawn from a NodePool.
// See node_pool.hpp for the memory-model rationale.

template <typename T, Layout L> class alignas(cacheLineSize) Queue {
private:
  struct Node {
    T value{};
    std::atomic<Node *> next{nullptr};
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = false;

  Queue(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads), m_head(&m_dummy), m_tail(&m_dummy) {
  }

  Queue(const Queue &) = delete;
  Queue &operator=(const Queue &) = delete;
  Queue(Queue &&) = delete;
  Queue &operator=(Queue &&) = delete;

  void enqueue(const T &value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = value;
    newNode->next.store(nullptr, std::memory_order_relaxed);

    Node *tail = nullptr;
    Node *tailNext = nullptr;

    while (true) {
      tail = m_tail.load(std::memory_order_relaxed);
      tailNext = tail->next.load(std::memory_order_relaxed);

      if (tail == m_tail.load(std::memory_order_relaxed)) {
        if (tailNext == nullptr) {
          // Try to link new node at the end
          if (tail->next.compare_exchange_weak(tailNext, newNode,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
            break;
          }
        } else {
          // Tail is behind, help advance it (pointer hint only)
          m_tail.compare_exchange_weak(tail, tailNext,
                                       std::memory_order_relaxed);
        }
      }
    }

    // Try to swing tail to the inserted node (pointer hint only)
    m_tail.compare_exchange_strong(tail, newNode, std::memory_order_relaxed);
  }

  void enqueue(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = std::move(value);
    newNode->next.store(nullptr, std::memory_order_relaxed);

    Node *tail = nullptr;
    Node *tailNext = nullptr;

    while (true) {
      tail = m_tail.load(std::memory_order_relaxed);
      tailNext = tail->next.load(std::memory_order_relaxed);

      if (tail == m_tail.load(std::memory_order_relaxed)) {
        if (tailNext == nullptr) {
          // Try to link new node at the end
          if (tail->next.compare_exchange_weak(tailNext, newNode,
                                               std::memory_order_release,
                                               std::memory_order_relaxed)) {
            break;
          }
        } else {
          // Tail is behind, help advance it (pointer hint only)
          m_tail.compare_exchange_weak(tail, tailNext,
                                       std::memory_order_relaxed);
        }
      }
    }

    // Try to swing tail to the inserted node (pointer hint only)
    m_tail.compare_exchange_strong(tail, newNode, std::memory_order_relaxed);
  }

  std::optional<T> dequeue() {
    Node *head = nullptr;
    Node *tail = nullptr;
    Node *headNext = nullptr;
    std::optional<T> returnValue = std::nullopt;
    while (true) {
      head = m_head.load(std::memory_order_relaxed);
      tail = m_tail.load(std::memory_order_relaxed);
      // Acquire: pairs with the enqueue link-in release so that the plain
      // headNext->value read below is guaranteed to see the enqueued value.
      headNext = head->next.load(std::memory_order_acquire);
      if (head == m_head.load(std::memory_order_relaxed)) {
        if (head == tail) {
          if (headNext == nullptr) {
            return std::nullopt;
          }
          m_tail.compare_exchange_weak(tail, headNext,
                                       std::memory_order_relaxed);
        } else {
          // Read by copy, not move: this happens BEFORE the CAS that claims
          // headNext, so multiple threads may read the same value concurrently.
          // Moving would let losers of the CAS leave headNext->value in a
          // moved-from state, corrupting the winner's result for non-trivial T.
          returnValue = headNext->value;
          if (m_head.compare_exchange_weak(head, headNext,
                                           std::memory_order_relaxed)) {
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
  alignas(syncAlign(L)) std::atomic<Node *> m_head;
  alignas(syncAlign(L)) std::atomic<Node *> m_tail;
};

} // namespace lockfree

} // namespace cpu
