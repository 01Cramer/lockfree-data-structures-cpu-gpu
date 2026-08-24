#pragma once

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"

namespace cpu {

namespace mutex {

// Lock-based FIFO queue guarded by a single mutex. Nodes are drawn from
// a NodePool outside the critical section; the mutex protects only the
// linked-list update.
//
// A dummy sentinel (a non-pool Node member) keeps m_head and m_tail
// non-nullable, so enqueue and dequeue never branch on emptiness. This
// matches the structural shape of QueueTwoLock and lockfree::Queue,
// isolating the locking strategy as the only meaningful difference
// between the three queue variants.
// See node_pool.hpp for the memory-model rationale and cache.hpp for the
// Layout parameter. One synchronization point: m_queueMutex guards both ends,
// so m_head and m_tail form a single group and only padLockFromData can split
// the mutex away from them.

template <typename T, Layout L> class alignas(cacheLineSize) Queue {
private:
  struct Node {
    T value{};
    Node *next = nullptr;
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = true;

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
    newNode->next = nullptr;

    const std::lock_guard<std::mutex> lock(m_queueMutex);
    m_tail->next = newNode;
    m_tail = newNode;
  }

  void enqueue(T &&value, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->value = std::move(value);
    newNode->next = nullptr;

    const std::lock_guard<std::mutex> lock(m_queueMutex);
    m_tail->next = newNode;
    m_tail = newNode;
  }

  std::optional<T> dequeue() {
    const std::lock_guard<std::mutex> lock(m_queueMutex);
    Node *next = m_head->next;
    if (next == nullptr) {
      return std::nullopt;
    }
    std::optional<T> result(std::move(next->value));
    m_head = next;
    return result;
  }

private:
  NodePool<Node> m_pool;
  Node m_dummy;
  alignas(syncAlign(L)) Node *m_head;
  Node *m_tail;
  alignas(lockAlign(L)) std::mutex m_queueMutex;
};

} // namespace mutex

} // namespace cpu
