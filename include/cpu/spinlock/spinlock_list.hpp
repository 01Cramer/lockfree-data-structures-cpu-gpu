#pragma once

#include <cstddef>
#include <mutex>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"
#include "cpu/shared/validation.hpp"
#include "cpu/spinlock/spinlock.hpp"

namespace cpu {

namespace spinlock {

// Spinlock-guarded sorted linked list implementing a set ADT. Identical to
// mutex::List except the single mutex is replaced by cpu::Spinlock -- the
// busy-wait synchronization category.
//
// Invariants:
//   head -> ... -> tail  (head and tail are sentinels, not part of the set)
//   strictly sorted by key
//   no duplicate keys
// See node_pool.hpp for the memory-model rationale and cache.hpp for the
// Layout parameter.

template <typename T, Layout L> class alignas(cacheLineSize) List {
private:
  struct Node {
    T key{};
    Node *next = nullptr;
  };

  struct SearchResult {
    Node *left;
    Node *right;
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = true;

  List(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads) {
    m_headSentinel.next = &m_tailSentinel;
  }

  List(const List &) = delete;
  List &operator=(const List &) = delete;
  List(List &&) = delete;
  List &operator=(List &&) = delete;

  bool insert(const T &key, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->key = key;

    const std::lock_guard<Spinlock> lock(m_listLock);
    auto [left, right] = search(key);
    if (right != &m_tailSentinel && right->key == key) {
      return false;
    }
    newNode->next = right;
    left->next = newNode;
    return true;
  }

  bool insert(T &&key, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->key = std::move(key);

    const std::lock_guard<Spinlock> lock(m_listLock);
    auto [left, right] = search(newNode->key);
    if (right != &m_tailSentinel && right->key == newNode->key) {
      return false;
    }
    newNode->next = right;
    left->next = newNode;
    return true;
  }

  bool remove(const T &key) {
    const std::lock_guard<Spinlock> lock(m_listLock);
    auto [left, right] = search(key);

    if (right == &m_tailSentinel || right->key != key) {
      return false;
    }

    left->next = right->next;

    return true;
  }

  bool find(const T &key) {
    const std::lock_guard<Spinlock> lock(m_listLock);
    const auto result = search(key);
    const Node *right = result.right;
    return right != &m_tailSentinel && right->key == key;
  }

  // Structural validator for the tests (see cpu/shared/validation.hpp). Walks
  // the chain once and reports sortedness, termination and the element count --
  // none of which find() can observe.
  Validation validate(std::size_t maxSteps) {
    const std::lock_guard<Spinlock> lock(m_listLock);
    Validation v;
    const Node *prev = nullptr;
    const Node *cur = m_headSentinel.next;
    while (cur != &m_tailSentinel) {
      if (cur == nullptr || static_cast<std::size_t>(v.count) >= maxSteps) {
        v.terminated = false;
        return v;
      }
      if (prev != nullptr && !(prev->key < cur->key)) {
        v.sorted = false;
      }
      prev = cur;
      cur = cur->next;
      ++v.count;
    }
    return v;
  }

private:
  SearchResult search(const T &key) {
    Node *left = &m_headSentinel;
    Node *right = left->next;

    while (right != &m_tailSentinel && right->key < key) {
      left = right;
      right = right->next;
    }

    return {left, right};
  }

private:
  NodePool<Node> m_pool;
  alignas(syncAlign(L)) Node m_headSentinel;
  Node m_tailSentinel;
  alignas(lockAlign(L)) Spinlock m_listLock;
};

} // namespace spinlock

} // namespace cpu
