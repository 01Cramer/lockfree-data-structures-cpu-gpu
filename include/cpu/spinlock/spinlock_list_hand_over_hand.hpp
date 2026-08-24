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

// Fine-grained (hand-over-hand / lock-coupling) sorted linked list implementing
// a set ADT. Identical to mutex::HandOverHandList except the per-node
// std::mutex is replaced by cpu::Spinlock -- the busy-wait synchronization
// category. Traversal couples locks so that concurrent operations on disjoint
// regions proceed in parallel (Bayer & Schkolnick, 1977).
//
// Invariants:
//   head -> ... -> tail  (head and tail are sentinels, not part of the set)
//   strictly sorted by key
//   no duplicate keys
// See node_pool.hpp for the memory-model rationale and cache.hpp for the
// Layout parameter.

template <typename T, Layout L> class alignas(cacheLineSize) HandOverHandList {
private:
  struct Node {
    T key{};
    Node *next = nullptr;
    Spinlock lock;
  };
  struct Window {
    Node *pred;
    Node *curr;
    std::unique_lock<Spinlock> predLock;
    std::unique_lock<Spinlock> currLock;
  };

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  // The lock lives inside each Node, so Layout::padLockFromData has no global
  // lock word to move. Nor should it move the per-node one: padding a mutex off
  // its key would add a cache line to every node, inflating the footprint many
  // times over and confounding the comparison the layout sweep exists for.
  static constexpr bool hasLockWord = false;

  HandOverHandList(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads) {
    m_headSentinel.next = &m_tailSentinel;
  }

  HandOverHandList(const HandOverHandList &) = delete;
  HandOverHandList &operator=(const HandOverHandList &) = delete;
  HandOverHandList(HandOverHandList &&) = delete;
  HandOverHandList &operator=(HandOverHandList &&) = delete;

  bool insert(const T &key, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->key = key;

    Window window = search(key);
    if (window.curr != &m_tailSentinel && window.curr->key == key) {
      return false;
    }
    newNode->next = window.curr;
    window.pred->next = newNode;
    return true;
  }

  bool insert(T &&key, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->key = std::move(key);

    Window window = search(newNode->key);
    if (window.curr != &m_tailSentinel && window.curr->key == newNode->key) {
      return false;
    }
    newNode->next = window.curr;
    window.pred->next = newNode;
    return true;
  }

  bool remove(const T &key) {
    Window window = search(key);
    if (window.curr == &m_tailSentinel || window.curr->key != key) {
      return false;
    }
    window.pred->next = window.curr->next;
    return true;
  }

  bool find(const T &key) {
    Window window = search(key);
    return window.curr != &m_tailSentinel && window.curr->key == key;
  }

  // Structural validator for the tests (see cpu/shared/validation.hpp). Walks
  // the chain once and reports sortedness, termination and the element count --
  // none of which find() can observe.
  Validation validate(std::size_t maxSteps) {
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
  // Lock-coupling traversal. Returns with pred and curr locked, positioned so
  // that pred->key < key <= curr->key (or curr == tail sentinel).
  Window search(const T &key) {
    Node *pred = &m_headSentinel;
    std::unique_lock<Spinlock> predLock(pred->lock);

    Node *curr = pred->next;
    std::unique_lock<Spinlock> currLock(curr->lock);

    while (curr != &m_tailSentinel && curr->key < key) {
      // Release old pred; curr becomes the new pred (keep its lock), then lock
      // the new curr while still holding pred -- never overtaken.
      predLock = std::move(currLock);
      pred = curr;
      curr = curr->next;
      currLock = std::unique_lock<Spinlock>(curr->lock);
    }

    return {pred, curr, std::move(predLock), std::move(currLock)};
  }

private:
  NodePool<Node> m_pool;
  alignas(syncAlign(L)) Node m_headSentinel;
  alignas(syncAlign(L)) Node m_tailSentinel;
};

} // namespace spinlock

} // namespace cpu
