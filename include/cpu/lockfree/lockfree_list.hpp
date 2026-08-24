#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "cpu/shared/cache.hpp"
#include "cpu/shared/node_pool.hpp"
#include "cpu/shared/validation.hpp"

namespace cpu {

namespace lockfree {

// Lock-free sorted linked list implementing a set ADT (Harris, "A Pragmatic
// Implementation of Non-Blocking Linked-Lists", 2001 -- often called the
// Harris-Michael list). Same set contract as mutex::List / spinlock::List:
//
// Invariants:
//   head -> ... -> tail  (head and tail are sentinels, not part of the set)
//   strictly sorted by key
//   no duplicate keys
//   insert / remove / find each return bool
// See node_pool.hpp for the memory-model rationale and cache.hpp for the
// Layout parameter.

template <typename T, Layout L> class alignas(cacheLineSize) List {
private:
  struct Node {
    T key{};
    std::atomic<Node *> next{nullptr};
  };

  static_assert(alignof(Node) > 1,
                "need a free low bit in Node* to steal for the delete mark");

  struct SearchResult {
    Node *left;
    Node *right;
  };

  static constexpr std::uintptr_t kMark = 1;

  static bool isMarked(Node *p) {
    return (reinterpret_cast<std::uintptr_t>(p) & kMark) != 0;
  }
  static Node *marked(Node *p) {
    return reinterpret_cast<Node *>(reinterpret_cast<std::uintptr_t>(p) |
                                    kMark);
  }
  static Node *unmarked(Node *p) {
    return reinterpret_cast<Node *>(reinterpret_cast<std::uintptr_t>(p) &
                                    ~kMark);
  }

public:
  static constexpr std::size_t nodeBytes = sizeof(Node);
  static constexpr bool hasLockWord = false;

  List(std::size_t getNodesPerThread, std::size_t numThreads)
      : m_pool(getNodesPerThread, numThreads) {
    m_head.next.store(&m_tail, std::memory_order_relaxed);
  }

  List(const List &) = delete;
  List &operator=(const List &) = delete;
  List(List &&) = delete;
  List &operator=(List &&) = delete;

  bool insert(const T &key, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->key = key;

    while (true) {
      const SearchResult window = search(key);
      Node *rightNode = window.right;
      if (rightNode != &m_tail && rightNode->key == key) { // T1: duplicate
        return false;
      }
      newNode->next.store(rightNode, std::memory_order_relaxed);
      // C2: publish newNode (release makes newNode->key visible to a traverser
      // that acquire-loads this next and reads the key).
      if (window.left->next.compare_exchange_strong(
              rightNode, newNode, std::memory_order_release,
              std::memory_order_relaxed)) {
        return true;
      }
      // CAS failed: left->next moved under us, retry (B3).
    }
  }

  bool insert(T &&key, std::size_t threadId) {
    Node *newNode = m_pool.takeNode(threadId);
    newNode->key = std::move(key);

    while (true) {
      const SearchResult window = search(newNode->key);
      Node *rightNode = window.right;
      if (rightNode != &m_tail && rightNode->key == newNode->key) { // T1
        return false;
      }
      newNode->next.store(rightNode, std::memory_order_relaxed);
      if (window.left->next.compare_exchange_strong( // C2
              rightNode, newNode, std::memory_order_release,
              std::memory_order_relaxed)) {
        return true;
      }
    }
  }

  bool remove(const T &key) {
    Node *leftNode = nullptr;
    Node *rightNode = nullptr;
    Node *rightNodeNext = nullptr;

    while (true) { // B4
      const SearchResult window = search(key);
      leftNode = window.left;
      rightNode = window.right;
      if (rightNode == &m_tail || rightNode->key != key) { // T1: absent
        return false;
      }
      // Acquire: rightNodeNext is republished by C4 below, so we must carry
      // its key's happens-before into this thread before re-releasing it.
      rightNodeNext = rightNode->next.load(std::memory_order_acquire);
      if (!isMarked(rightNodeNext)) {
        // C3: logical delete. Relaxed is sufficient -- this RMW extends the
        // release sequence on rightNode->next, so acquire loads of the marked
        // value still synchronize with rightNodeNext's original publisher.
        if (rightNode->next.compare_exchange_strong(
                rightNodeNext, marked(rightNodeNext), std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          break;
        }
      }
      // rightNode already marked, or its next changed: retry.
    }

    // C4: physical unlink (release republishes rightNodeNext). If it fails, a
    // concurrent search has changed left->next; trigger a search to physically
    // excise the marked node and move on -- the node is already logically gone.
    Node *expected = rightNode;
    if (!leftNode->next.compare_exchange_strong(expected, rightNodeNext,
                                                std::memory_order_release,
                                                std::memory_order_relaxed)) {
      search(rightNode->key);
    }
    return true;
  }

  bool find(const T &key) {
    const SearchResult window = search(key);
    return window.right != &m_tail && window.right->key == key;
  }

  // Structural validator for the tests (see cpu/shared/validation.hpp). This is
  // the variant it exists for: Harris is the only list here where a node can be
  // logically deleted but still reachable, and where the published algorithm's
  // correctness rests on two CASes that are each allowed to fail.
  Validation validate(std::size_t maxSteps) {
    Validation v;
    const Node *prev = nullptr;
    Node *cur = unmarked(m_head.next.load(std::memory_order_acquire));
    while (cur != &m_tail) {
      if (cur == nullptr || static_cast<std::size_t>(v.count) >= maxSteps) {
        v.terminated = false;
        return v;
      }
      Node *next = cur->next.load(std::memory_order_acquire);
      if (isMarked(next)) {
        v.noMarked = false;
      }
      if (prev != nullptr && !(prev->key < cur->key)) {
        v.sorted = false;
      }
      prev = cur;
      cur = unmarked(next);
      ++v.count;
    }
    return v;
  }

private:
  // Harris search (Fig. 5): returns (left, right) with left the last unmarked
  // node whose key < search key and right the first node with key >= search
  // key, guaranteeing right is the immediate successor of left (physically
  // unlinking any marked run between them via C1).
  SearchResult search(const T &key) {
    while (true) { // restart point (search_again / B2)
      Node *leftNode = &m_head;
      Node *leftNodeNext = nullptr;

      // 1: locate left and right.
      Node *t = &m_head;
      Node *tNext = m_head.next.load(std::memory_order_acquire);
      do {
        if (!isMarked(tNext)) {
          leftNode = t;
          leftNodeNext = tNext;
        }
        t = unmarked(tNext);
        if (t == &m_tail) {
          break;
        }
        tNext = t->next.load(std::memory_order_acquire);
      } while (isMarked(tNext) || t->key < key); // B1
      Node *rightNode = t;

      // 2: already adjacent?
      if (leftNodeNext == rightNode) {
        if (rightNode != &m_tail &&
            isMarked(rightNode->next.load(std::memory_order_relaxed))) {
          continue; // G1: right is being deleted, restart
        }
        return {leftNode, rightNode}; // R1
      }

      // 3: unlink the marked run leftNodeNext..rightNode in one CAS.
      if (leftNode->next.compare_exchange_strong(leftNodeNext, rightNode,
                                                 std::memory_order_release,
                                                 std::memory_order_relaxed)) {
        if (rightNode != &m_tail &&
            isMarked(rightNode->next.load(std::memory_order_relaxed))) {
          continue; // G2
        }
        return {leftNode, rightNode}; // R2
      }
      // C1 failed: list changed, restart (B2).
    }
  }

private:
  NodePool<Node> m_pool;
  alignas(syncAlign(L)) Node m_head;
  Node m_tail;
};

} // namespace lockfree

} // namespace cpu
