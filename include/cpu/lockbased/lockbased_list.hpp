#pragma once

#include <cstddef>
#include <mutex>
#include <utility>

#include "cpu/node_pool.hpp"

namespace cpu {

namespace lockbased {

// Lock-based sorted linked list implementing a set ADT, structurally
// aligned with the list used in Timothy L. Harris's paper
// "A Pragmatic Implementation of Non-Blocking Linked Lists".
//
// Invariants:
//   head -> ... -> tail  (head and tail are sentinels, not part of the set)
//   strictly sorted by key
//   no duplicate keys
//
// A single mutex serialises list operations. Nodes are drawn from a NodePool
// outside the critical section; head and tail sentinels are non-pool members.
// A duplicate insert() consumes one slice slot — size the pool accordingly
// for duplicate-heavy workloads.
// See node_pool.hpp for the memory-model rationale.

template <typename T> class List {
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
  List(std::size_t nodesPerThread, std::size_t numThreads)
      : m_pool(nodesPerThread, numThreads) {
    m_headSentinel.next = &m_tailSentinel;
  }

  List(const List &) = delete;
  List &operator=(const List &) = delete;
  List(List &&) = delete;
  List &operator=(List &&) = delete;

  bool insert(const T &key) {
    Node *newNode = m_pool.acquire();
    newNode->key = key;

    const std::lock_guard<std::mutex> lock(m_listMutex);
    auto [left, right] = search(key);
    if (right != &m_tailSentinel && right->key == key) {
      return false;
    }
    newNode->next = right;
    left->next = newNode;
    return true;
  }

  bool insert(T &&key) {
    Node *newNode = m_pool.acquire();
    newNode->key = std::move(key);

    const std::lock_guard<std::mutex> lock(m_listMutex);
    auto [left, right] = search(newNode->key);
    if (right != &m_tailSentinel && right->key == newNode->key) {
      return false;
    }
    newNode->next = right;
    left->next = newNode;
    return true;
  }

  bool remove(const T &key) {
    const std::lock_guard<std::mutex> lock(m_listMutex);
    auto [left, right] = search(key);

    if (right == &m_tailSentinel || right->key != key) {
      return false;
    }

    left->next = right->next;

    return true;
  }

  bool contains(const T &key) {
    const std::lock_guard<std::mutex> lock(m_listMutex);
    const auto result = search(key);
    const Node *right = result.right;
    return right != &m_tailSentinel && right->key == key;
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
  Node m_headSentinel;
  Node m_tailSentinel;
  std::mutex m_listMutex;
};

} // namespace lockbased

} // namespace cpu
