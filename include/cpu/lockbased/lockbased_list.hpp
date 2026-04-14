#pragma once

#include <mutex>
#include <utility>
#include <vector>

namespace cpu {

namespace lockbased {

// Lock-based sorted linked-list implementing a set abstract data type,
// structurally aligned with the List used in Timothy L. Harris paper
// "A Pragmatic Implementation of Non-Blocking Linked Lists".
// Invariant:
// head -> ... -> tail
// List is strictly sorted
// No duplicate keys
// Sentinel nodes are not part of the logical set
// Using a single mutex for synchronization.
// The structure owns all allocated nodes and uses deferred memory reclamation.
// Memory is not freed during execution to ensure consistency with lock-free
// implementations, where safe reclamation is more complex.
// This design isolates synchronization costs and enables fair benchmarking.

template <typename T> class List {
private:
  struct Node {
    Node(const T &key) : key(key) {}
    Node(T &&key) : key(std::move(key)) {}

    T key = {};
    Node *next = nullptr;
  };

  struct SearchResult {
    Node *left;
    Node *right;
  };

public:
  List() {
    m_head = new Node(T{});
    m_tail = new Node(T{});
    m_head->next = m_tail;

    // Preallocates storage to reduce allocator noise during benchmarking
    m_allNodes.reserve(defaultNodesSize);

    m_allNodes.push_back(m_head);
    m_allNodes.push_back(m_tail);
  }

  ~List() { deferredMemoryReclamation(); }

  List(const List &) = delete;
  List &operator=(const List &) = delete;
  List(List &&) = delete;
  List &operator=(List &&) = delete;

  bool insert(const T &key) {
    const std::lock_guard<std::mutex> lock(m_listMutex);
    auto [left, right] = search(key);
    if (right != m_tail && right->key == key) {
      return false;
    }

    Node *newNode = new Node(key);
    newNode->next = right;
    left->next = newNode;

    m_allNodes.push_back(newNode);
    return true;
  }

  bool insert(T &&key) {
    const std::lock_guard<std::mutex> lock(m_listMutex);
    auto [left, right] = search(key);
    if (right != m_tail && right->key == key) {
      return false;
    }

    Node *newNode = new Node(std::move(key));
    newNode->next = right;
    left->next = newNode;

    m_allNodes.push_back(newNode);
    return true;
  }

  bool remove(const T &key) {
    const std::lock_guard<std::mutex> lock(m_listMutex);
    auto [left, right] = search(key);

    if (right == m_tail || right->key != key) {
      return false;
    }

    left->next = right->next;

    return true;
  }

  bool contains(const T &key) {
    const std::lock_guard<std::mutex> lock(m_listMutex);
    const auto result = search(key);
    const Node *right = result.right;
    return right != m_tail && right->key == key;
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
  SearchResult search(const T &key) {
    Node *left = m_head;
    Node *right = left->next;

    while (right != m_tail && right->key < key) {
      left = right;
      right = right->next;
    }

    return {left, right};
  }

private:
  Node *m_head; // Sentinel node
  Node *m_tail; // Sentinel node

  std::mutex m_listMutex;

  // Track allocated nodes for deferred reclamation
  std::vector<Node *> m_allNodes;

  // Initial reservation size to reduce reallocations during benchmarks
  static constexpr size_t defaultNodesSize =
      100000 + 2; // + 2 for head and tail sentinels
};

} // namespace lockbased

} // namespace cpu