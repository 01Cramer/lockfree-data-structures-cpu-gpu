// Compilation smoke test: instantiates every CPU structure with int and
// exercises each public operation once.

#include "cpu/lockbased/lockbased_list.hpp"
#include "cpu/lockbased/lockbased_queue.hpp"
#include "cpu/lockbased/lockbased_queue_two_lock.hpp"
#include "cpu/lockbased/lockbased_stack.hpp"
#include "cpu/lockfree/lockfree_queue.hpp"
#include "cpu/lockfree/lockfree_stack.hpp"

int main() {
  constexpr std::size_t nodesPerThread = 16;
  constexpr std::size_t numThreads = 1;

  cpu::lockbased::Stack<int> lbStack(nodesPerThread, numThreads);
  cpu::lockbased::Queue<int> lbQueue(nodesPerThread, numThreads);
  cpu::lockbased::QueueTwoLock<int> lbQueueTwoLock(nodesPerThread, numThreads);
  cpu::lockbased::List<int> lbList(nodesPerThread, numThreads);
  cpu::lockfree::Stack<int> lfStack(nodesPerThread, numThreads);
  cpu::lockfree::Queue<int> lfQueue(nodesPerThread, numThreads);

  lbStack.push(1);
  (void)lbStack.pop();

  lbQueue.enqueue(1);
  (void)lbQueue.dequeue();

  lbQueueTwoLock.enqueue(1);
  (void)lbQueueTwoLock.dequeue();

  (void)lbList.insert(1);
  (void)lbList.contains(1);
  (void)lbList.remove(1);

  lfStack.push(1);
  (void)lfStack.pop();

  lfQueue.enqueue(1);
  (void)lfQueue.dequeue();

  return 0;
}
