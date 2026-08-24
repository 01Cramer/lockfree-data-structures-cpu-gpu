// The correctness suites do not vary layout -- they check semantics, and no
// Layout cell changes those. So they name one cell here, once, instead of
// repeating it at every instantiation.
//
// cpu_compilation_test.cpp is the exception and must NOT use these aliases:
// its whole job is to instantiate every cell explicitly.

#pragma once

#include "cpu/lockfree/lockfree_list.hpp"
#include "cpu/lockfree/lockfree_queue.hpp"
#include "cpu/lockfree/lockfree_stack.hpp"
#include "cpu/mutex/mutex_list.hpp"
#include "cpu/mutex/mutex_list_hand_over_hand.hpp"
#include "cpu/mutex/mutex_queue.hpp"
#include "cpu/mutex/mutex_queue_two_lock.hpp"
#include "cpu/mutex/mutex_stack.hpp"
#include "cpu/spinlock/spinlock_list.hpp"
#include "cpu/spinlock/spinlock_list_hand_over_hand.hpp"
#include "cpu/spinlock/spinlock_queue.hpp"
#include "cpu/spinlock/spinlock_queue_two_lock.hpp"
#include "cpu/spinlock/spinlock_stack.hpp"

#include "cpu/shared/cache.hpp"

namespace cpu_test {

namespace baseline {

// The baseline is the algorithm as the papers publish it: no padding. The
// three padded cells are treatments measured against this one.
inline constexpr cpu::Layout layout = cpu::NoPad;

namespace mutex {

template <typename T> using Stack = cpu::mutex::Stack<T, layout>;
template <typename T> using Queue = cpu::mutex::Queue<T, layout>;
template <typename T> using QueueTwoLock = cpu::mutex::QueueTwoLock<T, layout>;
template <typename T> using List = cpu::mutex::List<T, layout>;
template <typename T>
using HandOverHandList = cpu::mutex::HandOverHandList<T, layout>;

} // namespace mutex

namespace spinlock {

template <typename T> using Stack = cpu::spinlock::Stack<T, layout>;
template <typename T> using Queue = cpu::spinlock::Queue<T, layout>;
template <typename T>
using QueueTwoLock = cpu::spinlock::QueueTwoLock<T, layout>;
template <typename T> using List = cpu::spinlock::List<T, layout>;
template <typename T>
using HandOverHandList = cpu::spinlock::HandOverHandList<T, layout>;

} // namespace spinlock

namespace lockfree {

template <typename T> using Stack = cpu::lockfree::Stack<T, layout>;
template <typename T> using Queue = cpu::lockfree::Queue<T, layout>;
template <typename T> using List = cpu::lockfree::List<T, layout>;

} // namespace lockfree

} // namespace baseline

} // namespace cpu_test
