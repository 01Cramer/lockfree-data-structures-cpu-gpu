// The Spinlock below is adapted from Fedor Pikus's "LockFree" repository
// (https://github.com/fpikus/LockFree).
// The adaptive Spinlock class benchmarked as the winner in his
// "Lock-Free Programming and Concurrent Data Structures on Modern Hardware"
// talk. It is used and modified here under the terms of its MIT license:

//
//   MIT License
//   Copyright (c) 2026 fpikus
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the "Software"),
//   to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.

#pragma once

#include <atomic>
#include <thread>

// Backoff primitive. Fedor's original escalates to nanosleep({0,1}), a POSIX
// call that deschedules the thread via the OS scheduler. nanosleep does not
// exist on Windows/MSVC, so there we fall back to std::this_thread::yield().
// Benchmarks run on Linux (faithful nanosleep); the Windows path is for local
// dev / smoke tests only.
#ifndef _WIN32
#include <time.h>
#endif

namespace cpu {

namespace detail {

// One backoff step, taken after the spin budget is exhausted. Both variants
// hand the core back to the OS scheduler.
inline void spinlockBackoff() {
#ifdef _WIN32
  std::this_thread::yield();
#else
  static const timespec ns = {0, 1};
  ::nanosleep(&ns, nullptr);
#endif
}

} // namespace detail

class Spinlock {
public:
  Spinlock() = default;

  Spinlock(const Spinlock &) = delete;
  Spinlock &operator=(const Spinlock &) = delete;
  Spinlock(Spinlock &&) = delete;
  Spinlock &operator=(Spinlock &&) = delete;

  void lock() {
    for (int i = 0; m_locked.load(std::memory_order_relaxed) ||
                    m_locked.exchange(1, std::memory_order_acquire);
         ++i) {
      if (i == kSpinBudget) {
        i = 0;
        detail::spinlockBackoff();
      }
    }
  }

  void unlock() { m_locked.store(0, std::memory_order_release); }

private:
  static constexpr int kSpinBudget = 8;
  std::atomic<unsigned int> m_locked{0};
};

} // namespace cpu
