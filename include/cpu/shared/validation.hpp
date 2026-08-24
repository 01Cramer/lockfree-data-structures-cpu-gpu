#pragma once

#include <cstddef>
#include <cstdint>

namespace cpu {

// Result of a list's structural validator (see List::validate in each variant)
struct Validation {
  // Elements strictly between the two sentinels.
  //
  // Fixed width on purpose: `long` is 32-bit under MSVC and 64-bit under the
  // Linux ABI, and this project builds on both. A field whose range depends on
  // the toolchain does not belong in the one check that exists to catch
  // structural corruption.
  std::int64_t count = 0;

  // Keys strictly increase along the chain.
  bool sorted = true;

  // The walk reached the tail sentinel within the caller's step budget. A false
  // here means a cycle, or a chain that ran off into a null next.
  bool terminated = true;

  // No reachable node carries the delete mark. Lock-free variant only; always
  // true for the lock-based variants, which unlink under a lock and have no
  // mark bit. A false means a logically-deleted node is still reachable after
  // the structure went quiescent, i.e. a physical unlink was lost.
  bool noMarked = true;
};

} // namespace cpu
