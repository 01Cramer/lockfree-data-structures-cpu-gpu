// Cache-line layout is a first-order term in the measured cost of a concurrent
// data structures. This header makes layout explicit, identical across
// variants, and selectable at compile time.

#pragma once

#include <cstddef>
#include <new>

namespace cpu {

#ifdef __cpp_lib_hardware_interference_size
inline constexpr std::size_t cacheLineSize =
    std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t cacheLineSize = 64; // bytes
#endif

// Two orthogonal layout knobs.
//
// padSyncPoints -- each synchronization point starts on its own cache line.
//
// padLockFromData -- within a synchronization point, the lock word additionally
// gets its own line, separate from the pointer(s) it guards.

struct Layout {
  bool padSyncPoints;
  bool padLockFromData;
};

inline constexpr Layout NoPad{false, false};
inline constexpr Layout PadLockFromData{false, true};
inline constexpr Layout PadSyncPoints{true, false};
inline constexpr Layout PadSyncPointsAndLockFromData{true, true};

// alignas() arguments derived from a Layout. alignas(0) is defined to have no
// effect, so the "off" setting costs nothing in size or alignment.

inline constexpr std::size_t syncAlign(Layout layout) {
  return layout.padSyncPoints ? cacheLineSize : 0;
}

inline constexpr std::size_t lockAlign(Layout layout) {
  return layout.padLockFromData ? cacheLineSize : 0;
}

} // namespace cpu
