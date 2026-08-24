// Design rationale
//
// Every shared-state access in the GPU data structures goes through the
// aliases in this header. Three reasons:
//
//   1. Scope is the decision that is easy to get silently wrong. A grid-wide
//      queue is read and written by threads in different blocks, so every
//      atomic on it must be thread_scope_device. thread_scope_block compiles,
//      is measurably cheaper (the operation can be serviced in the SM rather
//      than at the L2), and is wrong -- it would produce a faster number for a
//      structure that does not work. Naming the scope once, here, removes the
//      opportunity to write it per call site.
//
//   2. The CPU half of this project uses std::atomic with explicit acquire /
//      release and a documented "minimal justified fence" convention. libcu++
//      gives the device the same vocabulary, so the convention transfers
//      verbatim and the two halves can be read side by side.
//
//   3. It is the one place to record what we deliberately do NOT do:
//      no bare __threadfence(), and no `volatile`. See the note below.
//
// On volatile and __threadfence
//
// A recurring pattern in the GPU queue literature (Zhang et al., IEICE 2014)
// is to mark the shared arrays `volatile` and use no fences at all. volatile
// suppresses caching of the access, which is a code-generation property; it
// orders nothing. In that code nothing separates the write of a node's payload
// from the CAS that links the node into the list, so a consumer may observe a
// linked node whose value has not landed. A release on the linking CAS is
// exactly the fence that is missing, and it is what the queues here use.
//
// Bare __threadfence() would also work, but it is a full fence in both
// directions at every use, whereas acquire/release says which direction is
// needed and lets the compiler emit the cheaper of the two. Since the point of
// the project is to measure the cost of synchronization, the synchronization
// should not be gratuitously stronger than the algorithm requires.

#pragma once

#if !__has_include(<cuda/atomic>)
#error "The GPU targets need libcu++ (<cuda/atomic>), shipped with the CUDA \
Toolkit since 10.2. Check that the CUDA include directory is on the path."
#endif

#include <cuda/atomic>

namespace gpu {

// Independent thread scheduling (Volta, CC 7.0) is a hard requirement, not a
// preference. On earlier hardware the threads of a warp share one program
// counter, so a lane that acquires a spinlock cannot reach the release while
// its peers are still spinning in the same warp: the lock-based variants
// deadlock unconditionally at activeLanes > 1. That is why Cederman et al.
// (Euro-Par 2012) ran one operating thread per block -- and why the
// activeLanes axis this project sweeps was not available to them.
//
// __nanosleep (spinlock.cuh) also requires CC 7.0.
#ifdef __CUDA_ARCH__
static_assert(__CUDA_ARCH__ >= 700,
              "gpu:: requires compute capability 7.0 or newer (independent "
              "thread scheduling). Build with -DCMAKE_CUDA_ARCHITECTURES=70 "
              "or higher.");
#endif

// All shared state in a grid-wide structure is device-scoped. Block scope is
// never correct here; system scope would be paid for and unused (no host or
// peer-device thread touches the structure while a kernel is running).
template <typename T>
using DeviceAtomicRef = cuda::atomic_ref<T, cuda::thread_scope_device>;

// Separation granularity for independently-contended words.
//
// The CPU half of this project measured cache-line padding changing the
// *ranking* of the variants, not just their absolute cost, so the device side
// gets the same treatment rather than being left to chance. On NVIDIA GPUs the
// L2 cache line is 128 B (fetched in 32 B sectors), and device-scope atomics
// are executed at the L2, so two independently-contended words in one line are
// serialized through one L2 slice.
//
// Unlike the CPU side this is NOT a swept knob: it is applied identically to
// all three variants and held fixed, because the experiment varies the
// synchronization mechanism only. What it buys is that no variant is
// accidentally penalized by a layout accident.
inline constexpr int kDeviceCacheLineBytes = 128;

} // namespace gpu
