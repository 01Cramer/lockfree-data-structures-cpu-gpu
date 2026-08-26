cmake_minimum_required(VERSION 3.20)

project(ConcurrentDataStructures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Build type (Release / Debug / RelWithDebInfo / MinSizeRel)" FORCE)
  message(STATUS
      "CMAKE_BUILD_TYPE was not set; defaulting to Release. "
      "Pass -DCMAKE_BUILD_TYPE=<type> to override.")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(PROJECT_TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

set(CPU_HEADERS
    "${PROJECT_INCLUDE_DIR}/cpu/shared/cache.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/shared/node_pool.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/shared/validation.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/mutex/mutex_stack.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/mutex/mutex_queue.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/mutex/mutex_queue_two_lock.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/mutex/mutex_list.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/mutex/mutex_list_hand_over_hand.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockfree/lockfree_stack.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockfree/lockfree_queue.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockfree/lockfree_list.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/spinlock/spinlock.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/spinlock/spinlock_stack.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/spinlock/spinlock_queue.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/spinlock/spinlock_queue_two_lock.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/spinlock/spinlock_list.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/spinlock/spinlock_list_hand_over_hand.hpp"
)

add_executable(CPU_COMPILATION
    ${PROJECT_TESTS_DIR}/cpu_compilation_test.cpp
    ${CPU_HEADERS}
)

target_include_directories(CPU_COMPILATION PRIVATE ${PROJECT_INCLUDE_DIR})
target_link_libraries(CPU_COMPILATION PRIVATE Threads::Threads)

# --- Correctness tests ---

option(ENABLE_TSAN "Build the correctness tests with ThreadSanitizer" OFF)

set(TEST_SUPPORT
    "${PROJECT_TESTS_DIR}/support/test_harness.hpp"
    "${PROJECT_TESTS_DIR}/support/baseline_types.hpp"
)

function(configure_cpu_test target)
  target_include_directories(${target}
      PRIVATE ${PROJECT_INCLUDE_DIR} ${PROJECT_TESTS_DIR})
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(ENABLE_TSAN)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(${target} PRIVATE -fsanitize=thread -g -O1)
      target_link_options(${target} PRIVATE -fsanitize=thread)
    else()
      message(WARNING
          "ENABLE_TSAN ignored: ThreadSanitizer needs Clang/GCC.")
    endif()
  endif()
endfunction()

add_executable(cpu_sequential_test
    ${PROJECT_TESTS_DIR}/cpu_sequential_test.cpp
    ${TEST_SUPPORT}
    ${CPU_HEADERS}
)
configure_cpu_test(cpu_sequential_test)

add_executable(cpu_stress_test
    ${PROJECT_TESTS_DIR}/cpu_stress_test.cpp
    ${TEST_SUPPORT}
    ${CPU_HEADERS}
)
configure_cpu_test(cpu_stress_test)

enable_testing()
add_test(NAME cpu_sequential COMMAND cpu_sequential_test)
add_test(NAME cpu_stress COMMAND cpu_stress_test)

# --- Benchmarks ---

# Energy measurement reads the Intel RAPL counters through powercap sysfs,
# which exists only on Linux and (since the Platypus disclosure) only for root.
# Off by default so the development build stays portable; turn it on for the
# measurement runs on the server.
option(ENABLE_RAPL
    "Measure energy via the Linux powercap/RAPL interface (needs root)" OFF)

# Deliberately fatal rather than a warning, unlike ENABLE_TSAN above. Ignoring
# the knob would produce a complete-looking sweep with a silently missing
# energy column, and that is only discovered after the run.
if(ENABLE_RAPL AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
      "ENABLE_RAPL requires Linux; there is no powercap sysfs on "
      "${CMAKE_SYSTEM_NAME}. Note that WSL does not expose it either, so the "
      "energy runs must happen on the server.")
endif()

set(PROJECT_BENCHMARKS_DIR ${CMAKE_SOURCE_DIR}/benchmarks)

# Header-only, so these are listed for dependency tracking and IDE visibility
# rather than compilation: editing one has to trigger a rebuild.
set(BENCHMARK_SUPPORT
    "${PROJECT_BENCHMARKS_DIR}/support/energy_meter.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/experiment.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/latency.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/perf_counters.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/thread_cpu.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/topology.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/variants.hpp"
    "${PROJECT_BENCHMARKS_DIR}/support/workload.hpp"
)

function(configure_benchmark target)
  target_include_directories(${target}
      PRIVATE ${PROJECT_INCLUDE_DIR} ${PROJECT_BENCHMARKS_DIR})
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(ENABLE_RAPL)
    target_compile_definitions(${target} PRIVATE CPU_ENABLE_RAPL)
  endif()
endfunction()

add_executable(energy_probe
    ${PROJECT_BENCHMARKS_DIR}/energy_probe.cpp
    ${BENCHMARK_SUPPORT}
)
configure_benchmark(energy_probe)

# The measurement harnesses need Google Benchmark; the energy probe above does
# not, so it stays available on a machine that has not installed it.
option(ENABLE_BENCHMARKS "Build the Google Benchmark measurement harnesses" ON)

if(ENABLE_BENCHMARKS)
  find_package(benchmark CONFIG REQUIRED)

  # Debug information on an optimized build. Needed for symbolized `perf`
  # output on the PMU counters, and it does not change codegen -- unlike
  # switching to RelWithDebInfo, which would silently drop -O3 to -O2.
  function(configure_measurement target)
    configure_benchmark(${target})
    target_link_libraries(${target} PRIVATE benchmark::benchmark)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      target_compile_options(${target} PRIVATE -g)
    endif()
  endfunction()

  foreach(adt stack queue list)
    add_executable(bench_${adt}
        ${PROJECT_BENCHMARKS_DIR}/bench_${adt}.cpp
        ${BENCHMARK_SUPPORT}
        ${CPU_HEADERS}
    )
    configure_measurement(bench_${adt})
  endforeach()
endif()

# --- GPU (CUDA) ---
#
# Off by default, and the CPU half must keep building on a machine with no
# toolkit: the CUDA work is developed and run on a different host, so the
# common case here is a checkout with no nvcc in sight.
option(ENABLE_CUDA "Build the GPU (CUDA) targets" OFF)

if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
        "ENABLE_CUDA is ON but no CUDA compiler was found. Put nvcc on PATH, "
        "or point CMake at it with -DCMAKE_CUDA_COMPILER=<path to nvcc>.")
  endif()
  enable_language(CUDA)

  # C++17 rather than the CPU half's 20. nvcc's device-side C++20 support
  # arrived late and still varies by toolkit version, and nothing under
  # include/gpu needs it. Keeping the device standard lower costs nothing and
  # removes a class of "works on my toolkit" failure from the remote loop.
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Volta is a hard floor, not a preference. Below compute 7.0 the threads of a
  # warp share a program counter, so a lane holding a spinlock cannot reach the
  # release while its peers spin: both lock-based variants deadlock
  # unconditionally when multiple lanes in one warp contend. Building for a
  # lower architecture would produce a binary that hangs rather than one that
  # runs slowly, so it is
  # refused here as well as static_asserted in include/gpu/shared/gpu_atomics.cuh.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS
        "CMAKE_CUDA_ARCHITECTURES was not set; defaulting to 70 (Volta). Set "
        "it to the target device's compute capability for the measurement "
        "runs -- e.g. 86 for Ampere GeForce, 89 for Ada, 90 for Hopper.")
  endif()

  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^[0-9]+" arch_number "${arch}")
    if(arch_number AND arch_number LESS 70)
      message(FATAL_ERROR
          "CMAKE_CUDA_ARCHITECTURES contains ${arch}, below the compute 7.0 "
          "floor. The lock-based queue variants require independent thread "
          "scheduling and would deadlock, not merely run slowly.")
    endif()
  endforeach()

  set(GPU_HEADERS
      "${PROJECT_INCLUDE_DIR}/gpu/shared/gpu_atomics.cuh"
      "${PROJECT_INCLUDE_DIR}/gpu/shared/gpu_cuda_utils.cuh"
      "${PROJECT_INCLUDE_DIR}/gpu/shared/gpu_node_pool.cuh"
      "${PROJECT_INCLUDE_DIR}/gpu/shared/gpu_spinlock.cuh"
      "${PROJECT_INCLUDE_DIR}/gpu/spinlock/gpu_spinlock_queue.cuh"
      "${PROJECT_INCLUDE_DIR}/gpu/spinlock/gpu_spinlock_queue_two_lock.cuh"
      "${PROJECT_INCLUDE_DIR}/gpu/lockfree/gpu_lockfree_queue.cuh"
  )

  set(GPU_TEST_SUPPORT
      "${PROJECT_TESTS_DIR}/support/gpu_test_harness.cuh"
      "${PROJECT_TESTS_DIR}/support/gpu_queue_oracles.hpp"
      "${PROJECT_BENCHMARKS_DIR}/support/gpu_queue_workload.cuh"
  )

  function(configure_gpu_target target)
    target_include_directories(${target} PRIVATE ${PROJECT_INCLUDE_DIR})
    # -lineinfo so compute-sanitizer reports a source line instead of an
    # address. It does not change codegen, unlike -G, which disables device
    # optimization entirely and would make every measurement meaningless.
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endfunction()

  foreach(gpu_test gpu_spinlock_progress_test gpu_queue_test)
    add_executable(${gpu_test}
        ${PROJECT_TESTS_DIR}/${gpu_test}.cu
        ${GPU_TEST_SUPPORT}
        ${TEST_SUPPORT}
        ${GPU_HEADERS}
    )
    configure_gpu_target(${gpu_test})
    target_include_directories(${gpu_test}
        PRIVATE ${PROJECT_TESTS_DIR} ${PROJECT_BENCHMARKS_DIR})
  endforeach()

  add_executable(gpu_bench_queue
      ${PROJECT_BENCHMARKS_DIR}/gpu_bench_queue.cu
      ${PROJECT_BENCHMARKS_DIR}/support/gpu_queue_workload.cuh
      ${PROJECT_BENCHMARKS_DIR}/support/gpu_workload.cuh
      ${PROJECT_BENCHMARKS_DIR}/support/gpu_energy.cuh
      ${GPU_HEADERS}
  )
  configure_gpu_target(gpu_bench_queue)
  target_include_directories(gpu_bench_queue
      PRIVATE ${PROJECT_BENCHMARKS_DIR})

  # --- Energy (NVML) ---
  #
  # On by default, unlike the CPU side's ENABLE_RAPL. There the energy build was
  # a separate binary because reading RAPL needs a readable sysfs path that the
  # development machine does not have; NVML ships with the driver that is
  # already required to run any of this, so there is no configuration where the
  # GPU targets build but NVML plausibly does not.
  #
  # Not fatal when missing, also unlike ENABLE_RAPL: a throughput-only run is
  # still a useful run, and the binary reports on stderr which mode it opened
  # in. Every energy column is then empty rather than zero, so a run without a
  # meter cannot be mistaken for a run that measured no energy.
  option(ENABLE_NVML "Measure GPU energy with NVML" ON)

  if(ENABLE_NVML)
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::nvml)
      target_compile_definitions(gpu_bench_queue PRIVATE GPU_ENABLE_NVML)
      target_link_libraries(gpu_bench_queue PRIVATE CUDA::nvml)
      message(STATUS "NVML energy measurement: enabled")
    else()
      message(WARNING
          "ENABLE_NVML is ON but the CUDA::nvml target was not found. Building "
          "without energy measurement; throughput is unaffected. On Linux the "
          "library is libnvidia-ml.so from the driver package.")
    endif()
  endif()

  # Ordered deliberately: the spinlock progress test is the precondition for
  # the lock-based queues, so if it fails there is no point reading the queue
  # results.
  add_test(NAME gpu_spinlock_progress COMMAND gpu_spinlock_progress_test)
  add_test(NAME gpu_queue COMMAND gpu_queue_test)
endif()
