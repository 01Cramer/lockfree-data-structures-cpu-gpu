cmake_minimum_required(VERSION 3.20)

project(ConcurrentDataStructures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(PROJECT_TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

set(CPU_HEADERS
    "${PROJECT_INCLUDE_DIR}/cpu/node_pool.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_stack.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_queue.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_queue_two_lock.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_list.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockfree/lockfree_stack.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockfree/lockfree_queue.hpp"
)

add_executable(CPU_COMPILATION
    ${PROJECT_TESTS_DIR}/cpu_compilation_test.cpp
    ${CPU_HEADERS}
)

target_include_directories(CPU_COMPILATION PRIVATE ${PROJECT_INCLUDE_DIR})
