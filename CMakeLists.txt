cmake_minimum_required(VERSION 3.20)

project(ConcurrentDataStructures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(PROJECT_TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

set(CPU_HEADERS
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_stack.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_queue.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockbased/lockbased_list.hpp"
    "${PROJECT_INCLUDE_DIR}/cpu/lockfree/lockfree_stack.hpp"
)

add_executable(CPU
    ${PROJECT_TESTS_DIR}/cpu_compilation_test.cpp
    ${CPU_HEADERS}
)

target_include_directories(CPU PRIVATE ${PROJECT_INCLUDE_DIR})
