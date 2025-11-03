cmake_minimum_required(VERSION 3.14)
project(rbtree_map_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED RBTREE_MAP_INCLUDE_DIR)
  set(RBTREE_MAP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(rbtree_map_tests
  tests/rbtree_map_tests.cpp
  tests/rbtree_map_invariant_stress.cpp
)

target_include_directories(rbtree_map_tests PRIVATE
  ${RBTREE_MAP_INCLUDE_DIR}
)

target_link_libraries(rbtree_map_tests PRIVATE
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(rbtree_map_tests)
