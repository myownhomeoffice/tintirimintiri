# RBTreeMapNH - Unit Tests

This repository contains a GoogleTest-based unit test suite for the RBTreeMapNH red-black tree map implementation (node_handle with allocator propagation).

Files added:
- `CMakeLists.txt` - top-level CMake config which fetches GoogleTest and builds tests.
- `tests/rbtree_map_tests.cpp` - comprehensive test suite.
- `README.md` - this file.

Prerequisites:
- CMake >= 3.14
- A C++17-capable compiler (g++/clang++/MSVC)
- Network access to fetch GoogleTest via CMake's FetchContent (or you can provide a local GoogleTest installation).

Project layout:
- Put the RBTreeMapNH header (e.g. `red_black_tree_map_nodehandle.hpp`) into `include/`.
  - Example: `include/red_black_tree_map_nodehandle.hpp`

Build & run tests:
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
ctest --output-on-failure
# or run the test executable directly:
# ./rbtree_map_tests
