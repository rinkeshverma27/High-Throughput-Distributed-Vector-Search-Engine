cmake_minimum_required(VERSION 3.16)
project(VectorDB VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra)

include_directories(include)

add_library(vectordb_core STATIC
    src/engine/distance.cpp
    src/engine/hnsw_index.cpp
    src/engine/ivf_index.cpp
    src/storage/wal.cpp
    src/cluster/consistent_hash.cpp
)

add_executable(vectordb src/server/main.cpp)
target_link_libraries(vectordb PRIVATE vectordb_core pthread)

add_executable(demo examples/demo_semantic_search.cpp)
target_link_libraries(demo PRIVATE vectordb_core pthread)

enable_testing()
add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE vectordb_core pthread)
add_test(NAME DistanceTest COMMAND test_distance)

add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE vectordb_core pthread)
add_test(NAME HNSWTest COMMAND test_hnsw)

add_executable(test_ivf tests/test_ivf.cpp)
target_link_libraries(test_ivf PRIVATE vectordb_core pthread)
add_test(NAME IVFTest COMMAND test_ivf)
