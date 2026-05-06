cmake_minimum_required(VERSION 3.10)
project(SpannAugmentor VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化：O3, 本机指令集优化, OpenMP支持
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -march=native -fopenmp -Wall -Wextra)
endif()

set(CORE_SOURCES
    src/spann_index.cpp
    src/spann_index_reader.cpp
    src/spann_index_builder.cpp
    src/spann_index_serializer.cpp
)

add_library(spann_core STATIC ${CORE_SOURCES})

# spann_aug 自身的头文件
target_include_directories(spann_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 复用主项目的 hnswlib (include/util/hnswlib/*.hpp)
# 通过将 include/util/ 加入搜索路径，使 #include "hnswlib/hnswlib.hpp" 可用
target_include_directories(spann_core
    SYSTEM PUBLIC
        ${CMAKE_SOURCE_DIR}/include/util
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spann_core PRIVATE OpenMP::OpenMP_CXX)
endif()
