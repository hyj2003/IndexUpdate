cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(P2H LANGUAGES CXX)

find_package(OpenMP REQUIRED)
set(ENV{MKLROOT} "/opt/intel/oneapi/mkl/latest")
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
message(STATUS "${MKL_IMPORTED_TARGETS}")


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-std=c++20)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-std=c++20 -march=native  -ftree-vectorize -ftree-vectorizer-verbose=0)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(-std=c++20)
endif ()

include(FetchContent)

FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(fmt)


# add_subdirectory(utils)
# add_subdirectory(ivf)
add_subdirectory(hnsw)
add_subdirectory(FreshDISKANN)
add_subdirectory(nsg)
# add_subdirectory(efanna_graph)
# add_subdirectory(SPFresh)