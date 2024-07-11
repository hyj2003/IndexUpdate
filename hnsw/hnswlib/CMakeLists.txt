file(GLOB_RECURSE srcs CONFIGURE_DEPENDS include/*.h)

add_library(hnswlib STATIC ${srcs})
target_include_directories(hnswlib PUBLIC include)
#target_compile_options(hnswlib PUBLIC -O3 -std=c++17 -fopenmp -march=native -ftree-vectorize -ftree-vectorizer-verbose=0)
#target_compile_options(hnswlib PUBLIC -O0 -g -std=c++17 -fopenmp -march=native -ftree-vectorize -ftree-vectorizer-verbose=0)
target_compile_options(hnswlib PUBLIC -std=c++17 -fopenmp -march=native -ftree-vectorize -ftree-vectorizer-verbose=0)
target_link_libraries(hnswlib PUBLIC utils)
