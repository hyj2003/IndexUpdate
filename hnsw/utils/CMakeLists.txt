file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)

add_library(utils STATIC ${srcs})
target_include_directories(utils PUBLIC include)
target_link_libraries(utils fmt OpenMP::OpenMP_CXX MKL::MKL)

#target_compile_options(utils PUBLIC -O0 -g -std=c++17 -fopenmp -march=native -ftree-vectorize -ftree-vectorizer-verbose=0)
#target_compile_options(utils PUBLIC -O3 -std=c++17 -fopenmp -march=native -ftree-vectorize -ftree-vectorizer-verbose=0)