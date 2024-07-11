add_executable(main main.cpp)
target_link_libraries(main hnswlib utils OpenMP::OpenMP_CXX)

add_executable(search search.cpp)
target_link_libraries(search hnswlib utils OpenMP::OpenMP_CXX)

add_executable(delete_test delete_test.cpp)
target_link_libraries(delete_test hnswlib utils OpenMP::OpenMP_CXX)

# SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O3 -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )
