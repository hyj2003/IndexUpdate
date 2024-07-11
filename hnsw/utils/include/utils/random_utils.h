#pragma once

#include <random>
#include <vector>
#include <algorithm>

namespace utils
{
    int gen_rand(int n);

    float gen_rand_float(const float min, const float max);

    template <typename T>
    void shuffle_vector(std::vector<T>& vec) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(vec.begin(), vec.end(), g);
    }
} // namespace utils

