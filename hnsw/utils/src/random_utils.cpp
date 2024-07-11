#include <random>
#include <utils/random_utils.h>

namespace utils {

int gen_rand(int n) {
    // std::random_device rd;  // a seed source for the random number engine

    static std::mt19937 gen(0); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, n);

    return distrib(gen);
}

float gen_rand_float(const float min, const float max) {
    static std::mt19937 gen(0);
    std::uniform_real_distribution<float> dis(min, max);
    return dis(gen);
}

} // namespace utils


