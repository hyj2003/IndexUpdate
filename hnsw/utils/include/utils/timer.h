#pragma once

#include <chrono>
#include <cmath>
#include <iostream>

namespace utils {
    template <typename T = std::chrono::nanoseconds>
    class Timer {
        std::chrono::steady_clock::time_point last_point_;
        // std::chrono::duration<double, T> elapsed_time_{0};
        std::chrono::steady_clock::duration elapsed_time_{0};
        // double e_t{0};

    public:
        Timer() = default;

        inline T getElapsedTime() {
            // auto time_end = std::chrono::steady_clock::now();
            // return std::chrono::duration_cast<T>(time_end - time_begin_);
            return std::chrono::duration_cast<T>(elapsed_time_);
        }

        inline void reset() {
            elapsed_time_ = std::chrono::steady_clock::duration::zero();
            last_point_ = std::chrono::steady_clock::now();
        }

        inline void start() {
            last_point_ = std::chrono::steady_clock::now();
        }


        inline void end() {
            const auto time_end = std::chrono::steady_clock::now();
            elapsed_time_ += time_end - last_point_;
            // e_t += std::chrono::duration_cast<T>(time_end - time_begin_).count();
            // if (std::fabs(elapsed_time_.count()*1e9 - e_t) > 1) {
            //     std::cout << "not equal, e_t: " << e_t 
            //               << ", elapsed_time: " << elapsed_time_.count() << std::endl;
            // }
        }


        inline double total() {
            return elapsed_time_.count();
        }
    };
}
