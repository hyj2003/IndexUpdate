#pragma once

#include <chrono>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>


namespace utils {
    template <typename ClockType = std::chrono::steady_clock, typename Rep = double, typename Period = std::nano>
    class TimerOpt {
        ClockType::time_point time_begin_;
        std::vector<std::chrono::duration<Rep, Period>> elapsed_time_;
        
        // std::chrono::duration<Rep, Period> elapsed_time_{0};
        // std::chrono::duration<double> elapsed_time_{0};

    public:
        TimerOpt() = default;

        inline std::chrono::duration<Rep, Period> getElapsedTime() {
            auto time_end = ClockType::now();
            return time_end - time_begin_;
        }

        void tick() {

        }

        inline void end() {
            const auto time_end = ClockType::now();
            // elapsed_time_ += time_end - time_begin_;
            elapsed_time_.emplace_back(time_end - time_begin_);
        }

        inline std::chrono::duration<Rep, Period> total() {
            // return elapsed_time_;
            return std::accumulate(elapsed_time_.begin(), elapsed_time_.end(), std::chrono::duration<Rep, Period>(0));
        }
        // inline double total() {
        //     return elapsed_time_.count();
        // }

        inline void reset() {
            time_begin_ = ClockType::now();
        }
    };

    template <typename TimerType>
    class TimerGuard {
        TimerType& timer_;

    public:
        TimerGuard(TimerType& timer): timer_(timer) {
            timer_.reset();
        }
        TimerGuard(const TimerGuard&) = delete;
        TimerGuard(TimerGuard&&) = delete;
        TimerGuard& operator=(const TimerGuard&) = delete;
        TimerGuard& operator=(TimerGuard&&) = delete;
        ~TimerGuard() {
            timer_.end();
        }
    };
}