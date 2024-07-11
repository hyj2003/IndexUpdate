#pragma once

#include <vector>
#include <tuple>
#include <chrono>
// #include <span>
#include <iostream>
#include <map>
#include <random>

template <int max_track, typename ClockType = std::chrono::high_resolution_clock, typename Rep = double, typename Period = std::nano>
struct TrackTimer {

typename ClockType::time_point start_time_, end_time_;
std::optional<std::tuple<int, typename ClockType::time_point>> last_time_;
std::array<std::chrono::duration<Rep, Period>, max_track> durations_;

TrackTimer() {
    durations_.fill(std::chrono::duration<Rep, Period>(0));
}

void track(int track_type) {
    auto time = ClockType::now();
    if (last_time_) {
        auto& [last_track_type, last_time] = *last_time_;
        durations_[last_track_type] += time - last_time;
    } else {
        start_time_ = time;
    }
    // last_time_ = std::make_tuple(track_type, ClockType::now());
    // end_time_ = ClockType::now();
    last_time_ = std::make_tuple(track_type, time);
}

void end() {
    auto time = ClockType::now();
    if (last_time_) {
        auto& [last_track_type, last_time] = *last_time_;
        durations_[last_track_type] += time - last_time;
    }
    end_time_ = time;
}

void print() {
    using namespace std::string_view_literals;
    std::string_view s = "s"sv;
    if constexpr (std::is_same_v<Period, std::milli>) {
        s = "ms"sv;
    } else if constexpr (std::is_same_v<Period, std::micro>) {
        s = "us"sv;
    } else if constexpr (std::is_same_v<Period, std::nano>) {
        s = "ns"sv;
    }
    int i = 0;
    for (auto& duration: durations_) {
        std::cout << "Track type: " << i << " duration: " << duration.count() << " " << s << "\n";
        ++i;
    }
    std::cout << "Total duration: " << std::chrono::duration<Rep, Period>(end_time_ - start_time_).count() << " " << s << "\n";
}

};