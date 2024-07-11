#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <limits>
#include <atomic>
#include <utils/kmeans.h>
#include <utils/dist_func.h>
#include <fmt/core.h>
#include <fmt/chrono.h>

namespace utils{
void kmeans(std::vector<std::vector<float>>& train,
            std::vector<std::vector<float>>& centroid,
            unsigned& cluster_num,
            const unsigned kmeans_iter);

void kmeans_assign(const std::vector<std::vector<float>>& centroid,
                   std::vector<std::vector<float>>& data,
                   std::vector<std::vector<unsigned>>& bucket_ids,
                   std::vector<uint8_t>& id);

void kmeans_assign(const std::vector<std::vector<float>> &centroid,
                   std::vector<std::vector<float>> &data,
                   std::vector<std::vector<unsigned int>> &bucket_ids,
                   std::vector<unsigned> &id);

void kmeans(const float* data,
            const size_t data_num,
            const size_t data_dim,
            std::vector<std::vector<float> >& centroid,
            unsigned& cluster_num,
            const unsigned kmeans_iter);

void cos_kmeans(
    std::vector<std::vector<float>>& train,
    std::vector<std::vector<float>>& centroid,
    unsigned cluster_num,
    const unsigned kmeans_iter);

int split_bucket(
    std::vector<std::vector<float>>& t_centroid,
    const std::vector<std::vector<float>>& train,
    const std::vector<unsigned>& split_ivf);

int split_bucket(
    std::vector<std::vector<float>>& t_centroid,
    const std::vector<std::vector<float>>& train,
    const std::vector<unsigned>& split_ivf);
} // namespace utils
