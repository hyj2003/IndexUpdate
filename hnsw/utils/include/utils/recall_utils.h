#pragma once

#include <vector>
#include <queue>
#include <cmath>
#include <utils/result_utils.h>
#include <utils/heap.h>

typedef std::vector<std::priority_queue<std::pair<float, std::size_t>>> hnsw_res_t;

namespace utils {
    float get_recall_by_id(const std::size_t kQueryNum, 
                           const std::size_t kGtSize, 
                           const unsigned *kGtIds,
                           hnsw_res_t &res);


    float get_recall_by_val(const std::size_t kQueryNum,
                            const std::size_t kGtSize,
                            const float *kGtVal,
                            const unsigned topk,
                            std::vector<std::vector<utils::HeapItem<float, unsigned>>> &res);


    float get_recall(const float kGtLower,
                     const unsigned kTopK,
                     const ResultPool<float> &result);


    float get_recall(const std::size_t kQueryNum,
                     const std::size_t kGtSize,
                     const float *kGtVal,
                     const unsigned topk,
                     std::vector<utils::ResultPool<float>> &res);


    float get_recall_by_val(const std::size_t kQueryNum,
                            const std::size_t kGtSize,
                            const float *kGtVal,
                            hnsw_res_t &res);

    float get_recall_by_id(const std::size_t kQueryNum,
                          const std::size_t kGtSize,
                          const unsigned *kGtIds,
                          const unsigned kTopK,
                          std::vector<utils::ResultPool<float>> &res);

    float get_recall_by_val(const std::size_t kQueryNum,
                            const std::size_t kGtSize,
                            const float *kGtVal,
                            const unsigned kTopK,
                            std::vector<utils::ResultPool<float>> &res);

    float get_recall(const std::size_t kQueryNum,
                     const std::size_t kGtSize,
                     const float *kGtVal,
                     const unsigned *kGtIds,
                     hnsw_res_t &res);

    float get_recall_by_val(
      const std::vector<std::vector<float>> &kGtVal,
      const unsigned kTopK,
      const std::vector<utils::ResultPool<float>> &res);
}