#include "fmt/core.h"
#include <utils/recall_utils.h>

namespace utils {
    float get_recall_by_id(const std::size_t kQueryNum, 
                           const std::size_t kGtSize, 
                           const unsigned *kGtIds,
                           hnsw_res_t &res) {
        float correct_num = 0;
        std::size_t r_size = res[0].size();
        // std::cout << res[0].size() << " " << kQueryNum << std::endl;
        for (std::size_t q=0; q<kQueryNum; ++q) {
            std::vector<bool> flag(r_size, true);
            for (; !res[q].empty(); res[q].pop()) {
                const auto& r=res[q].top();
                for (std::size_t i=0; i<r_size; ++i) {
                    if (flag[i] && r.second == kGtIds[i]) {
                        ++correct_num;
                        flag[i] = false;
                        break;
                    }
                }
            }
            // std::cout << correct_num << std::endl;
            kGtIds += kGtSize;
        }
        return correct_num / (kQueryNum*r_size);
    }

    float get_recall_by_val(const std::size_t kQueryNum,
                            const std::size_t kGtSize,
                            const float *kGtVal,
                            const unsigned topk,
                            std::vector<std::vector<utils::HeapItem<float, unsigned>>> &res) {
        float result = 0;
        for (std::size_t q=0; q<kQueryNum; ++q) {
            float gt_val = kGtVal[q*kGtSize+topk-1];
            for (std::size_t k=topk; k>0; --k) {
                if (res[q][k-1].dist_ <= gt_val) {
                    result += (float)k / topk;
                    break;
                }
            }
        }
        return result / kQueryNum;
    }

    float get_recall(const float kGtLower,
                     const unsigned kTopK,
                     const ResultPool<float> &result) {
        for (std::size_t i=kTopK; i>0; --i) {
            if (result.pool_[i-1].dist_ <= kGtLower) {
                return (float)i / kTopK;
            }
        }
        return 0;
    }

    float get_recall_by_id(const std::size_t kQueryNum,
                          const std::size_t kGtSize,
                          const unsigned *kGtIds,
                          const unsigned kTopK,
                          std::vector<utils::ResultPool<float>> &res) {
        float recall = 0;
        for (int i=0; i<kQueryNum; ++i) {
            const unsigned *gt = kGtIds + i * kGtSize;
            for (int k=0; k<kTopK; ++k) {
                for (int j=0; j<kTopK; ++j) {
                    if (res[i].pool_[j].id_ == gt[k]) {
                        ++recall;
                        break;
                    }
                }
            }
        }
        return recall / kQueryNum / kTopK;
    }

    float get_recall(const std::size_t kQueryNum,
                     const std::size_t kGtSize,
                     const float *kGtVal,
                     const unsigned kTopK,
                     std::vector<utils::ResultPool<float>> &res) {
        float recall = 0;
        for (size_t i=0; i<kQueryNum; ++i) {
            float gt = kGtVal[i*kGtSize+kTopK-1];
            for (unsigned k=kTopK; k>0; --k) {
                if (res[i].pool_[k-1].dist_ <= gt) {
                    recall += (float)k / (float)kTopK;
                    break;
                }
            }
        }
        return recall / kQueryNum;
    }

    float get_recall_by_val(const std::size_t kQueryNum,
                            const std::size_t kGtSize,
                            const float *kGtVal,
                            const unsigned kTopK,
                            std::vector<utils::ResultPool<float>> &res) {
        float recall = 0;
        for (size_t i=0; i<kQueryNum; ++i) {
            const float *gt = kGtVal + i * kGtSize;
            const float kGtLower = gt[kTopK-1];
            for (std::size_t k=kTopK; k>0; --k) {
                if (res[i].pool_[k-1].dist_ <= kGtLower) {
                    recall += (float)k / kTopK;
                }
            }
            // std::vector<bool> flag(kTopK, true);
            // const float *gt = kGtVal + i * kGtSize;
            // for (unsigned k=0; k<kTopK; ++k) {
            //     for (unsigned j=0; j<kTopK; ++j) {
            //         if (flag[k] && fabs(res[i].pool_[j].dist_ - gt[k]) < 1e-5) {
            //             ++recall;
            //             flag[k] = false;
            //             break;
            //         }
            //     }
            // }
        }
        return recall / kQueryNum / kTopK;

    }

    float get_recall_by_val(const std::size_t kQueryNum,
                            const std::size_t kGtSize,
                            const float *kGtVal,
                            hnsw_res_t &res) {
        float correct_num = 0;
        std::size_t r_size = res[0].size();
        for (std::size_t q=0; q<kQueryNum; ++q) {
            std::vector<bool> flag(r_size, true);
            for (; !res[q].empty(); res[q].pop()) {
                const auto& r=res[q].top();
                for (std::size_t i=0; i<r_size; ++i) {
                    if (flag[i] && fabs(r.first - kGtVal[i])<1e-4) {
                        ++correct_num;
                        flag[i] = false;
                        break;
                    }
                }
            }
            kGtVal += kGtSize;
        }
        return correct_num / (kQueryNum*r_size);
    }

    float get_recall(const std::size_t kQueryNum,
                     const std::size_t kGtSize,
                     const float *kGtVal,
                     const unsigned *kGtIds,
                     hnsw_res_t &res) {
        float correct_num = 0;
        std::size_t r_size = res[0].size();
        for (std::size_t q=0; q<kQueryNum; ++q) {
            std::vector<bool> flag(r_size, true);
            for (; !res[q].empty(); res[q].pop()) {
                const auto& r=res[q].top();
                for (std::size_t i=0; i<r_size; ++i) {
                    if (flag[i] && ((r.second == kGtIds[i]) || fabs(r.first - kGtVal[i])<1e-4)) {
                        ++correct_num;
                        flag[i] = false;
                        break;
                    }
                }
            }
            kGtVal += kGtSize;
            kGtIds += kGtSize;
        }
        return correct_num / (kQueryNum*r_size);
    }

}