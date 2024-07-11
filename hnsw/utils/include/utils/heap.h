#pragma once

#include <vector>
#include <algorithm>
#include <iostream>

namespace utils {
    template<typename DistType, typename IdType>
    struct HeapItem {
        DistType dist_;
        IdType id_;

        HeapItem() = default;
        HeapItem(DistType dist, IdType id) : dist_(dist), id_(id) {}

        inline bool operator<(const HeapItem &other) const {
            return dist_ < other.dist_;
        }
    };


    template<typename DistType, typename IdType>
    struct HeapMax {
        constexpr bool 
        operator()(HeapItem<DistType, IdType> const&a, HeapItem<DistType, IdType> const&b) const noexcept {
            return a.dist_ < b.dist_;
        }
    };


    template<typename DistType, typename IdType>
    struct HeapMin {
        constexpr bool 
        operator()(HeapItem<DistType, IdType> const&a, HeapItem<DistType, IdType> const&b) const noexcept {
            return a.dist_ > b.dist_;
        }
    };


    template<typename DistType, typename IdType>
    class Heap {
    public:
        std::vector<HeapItem<DistType, IdType> > mass_;
        IdType max_size_;
        Heap(IdType max_size) : max_size_(max_size) {
            mass_.reserve(max_size);
        }

        void push(DistType dist, IdType id) {
            if (mass_.size() < max_size_) {
                mass_.emplace_back(dist, id);
                std::push_heap(mass_.begin(), mass_.end());
            } else if (dist < mass_[0].dist_) {
                mass_[0].dist_ = dist;
                mass_[0].id_ = id;
                std::size_t idx = 0;
                std::size_t child_idx = 2 * idx + 1; // left child
                // std::size_t child_idx = 2 * (idx + 1);
                while (child_idx <= mass_.size() - 1) {
                    if (child_idx+1<=mass_.size()-1 && mass_[child_idx]<mass_[child_idx+1]) {
                        ++child_idx;
                    }
                    if (mass_[idx] < mass_[child_idx]) {
                        std::swap(mass_[child_idx], mass_[idx]);
                    } else {
                        break;
                    }
                    idx = child_idx;
                    child_idx = 2 * idx + 1;
                }
            }
        }

        // HeapItem<DistType, IdType>
        // pop() {

        // }

        DistType top_dist() {
            return mass_[0].dist_;
        }

        DistType get_dist(const std::size_t idx) {
            return mass_[idx].dist_;
        }

        std::size_t size() {
            return mass_.size();
        }

        bool fill() {
            return mass_.size() < max_size_ ? false : true;
        }

        std::vector<HeapItem<DistType, IdType> >
        get_topk(int topk) {
            std::vector<HeapItem<DistType, IdType> > result(topk);
            std::partial_sort_copy(mass_.begin(), mass_.end(), result.begin(), result.end());
            return result;
        }

        void print() {
            for (const auto &a:mass_) {
                std::cout << "(" << a.dist_ << ", " << a.id_ << ") ";
            }
            std::cout << std::endl;
        }
    };
} // namespace utils