#pragma once
#include <memory_resource>
#include <vector>
#include <algorithm>
#include <limits>
#include <fmt/core.h>

namespace utils {
    template<typename DistType>
    struct Node {
        std::size_t id_;
        DistType dist_;

        explicit Node() : id_(0), dist_(0) {};

        explicit Node(std::size_t id, DistType dist) : id_(id), dist_(dist){}
    
        Node(unsigned c1, unsigned c2, DistType dist): dist_(dist) {
            id_= 0;
            id_ |= (1LL<<63);
            id_ += (c1<<16) + c2;
        }

        Node(DistType dist, unsigned idx): dist_(dist) {
            id_ = 0;
            id_ |= (1LL<<63);
            id_ += idx;
        }
        
        // Node(DistType dist): id_(1LL<<63), dist_(dist) {}

        inline void parse_id(int &c1, int &c2) {
            c2 = id_ & 0x0ffff;
            c1 = (id_>>16) & 0x0ffff;
        }

        inline void parse_id(int &idx) {
            idx = id_ & 0x0ffffffff;
        }
        
        inline int parse_id() const {
            return (int) id_ & 0x0ffffffff;
        }

        inline bool operator<(const Node &other) const {
            return dist_ < other.dist_;
        }
        
        inline bool first_bit() const {
            if ((id_ & (1LL<<63)) == 0) {
                return false;
            } else {
                return true;
            }
        }
    };


    template<typename DistType>
    struct MinHeap {
        constexpr bool operator()(Node<DistType> const&a, Node<DistType> const&b) const noexcept {
            return a.dist_ > b.dist_;
        }
    };


    template<typename DistType>
    struct MaxHeap {
        constexpr bool operator()(Node<DistType> const&a, Node<DistType> const&b) const noexcept {
            return a.dist_ < b.dist_;
        }
    };


    template<typename DistType>
    class ResultPool {
    public:
        int topk_ = 0;
        int size_ = 0;
        Node<DistType> *pool_ = nullptr;

        ResultPool() = default;
        ResultPool(unsigned topk) : topk_(topk), size_(0) {
            pool_ = new Node<DistType>[topk+1];
            // std::cout << "size: " << topk+1 << std::endl;
            // pool_.reserve(topk+1);
        }
        ResultPool(const ResultPool& other) : topk_(other.topk_), size_(other.size_) {
            pool_ = new Node<DistType>[other.topk_+1];
            std::copy(&other.pool_[0], &other.pool_[other.size_], &pool_[0]);
            // std::cout << "copy construction: " << " topk: " << topk_ << ", size: " << size_ << std::endl;
        }
        ResultPool(ResultPool&& other) :
            topk_(other.topk_),
            size_(other.size_),
            pool_(other.pool_) {
                other.pool_ = nullptr;
                other.topk_ = 0;
                other.size_ = 0;
                // std::cout << "move construction" << std::endl;
        }
        ResultPool& operator=(const ResultPool& other) {
            if (this != &other) {
                if (pool_!=nullptr) {
                    delete [] pool_;
                }
                topk_ = other.topk_;
                size_ = other.size_;
                std::copy(&other.pool_[0], &other.pool_[other.size_], &pool_[0]);
            }
            // std::cout << "operator=" << std::endl;
            return *this;
        }
        ResultPool& operator=(ResultPool&& other) {
            if (this != other) {
                if (pool_ != nullptr) {
                    delete [] pool_;
                }
                pool_ = other.pool_;
                topk_ = other.topk_;
                size_ = other.size_;

                other.pool_ = nullptr;
                other.topk_ = 0;
                other.size_ = 0;
            }
            // std::cout << "move operator=" << std::endl;
            return *this;
        }

        ~ResultPool() {
            if (pool_!=nullptr) {
                delete[] pool_;
            }
        }

        DistType max_dist() {
            if (size_ == 0) return std::numeric_limits<DistType>::max();
            return pool_[size_-1].dist_;
        }

        DistType topk_dist() {
            if (size_ == 0) return 0;
            if (size_ < topk_) {
                return pool_[size_ -1].dist_;
            }
            return pool_[topk_-1].dist_;
        }

        unsigned insert(std::size_t id, DistType dist) {
            if (size_ == topk_) [[likely]] {
                if (dist >= this->max_dist()) [[likely]] return topk_+1;
                if (pool_[0].dist_ > dist) {
                    std::copy_backward(&pool_[0], &pool_[topk_], &pool_[topk_+1]);
                    pool_[0].id_ = id;
                    pool_[0].dist_ = dist;
                    return 0;
                }
                int l=0, r=topk_;
                while(l < r) {
                    int mid = (l+r)/2;
                    if (pool_[mid].dist_ <= dist) {
                        l = mid+1;
                    } else {
                        r = mid;
                    }
                } // l is first pool_[l].dist >= dist
                std::copy_backward(&pool_[l], &pool_[topk_], &pool_[topk_+1]);
                pool_[l].id_ = id;
                pool_[l].dist_ = dist;
                return l;

                // if (dist < this->max_dist()) {
                //     int i;
                //     for (i=size_-1; i>=0; --i) {
                //         if (dist < pool_[i].dist_) {
                //             pool_[i+1] = pool_[i];
                //         } else {
                //             break;
                //         }
                //     }
                //     pool_[i+1].id_ = id;
                //     pool_[i+1].dist_ = dist;
                // }
            } else [[unlikely]] {
                int l=0, r=size_;
                while (l < r) {
                    int mid = (l+r)/2;
                    if (pool_[mid].dist_ <= dist) {
                        l = mid+1;
                    } else {
                        r = mid;
                    }
                }
                std::copy_backward(&pool_[l], &pool_[size_], &pool_[size_+1]);
                pool_[l].id_ = id;
                pool_[l].dist_ = dist;
                ++size_;
                return l;
                // int i;
                // for (i=size_-1; i>=0; --i) {
                //     if (dist < pool_[i].dist_) {
                //         pool_[i+1] = pool_[i];
                //     } else {
                //         break;
                //     }
                // }
                // pool_[i+1].id_ = id;
                // pool_[i+1].dist_ = dist;
                // ++size_;
            }
        }

        std::size_t get_id(int loc) {
            return pool_[loc].id_;
        }

        void print() {
            for (size_t i=0; i<size_; ++i) {
                fmt::print("(dist: {}, id: {}), ", pool_[i].dist_, pool_[i].id_);
            }
            fmt::print("\n");
        }
        
        void emplace(std::size_t id, DistType dist, std::size_t loc) {
            pool_[loc].id_  = id;
            pool_[loc].dist_ = dist;
            ++size_;
        }

        unsigned get_size() {
            return size_;
        }

        void sort() {
            std::sort(pool_, pool_+size_);
        }
    };

} // namespace utils