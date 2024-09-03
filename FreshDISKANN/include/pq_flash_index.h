#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <shared_mutex>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "windows_customizations.h"

#define MAX_N_CMPS 16384
#define SECTOR_LEN (uint64_t) 4096
#define MAX_N_SECTOR_READS 128
#define MAX_PQ_CHUNKS 480
#define SECTORS_PER_MERGE (uint64_t)65536

namespace diskann {
  template<typename T>
  struct QueryScratch {
    T *  coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_scratch = nullptr;  // MUST BE AT LEAST [aligned_dim]
    float *aligned_pqtable_dist_scratch =
        nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    // uint8_t *aligned_quanted_dist_scratch =
    //     nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch =
        nullptr;  // MUST BE AT LEAST diskann MAX_DEGREE
    _u8 *aligned_pq_coord_scratch =
        nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
    T *    aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;

    void reset() {
      coord_idx = 0;
      sector_idx = 0;
    }
  };

  template<typename T>
  struct DiskNode {
    uint32_t  id = 0;
    T *       coords = nullptr;
    uint32_t  nnbrs;
    uint32_t *nbrs;

    // id : id of node
    // sector_buf : sector buf containing `id` data
    DiskNode(uint32_t id, T *coords, uint32_t *nhood);
  };

  template<typename T>
  struct ThreadData {
    QueryScratch<T> scratch;
    IOContext       ctx;
  };
  template<typename T>
  struct InsertCache {
    // std::pair<std::pair<_u32, _u32 *>, _u32>: <nbrs, nbrs*, vis_num>
    // NOT thread safe
    struct Node {
      _u32 cnt_ = 0;
      std::pair<_u32, _u32 *> nei_;
      T *coord_ = nullptr;
      Node() {}
      Node(_u32 cnt, 
           const std::pair<_u32, _u32 *> &nei, 
           T *coord) : cnt_(cnt), nei_(nei), coord_(coord) {}
    };
    tsl::robin_map<_u32, Node> insert_cache_;
    std::vector<std::vector<Node>> insert_search_info_;
    std::queue<_u32> q_;
    std::queue<_u32 *> q_u32_;
    std::queue<T *> q_T_;
    std::vector<std::queue<std::pair<_u32 *, T*>>> thread_buffer_;
    std::mutex mu_;
    _u32 *u32_buffer_;
    T *T_buffer_;
    int cap_ = 0, cur_ = 0, dim_ = 0, max_degree_; // nodes to cache, current cached number
    _u32 max_thread_size_ = 0, L_size_ = 200; // max thread size
    InsertCache(_u32 cap, _u32 max_thread_size, int dim, int max_degree) 
    : cap_(cap), dim_(dim), max_degree_(max_degree), max_thread_size_(max_thread_size) {
      insert_cache_.reserve(cap);
      insert_search_info_.reserve(max_thread_size);
      insert_search_info_.resize(max_thread_size);
      for (_u32 i = 0; i < max_thread_size; i++) {
        insert_search_info_[i].reserve(cap);
      }
      _u32 cap_add = cap + max_thread_size * L_size_;
      u32_buffer_ = new _u32[max_degree * cap_add];
      T_buffer_ = new T[dim * cap_add];
      for (_u32 i = 0; i < cap_add; i++) {
        q_u32_.emplace(u32_buffer_ + i * max_degree);
        q_T_.emplace(T_buffer_ + i * dim);
      }
      thread_buffer_.reserve(max_thread_size);
      thread_buffer_.resize(max_thread_size);
      for (_u32 i = 0; i < max_thread_size; i++) {
        while (thread_buffer_[i].size() < L_size_) {
          thread_buffer_[i].emplace(q_u32_.front(), q_T_.front());
          q_u32_.pop(), q_T_.pop();
        }
      }
    }
    ~InsertCache() {
      // for (auto &iter : insert_cache_) {
      //   delete iter.second.nei_.second;
      //   delete iter.second.coord_;
      // }
      delete u32_buffer_;
      delete T_buffer_;
    }
    void ClearSearchInfo() {
      for (_u32 i = 0; i < max_thread_size_; i++) {
        std::vector<Node> empty_vector;
        insert_search_info_[i].swap(empty_vector);
        while (thread_buffer_[i].size() < L_size_) {
          thread_buffer_[i].emplace(q_u32_.front(), q_T_.front());
          q_u32_.pop(), q_T_.pop();
        }
      }
    }
    void Evict() {
      while (!q_.empty() && cur_ > cap_) {
        _u32 p = q_.front(); q_.pop();
        auto iter = insert_cache_.find(p);
        assert(iter != insert_cache_.end() && iter->second.cnt_ != 0);
        if (iter->second.cnt_ == 1) {
          // delete iter->second.nei_.second;
          // delete iter->second.coord_;
          q_u32_.emplace(iter->second.nei_.second);
          q_T_.emplace(iter->second.coord_);
          insert_cache_.erase(iter);
          cur_--;
        } else {
          insert_cache_[iter->first] = Node(iter->second.cnt_ - 1, iter->second.nei_, iter->second.coord_);
        }
      }
    }
    void Insert(_u32 thread_id, _u32 node_id, const std::pair<_u32, _u32 *> &nei, T *coord) {
      auto iter = insert_cache_.find(node_id);
      if (iter == insert_cache_.end()) {
        // _u32 *n = new _u32[nei.first];
        // T    *c = new T[dim_];
        _u32 *n = nullptr;
        T    *c = nullptr;
        // {
        //   std::lock_guard guard(mu_);
        //   n = q_u32_.front(); q_u32_.pop();
        //   c = q_T_.front(); q_T_.pop();
        // }
        n = thread_buffer_[thread_id].front().first;
        c = thread_buffer_[thread_id].front().second;
        thread_buffer_[thread_id].pop();
        memcpy(n, nei.second, sizeof(_u32) * nei.first);
        memcpy(c, coord, sizeof(T) * dim_);
        insert_search_info_[thread_id].emplace_back(Node(node_id, std::make_pair(nei.first, n), c));
      } else {
        insert_search_info_[thread_id].emplace_back(Node(node_id, iter->second.nei_, iter->second.coord_));
      }
      // Evict();
    }
    bool GetNeighbor(_u32 id, std::pair<_u32, _u32 *> &nei) {
      auto iter = insert_cache_.find(id);
      if (iter == insert_cache_.end()) {
        return false;
      } else {
        nei = iter->second.nei_;
      }
      return true;
    }
    bool GetCoord(_u32 id, T* &coord) {
      auto iter = insert_cache_.find(id);
      if (iter == insert_cache_.end()) {
        return false;
      } else {
        coord = iter->second.coord_;
      }
      return true;
    }
    // Remember to merge after each round of searching
    void MergeSearchInfo() {
      tsl::robin_map<_u32, Node> merged;
      for (_u32 i = 0; i < max_thread_size_; i++) {
        for (auto &info : insert_search_info_[i]) {
          // auto iter = insert_cache_.find(info.cnt_);
          // if (iter == insert_cache_.end()) {
          //   _u32 *nei = new _u32[info.nei_.first];
          //   T *coord = new T[dim_];
          //   insert_cache_[info.cnt_] = Node(_u32(1), std::make_pair(info.nei_.first, nei), coord);
          //   cur_++;
          //   Evict();
          // } else {
          //   insert_cache_[info.cnt_] = Node(iter->second.cnt_, iter->second.nei_, iter->second.coord_);
          // }
          // q_.emplace(info.cnt_);
          auto iter_i = insert_cache_.find(info.cnt_);
          auto iter_m = merged.find(info.cnt_);
          if (iter_i != insert_cache_.end()) {
            insert_cache_[info.cnt_] = Node(iter_i->second.cnt_ + 1, iter_i->second.nei_, iter_i->second.coord_);
          } else if (iter_m != merged.end()) {
            // delete info.nei_.second;
            // delete info.coord_;
            q_u32_.emplace(info.nei_.second);
            q_T_.emplace(info.coord_);
            merged[info.cnt_] = Node(iter_m->second.cnt_ + 1, iter_m->second.nei_, iter_m->second.coord_);
          } else {
            merged[info.cnt_] = Node(_u32(1), info.nei_, info.coord_);
            cur_++;
          }
          q_.emplace(info.cnt_);
        }
      }
      for (auto &iter : merged) {
        insert_cache_.insert(iter);
      }
      Evict();
      ClearSearchInfo();
    }
  };

  template<typename T, typename TagT = uint32_t>
  class PQFlashIndex {
   public:
    // Gopal. Adapting to the new Bing interface. Since the DiskPriorityIO is
    // now a singleton, we have to take it in the DiskANNInterface and
    // pass it around. Since I don't want to pollute this interface with Bing
    // classes, this class takes an AlignedFileReader object that can be
    // created the way we need. Linux will create a simple AlignedFileReader
    // and pass it. Regular Windows code should create a BingFileReader using
    // the DiskPriorityIOInterface class, and for running on XTS, create a
    // BingFileReader
    // using the object passed by the XTS environment.
    // Freeing the reader object is now the client's (DiskANNInterface's)
    // responsibility.
    DISKANN_DLLEXPORT PQFlashIndex(
        std::shared_ptr<AlignedFileReader> &fileReader);
    DISKANN_DLLEXPORT ~PQFlashIndex();

    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *pq_prefix,
                               const char *disk_index_file,
                               bool        load_tags = false);
    DISKANN_DLLEXPORT void load_tags(const std::string &tag_file);

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);
    
    DISKANN_DLLEXPORT _u64 return_nd();
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        std::string sample_bin, _u64 l_search, _u64 beamwidth,
        _u64 num_nodes_to_cache, std::vector<uint32_t> &node_list);

    DISKANN_DLLEXPORT void cache_bfs_levels(_u64 num_nodes_to_cache,
                                            std::vector<uint32_t> &node_list);

    //    DISKANN_DLLEXPORT void cache_from_samples(const std::string
    //    sample_file, _u64 num_nodes_to_cache, std::vector<uint32_t>
    //    &node_list);

    //    DISKANN_DLLEXPORT void save_cached_nodes(_u64        num_nodes,
    //                                             std::string cache_file_path);

    // setting up thread-specific data

    // implemented
    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr,
        Distance<T> *output_dist_func = nullptr, TagT *res_tags = nullptr);

    // DISKANN_DLLEXPORT void filtered_beam_search(
    //     const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
    //     float *res_dists, const _u64 beam_width, QueryStats *stats = nullptr,
    //     Distance<T> *output_dist_func = nullptr, TagT *res_tags = nullptr);
    std::shared_ptr<AlignedFileReader> reader = nullptr;

    /* diskv2 extra API requirements */
    /* --------------------------------------------------------------------------------------------
     */
    DISKANN_DLLEXPORT void disk_iterate_to_fixed_point(
        const T *vec, const uint32_t Lsize, const uint32_t beam_width,
        std::vector<Neighbor> &expanded_nodes_info,
        tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
        Distance<T> *output_dist_func = nullptr, QueryStats *stats = nullptr,
        ThreadData<T> *           passthrough_data = nullptr,
        tsl::robin_set<uint32_t> *exclude_nodes = nullptr);
    DISKANN_DLLEXPORT void cached_disk_iterate_to_fixed_point(
        const T *vec, const uint32_t Lsize, const uint32_t beam_width,
        std::vector<Neighbor> &expanded_nodes_info,
        tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
        Distance<T> *output_dist_func = nullptr, QueryStats *stats = nullptr,
        ThreadData<T> *           passthrough_data = nullptr,
        tsl::robin_set<uint32_t> *exclude_nodes = nullptr, const _u32 t_id = 0, const std::vector<_u32> &enters = std::vector<_u32>());
    // DISKANN_DLLEXPORT void filter_disk_iterate_to_fixed_point(
    //     const T *vec, const uint32_t Lsize, const uint32_t beam_width,
    //     std::vector<Neighbor> &expanded_nodes_info,
    //     tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
    //     Distance<T> *output_dist_func = nullptr, QueryStats *stats = nullptr,
    //     ThreadData<T> *           passthrough_data = nullptr,
    //     tsl::robin_set<uint32_t> *exclude_nodes = nullptr);
    // DISKANN_DLLEXPORT void Expand() 
    std::vector<uint32_t> get_init_ids() {
      return std::vector<uint32_t>(this->medoids,
                                   this->medoids + this->num_medoids);
    }
    // gives access to backing thread data buf for easy parallelization
    std::vector<ThreadData<T>> &get_thread_data() {
      return this->thread_data_backing_buf;
    }

    // computes PQ dists between src->[ids] into fp_dists (merge, insert)
    DISKANN_DLLEXPORT void compute_pq_dists(const _u32 src, const _u32 *ids,
                                            float *fp_dists, const _u32 count,
                                            uint8_t *aligned_scratch = nullptr);
    // computes PQ dists between aligned_query->[ids] into fp_dists (search)
    DISKANN_DLLEXPORT void compute_pq_dists(const T *query, const _u32 *ids,
                                            float *fp_dists, const _u32 count);

    // DISKANN_DLLEXPORT void compute_pq_dists_quanted(const T *query, const _u32 *ids,
    //                                                 float *fp_dists, const _u32 count,
    //                                                 uint16_t* scratch_u16);
    // read/write [start_id:end_id-1] points from disk
    // WARNING -- ensure (start_id,end_id) % nnodes_per_sector = 0,
    // aligned_scratch is SECTOR_LEN aligned
    // WARNING -- ensure aligned_scratch size is >((end_id -
    // start_id)/nnodes_per_sector) * SECTOR_LEN bytes
    DISKANN_DLLEXPORT _u32 merge_read(std::vector<DiskNode<T>> &disk_nodes,
                                      _u32 &start_id, const _u32 sector_count,
                                      char *scratch);
    DISKANN_DLLEXPORT void scan_deleted_nodes(
        const tsl::robin_set<uint32_t> &delete_set,
        std::vector<DiskNode<T>> &deleted_nodes, char *buf, char *backing_buf,
        const uint32_t sectors_per_scan);
    DISKANN_DLLEXPORT void reload_index(const std::string &new_disk_prefix,
                                        const std::string &new_pq_prefix,
                                        const uint32_t     new_max_pts);
    DISKANN_DLLEXPORT void passthrough_write(char *buf, const uint64_t offset,
                                             const uint64_t size);

    // deflates `vec` into PQ ids
    DISKANN_DLLEXPORT std::vector<_u8> deflate_vector(const T *vec);
    std::pair<_u8 *, _u32> get_pq_config() {
      return std::make_pair(this->data, (uint32_t) this->n_chunks);
    }
    TagT *get_tags() {
      return this->tags; 
    }

    DISKANN_DLLEXPORT SIMDPQTable<T>* GetPQTable() {
      return &pq_table;
    }
    DISKANN_DLLEXPORT _u64 GetDataDim() {
      return data_dim;
    }
    DISKANN_DLLEXPORT _u8* GetPQData() {
      return data;
    }
    DISKANN_DLLEXPORT InsertCache<T>* CreateInsertCache(_u32 num_to_cache, _u32 thread_num) {
      this->insert_cache_ = new InsertCache<T>(num_to_cache, thread_num, this->data_dim, this->max_degree);
      return this->insert_cache_;
    }
    DISKANN_DLLEXPORT void DeleteInsertCache() {
      delete this->insert_cache_;
    }
    void SetEntersNum(size_t number) {
      enters_ = number;
    }
    DISKANN_DLLEXPORT void CreateThreadPoints(size_t thread_num) {
      // insert_enters_.resize(thread_num);
      // insert_enters_.reserve(thread_num);
      insert_enters_.clear();
      insert_enters_tmp_.resize(thread_num);
      insert_enters_tmp_.reserve(thread_num);
      // insert_coords_.resize(thread_num * aligned_dim);
      // insert_coords_.reserve(thread_num * aligned_dim);
      insert_coords_.clear();
      insert_coords_tmp_.resize(thread_num * aligned_dim);
      insert_coords_tmp_.reserve(thread_num * aligned_dim);
    }
    DISKANN_DLLEXPORT void UpdateThreadPoints(size_t thread_num) {
      insert_enters_ = insert_enters_tmp_;
      insert_coords_ = insert_coords_tmp_;
      // for (size_t i = 0; i < thread_num; i++) {
      //   nearest_pt_[]
      // }
    }
    int GetNumMedoids() {
      return num_medoids;
    }
    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;
    _u64 cache_hit = 0, total_travel = 0;
    std::vector<_u32> traveled[100];
    size_t enters_;
    std::vector<_u32>  insert_enters_;
    std::vector<float> insert_coords_;
    std::vector<_u32>  insert_enters_tmp_;
    std::vector<float> insert_coords_tmp_;
    tsl::robin_map<_u32, _u32> nearest_pt_;
    double dist_1 = 0, dist_2 = 0;

    // void PushDelete(_u32 id, const std::pair<_u32, _u32 *> &p) {
    //   {
    //     std::unique_lock guard(delete_cache_lock_);
    //     delete_cache_.insert(std::make_pair(id, p));
    //   }
    //   del_filter_.set(id);
    // }

   protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

    // private:
    // data info
    _u64 num_points = 0;
    _u64 data_dim = 0;
    _u64 aligned_dim = 0;

    std::string disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *                data = nullptr;
    _u64                 chunk_size;
    _u64                 n_chunks;
    SIMDPQTable<T>       pq_table;
    // distance comparator
    std::shared_ptr<Distance<T>>     dist_cmp;
    std::shared_ptr<Distance<float>> dist_cmp_float;

    // medoid/start info
    uint32_t *medoids =
        nullptr;         // by default it is just one entry point of graph, we
                         // can optionally have multiple starting points
    size_t num_medoids;  // by default it is set to 1
    float *centroid_data =
        nullptr;  // by default, it is empty. If there are multiple
                  // centroids, we pick the medoid corresponding to the
                  // closest centroid as the starting point of search

    // nhood_cache
    unsigned *nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;

    // coord_cache
    T *coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;
    
    // insert_cache
    InsertCache<T> *insert_cache_ = nullptr;
    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    std::vector<ThreadData<T>>     thread_data_backing_buf;
    _u64                           max_nthreads;
    bool                           load_flag = false;
    bool                           count_visited_nodes = false;

    /* diskv2 extra API requirements */
    // ids that don't have disk nhoods, but have in-mem PQ
    tsl::robin_set<_u32> invalid_ids;
    std::mutex           invalid_ids_lock;

    // tags
//    std::vector<TagT> tags;
	  TagT   * tags = nullptr;
  };
}  // namespace diskann
