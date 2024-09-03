#pragma once
#include "pq_flash_index.h"
namespace diskann {
  template<typename T, typename TagT = uint32_t>
  class FreshPQFlashIndex : public PQFlashIndex<T, TagT> {
   public:
    FreshPQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader);
    ~FreshPQFlashIndex();
    void filtered_beam_search(
      const T *query, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, QueryStats *stats,
      Distance<T> *output_dist_func, TagT *res_tags);
    void filter_disk_iterate_to_fixed_point(
      const T *vec, const uint32_t Lsize, const uint32_t beam_width,
      std::vector<Neighbor> &expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
      Distance<T> *output_dist_func = nullptr, QueryStats *stats = nullptr,
      ThreadData<T> *           passthrough_data = nullptr,
      tsl::robin_set<uint32_t> *exclude_nodes = nullptr);
    void load_deleted_list(const char *deleted_tags_file);
    int load(uint32_t num_threads, const char *pq_prefix,
             const char *disk_index_file, 
             bool load_tags = false);
    void PruneNeighbors(_u64 id, std::vector<) {

    }
    void ProcessPage(_u64 sector_id, char *sector_buf) {
      char *tmp_buf = new char[SECTOR_LEN];
      memcpy(tmp_buf, sector_buf, SECTOR_LEN);
      assert(sector_id != 0);
      _u32 cur_node_id = (sector_id - 1) * this->nnodes_per_sector;
      std::vector<DiskNode<T>> disk_nodes;
      for (uint32_t i = 0; i < this->nnodes_per_sector && cur_node_id < this->num_points; i++) {
        disk_nodes.emplace_back(cur_node_id, 
                                OFFSET_TO_NODE_COORDS(tmp_buf),
                                OFFSET_TO_NODE_NHOOD(tmp_buf));
        cur_node_id++;
      }
      std::shared_lock<std::shared_mutex> delete_guard(this->delete_cache_lock_);
      std::shared_lock<std::shared_mutex> insert_guard(this->insert_edges_lock_);
      for (auto &disk_node : disk_nodes) {
        auto      id = disk_node.id;
        auto      nnbrs = disk_node.nnbrs;
        unsigned *nbrs = disk_node.nbrs;
        std::vector<_u32> new_nbrs;
        for (uint32_t i = 0; i < nnbrs; i++) {
          if (!del_filter_.get(nbrs[i])) {
            new_nbrs.emplace_back(nbrs[i]);
          } else {
            auto iter = this->delete_cache_.find(nbrs[i]);
            auto m = iter->second.first;
            auto nei = iter->second.second;
            for (_u32 j = 0; j < m; j++) {
              if (!del_filter_.get(nei[j])) {
                new_nbrs.emplace_back(nei[j]);
              }
            }
          }
        }
        auto iter = this->insert_edges_.find(id);
        if (iter != insert_edges_.end()) {
          for (auto nbrs : iter->second) {
            if (del_filter_.get(nbrs)) continue;
            new_nbrs.emplace_back(nbrs);
          }
        }
        
      }
      delete[] tmp_buf;
    }

   private:
    std::shared_mutex delete_cache_lock_;
    tsl::robin_map<_u32, std::pair<_u32, const _u32 *>> delete_cache_;
    const uint64_t MAX_NODE_NUMBER = uint64_t(10000005);
    VisitedList del_filter_ = VisitedList(MAX_NODE_NUMBER);
    _u32 two_hops_lim;
    std::shared_mutex insert_edges_lock_;
    tsl::robin_map<_u32, std::vector<_u32>> insert_edges_;
  };
  template<typename T>
  class Updater {
   public:
    std::shared_mutex delete_cache_lock_;
    tsl::robin_map<_u32, std::pair<_u32, const _u32 *>> delete_cache_;
    tsl::robin_map<_u64, char *> page_in_use;

  };
}  // namespace diskann
