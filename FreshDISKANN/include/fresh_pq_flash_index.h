#pragma once
#include "pq_flash_index.h"
namespace diskann {
  template<typename T>
  struct UpdateScratch {
    char *page_buf_ = nullptr;
    char *page_buf_copied_ = nullptr;
    uint8_t* scratch_ = nullptr;
    uint16_t* scratch_u16_ = nullptr;
  } ;
  template<typename T>
  struct UpdateThreadData {
    UpdateScratch<T> scratch;
    IOContext        ctx;
    std::fstream*    output_writer = nullptr;
  };
  template<typename T, typename TagT = uint32_t>
  class FreshPQFlashIndex : public PQFlashIndex<T, TagT> {
   public:
    FreshPQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader);
    ~FreshPQFlashIndex();
    void filtered_beam_search(
      const T *query, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, QueryStats *stats = nullptr,
      Distance<T> *output_dist_func = nullptr, TagT *res_tags = nullptr);
    void filter_disk_iterate_to_fixed_point(
      const T *vec, const uint32_t Lsize, const uint32_t beam_width,
      std::vector<Neighbor> &expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
      Distance<T> *output_dist_func = nullptr, QueryStats *stats = nullptr,
      ThreadData<T> *           passthrough_data = nullptr,
      tsl::robin_set<uint32_t> *exclude_nodes = nullptr);
    void load_deleted_list(const char *deleted_tags_file);
    int load(uint32_t num_threads, const char *pq_prefix,
             const char *disk_index_file, uint32_t max_index_num, 
             bool load_tags = false);
    void occlude_list_pq_simd(std::vector<Neighbor> &pool, 
                              const uint32_t id, 
                              std::vector<Neighbor> &result, 
                              std::vector<float> &occlude_factor, 
                              uint8_t* scratch,
                              uint16_t* scratch_u16);
    void PruneNeighbors(DiskNode<T> &disk_node, std::vector<_u32> &new_nbrs, _u8 *scratch, _u16 *scratch_u16);
    void ProcessPage(_u64 sector_id, UpdateThreadData<T> *upd);
    void PushVisitedPage(_u64 sector_id, char *sector_buf);
    void ProcessUpdateRequests(bool update_flag);
    void dump_to_disk(std::fstream     &output_writer,
                      const std::vector<DiskNode<T>> &disk_nodes, const uint32_t start_id, 
                      const char* buf, const uint32_t n_sectors) {
      assert(start_id % this->nnodes_per_sector == 0);
      uint32_t start_sector = (start_id / this->nnodes_per_sector) + 1;
      uint64_t start_off = start_sector * (uint64_t) SECTOR_LEN;

      // seek fp
      output_writer.seekp(start_off, std::ios::beg);

        // dump
      output_writer.write(buf, (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);

      uint64_t nb_written = (uint64_t) output_writer.tellp() - (uint64_t) start_off;
      assert(nb_written == (uint64_t) n_sectors * (uint64_t) SECTOR_LEN);
    }
    void PushDelete(_u32 external_id);
    void PushInsert(_u32 external_id, T *point);

    _u32 GetInternalId() {
      std::unique_lock<std::mutex> guard(free_ids_lock_);
      if (free_ids_.empty()) {
        return cur_max_id_++;
      }
      _u32 res = free_ids_.back();
      free_ids_.pop_back();
      return res;
    }
    void RecycleId(_u32 internal_id) {
      std::unique_lock<std::mutex> guard(free_ids_lock_);
      free_ids_.emplace_back(internal_id);
    }
    void ComputeInDegree();
    void SetParameters() {
      //
    }
    void SetUpdateThread(bool &update_flag) {
      while (update_flag) {
        ProcessUpdateRequests(update_flag);
      }
    }
   private:
    // Variables for deletion
    std::shared_mutex delete_cache_lock_;
    tsl::robin_map<_u32, std::pair<_u32, const _u32 *>> delete_cache_;
    const uint64_t MAX_NODE_NUMBER = uint64_t(10000005);    //
    VisitedList del_filter_ = VisitedList(MAX_NODE_NUMBER); // May need to be modified.
    _u32 two_hops_lim;
    // Variables for insertion
    std::shared_mutex insert_edges_lock_;
    tsl::robin_map<_u32, std::vector<_u32>> insert_edges_;

    // Variables for pruning
    const size_t maxc = 750, range = 64;
    const float alpha = 1.2;

    // Variables for page caching
    std::shared_mutex page_cache_lock_;
    tsl::robin_map<_u64, char *> page_cache_;
    // VisitedList del_filter_ = VisitedList(MAX_NODE_NUMBER);

    ConcurrentQueue<std::pair<_u64, UpdateThreadData<T>* >> reqs_queue_;
    ConcurrentQueue<UpdateThreadData<T>>        scratch_queue_;
    tsl::robin_set<_u64>                     page_in_process_;
    std::shared_mutex                        page_in_process_lock_;

    tsl::robin_map<_u32, _u32> in_degree_cnt_; // Use array for simple testing (without file systems)
    _u32 *in_degree_;
    std::shared_mutex          in_degree_lock_;
    _u32 *inv_tags;

    std::vector<_u32>          free_ids_;
    std::mutex                 free_ids_lock_;
    _u32                       cur_max_id_;
    uint32_t                   max_index_num_;
    // need further implement concurrent insert/delete queue to strengthen visibility

    uint8_t* update_thread_pq_scratch = nullptr;
    uint16_t* update_thread_pq_scratch_u16 = nullptr;
    uint32_t background_threads_num_;
    // std::shared_ptr<AlignedFileReader> fresh_reader = nullptr;
    // May need to be added
  };
  template<typename T>
  class Updater {
   public:
    std::shared_mutex delete_cache_lock_;
    tsl::robin_map<_u32, std::pair<_u32, const _u32 *>> delete_cache_;
    tsl::robin_map<_u64, char *> page_in_use;

  };
}  // namespace diskann
