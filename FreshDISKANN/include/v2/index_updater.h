#pragma once

#include "v2/graph_delta.h"
#include "v2/fs_allocator.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "fresh_pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "index.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <future>
namespace diskann {
  template<typename T, typename TagT=uint32_t>
  class IndexUpdater {
    public:
      // constructor to read a constructed index, allocated IDs
      // disk_in : SSD-DiskANN index to merge into
      // mem_in : list of mem-DiskANN indices to merge into disk_in 
      // disk_out : SSD-DiskANN index to write out
      // delete_list : list of IDs to delete from disk_in
      // ndims : dimensionality of full-prec vectors
      // dist : distance comparator -- WARNING :: assumed to be L2
      // beam_width : BW for search on disk_in
      // range : max out-degree
      // l_index : L param for indexing
      // maxc : max num of candidates to consider while pruning
      IndexUpdater(const char* disk_in, Distance<T>* dist, const uint32_t beam_width,
                  const uint32_t range, const uint32_t l_index, const float alpha, const uint32_t maxc);
      ~IndexUpdater();
      void UpdateThreadSetup(size_t background_threads_num);
      int insert(T *point, TagT id) {
        this->disk_index->PushInsert(id, point);
        return 0;
      }
      void remove(TagT id) {
        this->disk_index->PushDelete(id);
      }
      void search(const T* query, const uint64_t K, const uint64_t search_L,
                     TagT* tags, float * distances, QueryStats * stats) {
        size_t                                    searchK = K;
        std::vector<_u64>                         query_result_ids_64(searchK);
        std::vector<float>                        query_result_dists(searchK);
        std::vector<TagT>                         query_result_tags(searchK);
        disk_index->filtered_beam_search(
              query, searchK, search_L, query_result_ids_64.data(),
              query_result_dists.data(), this->beam_width, nullptr, nullptr, query_result_tags.data());
        for (size_t i = 0; i < searchK; i++) {
          tags[i] = query_result_tags[i];
          distances[i] = query_result_dists[i];
        }
      }
      void StopUpdate() {
        this->update_flag_ = false;
      }
    private:
      Distance<T> *dist_cmp;

      // allocators
      // FixedSizeAlignedAllocator<T> *fp_alloc = nullptr;
      // FixedSizeAlignedAllocator<uint8_t> *pq_alloc = nullptr;

      // vector info
      uint32_t ndims, aligned_ndims;
      // search + index params
      uint32_t beam_width;
      uint32_t l_index, range, maxc;
      float alpha;
      bool update_flag_;
      FreshPQFlashIndex<T, TagT> *disk_index = nullptr;
      std::vector<std::future<void>> back_threads_;
  };
}; // namespace diskann
