#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/index_updater.h"
#include "mem_aligned_file_reader.h"
#include <algorithm>
#include <cassert>
#include <csignal>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>
#include <omp.h>
#include <future>

#include "tcmalloc/malloc_extension.h"

#define SECTORS_PER_MERGE (uint64_t)65536
// max number of points per mem index being merged -- 32M
#define MAX_PTS_PER_MEM_INDEX (uint64_t)(1 << 25)
#define INDEX_OFFSET (uint64_t) (MAX_PTS_PER_MEM_INDEX * 4)
#define MAX_INSERT_THREADS (uint64_t) 40
#define MAX_N_THREADS (uint64_t) 60
#define PER_THREAD_BUF_SIZE (uint64_t) (65536 * 64 * 4)
#define PQ_FLASH_INDEX_CACHE_SIZE 200000

namespace diskann {
  template<typename T, typename TagT>
  IndexUpdater<T, TagT>::IndexUpdater(const char* disk_in, Distance<T>* dist, const uint32_t beam_width,
                           const uint32_t range, const uint32_t l_index, const float alpha, const uint32_t maxc) {
	  std::cout << l_index << " , " << range << " , " << alpha << " , " << maxc << " , " << beam_width << std::endl;
    // book keeping
    this->range = range;
    this->l_index = l_index;
    this->beam_width = beam_width;
    this->maxc = maxc;
    this->alpha = alpha;
    this->dist_cmp = dist;

    // std::cout << "Created PQFlashIndex inside index_merger " << std::endl;
    std::string pq_prefix = std::string(disk_in) + "_pq";
    std::string disk_index_file = std::string(disk_in) + "_disk.index";
    std::shared_ptr<AlignedFileReader> reader = std::make_shared<LinuxAlignedFileReader>();
    this->disk_index = new FreshPQFlashIndex<T, TagT>(reader);
    this->disk_index->load(MAX_N_THREADS, pq_prefix.c_str(), disk_index_file.c_str(), 10000000, true);
    std::cout << "Loaded PQFlashIndex" << std::endl;
    // std::vector<uint32_t> cache_node_list;
    // this->disk_index->cache_bfs_levels(PQ_FLASH_INDEX_CACHE_SIZE, cache_node_list);
    // this->disk_index->load_cache_list(cache_node_list);
    // this->disk_index->load_deleted_list(deleted_tags_file);
    this->disk_index->ComputeInDegree();
    // std::cout << "Allocating thread scratch space -- " << PER_THREAD_BUF_SIZE / (1<<20) << " MB / thread.\n";
    // alloc_aligned((void**) &this->thread_pq_scratch, MAX_N_THREADS * PER_THREAD_BUF_SIZE, SECTOR_LEN);
    // this->thread_bufs.resize(MAX_N_THREADS);
    // for(uint32_t i=0; i < thread_bufs.size(); i++) {
    //   this->thread_bufs[i] = this->thread_pq_scratch + i * PER_THREAD_BUF_SIZE;
    // }
    // alloc_aligned((void**) &this->thread_pq_scratch_u16, MAX_N_THREADS * PER_THREAD_BUF_SIZE, SECTOR_LEN);
    // this->thread_bufs_u16.resize(MAX_N_THREADS);
    // for(uint32_t i=0; i < thread_bufs_u16.size(); i++) {
    //   this->thread_bufs_u16[i] = this->thread_pq_scratch_u16 + i * PER_THREAD_BUF_SIZE / 2;
    // }
  }
  template<typename T, typename TagT>
  IndexUpdater<T, TagT>::~IndexUpdater() {
    
  }
  template<typename T, typename TagT>
  void IndexUpdater<T, TagT>::UpdateThreadSetup(size_t background_threads_num) {
    this->update_flag_ = true;
    this->back_threads_.reserve(background_threads_num);
    this->back_threads_.resize(background_threads_num);
    for (size_t i = 0; i < background_threads_num; i++) {
      this->back_threads_[i] = 
                std::async(std::launch::async, [this, i]() {
                  this->disk_index->SetUpdateThread(this->update_flag_);
                });
    }
  }

  // template class instantiations
  template class IndexUpdater<float, uint32_t>;
  template class IndexUpdater<uint8_t, uint32_t>;
  template class IndexUpdater<int8_t, uint32_t>;
  template class IndexUpdater<float, int64_t>;
  template class IndexUpdater<uint8_t, int64_t>;
  template class IndexUpdater<int8_t, int64_t>;
  template class IndexUpdater<float, uint64_t>;
  template class IndexUpdater<uint8_t, uint64_t>;
  template class IndexUpdater<int8_t, uint64_t>;
} // namespace diskann
