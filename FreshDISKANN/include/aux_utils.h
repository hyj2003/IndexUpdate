﻿#pragma once
#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef _WINDOWS
#include <Windows.h>
typedef HANDLE FileHandle;
#else
#include <unistd.h>
typedef int FileHandle;
#endif

#include "aligned_dtor.h"
#include "cached_io.h"
#include "common_includes.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "windows_customizations.h"

namespace diskann {
  const size_t   TRAINING_SET_SIZE = 1500000;
  const size_t   TRAINING_SET_SIZE_SMALL = 150000;
  const double   SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double   THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t NUM_NODES_TO_CACHE = 250000;
  const uint32_t WARMUP_L = 20;

  template<typename T, typename TagT>
  class PQFlashIndex;

  DISKANN_DLLEXPORT double calculate_recall(
      unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
      unsigned *our_results, unsigned dim_or, unsigned recall_at);

  DISKANN_DLLEXPORT double calculate_recall(
      unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
      unsigned *our_results, unsigned dim_or, unsigned recall_at,
      tsl::robin_set<unsigned> &deletion_set);

  DISKANN_DLLEXPORT void read_idmap(const std::string &    fname,
                                    std::vector<unsigned> &ivecs);

  template<typename T>
  DISKANN_DLLEXPORT T *load_warmup(const std::string &cache_warmup_file,
                                   uint64_t &warmup_num, uint64_t warmup_dim,
                                   uint64_t warmup_aligned_dim);

  DISKANN_DLLEXPORT int merge_shards(const std::string &nsg_prefix,
                                     const std::string &nsg_suffix,
                                     const std::string &idmaps_prefix,
                                     const std::string &idmaps_suffix,
                                     const _u64 nshards, unsigned max_degree,
                                     const std::string &output_nsg,
                                     const std::string &medoids_file);

  template<typename T>
  DISKANN_DLLEXPORT int build_merged_vamana_index(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_file,
      std::string centroids_file);

  template<typename T, typename TagT = uint32_t>
  DISKANN_DLLEXPORT uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T, TagT>> &_pFlashIndex,
      T *tuning_sample, _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim,
      uint32_t L, uint32_t start_bw = 2);

  template<typename T>
  DISKANN_DLLEXPORT bool build_disk_index(const char *    dataFilePath,
                                          const char *    indexFilePath,
                                          const char *    indexBuildParameters,
                                          diskann::Metric _compareMetric);
  template<typename T, typename TagT>
  DISKANN_DLLEXPORT bool build_disk_index_with_tags(const char *    dataFilePath,
                                          const char *    indexFilePath,
                                          const char *    indexBuildParameters,
                                          diskann::Metric _compareMetric, const int kvecs);

  template<typename T>
  DISKANN_DLLEXPORT void create_disk_layout(const std::string base_file,
                                            const std::string mem_index_file,
                                            const std::string output_file);

    // todo: pq_codes应该按照group_size×nchunk对齐，pq_codes的内存在load_bin_impl函数中分配
  DISKANN_DLLEXPORT void transpose_pq_codes(uint8_t* pq_codes, int n, int nchunk, int group_size);

}  // namespace diskann
