
#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "aux_utils.h"
#include "cached_io.h"
#include "index.h"
#include "mkl.h"
#include "omp.h"
#include "partition_and_pq.h"
#include "pq_flash_index.h"
#include "tsl/robin_set.h"
#include "utils.h"

#define NUM_KMEANS 3

namespace diskann {

  double get_memory_budget(const std::string &mem_budget_str) {
    double mem_ram_budget = atof(mem_budget_str.c_str());
    double final_index_ram_limit = mem_ram_budget;
    if (mem_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB >
        THRESHOLD_FOR_CACHING_IN_GB) {  // slack for space used by cached
                                        // nodes
      final_index_ram_limit = mem_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
    }
    return final_index_ram_limit * 1024 * 1024 * 1024;
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned recall_at) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        float *gt_dist_vec = gs_dist + dim_gs * i;
        tie_breaker = recall_at - 1;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned                  recall_at,
                          tsl::robin_set<unsigned> &inactive_locations) {
    double             total_recall = 0, total_active = 0;
    std::set<unsigned> gt, res;
    bool               printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      unsigned  active_points_count = 0;
      unsigned  cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (inactive_locations.find(gt_vec[cur_counter]) ==
            inactive_locations.end()) {
          gt.insert(gt_vec[cur_counter]);
          active_points_count++;
        }
        cur_counter++;
      }
      res.insert(res_vec, res_vec + recall_at);
      if (active_points_count < recall_at && !printed) {
        std::cout << "Warning: Couldn't find enough closest neighbors from "
                     "truthset. will result in under-reported value of recall."
                  << std::endl;
        std::cout << active_points_count << std::endl;
        // for (auto id : gt) {
        //   std::cout << id << " ";
        // }
        // std::cout << std::endl;
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1]) {
          if (inactive_locations.find(gt_vec[tie_breaker]) ==
              inactive_locations.end()) {
            gt.insert(gt_vec[tie_breaker]);
          }
          tie_breaker++;
        }
      }
      /*
         std::cout<<"ground_truth :: ";
                 float *gt_dist_vec = gs_dist + dim_gs * i;
                 for (_u32 t= 0; t < tie_breaker; t++) {
                     std::cout<<t <<": " << gt_vec[t] <<"," << gt_dist_vec[t]
         <<" ";
                 }
                 std::cout<<std::endl;
      */
      // gt.insert(gt_vec, gt_vec + tie_breaker);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end() && inactive_locations.find(v) == inactive_locations.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
      total_active += active_points_count;
    }
    return total_recall / total_active * recall_at;
  }

  template<typename T>
  T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num,
                 uint64_t warmup_dim, uint64_t warmup_aligned_dim) {
    T *      warmup = nullptr;
    uint64_t file_dim, file_aligned_dim;
    if (file_exists(cache_warmup_file)) {
      diskann::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num,
                                   file_dim, file_aligned_dim);
      if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim) {
        std::stringstream stream;
        stream << "Mismatched dimensions in sample file. file_dim = "
               << file_dim << " file_aligned_dim: " << file_aligned_dim
               << " index_dim: " << warmup_dim
               << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
        throw diskann::ANNException(stream.str(), -1);
      }
    } else {
      warmup_num = 100000;
      //      std::cout << "Generating random warmup file with dim " <<
      //      warmup_dim
      //                << " and aligned dim " << warmup_aligned_dim <<
      //                std::flush;
      diskann::alloc_aligned(((void **) &warmup),
                             warmup_num * warmup_aligned_dim * sizeof(T),
                             8 * sizeof(T));
      std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
      std::random_device              rd;
      std::mt19937                    gen(rd());
      std::uniform_int_distribution<> dis(-128, 127);
      for (uint32_t i = 0; i < warmup_num; i++) {
        for (uint32_t d = 0; d < warmup_dim; d++) {
          warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
        }
      }
      //      std::cout << "..done" << std::endl;
    }
    return warmup;
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t      npts32, dim;
    size_t        actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 ||
        actual_file_size !=
            ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
      std::stringstream stream;
      stream << "Error reading idmap file. Check if the file is bin file with "
                "1 dimensional data. Actual: "
             << actual_file_size
             << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t)
             << std::endl;

      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix,
                   const std::string &vamana_suffix,
                   const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards,
                   unsigned max_degree, const std::string &output_vamana,
                   const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string>           vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      vamana_names[shard] =
          vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix,
                 idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    std::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree
              << std::endl;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      std::cout << "Creating inverse map -- shard #" << shard << "\n";
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((_u32) node_id, (_u32) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(), [](const auto &left,
                                                       const auto &right) {
      return left.first < right.first ||
             (left.first == right.first && left.second < right.second);
    });
    std::cout << "Finished computing node -> shards map\n";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], 1024 * 1048576);
      size_t actual_file_size = get_file_size(vamana_names[i]);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
      if (actual_file_size != expected_file_size) {
        std::stringstream stream;
        stream << "Error in Vamana Index file " << vamana_names[i]
               << " Actual file size: " << actual_file_size
               << " does not match expected file size: " << expected_file_size
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
    }

    size_t merged_index_size = 16;
    // create cached vamana writers
    cached_ofstream diskann_writer(output_vamana, 1024 * 1048576);
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width =
          input_width > max_input_width ? input_width : max_input_width;
    }

    std::cout << "Max input width: " << max_input_width
              << ", output width: " << output_width << std::endl;

    diskann_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    _u32          nshards_u32 = (_u32) nshards;
    _u32          one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        diskann_writer.write((char *) &medoid, sizeof(unsigned));
    }
    medoid_writer.close();

    std::cout << "Starting merge\n";

    std::vector<bool>     nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        std::random_shuffle(final_nhood.begin(), final_nhood.end());
        nnbrs =
            (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
        diskann_writer.write((char *) final_nhood.data(),
                             nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          std::cout << "." << std::flush;
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(),
                                    shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    std::random_shuffle(final_nhood.begin(), final_nhood.end());
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    diskann_writer.write((char *) &nnbrs, sizeof(unsigned));
    diskann_writer.write((char *) final_nhood.data(), nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    std::cout << "Expected size: " << merged_index_size << std::endl;

    diskann_writer.reset();
    diskann_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    std::cout << "Finished merge\n";
    return 0;
  }

  template<typename T>
  int build_merged_vamana_index(std::string     base_file,
                                diskann::Metric _compareMetric, unsigned L,
                                unsigned R, double sampling_rate,
                                double ram_budget, std::string mem_index_path,
                                std::string medoids_file,
                                std::string centroids_file) {
    size_t base_num, base_dim;
    diskann::get_bin_metadata(base_file, base_num, base_dim);

    double full_index_ram =
        ESTIMATE_RAM_USAGE(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      std::cout << "Full index fits in RAM, building in one shot" << std::endl;
      diskann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 1);
      paras.Set<std::string>("save_path", mem_index_path);

      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(
              new diskann::Index<T>(_compareMetric, base_dim, base_num));
      _pvamanaIndex->build(base_file.c_str(), base_num, paras);
      _pvamanaIndex->save(mem_index_path.c_str());
      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return 0;
    }
    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int         num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget,
                                     2 * R / 3, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      diskann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", (2 * (R / 3)));
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);

      _u64 shard_base_dim, shard_base_pts;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(new diskann::Index<T>(
              _compareMetric, shard_base_dim, shard_base_pts));
      _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
      _pvamanaIndex->save(shard_index_file.c_str());
    }

    diskann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index",
                          merged_index_prefix + "_subshard-", "_ids_uint32.bin",
                          num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" +
                                  std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
    }
    return 0;
  }

  // General purpose support for DiskANN interface
  //
  //

  // optimizes the beamwidth to maximize QPS for a given L_search subject to
  // 99.9 latency not blowing up
  template<typename T>
  uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T>> &pFlashIndex, T *tuning_sample,
      _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim, uint32_t L,
      uint32_t start_bw) {
    uint32_t cur_bw = start_bw;
    double   max_qps = 0;
    uint32_t best_bw = start_bw;
    bool     stop_flag = false;

    while (!stop_flag) {
      std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
      std::vector<float>    tuning_sample_result_dists(tuning_sample_num, 0);
      diskann::QueryStats * stats = new diskann::QueryStats[tuning_sample_num];

      auto  s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
      for (_s64 i = 0; i < (int64_t) tuning_sample_num; i++) {
        pFlashIndex->cached_beam_search(
            tuning_sample + (i * tuning_sample_aligned_dim), 1, L,
            tuning_sample_result_ids_64.data() + (i * 1),
            tuning_sample_result_dists.data() + (i * 1), cur_bw, stats + i);
      }
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e - s;
      double qps = (1.0f * tuning_sample_num) / (1.0f * diff.count());

      double lat_999 = diskann::get_percentile_stats(
          stats, tuning_sample_num, 0.999,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      double mean_latency = diskann::get_mean_stats(
          stats, tuning_sample_num,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      // std::cout << "For bw: " << cur_bw << " qps: " << qps
      //          << " max_qps: " << max_qps << " mean_lat: " << mean_latency
      //          << " lat_999: " << lat_999 << std::endl;

      if (qps > max_qps && lat_999 < (15000) + mean_latency * 2) {
        //      if (qps > max_qps) {
        max_qps = qps;
        best_bw = cur_bw;
        //        std::cout<<"cur_bw: " << cur_bw <<", qps: " << qps <<",
        //        mean_lat: " << mean_latency/1000<<", 99.9lat: " <<
        //        lat_999/1000<<std::endl;
        cur_bw = (uint32_t)(std::ceil)((float) cur_bw * 1.1);
      } else {
        stop_flag = true;
        // std::cout << "Stopping at bw: " << best_bw << " max_qps: " << max_qps
        //          << std::endl;
        //        std::cout<<"cur_bw: " << cur_bw <<", qps: " << qps <<",
        //        mean_lat: " << mean_latency/1000<<", 99.9lat: " <<
        //        lat_999/1000<<std::endl;
      }
      if (cur_bw > 64)
        stop_flag = true;

      delete[] stats;
    }
    return best_bw;
  }

  template<typename T>
  void create_disk_layout(const std::string base_file,
                          const std::string mem_index_file,
                          const std::string output_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // create cached reader + writer
    size_t          actual_file_size = get_file_size(mem_index_file);
    std::ifstream   vamana_reader(mem_index_file, std::ios::binary);
    cached_ofstream diskann_writer(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t   index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size) {
      std::stringstream stream;
      stream << "Vamana Index file size does not match expected size per "
                "meta-data."
             << " file size from file: " << index_file_size
             << " actual file size: " << actual_file_size << std::endl;

      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));

    // compute
    _u64 medoid, max_node_len, nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
    max_node_len =
        (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));
    nnodes_per_sector = SECTOR_LEN / max_node_len;

    //    std::cout << "medoid: " << medoid << "B\n";
    //    std::cout << "max_node_len: " << max_node_len << "B\n";
    //    std::cout << "nnodes_per_sector: " << nnodes_per_sector << "B\n";

    // SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    unsigned &nnbrs = *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T));
    unsigned *nhood_buf =
        (unsigned *) (node_buf.get() + (ndims_64 * sizeof(T)) +
                      sizeof(unsigned));

    // number of sectors (1 for meta data)
    _u64 n_sectors = ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector;
    _u64 disk_index_file_size = (n_sectors + 1) * SECTOR_LEN;
    // write first sector with metadata
    *(_u64 *) (sector_buf.get() + 0 * sizeof(_u64)) = disk_index_file_size;
    *(_u64 *) (sector_buf.get() + 1 * sizeof(_u64)) = npts_64;
    //    *(_u64 *) (sector_buf.get() + 2 * sizeof(_u64)) = ndims_64;
    *(_u64 *) (sector_buf.get() + 2 * sizeof(_u64)) = medoid;
    *(_u64 *) (sector_buf.get() + 3 * sizeof(_u64)) = max_node_len;
    *(_u64 *) (sector_buf.get() + 4 * sizeof(_u64)) = nnodes_per_sector;
    diskann_writer.write(sector_buf.get(), SECTOR_LEN);

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    //    std::cout << "# sectors: " << n_sectors << "\n";
    _u64 cur_node_id = 0;
    for (_u64 sector = 0; sector < n_sectors; sector++) {
      //      if (sector % 100000 == 0) {
      //        std::cout << "Sector #" << sector << "written\n";
      //      }
      memset(sector_buf.get(), 0, SECTOR_LEN);
      for (_u64 sector_node_id = 0;
           sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
           sector_node_id++) {
        memset(node_buf.get(), 0, max_node_len);
        // read cur node's nnbrs
        vamana_reader.read((char *) &nnbrs, sizeof(unsigned));

        // sanity checks on nnbrs
        assert(nnbrs > 0);
        //        assert(nnbrs <= width_u32);

        // read node's nhood
        vamana_reader.read((char *) nhood_buf,
                           (std::min)(nnbrs, width_u32) * sizeof(unsigned));
        if (nnbrs > width_u32) {
          vamana_reader.seekg((nnbrs - width_u32) * sizeof(unsigned),
                              vamana_reader.cur);
        }

        // write coords of node first
        //  T *node_coords = data + ((_u64) ndims_64 * cur_node_id);
        base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
        memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

        // write nnbrs
        *(unsigned *) (node_buf.get() + ndims_64 * sizeof(T)) =
            (std::min)(nnbrs, width_u32);

        // write nhood next
        memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(unsigned),
               nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(unsigned));

        // get offset into sector_buf
        char *sector_node_buf =
            sector_buf.get() + (sector_node_id * max_node_len);

        // copy node buf into sector_node_buf
        memcpy(sector_node_buf, node_buf.get(), max_node_len);
        cur_node_id++;
      }
      // flush sector to disk
      diskann_writer.write(sector_buf.get(), SECTOR_LEN);
    }
    //    std::cout << "Output file written\n";
  }

  template<typename T>
  bool build_disk_index(const char *dataFilePath, const char *indexFilePath,
                        const char *    indexBuildParameters,
                        diskann::Metric _compareMetric) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 5) {
      std::cout
          << "Correct usage of parameters is R (max degree) "
             "L (indexing list size, better if >= R) B (RAM limit of final "
             "index in "

             "GB) M (memory limit while indexing) T (number of threads for "
             "indexing)"
          << std::endl;
      return false;
    }

    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_pq_compressed.bin";

    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";
    std::string sample_base_prefix = index_prefix_path + "_sample";

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      std::cerr << "Insufficient memory budget (or string was not in right "
                   "format). Should be > 0."
                << std::endl;
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      std::cerr << "Not building index. Please provide more RAM budget"
                << std::endl;
      return false;
    }
    _u32 num_threads = (_u32) atoi(param_list[4].c_str());

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
      mkl_set_num_threads(num_threads);
    }

    std::cout << "Starting index build: R=" << R << " L=" << L
              << " Query RAM budget: " << final_index_ram_limit
              << " Indexing ram budget: " << indexing_ram_budget
              << " T: " << num_threads << std::endl;

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    diskann::get_bin_metadata(dataFilePath, points_num, dim);

    std::cout << points_num << " " << dim << std::endl;
    size_t num_pq_chunks =
        (size_t)(std::floor)(_u64(final_index_ram_limit / points_num));

    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks =
        num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;
    // REMOVE
    num_pq_chunks = 32;

    std::cout << "Compressing " << dim << "-dimensional data into "
              << num_pq_chunks << " bytes per vector." << std::endl;

    size_t train_size, train_dim;
    float *train_data;

    double p_val = ((double) TRAINING_SET_SIZE_SMALL / (double) points_num);
    // generates random sample and sets it to train_data and updates train_size
    gen_random_slice<T>(dataFilePath, p_val, train_data, train_size, train_dim);

    std::cout << "Training data loaded of size " << train_size << std::endl;

    generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                       (uint32_t) num_pq_chunks, NUM_KMEANS, pq_pivots_path);
    generate_pq_data_from_pivots<T>(dataFilePath, 256, (uint32_t) num_pq_chunks,
                                    pq_pivots_path, pq_compressed_vectors_path);

    delete[] train_data;

    train_data = nullptr;

    diskann::build_merged_vamana_index<T>(
        dataFilePath, _compareMetric, L, R, p_val, indexing_ram_budget,
        mem_index_path, medoids_path, centroids_path);

    diskann::create_disk_layout<T>(dataFilePath, mem_index_path,
                                   disk_index_path);

    double sample_sampling_rate = (150000.0 / points_num);

    gen_random_slice<T>(dataFilePath, sample_base_prefix, sample_sampling_rate);

    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    return true;
  }

  template<typename T, typename TagT>
  bool build_disk_index_with_tags(const char *dataFilePath, const char *indexFilePath,
                        const char *    indexBuildParameters,
                        diskann::Metric _compareMetric, const int kvecs) {
    std::stringstream parser;
    parser << std::string(indexBuildParameters);
    std::string              cur_param;
    std::vector<std::string> param_list;
    while (parser >> cur_param)
      param_list.push_back(cur_param);

    if (param_list.size() != 5) {
      std::cout
          << "Correct usage of parameters is R (max degree) "
             "L (indexing list size, better if >= R) B (RAM limit of final "
             "index in "

             "GB) M (memory limit while indexing) T (number of threads for "
             "indexing)"
          << std::endl;
      return false;
    }

    std::string index_prefix_path(indexFilePath);
    std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
    std::string pq_compressed_vectors_path =
        index_prefix_path + "_pq_compressed.bin";

    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = index_prefix_path + "_disk.index";
    std::string medoids_path = disk_index_path + "_medoids.bin";
    std::string centroids_path = disk_index_path + "_centroids.bin";
    std::string sample_base_prefix = index_prefix_path + "_sample";

    unsigned R = (unsigned) atoi(param_list[0].c_str());
    unsigned L = (unsigned) atoi(param_list[1].c_str());

    double final_index_ram_limit = get_memory_budget(param_list[2]);
    if (final_index_ram_limit <= 0) {
      std::cerr << "Insufficient memory budget (or string was not in right "
                   "format). Should be > 0."
                << std::endl;
      return false;
    }
    double indexing_ram_budget = (float) atof(param_list[3].c_str());
    if (indexing_ram_budget <= 0) {
      std::cerr << "Not building index. Please provide more RAM budget"
                << std::endl;
      return false;
    }
    _u32 num_threads = (_u32) atoi(param_list[4].c_str());

    if (num_threads != 0) {
      omp_set_num_threads(num_threads);
      mkl_set_num_threads(num_threads);
    }

    std::cout << "Starting index build: R=" << R << " L=" << L
              << " Query RAM budget: " << final_index_ram_limit
              << " Indexing ram budget: " << indexing_ram_budget
              << " T: " << num_threads << std::endl;

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    diskann::get_bin_metadata(dataFilePath, points_num, dim);

    if ((int)points_num < kvecs) {
      std::cout << "Can't build index because number of file vectors isn't enough.\n";
      return false;
    }

    size_t tmp_points_num = points_num;
    points_num = kvecs;
    std::cout << points_num << " " << dim << std::endl;
    size_t num_pq_chunks =
        (size_t)(std::floor)(_u64(final_index_ram_limit / points_num));

    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks =
        num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;
    // REMOVE
    num_pq_chunks = 32;

    std::cout << "Compressing " << dim << "-dimensional data into "
              << num_pq_chunks << " bytes per vector." << std::endl;

    size_t train_size, train_dim;
    float *train_data;

    double p_val = ((double) TRAINING_SET_SIZE_SMALL / (double) points_num);
    // generates random sample and sets it to train_data and updates train_size
    std::string cut_off_datapath = index_prefix_path + "_cut_off_data";
    {
      T *data = new T[tmp_points_num * dim];
      diskann::load_bin<T>(dataFilePath, data, tmp_points_num, dim);
      std::ofstream data_writer(cut_off_datapath.c_str(), std::ios::binary);
      data_writer.write((char *) &points_num, sizeof(uint32_t));
      data_writer.write((char *) &dim, sizeof(uint32_t));
      data_writer.write((char *) data, points_num * dim * sizeof(T));
      data_writer.close();
    }
    gen_random_slice<T>(cut_off_datapath, p_val, train_data, train_size, train_dim);

    std::cout << "Training data loaded of size " << train_size << std::endl;

    generate_pq_pivots(train_data, train_size, (uint32_t) dim, 256,
                       (uint32_t) num_pq_chunks, NUM_KMEANS, pq_pivots_path);
    generate_pq_data_from_pivots<T>(cut_off_datapath, 256, (uint32_t) num_pq_chunks,
                                    pq_pivots_path, pq_compressed_vectors_path);

    delete[] train_data;

    train_data = nullptr;

    diskann::build_merged_vamana_index<T>(
        cut_off_datapath, _compareMetric, L, R, p_val, indexing_ram_budget,
        mem_index_path, medoids_path, centroids_path);

    diskann::create_disk_layout<T>(cut_off_datapath, mem_index_path,
                                   disk_index_path);

    double sample_sampling_rate = (150000.0 / points_num);

    gen_random_slice<T>(cut_off_datapath, sample_base_prefix, sample_sampling_rate);
    std::string disk_index_tag_path = index_prefix_path + "_disk.index.tags";
    std::ofstream tag_writer(disk_index_tag_path.c_str(), std::ios::binary);
    uint32_t ud1 = 1;
    tag_writer.write((char *) &tmp_points_num, sizeof(uint32_t));
    tag_writer.write((char *) &ud1, sizeof(uint32_t));
    TagT *tag = new TagT[tmp_points_num];
    for (int i = 0; i < (int)tmp_points_num; i++) {
      if (i < kvecs) tag[i] = i;
      else tag[i] = std::numeric_limits<TagT>::max();
    }
    tag_writer.write((char *)tag, tmp_points_num * sizeof(TagT));
    tag_writer.close();
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;

    std::cout << "Indexing time: " << diff.count() << "\n";

    return true;
  }
  DISKANN_DLLEXPORT void transpose_pq_codes(uint8_t* pq_codes, int n, int nchunk, int group_size) {
    //    assert(group_size * sizeof(uint8_t) * 8 == 512);
    std::vector<uint8_t> buffer(nchunk * group_size, 0);
    // todo：使用omp会引起并发问题，需要改一下代码
//#pragma omp parallel for num_threads(32)
    for(int i = 0; i < n; i += group_size) {
        uint8_t* group_codes = pq_codes + i * nchunk;
        int block_size = std::min(n - i, group_size);
        std::memcpy(buffer.data(), group_codes, nchunk * block_size);
        if(block_size != group_size) {
            std::memset(buffer.data() + nchunk * block_size, 0, nchunk * (group_size - block_size));
        }
        for(int j = 0; j < block_size; j++) {
            for(int k = 0; k < nchunk; k++) {
                group_codes[k * group_size + j] = buffer[j * nchunk + k];
            }
        }
    }
  }
  template DISKANN_DLLEXPORT void create_disk_layout<int8_t>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file);

  template DISKANN_DLLEXPORT void create_disk_layout<uint8_t>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file);
  template DISKANN_DLLEXPORT void create_disk_layout<float>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file);

  template DISKANN_DLLEXPORT int8_t *load_warmup<int8_t>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template DISKANN_DLLEXPORT uint8_t *load_warmup<uint8_t>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template DISKANN_DLLEXPORT float *load_warmup<float>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);

  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<int8_t>(
      std::unique_ptr<diskann::PQFlashIndex<int8_t>> &pFlashIndex,
      int8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<uint8_t>(
      std::unique_ptr<diskann::PQFlashIndex<uint8_t>> &pFlashIndex,
      uint8_t *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t start_bw);
  template DISKANN_DLLEXPORT uint32_t optimize_beamwidth<float>(
      std::unique_ptr<diskann::PQFlashIndex<float>> &pFlashIndex,
      float *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t start_bw);

  template DISKANN_DLLEXPORT bool build_disk_index<int8_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric);
  template DISKANN_DLLEXPORT bool build_disk_index<uint8_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric);
  template DISKANN_DLLEXPORT bool build_disk_index<float>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric);
  
  template DISKANN_DLLEXPORT bool build_disk_index_with_tags<int8_t, uint32_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      const int kvecs);
  template DISKANN_DLLEXPORT bool build_disk_index_with_tags<uint8_t, uint32_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      const int kvecs);
  template DISKANN_DLLEXPORT bool build_disk_index_with_tags<float, uint32_t>(
      const char *dataFilePath, const char *indexFilePath,
      const char *indexBuildParameters, diskann::Metric _compareMetric,
      const int kvecs);

  template DISKANN_DLLEXPORT int build_merged_vamana_index<int8_t>(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_path,
      std::string centroids_file);
  template DISKANN_DLLEXPORT int build_merged_vamana_index<float>(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_path,
      std::string centroids_file);
  template DISKANN_DLLEXPORT int build_merged_vamana_index<uint8_t>(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, double sampling_rate, double ram_budget,
      std::string mem_index_path, std::string medoids_path,
      std::string centroids_file);
};  // namespace diskann
