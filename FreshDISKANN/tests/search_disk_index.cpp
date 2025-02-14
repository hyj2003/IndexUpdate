#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#include "mem_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP true

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  std::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(8) << percentiles[s] << "%";
  }
  std::cout << std::endl;
  std::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(9) << results[s];
  }
  std::cout << std::endl;
}

template<typename T>
int search_disk_index(int argc, char** argv) {
  // load query bin
  T*                query = nullptr;
  unsigned*         gt_ids = nullptr;
  uint32_t*         gt_tags = nullptr;
  uint32_t*         gt_tags_tmp = nullptr;
  float*            gt_dists = nullptr;
  size_t            query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  std::vector<_u64> Lvec;

  std::string index_prefix_path(argv[2]);
  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  _u64        num_nodes_to_cache = std::atoi(argv[3]);
  _u32        num_threads = std::atoi(argv[4]);
  _u32        beamwidth = std::atoi(argv[5]);
  std::string query_bin(argv[6]);
  std::string truthset_bin(argv[7]);
  _u64        recall_at = std::atoi(argv[8]);
  std::string result_output_prefix(argv[9]);

  std::string disk_index_tag_file = disk_index_file + ".tags";

  tsl::robin_set<uint32_t> inactive_tags;

  if (file_exists(disk_index_tag_file)) {
    for (uint32_t i = 0; i < 1000000; i++)
      inactive_tags.insert(i);
  }
  uint32_t* tag_data;
  size_t    tag_num, tag_dim;
  diskann::load_bin<uint32_t>(disk_index_tag_file, tag_data, tag_num, tag_dim);
  for (size_t i = 0; i < tag_num; i++)
    inactive_tags.erase(tag_data[i]);
  delete[] tag_data;
  // /*
  //       std::ifstream tag_reader(disk_index_tag_file);
  //       uint32_t      cur_i = 0;
  //       uint32_t      cur_tag;
  //       while (tag_reader >> cur_tag) {
  //         inactive_tags.erase(cur_tag);
  //         cur_i++;
  //       }
  //       tag_reader.close();
  //       */
  std::cout << "Loaded " << tag_num << " tags from " << disk_index_tag_file
            << ", # inactive tags:" << inactive_tags.size() << std::endl;
  bool calc_recall_flag = false;

  for (int ctr = 10; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at"
              << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);
  if (file_exists(truthset_bin)) {
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                              &gt_tags);
    // diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
    //                        &gt_tags_tmp);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data"
                << std::endl;
    }
    calc_recall_flag = true;
  }
  std::cout << "----===----" << std::endl;
  std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
  reader.reset(new WindowsAlignedFileReader());
#else
  reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
  reader.reset(new LinuxAlignedFileReader());
//  reader.reset(new diskann::MemAlignedFileReader());
#endif

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader));
  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str(), true);
  if (res != 0) {
    return res;
  }
  // cache bfs levels
  std::vector<uint32_t> node_list;
  std::cout << "Caching " << num_nodes_to_cache << " BFS nodes around medoid(s)"
            << std::endl;
  _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
  //  std::cout << "WARNING: not generating cache list from sample queries\n";
  /*
  _pFlashIndex->generate_cache_list_from_sample_queries(
      warmup_query_file, 15, 6, num_nodes_to_cache, node_list); */
  _pFlashIndex->load_cache_list(node_list);
  node_list.clear();
  node_list.shrink_to_fit();

  omp_set_num_threads(num_threads);

  uint64_t warmup_L = 20;
  uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
  T*       warmup = nullptr;

  if (WARMUP) {
    if (file_exists(warmup_query_file)) {
      diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num,
                                   warmup_dim, warmup_aligned_dim);
    } else {
      warmup_num = (std::min)((_u32) 150000, (_u32) 15000 * num_threads);
      warmup_dim = query_dim;
      warmup_aligned_dim = query_aligned_dim;
      diskann::alloc_aligned(((void**) &warmup),
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
    }
    std::cout << "Warming up index... " << std::flush;
    std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
    std::vector<float>    warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
    for (_s64 i = 0; i < (int64_t) warmup_num; i++) {
      _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1,
                                       warmup_L,
                                       warmup_result_ids_64.data() + (i * 1),
                                       warmup_result_dists.data() + (i * 1), 4);
    }
    std::cout << "..done" << std::endl;
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
            << std::setw(16) << "QPS" << std::setw(16) << "Mean Latency"
            << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs"
            << std::setw(16) << "CPU (s)";
  if (calc_recall_flag) {
    std::cout << std::setw(16) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "==============================================================="
               "==========================================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  uint32_t optimized_beamwidth = 2;

  //  query_num = 1;
  for (int cas = 0; cas <= 5; cas++) {
  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (beamwidth <= 0) {
      std::cerr << "WARNING : Beamwidth tuning disabled.\n";
      optimized_beamwidth = 4;
      //    std::cout<<"Tuning beamwidth.." << std::endl;
      // optimized_beamwidth =
      //     optimize_beamwidth(_pFlashIndex, warmup, warmup_num,
      //                       warmup_aligned_dim, (_u32) L,
      //                       optimized_beamwidth);
    } else
      optimized_beamwidth = beamwidth;

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    diskann::QueryStats* stats = new diskann::QueryStats[query_num];

    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
    diskann::Timer timer;
    // auto                  s = std::chrono::high_resolution_clock::now();
#pragma omp               parallel for schedule(dynamic, 1)  // num_threads(1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex->cached_beam_search(
          query + (i * query_aligned_dim), recall_at, L,
          query_result_ids_64.data() + (i * recall_at),
          query_result_dists[test_id].data() + (i * recall_at),
          optimized_beamwidth, stats + i, nullptr,
          query_result_tags[test_id].data() + (i * recall_at));
    }
    // auto                          e = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = e - s;
    float diff = (float)timer.elapsed() / 1000 / 1000;
    float qps = (float) (1.0 * query_num) / diff;

    diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                               query_result_ids[test_id].data(),
                                               query_num, recall_at);

    float mean_latency = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_999 = (float) diskann::get_percentile_stats(
        stats, query_num, 0.999,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    float mean_cpuus = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });
    delete[] stats;

    float recall = 0;
    if (calc_recall_flag) {
      if (gt_tags == nullptr) {
        std::cout << "Testing recall using IDs.\n";
        recall = (float) diskann::calculate_recall(
            (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim,
            query_result_ids[test_id].data(), (_u32) recall_at,
            (_u32) recall_at);
      } else {
        std::cout << "Testing recall using tags.\n";
        recall = (float) diskann::calculate_recall(
            (_u32) query_num, gt_tags, gt_dists, (_u32) gt_dim,
            query_result_tags[test_id].data(), (_u32) recall_at,
            (_u32) recall_at, inactive_tags);
      }
    }

    std::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth
              << std::setw(16) << qps << std::setw(16) << mean_latency
              << std::setw(16) << latency_999 << std::setw(16) << mean_ios
              << std::setw(16) << mean_cpuus;
    if (calc_recall_flag) {
      std::cout << std::setw(16) << recall << std::endl;
    } else
      std::cout << std::endl;
  }
  }
  std::cout << "Done searching. Now saving results " << std::endl;
  _u64 test_id = 0;
  for (auto L : Lvec) {
    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);

    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_tags_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_tags[test_id].data(),
                            query_num, recall_at);
    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    diskann::save_bin<float>(cur_result_path,
                             query_result_dists[test_id++].data(), query_num,
                             recall_at);
  }
  diskann::aligned_free(query);
  if (warmup != nullptr)
    diskann::aligned_free(warmup);
  delete[] gt_ids;
  delete[] gt_tags;
  delete[] gt_tags_tmp;
  delete[] gt_dists;
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 11) {
    std::cout
        << "Usage: " << argv[0]
        << "  [index_type<float/int8/uint8>]  [index_prefix_path] "
           " [num_nodes_to_cache]  [num_threads]  [beamwidth (use 0 to "
           "optimize internally)] "
           " [query_file.bin]  [truthset.bin (use \"null\" for none)] "
           " [K]  [result_output_prefix] "
           " [L1]  [L2] etc.  See README for more information on parameters."
        << std::endl;
    exit(-1);
  }
  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8"
              << std::endl;
}
