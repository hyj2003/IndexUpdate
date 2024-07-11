#include <cstring>
#include <index.h>
#include <iomanip>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include <timer.h>
#include <string.h>

#include "aux_utils.h"
#include "utils.h"
#include "tsl/robin_set.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "memory_mapper.h"

template<typename T, typename TagT>
void search_kernel(
    T* query, size_t query_num, size_t query_aligned_dim, const int recall_at,
    std::vector<_u64> Lvec, diskann::Index<T, TagT>& index,
    const std::string&       truthset_file,
    tsl::robin_set<unsigned> delete_list = tsl::robin_set<unsigned>()) {
  unsigned* gt_ids = NULL;
  float*    gt_dists = NULL;
  size_t    gt_num, gt_dim;
  diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

  float*    query_result_dists = new float[recall_at * query_num];
  unsigned* query_result_ids = new unsigned[recall_at * query_num];
  TagT*     query_result_tags = new TagT[recall_at * query_num];
  memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
  memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);
  memset(query_result_ids, 0, sizeof(unsigned) * recall_at * query_num);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18)
            << "Mean Latency (ms)" << std::setw(15) << "99.9 Latency"
            << std::setw(12) << recall_string << std::endl;

  std::cout << "==============================================================="
               "==============="
            << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    std::vector<double> latency_stats(query_num, 0);
    memset(query_result_dists, 0, sizeof(float) * recall_at * query_num);
    memset(query_result_tags, 0, sizeof(TagT) * recall_at * query_num);
    memset(query_result_ids, 0, sizeof(unsigned) * recall_at * query_num);
    _u64    L = Lvec[test_id];
    auto    s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto qs = std::chrono::high_resolution_clock::now();
      index.search_with_tags(query + i * query_aligned_dim, recall_at, (_u32) L,
                             query_result_tags + i * recall_at, 1,
                             query_result_ids + i * recall_at);
      auto qe = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = qe - qs;
      latency_stats[i] = diff.count() * 1000;
      //      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    auto e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = e - s;
    float                         qps = (float) (query_num / diff.count());

    float recall;
    if (delete_list.size() > 0) {
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim, query_result_tags,
          (_u32) recall_at, (_u32) recall_at, delete_list);
    } else {
      recall = (float) diskann::calculate_recall(
          (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim, query_result_tags,
          (_u32) recall_at, (_u32) recall_at);
    }

    std::sort(latency_stats.begin(), latency_stats.end());
    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << std::accumulate(latency_stats.begin(), latency_stats.end(),
                                 0) /
                     (float) query_num
              << std::setw(15)
              << (float) latency_stats[(_u64)(0.999 * query_num)]
              << std::setw(12) << recall << std::endl;
  }
  delete[] query_result_dists;
  delete[] query_result_ids;
  delete[] query_result_tags;
}

template<typename T, typename TagT>
int build_incremental_index(const std::string& data_path,
                            const std::string& memory_index_file,
                            const unsigned L, const unsigned R,
                            const unsigned C, const unsigned num_rnds,
                            const float alpha, const std::string& save_path,
                            const unsigned num_cycles, unsigned num_del,
                            unsigned num_ins, const std::string& query_file,
                            const std::string& truthset_file,
                            const int recall_at, std::vector<_u64> Lvec) {
  diskann::Parameters paras;
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("C", C);
  paras.Set<float>("alpha", alpha);
  paras.Set<unsigned>("num_rnds", num_rnds);

  T*     data_load = NULL;
  size_t num_points, dim, aligned_dim;

  diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                               aligned_dim);

  diskann::Index<T, TagT> index(diskann::L2, dim, num_points, 1, true, true, 0);

  auto tag_path = memory_index_file + ".tags";
  auto data_file = memory_index_file + ".data";
  index.load(memory_index_file.c_str(), data_file.c_str(), true,
             tag_path.c_str());
  std::cout << "Loaded index and tags and data" << std::endl;

  tsl::robin_set<TagT> active_tags;
  index.get_active_tags(active_tags);
  std::cout << active_tags.size() << std::endl;

  tsl::robin_set<TagT> inactive_tags;
  for (_u64 p = 0; p < num_points; p++) {
    if (active_tags.find((TagT) p) == active_tags.end())
      inactive_tags.insert((TagT) p);
  }

  if ((active_tags.size() + inactive_tags.size()) != num_points) {
    std::cout << "Error in size of active tags and inactive tags.   : "
              << active_tags.size() << "  ,  " << inactive_tags.size()
              << std::endl;
    exit(-1);
  }
  T*     query = NULL;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Search on static index" << std::endl;
  search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec, index,
                truthset_file, inactive_tags);
  tsl::robin_set<TagT> new_active_tags;
  tsl::robin_set<TagT> new_inactive_tags;

  unsigned i = 0;
  while (i < num_cycles) {
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    tsl::robin_set<TagT>                  delete_vec;
    tsl::robin_set<TagT>                  insert_vec;

    new_active_tags.clear();
    new_inactive_tags.clear();

    delete_vec.clear();
    insert_vec.clear();

    float active_tags_sampling_rate =
        (float) ((std::min)((1.0 * num_del) / (1.0 * active_tags.size()), 1.0));

    for (auto iter = active_tags.begin(); iter != active_tags.end(); iter++) {
      if (dis(gen) < active_tags_sampling_rate) {
        delete_vec.insert(*iter);
        new_inactive_tags.insert(*iter);
      } else
        new_active_tags.insert(*iter);
    }

    float inactive_tags_sampling_rate = (float) ((std::min)(
        (1.0 * num_ins) / (1.0 * inactive_tags.size()), 1.0));

    for (auto iter = inactive_tags.begin(); iter != inactive_tags.end();
         iter++) {
      if (dis(gen) < inactive_tags_sampling_rate) {
        insert_vec.insert(*iter);
        new_active_tags.insert(*iter);
      } else
        new_inactive_tags.insert(*iter);
    }

    std::cout << "Preparing to insert " << insert_vec.size()
              << " points and delete  " << delete_vec.size() << " points. "
              << std::endl;
    {
      index.enable_delete();
      diskann::Timer    del_timer;
      std::vector<TagT> failed_tags;
      if (index.lazy_delete(delete_vec, failed_tags) < 0)
        std::cerr << "Error in delete_points" << std::endl;
      if (failed_tags.size() > 0)
        std::cerr << "Failed to delete " << failed_tags.size() << " tags"
                  << std::endl;
      std::cout << "completed in " << del_timer.elapsed() / 1000000.0
                << "sec.\n"
                << "Starting consolidation... " << std::flush;

      diskann::Timer timer;
      if (index.disable_delete(paras, true) != 0) {
        std::cerr << "Disable delete failed" << std::endl;
        return -1;
      }
      std::cout << "completed in " << timer.elapsed() / 1000000.0 << "sec.\n"
                << std::endl;
      index.compact_data_for_search();
    }

    {
      index.reposition_frozen_point_to_end();
      std::vector<unsigned> insert_vector;
      for (auto iter : insert_vec)
        insert_vector.emplace_back(iter);
      diskann::Timer timer;
#pragma omp          parallel for
      for (size_t i = 0; i < insert_vec.size(); i++) {
        unsigned p = insert_vector[i];
        index.insert_point(data_load + (size_t) p * (size_t) aligned_dim, paras,
                           p);
      }
      std::cout << "Re-incremental time: " << timer.elapsed() / 1000 << "ms\n";
      index.prune_all_nbrs(paras);
      index.compact_frozen_point();
    }

    inactive_tags.swap(new_inactive_tags);
    active_tags.swap(new_active_tags);

    search_kernel(query, query_num, query_aligned_dim, recall_at, Lvec, index,
                  truthset_file, inactive_tags);
    index.reposition_frozen_point_to_end();
    /*  auto save_path_reinc = save_path + ".reinc" + std::to_string(i);
      index.save(save_path_reinc.c_str());  */
    i++;
  }

  delete[] data_load;

  return 0;
}

int main(int argc, char** argv) {
  if (argc < 17) {
    std::cout << "Correct usage: " << argv[0]
              << " <type>[int8/uint8/float] <data_file> <index_file> <L> <R> "
                 "<C> <alpha> "
                 "<num_rounds> "
              << "<save_graph_file> <#batches> "
                 "<#batch_del_size> <#batch_ins_size> <query_file> "
                 "<truthset_file> <recall@> "
                 "<L1> <L2> ...."
              << std::endl;
    exit(-1);
  }

  int               arg_no = 4;
  unsigned          L = (unsigned) atoi(argv[arg_no++]);
  unsigned          R = (unsigned) atoi(argv[arg_no++]);
  unsigned          C = (unsigned) atoi(argv[arg_no++]);
  float             alpha = (float) std::atof(argv[arg_no++]);
  unsigned          num_rnds = (unsigned) std::atoi(argv[arg_no++]);
  std::string       save_path(argv[arg_no++]);
  unsigned          num_cycles = (unsigned) atoi(argv[arg_no++]);
  unsigned          num_del = (unsigned) atoi(argv[arg_no++]);
  unsigned          num_ins = (unsigned) atoi(argv[arg_no++]);
  std::string       query_file(argv[arg_no++]);
  std::string       truthset(argv[arg_no++]);
  int               recall_at = (int) std::atoi(argv[arg_no++]);
  std::vector<_u64> Lvec;

  for (int ctr = 16; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at."
              << std::endl;
    return -1;
  }

  if (std::string(argv[1]) == std::string("int8"))
    build_incremental_index<int8_t, unsigned>(
        argv[2], argv[3], L, R, C, num_rnds, alpha, save_path, num_cycles,
        num_del, num_ins, query_file, truthset, recall_at, Lvec);
  else if (std::string(argv[1]) == std::string("uint8"))
    build_incremental_index<uint8_t, unsigned>(
        argv[2], argv[3], L, R, C, num_rnds, alpha, save_path, num_cycles,
        num_del, num_ins, query_file, truthset, recall_at, Lvec);
  else if (std::string(argv[1]) == std::string("float"))
    build_incremental_index<float, unsigned>(
        argv[2], argv[3], L, R, C, num_rnds, alpha, save_path, num_cycles,
        num_del, num_ins, query_file, truthset, recall_at, Lvec);
  else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
