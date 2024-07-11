#include "v2/index_merger.h"

#include <numeric>
#include <random>
#include <omp.h>
#include <cstring>
#include <ctime>
#include <timer.h>
#include <iomanip>

#include "aux_utils.h"
#include "mem_aligned_file_reader.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// random number generator
std::random_device dev;
std::mt19937       rng(dev());

std::vector<uint32_t> Lvec;
tsl::robin_map<std::string, uint32_t> params;
float    mem_alpha, merge_alpha;
uint32_t medoid_id = std::numeric_limits<uint32_t>::max();

template<typename T, typename TagT = uint32_t>
void seed_iter(tsl::robin_set<uint32_t> &active_set,
               tsl::robin_set<uint32_t> &inactive_set, T *all_points,
               const std::string &inserted_points_file,
               const std::string &inserted_tags_file,
               const std::string &deleted_tags_file) {
  const uint32_t insert_count = params[std::string("insert_count")];
  const uint32_t delete_count = params[std::string("delete_count")];
  const uint32_t ndims = params[std::string("ndims")];
  std::cout << "ITER: start = " << active_set.size() << ", "
            << inactive_set.size() << "\n";

  // pick `delete_count` tags
  std::vector<uint32_t> active_vec(active_set.begin(), active_set.end());
  std::shuffle(active_vec.begin(), active_vec.end(), rng);
  std::vector<uint32_t> delete_vec;
  if (active_vec.size() < delete_count)
    delete_vec.insert(delete_vec.end(), active_vec.begin(), active_vec.end());
  else
    delete_vec.insert(delete_vec.end(), active_vec.begin(),
                      active_vec.begin() + delete_count);
  // assert(delete_vec.size() == delete_count);
  active_set.clear();
  active_set.insert(active_vec.begin() + delete_vec.size(), active_vec.end());
  std::cout << "ITER: DELETE - " << delete_vec.size() << " IDs in "
            << deleted_tags_file << "\n";
  TagT *del_tags = new TagT[delete_vec.size()];
  for (size_t i = 0; i < delete_vec.size(); i++)
    *(del_tags + i) = delete_vec[i];
  diskann::save_bin<TagT>(deleted_tags_file, del_tags, delete_vec.size(), 1);
  delete[] del_tags;
  /*  std::ofstream deleted_tags_writer(deleted_tags_file, std::ios::trunc);
    for (auto &id : delete_vec) {
      deleted_tags_writer << id << std::endl;
    }
    deleted_tags_writer.close(); */
  // pick `insert_count` tags
  std::vector<uint32_t> inactive_vec(inactive_set.begin(), inactive_set.end());
  std::shuffle(inactive_vec.begin(), inactive_vec.end(), rng);
  std::vector<uint32_t> insert_vec;
  if (inactive_vec.size() < insert_count)
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.end());
  else
    insert_vec.insert(insert_vec.end(), inactive_vec.begin(),
                      inactive_vec.begin() + insert_count);
  inactive_set.clear();
  std::cout << "ITER: INSERT - " << insert_vec.size() << " IDs in "
            << inserted_tags_file << "\n";
  inactive_set.insert(inactive_vec.begin() + insert_vec.size(),
                      inactive_vec.end());
  std::sort(insert_vec.begin(), insert_vec.end());
  TagT *ins_tags = new TagT[insert_vec.size()];
  for (size_t i = 0; i < insert_vec.size(); i++)
    *(ins_tags + i) = insert_vec[i];
  diskann::save_bin<TagT>(inserted_tags_file, ins_tags, insert_vec.size(), 1);
  delete[] ins_tags;
  /*  std::ofstream inserted_tags_writer(inserted_tags_file, std::ios::trunc);
    for (auto &id : insert_vec) {
      inserted_tags_writer << id << std::endl;
    }
    inserted_tags_writer.close();
    */
  std::ofstream inserted_points_writer(inserted_points_file, std::ios::binary);
  T *new_pts = new T[(uint64_t) insert_vec.size() * (uint64_t) ndims];
  for (uint64_t idx = 0; idx < insert_vec.size(); idx++) {
    uint32_t actual_idx = insert_vec[idx];
    T *      src_ptr = all_points + actual_idx * (uint64_t) ndims;
    T *      dest_ptr = new_pts + idx * (uint64_t) ndims;
    std::memcpy(dest_ptr, src_ptr, ndims * sizeof(T));
  }
  uint32_t npts_u32 = insert_vec.size(), ndims_u32 = ndims;
  inserted_points_writer.write((char *) &npts_u32, sizeof(uint32_t));
  inserted_points_writer.write((char *) &ndims_u32, sizeof(uint32_t));
  inserted_points_writer.write(
      (char *) new_pts,
      (uint64_t) insert_vec.size() * (uint64_t) ndims * sizeof(T));
  inserted_points_writer.close();
  delete[] new_pts;

  // balance tags
  inactive_set.insert(delete_vec.begin(), delete_vec.end());
  active_set.insert(insert_vec.begin(), insert_vec.end());
  std::cout << "ITER: end = " << active_set.size() << ", "
            << inactive_set.size() << "\n";
  malloc_stats();
}

float compute_active_recall(const uint32_t *result_tags,
                            const uint32_t  result_count,
                            const uint32_t *gs_tags, const uint32_t gs_count,
                            const tsl::robin_set<uint32_t> &inactive_set) {
  tsl::robin_set<uint32_t> active_gs;
  for (uint32_t i = 0; i < gs_count && active_gs.size() < result_count; i++) {
    auto iter = inactive_set.find(gs_tags[i]);
    if (iter == inactive_set.end()) {
      active_gs.insert(gs_tags[i]);
    }
  }
  uint32_t match = 0;
  for (uint32_t i = 0; i < result_count; i++) {
    match += (active_gs.find(result_tags[i]) != active_gs.end());
  }
  return ((float) match / (float) result_count) * 100;
}

// borrowed from tests/search_disk_index.cpp
template<typename T, typename TagT = uint32_t>
void search_disk_index(const std::string &             index_prefix_path,
                       const tsl::robin_set<uint32_t> &inactive_tags,
                       const std::string &             query_path,
                       const std::string &             gs_path) {
  std::string pq_prefix = index_prefix_path + "_pq";
  std::string disk_index_file = index_prefix_path + "_disk.index";
  std::string warmup_query_file = index_prefix_path + "_sample_data.bin";
  //  uint64_t    num_nodes_to_cache =
  //    params[std::string("disk_search_node_cache_count")];
  uint32_t    num_threads = params[std::string("disk_search_nthreads")];
  uint32_t    beamwidth = params[std::string("beam_width")];
  std::string query_bin = query_path;
  std::string truthset_bin = gs_path;
  uint64_t    recall_at = params[std::string("recall_k")];
  //  uint64_t    search_L = params[std::string("search_L")];

  // hold data
  T *       query = nullptr;
  unsigned *gt_ids = nullptr;
  uint32_t *gt_tags = nullptr;
  float *   gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;

  // load query + truthset
  diskann::load_aligned_bin<T>(query_bin, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Loaded query file" << std::endl;
  diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                         &gt_tags);
  std::cout << "Loaded truthset" << std::endl;
  if (gt_num != query_num) {
    std::cout << "Error. Mismatch in number of queries and ground truth data"
              << std::endl;
  }

  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  std::unique_ptr<diskann::PQFlashIndex<T,uint32_t>> _pFlashIndex(
      new diskann::PQFlashIndex<T,uint32_t>(reader));
  int res = _pFlashIndex->load(num_threads, pq_prefix.c_str(),
                               disk_index_file.c_str(), true);
  if (res != 0) {
    std::cerr << "Failed to load index.\n";
    exit(-1);
  }

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "QPS" << std::setw(16) << "Mean Latency" << std::setw(16) << "90 Latency" << std::setw(16) << "95 Latency" << std::setw(16) << "99 Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16) << recall_string << std::setw(16) << "CPU (s)" << std::endl;
  // prep for search
  std::vector<uint32_t> query_result_ids;
  std::vector<uint32_t> query_result_tags;
  std::vector<float>    query_result_dists;
  query_result_ids.resize(recall_at * query_num);
  query_result_dists.resize(recall_at * query_num);
  query_result_tags.resize(recall_at * query_num);
  std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
  for (size_t test_id = 0; test_id < ::Lvec.size(); test_id++) {
    diskann::QueryStats *stats = new diskann::QueryStats[query_num];
    uint32_t             L = Lvec[test_id];
    auto                 s = std::chrono::high_resolution_clock::now();
#pragma omp              parallel for schedule(dynamic, 1)  // num_threads(1)
    for (_s64 i = 0; i < (int64_t) query_num; i++) {
      _pFlashIndex->cached_beam_search(
          query + (i * query_aligned_dim), recall_at, L,
          query_result_ids_64.data() + (i * recall_at),
          query_result_dists.data() + (i * recall_at), beamwidth, stats + i,
          nullptr, query_result_tags.data() + (i * recall_at));
    }

    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) ((1.0 * query_num) / (1.0 * diff.count()));
    // compute mean recall, IOs
    float mean_recall = 0.0f;
    for (uint32_t i = 0; i < query_num; i++) {
      auto *result_tags = query_result_tags.data() + (i * recall_at);
      auto *gs_tags = gt_tags + (i * gt_dim);
      float query_recall = compute_active_recall(
          result_tags, recall_at, gs_tags, gt_dim, inactive_tags);
      mean_recall += query_recall;
    }
    mean_recall /= query_num;

    float mean_latency = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_90 = (float) diskann::get_percentile_stats(
        stats, query_num, 0.900,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_95 = (float) diskann::get_percentile_stats(
        stats, query_num, 0.950,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_99 = (float) diskann::get_percentile_stats(
        stats, query_num, 0.990,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float latency_999 = (float) diskann::get_percentile_stats(
        stats, query_num, 0.999,
        [](const diskann::QueryStats& stats) { return stats.total_us; });

    float mean_ios = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_ios; });

    float mean_cpuus = (float) diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.cpu_us; });
/*    std::cout << "PQFlashIndex :: Ls - " << L << " , recall-" << recall_at
              << "@" << recall_at << ": " << mean_recall
              << ", mean IOs: " << mean_ios << "\n";
*/
    std::cout << std::setw(6) << L << std::setw(12) << qps << std::setw(16) << mean_latency
              << std::setw(16) << latency_90 << std::setw(16) << latency_95 << std::setw(16) << latency_99 << std::setw(16) << latency_999 << std::setw(16) << mean_ios
              << std::setw(16) << mean_recall << std::setw(16) << mean_cpuus << std::endl;
    delete[] stats;
  }
  diskann::aligned_free(query);
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
}

// borrowed from tests/build_memory_index.cpp
template<typename T>
void build_in_memory_index(const std::string &data_path,
                           const std::string &save_path) {
  const uint32_t R = params[std::string("range")];
  const uint32_t L = params[std::string("mem_l_index")];
  const float    alpha = ::mem_alpha;
  const uint32_t num_threads = params[std::string("mem_nthreads")];

  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  // maximum candidate set size during pruning procedure
  paras.Set<unsigned>("C", 750);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", 0);
  paras.Set<unsigned>("num_threads", num_threads);

  _u64 data_num, data_dim;
  diskann::get_bin_metadata(data_path, data_num, data_dim);
  diskann::Index<T> index(diskann::L2, data_dim, data_num);
  auto              s = std::chrono::high_resolution_clock::now();
  index.build(data_path.c_str(), data_num, paras);
  index.check_graph_quality(1000, paras);
  std::chrono::duration<double> diff =
      std::chrono::high_resolution_clock::now() - s;

  std::cout << "Indexing time: " << diff.count() << "\n";
  index.save(save_path.c_str(), true, true);
}

// borrowed from tests/test_index_merger.cpp
template<typename T>
void run_index_merger(const char *                    disk_in,
                      const std::vector<std::string> &mem_in,
                      const char *disk_out, const char *deleted_tags,
                      diskann::Distance<T> *dist) {
  const uint32_t ndims = params[std::string("ndims")];
  const uint32_t beam_width = params[std::string("beam_width")];
  const uint32_t maxc = params[std::string("merge_maxc")];
  const float    alpha = ::merge_alpha;
  const uint32_t l_index = params[std::string("merge_l_index")];
  const uint32_t range = params[std::string("range")];
  std::cout << "ITER: MERGE " << disk_in << " + " << mem_in[0] << " -> "
            << disk_out << "\n";

  std::cout << "Instantiating IndexMerger\n";
  diskann::IndexMerger<T, uint32_t> merger(disk_in, mem_in, disk_out, deleted_tags, ndims,
                                 dist, beam_width, range, l_index, alpha, maxc);

  std::cout << "Starting merge\n";
  merger.merge();

  std::cout << "Finished merging\n";
}

template<typename T>
void run_iter(const std::string &base_prefix, const std::string &merge_prefix,
              const std::string &mem_prefix, const std::string &deleted_tags,
              tsl::robin_set<uint32_t> &active_set,
              tsl::robin_set<uint32_t> &inactive_set,
              const std::string &query_file, const std::string &gs_file,
              T *data, diskann::Distance<T> *dist_cmp) {
  uint64_t initial_count = active_set.size() + inactive_set.size();
  // files for mem-DiskANN
  std::string mem_pts_file = mem_prefix + ".index.data";
  std::string mem_index_file = mem_prefix + ".index";
  std::string mem_tags_file = mem_prefix + ".index.tags";

  std::cout << "ITER: Seeding iteration"
            << "\n";
  // seed the iteration
  seed_iter(active_set, inactive_set, data, mem_pts_file, mem_tags_file,
            deleted_tags);
  assert(active_set.size() + inactive_set.size() == initial_count);

  // std::cout << "ITER: Building memory index for inserted points"
  //         << "\n";
  // build in-memory index
  //  build_in_memory_index<T>(mem_pts_file, mem_index_file);
  std::cout << "waiting for input: \n";

  // run merge
  std::vector<std::string> mem_in;
  mem_in.push_back(mem_index_file);
  std::cout << "ITER: Folding mem-DiskANN into SSD-DiskANN"
            << "\n";
  run_index_merger<T>(base_prefix.c_str(), mem_in, merge_prefix.c_str(),
                      deleted_tags.c_str(), dist_cmp);

  // search PQ Flash Index
  std::cout << "ITER: Searching SSD-DiskANN"
            << "\n";
  search_disk_index<T>(merge_prefix, inactive_set, query_file, gs_file);
}

template<typename T, typename TagT = uint32_t>
void run_all_iters(const std::string &base_prefix,
                   const std::string &merge_prefix,
                   const std::string &mem_prefix,
                   const std::string &deleted_tags,
                   const std::string &data_file,
                   const std::string &active_tags_file,
                   const std::string &query_file, const std::string &gs_file,
                   diskann::Distance<T> *dist_cmp, const uint32_t n_iters) {
  // load all data points
  std::unique_ptr<T[]> data;
  uint64_t             npts = 0, ndims = 0;
  diskann::load_bin<T>(data_file, data, npts, ndims);
  params[std::string("ndims")] = ndims;

  // load active tags
  size_t tags_pts, tags_dim;
  TagT * active_tags_data;

  diskann::load_bin<TagT>(active_tags_file, active_tags_data, tags_pts,
                          tags_dim);
  std::cout << "Tag dim = " << tags_dim << ", Tags pts = " << tags_pts
            << std::endl;

  tsl::robin_set<uint32_t> active_tags;
  for (size_t i = 0; i < tags_pts; i++)
    active_tags.insert(*(active_tags_data + i));

  // read medoid ID from base_prefix
  std::ifstream disk_reader(base_prefix + "_disk.index", std::ios::binary);
  disk_reader.seekg(2 * sizeof(uint64_t), std::ios::beg);
  uint64_t medoid = std::numeric_limits<uint64_t>::max();
  disk_reader.read((char *) &medoid, sizeof(uint64_t));
  std::cout << "Detected medoid = " << medoid
            << " ==> excluding from insert/deletes.\n";
  ::medoid_id = medoid;

  // generate inactive tags
  tsl::robin_set<uint32_t> inactive_tags;
  for (uint32_t i = 0; i < npts; i++) {
    auto iter = active_tags.find(i);
    if (iter == active_tags.end()) {
      inactive_tags.insert(i);
    }
  }

  // remove medoid from active_set
  active_tags.erase(::medoid_id);

  for (uint32_t i = 0; i < n_iters; i++) {
    std::cout << "ITER: Iteration #" << i + 1 << "\n";
    std::string base, merge;
    if (i % 2 == 0) {
      base = base_prefix;
      merge = merge_prefix;
    } else {
      base = merge_prefix;
      merge = base_prefix;
    }
    run_iter<T>(base, merge, mem_prefix, deleted_tags, active_tags,
                inactive_tags, query_file, gs_file, data.get(), dist_cmp);
  }
}

int main(int argc, char **argv) {
  if (argc < 16) {
    std::cout << "Correct usage: " << argv[0]
              << " <type[int8/uint8/float]> <WORKING_FOLDER> <base_prefix> "
                 "<merge_prefix> <mem_prefix>"
              << " <deleted_tags_file> <full_data_bin> <query_bin> <truthset>"
              << " <niters> <insert_count> <delete_count> <range> <recall_k> "
                 "<search_L1> <search_L2> ..."
              << "\n WARNING: Other parameters set inside CPP source."
              << std::endl;
    exit(-1);
  }
  std::cout.setf(std::ios::unitbuf);

  int         arg_no = 1;
  std::string index_type = argv[arg_no++];
  TMP_FOLDER = argv[arg_no++];
  // assert(index_type == std::string("float"));
  std::string base_prefix(argv[arg_no++]);
  std::string merge_prefix(argv[arg_no++]);
  std::string mem_prefix(argv[arg_no++]);
  std::string deleted_tags_file(argv[arg_no++]);
  std::string data_bin(argv[arg_no++]);
  std::string query_bin(argv[arg_no++]);
  std::string gt_bin(argv[arg_no++]);
  uint32_t    n_iters = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    insert_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    delete_count = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    range = (uint32_t) atoi(argv[arg_no++]);
  uint32_t    recall_k = (uint32_t) atoi(argv[arg_no++]);

  for (int ctr = arg_no; ctr < argc; ctr++) {
    _u64 curL = std::atoi(argv[ctr]);
    if (curL >= recall_k)
      ::Lvec.push_back(curL);
  }
  params[std::string("insert_count")] = insert_count;
  params[std::string("delete_count")] = delete_count;
  params[std::string("range")] = range;
  params[std::string("recall_k")] = recall_k;
  //  params[std::string("search_L")] = search_L;

  // hard-coded params
  params[std::string("disk_search_node_cache_count")] = 200000;
  params[std::string("disk_search_nthreads")] = 96;
  params[std::string("beam_width")] = 4;
  params[std::string("mem_l_index")] = 75;
  mem_alpha = 1.2;
  merge_alpha = 1.2;
  params[std::string("mem_nthreads")] = 64;
  params[std::string("merge_maxc")] = (uint32_t)(range * 2.5);
  params[std::string("merge_l_index")] = 75;

  if (index_type == std::string("float")) {
    diskann::DistanceL2 dist_cmp;
    run_all_iters<float>(base_prefix, merge_prefix, mem_prefix,
                         deleted_tags_file, data_bin,
                         base_prefix + "_disk.index.tags", query_bin, gt_bin,
                         &dist_cmp, n_iters);
  } else if (index_type == std::string("uint8")) {
    diskann::DistanceL2UInt8 dist_cmp;
    run_all_iters<uint8_t>(base_prefix, merge_prefix, mem_prefix,
                           deleted_tags_file, data_bin,
                           base_prefix + "_disk.index.tags", query_bin, gt_bin,
                           &dist_cmp, n_iters);
  } else if (index_type == std::string("int8")) {
    diskann::DistanceL2Int8 dist_cmp;
    run_all_iters<int8_t>(base_prefix, merge_prefix, mem_prefix,
                          deleted_tags_file, data_bin,
                          base_prefix + "_disk.index.tags", query_bin, gt_bin,
                          &dist_cmp, n_iters);
  } else {
    std::cout << "Unsupported type : " << index_type << "\n";
  }
  std::cout << "Exiting\n";
}
