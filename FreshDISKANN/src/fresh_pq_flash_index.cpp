#include "fresh_pq_flash_index.h"
#include "timer.h"

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present
#define NODE_SECTOR_NO(node_id) (((_u64)(node_id)) / this->nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((_u64) node_id) % this->nnodes_per_sector) * this->max_node_len)

// offset into sector where node_id's nhood starts
#define NODE_SECTOR_OFFSET(sector_buf, node_id) \
  ((char *) sector_buf +                        \
   ((((_u64) node_id) % this->nnodes_per_sector) * this->max_node_len))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + this->data_dim * sizeof(T))

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

namespace {
  void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                        const _u8 *all_coords, const _u64 ndims, _u8 *out) {
    for (_u64 i = 0; i < n_ids; i++) {
      memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
    }
  }

  void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                      const _u64 pq_nchunks, const float *pq_dists,
                      float *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
      const float *chunk_dists = pq_dists + 256 * chunk;
      if (chunk < pq_nchunks - 1) {
        _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
      }
      for (_u64 idx = 0; idx < n_pts; idx++) {
        _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
        dists_out[idx] += chunk_dists[pq_centerid];
      }
    }
  }

  template<typename T>
  std::shared_ptr<diskann::Distance<T>> get_distance_function();

  template<>
  std::shared_ptr<diskann::Distance<float>> get_distance_function() {
    return std::shared_ptr<diskann::Distance<float>>(new diskann::DistanceL2());
  }
  template<>
  std::shared_ptr<diskann::Distance<uint8_t>> get_distance_function() {
    return std::shared_ptr<diskann::Distance<uint8_t>>(
        new diskann::DistanceL2UInt8());
  }
  template<>
  std::shared_ptr<diskann::Distance<int8_t>> get_distance_function() {
    return std::shared_ptr<diskann::Distance<int8_t>>(
        new diskann::DistanceL2Int8());
  }

}  // namespace

namespace diskann {
  template<typename T, typename TagT>
  FreshPQFlashIndex<T, TagT>::FreshPQFlashIndex(
      std::shared_ptr<AlignedFileReader> &fileReader)
      : PQFlashIndex<T, TagT>(fileReader) {
  }

  template<typename T, typename TagT>
  FreshPQFlashIndex<T, TagT>::~FreshPQFlashIndex() {
    for (auto iter : delete_cache_) {
      delete[] iter.second.second;
    }
  }
  template<typename T, typename TagT>
  int FreshPQFlashIndex<T, TagT>::load(uint32_t num_threads, const char *pq_prefix,
                                  const char *disk_index_file, bool load_tags) {
    std::string pq_table_bin = std::string(pq_prefix) + "_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(pq_prefix) + "_compressed.bin";
    std::string medoids_file = std::string(disk_index_file) + "_medoids.bin";
    std::string centroids_file =
        std::string(disk_index_file) + "_centroids.bin";

    size_t pq_file_dim, pq_file_num_centroids;
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);

    this->disk_index_file = std::string(disk_index_file);

    if (pq_file_num_centroids != 256) {
      std::cout << "Error. Number of PQ centroids is not 256. Exitting."
                << std::endl;
      return -1;
    }

    this->data_dim = pq_file_dim;
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

    this->pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
    std::cout << "load_pq_centroid_bin" << std::endl;
    std::ifstream nsg_meta(disk_index_file, std::ios::binary);

    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(nsg_meta, expected_file_size);
    if (actual_index_size != expected_file_size) {
      std::cout << "File size mismatch for " << disk_index_file
                << " (size: " << actual_index_size << ")"
                << " with meta-data size: " << expected_file_size << std::endl;
      return -1;
    }

    _u64 disk_nnodes;
    READ_U64(nsg_meta, disk_nnodes);
    if (disk_nnodes != this->num_points) {
      std::cout << "Mismatch in #points for compressed data file and disk "
                   "index file: "
                << disk_nnodes << " vs " << this->num_points << std::endl;
      return -1;
    }

    size_t medoid_id_on_file;  // ndims;
    // read u64 to read ndims_64 of data points
    //    READ_U64(nsg_meta, ndims);
    READ_U64(nsg_meta, medoid_id_on_file);
    READ_U64(nsg_meta, this->max_node_len);
    READ_U64(nsg_meta, this->nnodes_per_sector);
    /*
        std::cout << "Disk num_points: " << disk_nnodes
                  << ", medoid_id_on_file: " << medoid_id_on_file
                  << ", max_node_len: " << max_node_len
                  << ", num_nodes_in_sector: " << nnodes_per_sector <<
       std::endl;
                  */

    this->max_degree = ((this->max_node_len - this->data_dim * sizeof(T)) / sizeof(unsigned)) - 1;
    this->two_hops_lim = this->max_degree * 2;
    /*
        std::cout << "Disk-Index File Meta-data: ";
        std::cout << "# nodes per sector: " << nnodes_per_sector;
        std::cout << ", max node len (bytes): " << max_node_len;
        std::cout << ", max node degree: " << max_degree << std::endl;
        */
    nsg_meta.close();

    // open AlignedFileReader handle to nsg_file
    std::string nsg_fname(disk_index_file);
    this->reader->open(nsg_fname, false, false);

    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    if (file_exists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(medoids_file, this->medoids, this->num_medoids, tmp_dim);

      if (tmp_dim != 1) {
        std::stringstream stream;
        stream << "Error loading medoids file. Expected bin format of m times "
                  "1 vector of uint32_t."
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      if (!file_exists(centroids_file)) {
        //        std::cout
        //            << "Centroid data file not found. Using corresponding
        //            vectors "
        //               "for the medoids "
        //            << std::endl;
        this->use_medoids_data_as_centroids();
      } else {
        size_t num_centroids, aligned_tmp_dim;
        diskann::load_aligned_bin<float>(centroids_file, this->centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
        if (aligned_tmp_dim != this->aligned_dim || num_centroids != this->num_medoids) {
          std::stringstream stream;
          stream << "Error loading centroids data file. Expected bin format of "
                    "m times data_dim vector of float, where m is number of "
                    "medoids "
                    "in medoids file."
                 << std::endl;
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    } else {
      this->num_medoids = 1;
      this->medoids = new uint32_t[1];
      this->medoids[0] = (_u32)(medoid_id_on_file);
      this->use_medoids_data_as_centroids();
    }
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // num_medoids = 1;
    // load tags
    if (load_tags) {
      std::string tag_file = disk_index_file;
      tag_file = tag_file + ".tags";
      this->load_tags(tag_file);
    }
    return 0;
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::load_deleted_list(
    const char *deleted_tags_file) {
    std::cout << "Loading deleted list from " << deleted_tags_file << std::endl;
    size_t tag_num, tag_dim;
    TagT * tag_data;
    diskann::load_bin<TagT>(deleted_tags_file, tag_data, tag_num, tag_dim);
    tsl::robin_set<TagT> deleted_tags;
    tsl::robin_set<uint32_t> disk_deleted_ids;
    for(size_t i = 0; i < tag_num; i++) {
      deleted_tags.insert(*(tag_data + i));
    }
    for(uint32_t i=0; i < this->num_points; i++) {
      TagT i_tag = this->tags[i];
      if (deleted_tags.find(i_tag) != deleted_tags.end()) {
        disk_deleted_ids.insert(i);
      }
    }
    char* buf = nullptr;
    alloc_aligned((void**)&buf, SECTORS_PER_MERGE * SECTOR_LEN, SECTOR_LEN);

    // scan deleted nodes and get 
    std::vector<DiskNode<T>> deleted_nodes;
    uint64_t backing_buf_size = (uint64_t) disk_deleted_ids.size() * ROUND_UP(this->max_node_len, 32);
    backing_buf_size = ROUND_UP(backing_buf_size, 256);
    // std::cout << "ALLOC: " << (backing_buf_size << 10) << "KiB aligned buffer for deletes.\n";
    char* delete_backing_buf = nullptr;
    alloc_aligned((void**) &delete_backing_buf, backing_buf_size, 256);
    memset(delete_backing_buf, 0, backing_buf_size);
    std::cout << "Starting reading neighbors ...";
    this->scan_deleted_nodes(disk_deleted_ids, deleted_nodes, buf, delete_backing_buf, SECTORS_PER_MERGE);
    std::cout << " done." << std::endl;
    // insert into delete_cache_
    this->delete_cache_.clear();
    for(auto &nhood : deleted_nodes) {
      // WARNING :: ASSUMING DISK GRAPH DEGREE NEVER GOES OVER 512
      assert(nhood.nnbrs < 512);
      std::vector<uint32_t> non_deleted_nbrs;
      for(uint32_t i=0;i<nhood.nnbrs;i++){
        uint32_t id = nhood.nbrs[i];
        auto iter = disk_deleted_ids.find(id);
        if (iter == disk_deleted_ids.end()) {
          non_deleted_nbrs.push_back(id);
        }
      }
      uint32_t *nbr = new uint32_t[non_deleted_nbrs.size()];
      memcpy(nbr, non_deleted_nbrs.data(), non_deleted_nbrs.size() * sizeof(uint32_t));
      this->delete_cache_.insert(std::make_pair(nhood.id, std::make_pair(non_deleted_nbrs.size(), nbr)));
      this->del_filter_.set(nhood.id);
    }
    std::cout << "Deleted nodes cached with " << this->delete_cache_.size() << " nodes." << std::endl;
    // free buf
    aligned_free((void*) buf);
    assert(deleted_nodes.size() == disk_deleted_ids.size());
    assert(disk_deleted_nhoods.size() == disk_deleted_ids.size());
  }

  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::filtered_beam_search(
      const T *query, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, QueryStats *stats,
      Distance<T> *output_dist_func, TagT *res_tags) {
    // iterate to fixed point
    std::vector<Neighbor> expanded_nodes_info;
    this->filter_disk_iterate_to_fixed_point(query, l_search, beam_width,
                                      expanded_nodes_info, nullptr,
                                      output_dist_func, stats);

    // fill in `indices`, `distances`
    for (uint32_t i = 0; i < k_search; i++) {
      indices[i] = expanded_nodes_info[i].id;
      if (distances != nullptr) {
        distances[i] = expanded_nodes_info[i].distance;
      }
      if (res_tags != nullptr && this->tags != nullptr) {
        res_tags[i] = this->tags[indices[i]];
      }
    }
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::filter_disk_iterate_to_fixed_point(
      const T *query1, const uint32_t l_search, const uint32_t beam_width,
      std::vector<Neighbor> &expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> *coord_map, Distance<T> *output_dist_func,
      QueryStats *stats, ThreadData<T> *passthrough_data,
      tsl::robin_set<uint32_t> *exclude_nodes) {
    // only pull from sector scratch if ThreadData<T> not passed as arg
    ThreadData<T> data;
    if (passthrough_data == nullptr) {
      data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
    } else {
      data = *passthrough_data;
    }

    for (uint32_t i = 0; i < this->data_dim; i++) {
      data.scratch.aligned_query_float[i] = query1[i];
    }
    memcpy(data.scratch.aligned_query_T, query1, this->data_dim * sizeof(T));
    const T *    query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    IOContext ctx = data.ctx;
    auto      query_scratch = &(data.scratch);

    // reset query
    query_scratch->reset();

    // scratch space to compute distances between FP32 Query and INT8 data
    float *scratch = query_scratch->aligned_scratch;
    _mm_prefetch((char *) scratch, _MM_HINT_T0);

    // pointers to buffers for data
    T *   data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    this->pq_table.populate_chunk_distances(query, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](
        const unsigned *ids, const _u64 n_ids, float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                         pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                       dists_out);
    };

    Timer                 query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(l_search + 1);
    tsl::robin_set<_u64> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(4096);
    _u32                        best_medoid = 0;
    float                       best_dist = (std::numeric_limits<float>::max)();
    std::vector<SimpleNeighbor> medoid_dists;
    for (_u64 cur_m = 0; cur_m < this->num_medoids; cur_m++) {
      float cur_expanded_dist = this->dist_cmp_float->compare(
          query_float, this->centroid_data + this->aligned_dim * cur_m,
          (unsigned) this->aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = this->medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset[0].id = best_medoid;
    retset[0].distance = dist_scratch[0];
    retset[0].flag = true;
    visited.insert(best_medoid);

    unsigned cur_list_size = 1;

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;
    unsigned two_hops = 0;
    int n_ops = 0;
    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    std::vector<AlignedRead> frontier_read_reqs;
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    std::vector<_u32> n_ids;
    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      n_ids.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: _u64 marker = k - 1;
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          auto iter = this->nhood_cache.find(retset[marker].id);
          if (iter != this->nhood_cache.end()) {
            cached_nhoods.push_back(
                std::make_pair(retset[marker].id, iter->second));
            // #pragma omp atomic 
            //   cache_hit++;
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
          } else {
            frontier.push_back(retset[marker].id);
          }
          retset[marker].flag = false;
          if (this->count_visited_nodes) {
            reinterpret_cast<std::atomic<_u32> &>(
                this->node_visit_counter[retset[marker].id].second)
                .fetch_add(1);
          }
        }
        marker++;
      }
      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              NODE_SECTOR_NO(((size_t) id)) * SECTOR_LEN, SECTOR_LEN,
              fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
        n_ops = this->reader->submit_reqs(frontier_read_reqs, ctx);  
        // synchronous IO linux
        // need implementation  of async to overlap the cost of locks
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
      }
      {
      std::shared_lock<std::shared_mutex> guard(this->delete_cache_lock_);
      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto global_cache_iter = this->coord_cache.find(cached_nhood.first);
        T *  node_fp_coords = global_cache_iter->second;
        T *  node_fp_coords_copy = data_buf + (data_buf_idx * this->aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, this->data_dim * sizeof(T));
        float cur_expanded_dist = this->dist_cmp->compare(query, node_fp_coords_copy,
                                                    (unsigned) this->aligned_dim);
        bool exclude_cur_node = false;
        if (exclude_nodes != nullptr) {
          exclude_cur_node =
              (exclude_nodes->find(cached_nhood.first) != exclude_nodes->end());
        }
        // only figure in final list if
        if (!exclude_cur_node) {
          // added for IndexMerger calls
          if (coord_map != nullptr) {
            coord_map->insert(
                std::make_pair(cached_nhood.first, node_fp_coords_copy));
          }
          full_retset.push_back(
              Neighbor((unsigned) cached_nhood.first, cur_expanded_dist, true));
        }
        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        // compute node_nbrs <-> query dists in PQ space
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            if (!this->del_filter_.get(id)) {
              n_ids.push_back(id);
              visited.insert(id);
            } else if (two_hops <= this->two_hops_lim) {
              auto iter = this->delete_cache_.find(id);
              auto sz = iter->second.first;
              auto ne = iter->second.second;
              for (_u32 i = 0; i < sz && two_hops <= this->two_hops_lim; i++) {
                if (visited.find(ne[i]) == visited.end() &&
                    !this->del_filter_.get(ne[i])) {
                  n_ids.push_back(ne[i]);
                  two_hops++;
                  visited.insert(ne[i]);
                }
              }
            }
          }
        }
        cpu_timer.reset();
        compute_dists(n_ids.data(), n_ids.size(), dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += nnbrs;
          stats->cpu_us += cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < (_u64)n_ids.size(); ++m) {
          unsigned id = n_ids[m];
          cmps++;
          float dist = dist_scratch[m];
          // std::cerr << "dist: " << dist << std::endl;
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search))
            continue;
          Neighbor nn(id, dist, true);
          auto     r = InsertIntoPool(
              retset.data(), cur_list_size,
              nn);  // Return position in sorted list where nn inserted.
          if (cur_list_size < l_search)
            ++cur_list_size;
          if (r < nk)
            nk = r;  // nk logs the best position in the retset that was
                     // updated
                     // due to neighbors of n.
        }
        n_ids.clear();
      }
      if (!frontier.empty()) {
        this->reader->get_events(ctx, n_ops);
      }
      for (auto &frontier_nhood : frontier_nhoods) {
        char *node_disk_buf =
            OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
        unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
        _u64      nnbrs = (_u64)(*node_buf);
        T *       node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);

        T *node_fp_coords_copy = data_buf + (data_buf_idx * this->aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, this->data_dim * sizeof(T));
        float cur_expanded_dist =
            this->dist_cmp->compare(query, node_fp_coords_copy, (unsigned) this->data_dim);
        bool exclude_cur_node = false;
        if (exclude_nodes != nullptr) {
          exclude_cur_node = (exclude_nodes->find(frontier_nhood.first) !=
                              exclude_nodes->end());
        }
        // if node is to be excluded from final search results
        if (!exclude_cur_node) {
          // added for IndexMerger calls
          if (coord_map != nullptr) {
            coord_map->insert(
                std::make_pair(frontier_nhood.first, node_fp_coords_copy));
          }
          full_retset.push_back(
              Neighbor(frontier_nhood.first, cur_expanded_dist, true));
        }
        unsigned *node_nbrs = (node_buf + 1);
        // compute node_nbrs <-> query dists in PQ space
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          } else {
            if (!this->del_filter_.get(id)) {
              n_ids.push_back(id);
              visited.insert(id);
            } else if (two_hops <= this->two_hops_lim) {
              auto iter = this->delete_cache_.find(id);
              auto sz = iter->second.first;
              auto ne = iter->second.second;
              for (_u32 i = 0; i < sz && two_hops <= this->two_hops_lim; i++) {
                if (visited.find(ne[i]) == visited.end() &&
                    !this->del_filter_.get(ne[i])) {
                  n_ids.push_back(ne[i]);
                  two_hops++;
                  visited.insert(ne[i]);
                }
              }
            }
          }
        }
        cpu_timer.reset();
        compute_dists(n_ids.data(), n_ids.size(), dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += nnbrs;
          stats->cpu_us += cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < (_u64)n_ids.size(); ++m) {
          unsigned id = n_ids[m];
          cmps++;
          float dist = dist_scratch[m];
          // std::cerr << "dist: " << dist << std::endl;
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search))
            continue;
          Neighbor nn(id, dist, true);
          auto     r = InsertIntoPool(
              retset.data(), cur_list_size,
              nn);  // Return position in sorted list where nn inserted.
          if (cur_list_size < l_search)
            ++cur_list_size;
          if (r < nk)
            nk = r;  // nk logs the best position in the retset that was
                     // updated
                     // due to neighbors of n.
        }
        n_ids.clear();
      }
      }
      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
    }
    // std::cout << two_hops << std::endl;
    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    // return data to ConcurrentQueue only if popped from it
    if (passthrough_data == nullptr) {
      this->thread_data.push(data);
      this->thread_data.push_notify_all();
    }

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  // instantiations
  template class FreshPQFlashIndex<float, int32_t>;
  template class FreshPQFlashIndex<_s8, int32_t>;
  template class FreshPQFlashIndex<_u8, int32_t>;
  template class FreshPQFlashIndex<float, uint32_t>;
  template class FreshPQFlashIndex<_s8, uint32_t>;
  template class FreshPQFlashIndex<_u8, uint32_t>;
  template class FreshPQFlashIndex<float, int64_t>;
  template class FreshPQFlashIndex<_s8, int64_t>;
  template class FreshPQFlashIndex<_u8, int64_t>;
  template class FreshPQFlashIndex<float, uint64_t>;
  template class FreshPQFlashIndex<_s8, uint64_t>;
  template class FreshPQFlashIndex<_u8, uint64_t>;
}  // namespace diskann
