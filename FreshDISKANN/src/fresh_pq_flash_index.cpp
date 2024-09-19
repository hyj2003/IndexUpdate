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
#define MAX_PROCESSING_PAGES 65536

#define THREAD_INSERT 5
#define THREAD_DELETE 5

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
#define PER_THREAD_BUF_SIZE (uint64_t) (65536 * 64 * 4)

namespace diskann {
  template<typename T, typename TagT>
  FreshPQFlashIndex<T, TagT>::FreshPQFlashIndex(
      std::shared_ptr<AlignedFileReader> &fileReader)
      : PQFlashIndex<T, TagT>(fileReader), page_locks_write_(kPageLocks), page_locks_read_(kPageLocks) {
  }

  template<typename T, typename TagT>
  FreshPQFlashIndex<T, TagT>::~FreshPQFlashIndex() {
    while (!scratch_queue_.empty()) {
      scratch_queue_.pop().output_writer->close();
    }
    delete this->update_thread_pq_scratch;
    delete this->update_thread_pq_scratch_u16;
    delete this->update_thread_page_scratch;
  }
  template<typename T, typename TagT>
  int FreshPQFlashIndex<T, TagT>::load(uint32_t num_threads, const char *pq_prefix,
                                  const char *disk_index_file, uint32_t max_index_num, 
                                  bool load_tags) {
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

    // max_index_num = this->num_points = npts_u64; // Used when don't want to affect the original index file.
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
    this->inv_tags = new _u32[max_index_num];
    this->max_index_num_ = max_index_num;
    if (load_tags) {
      std::string tag_file = disk_index_file;
      tag_file = tag_file + ".tags";
      this->load_tags(tag_file);
      TagT *new_tags = new TagT[max_index_num];
      memcpy(new_tags, this->tags, sizeof(TagT) * this->num_points);
      delete[] this->tags;
      this->tags = new_tags;
      this->cur_max_id_ = 0;
      for (_u64 i = 0; i < this->num_points; i++) {
        if (uint32_t(this->tags[i]) == std::numeric_limits<uint32_t>::max()) {
          continue;
        }
        this->inv_tags[this->tags[i]] = i;
        this->cur_max_id_ = std::max(this->cur_max_id_, (_u32)this->tags[i] + 1);
      }
      std::cout << "Current max id: "<< this->cur_max_id_ << std::endl;
    }
    /* Extend disk index */
    _u64 new_n_sectors = ROUND_UP(this->max_index_num_, this->nnodes_per_sector) / this->nnodes_per_sector;
    _u64 new_disk_index_file_size = (new_n_sectors + 1) * SECTOR_LEN;
    {
      std::fstream disk_writer(this->disk_index_file, std::ios::in | std::ios::out | std::ios::binary);
      std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
      *(_u64 *) (sector_buf.get() + 0 * sizeof(_u64)) = new_disk_index_file_size;
      *(_u64 *) (sector_buf.get() + 1 * sizeof(_u64)) = max_index_num;
      //    *(_u64 *) (sector_buf.get() + 2 * sizeof(_u64)) = ndims_64;
      *(_u64 *) (sector_buf.get() + 2 * sizeof(_u64)) = medoid_id_on_file;
      *(_u64 *) (sector_buf.get() + 3 * sizeof(_u64)) = this->max_node_len;
      *(_u64 *) (sector_buf.get() + 4 * sizeof(_u64)) = this->nnodes_per_sector;
      disk_writer.seekp(0, std::ios::beg);
      disk_writer.write(sector_buf.get(), SECTOR_LEN);
      disk_writer.close();
    }
    {
      std::fstream disk_writer(this->disk_index_file, std::ios::app | std::ios::out | std::ios::binary);
      _u16 old_n_sectors = ROUND_UP(this->num_points, this->nnodes_per_sector) / this->nnodes_per_sector;
      std::cout << (uint64_t)disk_writer.tellp() << std::endl;
      disk_writer.seekp(0, std::ios::end);
      std::cout << "Previous position: " << old_n_sectors << " " << (uint64_t)disk_writer.tellp() << std::endl;
      std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(SECTOR_LEN);
      memset(sector_buf.get(), 0, sizeof(SECTOR_LEN));
      size_t accumulate_pages = 0;
      for (_u64 i = old_n_sectors; i < new_n_sectors; i++) {
        disk_writer.write(sector_buf.get(), SECTOR_LEN);
        if (++accumulate_pages == 512) {
          disk_writer.flush();
          accumulate_pages = 0;
        }
      }
      std::cout << "Write position: " <<  new_n_sectors << " " << (uint64_t)disk_writer.tellp() << std::endl;
      disk_writer.flush();
      disk_writer.close();
    }
    /* Extend pq table */
    {
      auto old_pq = this->data;
      this->data = new _u8[this->n_chunks * max_index_num];
      memcpy(this->data, old_pq, this->n_chunks * max_index_num * sizeof(_u8));
      delete old_pq;
    }
    
    std::cout << "Allocating thread scratch space -- " << PER_THREAD_BUF_SIZE / (1<<20) << " MB / thread.\n";
    alloc_aligned((void**) &this->update_thread_pq_scratch, 4 * num_threads * PER_THREAD_BUF_SIZE, SECTOR_LEN);
    alloc_aligned((void**) &this->update_thread_pq_scratch_u16, 4 * num_threads * PER_THREAD_BUF_SIZE * 2, SECTOR_LEN);
    alloc_aligned((void**) &this->update_thread_page_scratch, MAX_PROCESSING_PAGES * SECTOR_LEN * 2, SECTOR_LEN);
    
    for (uint32_t i = 0; i < num_threads * 4; i++) {
      UpdateThreadData<T> u_data;
      u_data.scratch.scratch_ = this->update_thread_pq_scratch + PER_THREAD_BUF_SIZE * i;
      u_data.scratch.scratch_u16_ = this->update_thread_pq_scratch_u16 + PER_THREAD_BUF_SIZE * i;
      u_data.ctx = this->reader->get_ctx();
      u_data.output_writer = new std::fstream;
      u_data.output_writer->open(this->disk_index_file, std::ios::in | std::ios::out | std::ios::binary);
      scratch_queue_.push(u_data);
      scratch_queue_.push_notify_all();
    }
    for (uint32_t i = 0; i < MAX_PROCESSING_PAGES; i++) {
      std::pair<char*, char*> p;
      assert(p.first == nullptr);
      auto pair = PagePair(this->update_thread_page_scratch + SECTOR_LEN * (2 * i),
                           this->update_thread_page_scratch + SECTOR_LEN * (2 * i + 1));
      page_pairs_queue_.push(pair);
    }
    this->in_degree_ = new _u32[max_index_num];
    memset(this->in_degree_, 0, sizeof(_u32) * max_index_num);
    memset(last_hit, 0, sizeof(_u32) * this->num_points);
    // this->page_locks_.resize(kPageLocks);
    // this->page_locks_.reserve(kPageLocks);
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
      // RecycleId(*(tag_data + i));
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
      this->delete_cache_.insert(std::make_pair(nhood.id, std::move(non_deleted_nbrs.size())));
      this->del_filter_.set(nhood.id);
    }
    std::cout << "Deleted nodes cached with " << this->delete_cache_.size() << " nodes." << std::endl;
    // free buf
    aligned_free((void*) buf);
    assert(deleted_nodes.size() == disk_deleted_ids.size());
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::ComputeInDegree() {
    uint64_t sectors_per_scan = SECTORS_PER_MERGE;
    char* buf = nullptr;
    alloc_aligned((void**)&buf, sectors_per_scan * SECTOR_LEN, SECTOR_LEN);

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    IOContext ctx = data.ctx;
    uint32_t                 n_scanned = 0;
    uint32_t                 base_offset = NODE_SECTOR_NO(0) * SECTOR_LEN;
    std::vector<AlignedRead> reads(1);
    reads[0].buf = buf;
    reads[0].len = sectors_per_scan * SECTOR_LEN;
    reads[0].offset = base_offset;
    std::cout << this->num_points << std::endl;
    while (n_scanned < this->num_points) {
      memset(buf, 0, sectors_per_scan * SECTOR_LEN);
      assert(this->reader);

      this->reader->read(reads, ctx);
      // scan each sector
      for (uint32_t i = 0; i < sectors_per_scan && n_scanned < this->num_points;
           i++) {
        char *sector_buf = buf + i * SECTOR_LEN;
        // scan each node
        for (uint32_t j = 0;
             j < this->nnodes_per_sector && n_scanned < this->num_points; j++) {
          // std::cout << j << std::endl;
          char *node_buf = OFFSET_TO_NODE(sector_buf, n_scanned);
          // if in delete_set, add to deleted_nodes
          if (this->delete_cache_.find(n_scanned) == this->delete_cache_.end()) {
            // create disk node object from backing buf instead of `buf`
            DiskNode<T> node(n_scanned, OFFSET_TO_NODE_COORDS(node_buf),
                             OFFSET_TO_NODE_NHOOD(node_buf));
            uint32_t nnbrs = node.nnbrs;
            unsigned *nbrs = node.nbrs;
            for (uint32_t k = 0; k < nnbrs; k++) {
              this->in_degree_[nbrs[k]]++;
              out_edges[n_scanned].insert(nbrs[k]);
              in_nodes[nbrs[k]].insert(n_scanned);
              assert(this->in_degree_[nbrs[k]] == in_nodes[nbrs[k]].size());
              assert(nbrs[k] < this->num_points);
            }
          }
          n_scanned++;
        }
      }
      reads[0].offset = NODE_SECTOR_NO(n_scanned) * SECTOR_LEN;
    }
    std::cout << "Compute in-degree finished.\n";
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    aligned_free((void*) buf);
    // std::vector<_u32> deg(this->in_degree_, this->in_degree_ + this->num_points);
    // sort(deg.begin(), deg.end(), [](_u32 i, _u32 j) { return i > j; });
    // std::cout << "0 ~ 9:" << std::endl;
    // for (int i = 0; i < 10; i++) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "10 ~ 90:" << std::endl;
    // for (int i = 10; i < 100; i += 10) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "100 ~ 900:" << std::endl;
    // for (int i = 100; i < 1000; i += 100) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "1000 ~ 9000:" << std::endl;
    // for (int i = 1000; i < 10000; i += 1000) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "10000 ~ 90000:" << std::endl;
    // for (int i = 10000; i < 100000; i += 10000) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "100000 ~ 900000:" << std::endl;
    // for (int i = 100000; i < 1000000; i += 100000) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "1000000 ~ 9000000:" << std::endl;
    // for (int i = 1000000; i < 10000000; i += 1000000) {
    //   std::cout << deg[i] << " ";
    // }
    // std::cout << std::endl;
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
          {
            // std::shared_lock<std::shared_mutex> page_cache_guard(page_cache_lock_);
            // auto p_iter = page_cache_.find(retset[marker].id);
            if (iter != this->nhood_cache.end()) {
              cached_nhoods.push_back(
                  std::make_pair(retset[marker].id, iter->second));
              // #pragma omp atomic 
              //   cache_hit++;
              if (stats != nullptr) {
                stats->n_cache_hits++;
              }
            } else if (false) {

            } else {
              frontier.push_back(retset[marker].id);
            }
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
        {
          std::shared_lock<std::shared_mutex> guard(page_locks_read_[NODE_SECTOR_NO(((size_t) frontier[0])) & (kPageLocks - 1)]);
          // printf("node: %u\n", frontier[0]);
          this->reader->read(frontier_read_reqs, ctx);
        }
        // n_ops = this->reader->submit_reqs(frontier_read_reqs, ctx);  
        // asynchronous IO linux
        // may need implementation of async to overlap the cost of locks
        if (stats != nullptr) {
          stats->io_us += io_timer.elapsed();
        }
      }
      // printf("Get delete_cache_lock_ in search.\n");
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
        {
          std::shared_lock<std::shared_mutex> delete_guard(this->delete_cache_lock_);
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              visited.insert(id);
              assert(id < this->cur_max_id_);
              if (!this->del_filter_.get(id)) {
                n_ids.push_back(id);
              } else if (two_hops < this->two_hops_lim) {
                auto iter = this->delete_cache_.find(id);
                auto &vec = iter->second;
                for (auto nei : vec) {
                  if (two_hops >= this->two_hops_lim) break;
                  if (visited.find(nei) == visited.end() && !this->del_filter_.get(nei)) {
                    n_ids.push_back(nei);
                    visited.insert(nei);
                    two_hops++;
                  }
                }
              }
            }
          }
        }
        {
          std::shared_lock<std::shared_mutex> insert_guard(this->insert_edges_lock_);
          auto iter = this->insert_edges_.find(cached_nhood.first);
          if (iter != this->insert_edges_.end()) {
            for (auto id : iter->second) {
              if (visited.find(id) != visited.end() && del_filter_.get(id)) continue;
              assert(id < this->cur_max_id_);
              n_ids.emplace_back(id);
              visited.insert(id);
            }
          }
        }
        // assert(n_ids.size() < maxc);
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
      // if (!frontier.empty()) {
      //   this->reader->get_events(ctx, n_ops);
      // }
      for (auto &reqs : frontier_read_reqs) {
        // printf("Pass page: %llu\n", reqs.offset / SECTOR_LEN);
        PushVisitedPage(reqs.offset / SECTOR_LEN, (char *)reqs.buf); // send current page to page processor
      }
      for (auto &frontier_nhood : frontier_nhoods) {
        char *node_disk_buf =
            OFFSET_TO_NODE(frontier_nhood.second, frontier_nhood.first);
        unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
        _u64      nnbrs = (_u64)(*node_buf);
        T *       node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
        assert(data_buf_idx < MAX_N_CMPS);
        // printf("Frontier: %lu\n", frontier_nhood.first);
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
        {
          std::shared_lock<std::shared_mutex> delete_guard(this->delete_cache_lock_);
          for (_u64 m = 0; m < nnbrs; ++m) {
            unsigned id = node_nbrs[m];
            if (visited.find(id) != visited.end()) {
              continue;
            } else {
              assert(id < this->cur_max_id_);
              visited.insert(id);
              if (!this->del_filter_.get(id)) {
                n_ids.push_back(id);
              } else if (two_hops < this->two_hops_lim) {
                auto iter = this->delete_cache_.find(id);
                assert(iter != this->delete_cache_.end());
                auto &vec = iter->second;
                for (auto nei : vec) {
                  if (two_hops >= this->two_hops_lim) break;
                  if (visited.find(nei) == visited.end() && !this->del_filter_.get(nei)) {
                    n_ids.push_back(nei);
                    visited.insert(nei);
                    two_hops++;
                  }
                }
              }
            }
          }
        }
        {
          std::shared_lock<std::shared_mutex> insert_guard(this->insert_edges_lock_);
          auto iter = this->insert_edges_.find(frontier_nhood.first);
          if (iter != this->insert_edges_.end()) {
            for (auto id : iter->second) {
              if (visited.find(id) != visited.end() && del_filter_.get(id)) continue;
              n_ids.emplace_back(id);
              visited.insert(id);
              assert(id < this->cur_max_id_);
            }
          }
        }
        assert(n_ids.size() < maxc);
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
        // std::cout << n_ids.size() << std::endl;
        n_ids.clear();
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
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::occlude_list_pq_simd(std::vector<Neighbor> &pool, 
                                                        const uint32_t id, 
                                                        std::vector<Neighbor> &result, 
                                                        std::vector<float> &occlude_factor, 
                                                        uint8_t* scratch,
                                                        uint16_t* scratch_u16) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());
    const size_t kSize = 64;
    const _u64 nchunks = this->n_chunks;
    uint8_t *scratch_T = scratch + 256 * nchunks;
    { // reorder pq code for simd
      for (size_t i = 0; i < pool.size(); i += kSize) {
        size_t len = std::min(pool.size() - i, kSize);
        uint8_t* pq = scratch_T + i * nchunks;
        if (len != kSize) {
          memset(pq, 0, sizeof(uint8_t) * kSize * nchunks);
        }
        for (size_t j = 0; j < len; j++) {
          for (size_t k = 0; k < nchunks; k++) {
            pq[k * kSize + j] = this->data[nchunks * pool[i + j].id + k];
          }
        }
      }
    }
    PQComputer<T> *computer = 
      new PQComputer<T>(this->data,
                     scratch_T,
                     std::min(pool.size(), this->maxc),
                     nchunks,
                     scratch,
                     &this->pq_table);
    uint16_t *dist_u16 = scratch_u16;
    float cur_alpha = 1;
    size_t rerank_size = range + 20;
    while (cur_alpha <= alpha && result.size() < rerank_size) {
      uint32_t start = 0;
      while (result.size() < rerank_size && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        computer->compute_batch_dists(this->data + pool[start].id * nchunks, 
                                      start / 64 * 64, 
                                      dist_u16 + start / 64 * 64);
        for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          // djk = dist(p.id, pool[t.id])
          float djk = static_cast<float>(dist_u16[t]);
          occlude_factor[t] = (std::max)(occlude_factor[t], pool[t].distance / djk);
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
    for (auto &iter : result) {
      this->compute_pq_dists(id, &(iter.id), &iter.distance, 1, scratch);
    }
    sort(result.begin(), result.end());
    result.resize(std::min(result.size(), range));
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::PruneNeighbors(DiskNode<T> &disk_node, 
                                                  std::vector<_u32> &new_nbrs,
                                                  _u8 *scratch, _u16 *scratch_u16) {
    std::vector<float> id_nhood_dists(new_nbrs.size(), 0.0f);
    assert(scratch != nullptr);
    _u32 id = disk_node.id;
    {
      auto computer = new PQComputer(this->data, 
                                    nullptr, 
                                    new_nbrs.size(), 
                                    this->n_chunks, 
                                    nullptr, 
                                    &this->pq_table);
      computer->compute_pq_dists(id, new_nbrs.data(), id_nhood_dists.data(), new_nbrs.size());
    }
    std::vector<Neighbor> cand_nbrs(new_nbrs.size());
    for (size_t i = 0; i < std::min(this->maxc * 2, new_nbrs.size()); i++) {
      cand_nbrs[i].id = new_nbrs[i];
      cand_nbrs[i].distance = id_nhood_dists[i];
    }
    // sort and keep only maxc neighbors
    std::sort(cand_nbrs.begin(), cand_nbrs.end());
    if (cand_nbrs.size() > this->maxc) {
      cand_nbrs.resize(this->maxc);
    }
    std::vector<Neighbor> pruned_nbrs;
    std::vector<float> occlude_factor(cand_nbrs.size(), 0.0f);
    pruned_nbrs.reserve(this->range);
    this->occlude_list_pq_simd(cand_nbrs, id, pruned_nbrs, occlude_factor, scratch, scratch_u16);
    // copy back final nbrs
    disk_node.nnbrs = pruned_nbrs.size();
    *(disk_node.nbrs - 1) = disk_node.nnbrs;
    for (uint32_t i = 0; i < pruned_nbrs.size(); i++) {
      disk_node.nbrs[i] = pruned_nbrs[i].id;
    }
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::ProcessPage(_u64 sector_id, UpdateThreadData<T> *upd) {
    std::unique_lock<std::mutex> write_guard(page_locks_write_[sector_id & (kPageLocks - 1)]);
    char *sector_buf = upd->scratch.page_buf_;
    char *tmp_buf = upd->scratch.page_buf_copied_;
    // {
    //   std::vector<AlignedRead> read_reqs;
    //   read_reqs.emplace_back(sector_id * SECTOR_LEN, SECTOR_LEN, tmp_buf);
    //   this->reader->read(read_reqs, upd->ctx);
    // }
    assert(sector_buf != nullptr);
    // assert(tmp_buf != nullptr);
    memcpy(tmp_buf, sector_buf, SECTOR_LEN);
    assert(sector_id != 0);
    _u32 cur_node_id = (sector_id - 1) * this->nnodes_per_sector;
    std::vector<DiskNode<T>> disk_nodes;
    for (uint32_t i = 0; i < this->nnodes_per_sector && cur_node_id < this->cur_max_id_; i++) {
      char *node_disk_buf = OFFSET_TO_NODE(sector_buf, cur_node_id);
      disk_nodes.emplace_back(cur_node_id, 
                              OFFSET_TO_NODE_COORDS(node_disk_buf),
                              OFFSET_TO_NODE_NHOOD(node_disk_buf));
      last_hit[cur_node_id] = sector_id;
      cur_node_id++;
    }
    assert(insert_edges_.size() == 0);
    bool dump_flag = false;
    for (auto &disk_node : disk_nodes) {
      auto      id = disk_node.id;
      bool      change = false;
      auto      nnbrs = disk_node.nnbrs;
      unsigned *nbrs = disk_node.nbrs;
      std::vector<_u32> cand_nbrs;
      std::vector<_u32> old_nbrs;
      if (del_filter_.get(disk_node.id)) {
        continue;
      }
      {
        std::shared_lock<std::shared_mutex> delete_guard(this->delete_cache_lock_);
        // std::cout << id << " ";
        for (uint32_t i = 0; i < nnbrs; i++) {
          old_nbrs.push_back(nbrs[i]);
          if (nbrs[i] == id) {
            continue;
          }
          if (!del_filter_.get(nbrs[i])) {
            cand_nbrs.emplace_back(nbrs[i]);
          } else {
            // if (--mark[nbrs[i]] == 0) ... (atomic)
            change = true;
            auto iter = this->delete_cache_.find(nbrs[i]);
            if (iter == this->delete_cache_.end()) {
              printf("In nodes: %u\n", in_degree_[nbrs[i]]);
              for (auto id : in_nodes[nbrs[i]]) {
                printf("%u ", id);
              }
              std::cout << std::endl;
            }
            assert(iter != this->delete_cache_.end());
            auto &vec = iter->second;
            for (auto j : vec) {
              if (!del_filter_.get(j) && j != id) {
                cand_nbrs.emplace_back(j);
              }
            }
          }
        }
      }
      // printf("New nbrs size %lu\n", new_nbrs.size());
      {
        std::unique_lock<std::shared_mutex> insert_guard(insert_edges_lock_);
        std::shared_lock<std::shared_mutex> delete_guard(this->delete_cache_lock_);
        auto iter = this->insert_edges_.find(id);
        if (iter != insert_edges_.end()) {
          for (auto nbrs : iter->second) {
            old_nbrs.emplace_back(nbrs);
            if (del_filter_.get(nbrs)) 
              continue;
            change = true;
            cand_nbrs.emplace_back(nbrs);
          }
          this->insert_edges_.erase(iter);
        }
      }
      if (change) {
        // printf("Processing %lu\n", sector_id);
        uint8_t *scratch = upd->scratch.scratch_;
        uint16_t *scratch_u16 = upd->scratch.scratch_u16_;
        std::sort(cand_nbrs.begin(), cand_nbrs.end());
        cand_nbrs.erase(std::unique(cand_nbrs.begin(), cand_nbrs.end()), cand_nbrs.end());
        PruneNeighbors(disk_node, cand_nbrs, scratch, scratch_u16);
        dump_flag = true;
        auto      new_nnbrs = disk_node.nnbrs;
        unsigned *new_nbrs = disk_node.nbrs;
        std::unique_lock<std::shared_mutex> in_degree_guard(this->in_degree_lock_);
        tsl::robin_set<_u32> old_nodes(old_nbrs.begin(), old_nbrs.end());
        tsl::robin_set<_u32> new_nodes(new_nbrs, new_nbrs + new_nnbrs);
        // assert(old_nodes.size() == old_nbrs.size());
        assert(new_nodes.size() == new_nnbrs);
        // if (!SetEqual(old_nodes, out_edges[disk_node.id])) {
        //   printf("last hit %u is %u\n", disk_node.id, last_hit[disk_node.id]);
        //   std::vector<_u32> vec;

        //   for (auto id : out_edges[disk_node.id]) {
        //     vec.push_back(id);
        //   }
        //   sort(vec.begin(), vec.end());
        //   for (auto id : vec) {
        //     std::cout << id << " ";
        //   }
        //   vec.clear();
        //   std::cout << "\n";
        //   for (auto id : old_nodes) {
        //     vec.push_back(id);
        //   }
        //   sort(vec.begin(), vec.end());
        //   for (auto id : vec) {
        //     std::cout << id << " ";
        //   }
        //   std::cout << "\n";
        // }
        // assert(SetEqual(old_nodes, out_edges[disk_node.id]));
        // for (uint32_t i = 0; i < new_nnbrs; i++) {
        //   this->in_degree_[new_nbrs[i]]++;
        //   nodes.insert(new_nbrs[i]);
        //   // in_nodes[new_nbrs[i]].insert(disk_node.id);
        // }
        for (auto id : new_nodes) {
          this->in_degree_[id]++;
          // in_nodes[new_nbrs[i]].insert(disk_node.id);
        }
        for (auto id : old_nodes) {
          this->in_degree_[id]--;
          // if (new_nodes.find(id) != new_nodes.end()) {
          //   new_nodes.erase(id);
          // } else {
            // assert(in_nodes[id].find(disk_node.id) != in_nodes[id].end());
            // in_nodes[id].erase(disk_node.id);
          // }
          // in_nodes[id].erase(in_nodes[id].find(disk_node.id));
          CheckAndRecycle(id);
        }
        for (auto id : new_nodes) {
          in_nodes[id].insert(disk_node.id);
        }
        out_edges[disk_node.id] = std::move(new_nodes);
      }
      // printf("change: %d\n", change);
    }
    // {
    //   std::unique_lock<std::shared_mutex> page_cache_guard(this->page_cache_lock_);
    //   page_cache_.insert(std::make_pair(sector_id, tmp_buf));
    // }
    /* write on page[sector_id] */
    assert(upd->output_writer != nullptr);
    if (dump_flag) {
      std::fstream &output_writer = *upd->output_writer;
      output_writer.seekp(sector_id * (uint64_t) SECTOR_LEN, std::ios::beg);
      uint64_t prev_pos = output_writer.tellp();
      std::unique_lock<std::shared_mutex> read_guard(page_locks_read_[sector_id & (kPageLocks - 1)]);
      dump_to_disk(output_writer, (sector_id - 1) * this->nnodes_per_sector, sector_buf, 1);
      output_writer.flush();
      uint64_t cur_pos = output_writer.tellp();
      assert(cur_pos - prev_pos == SECTOR_LEN);
      // bool flag = false;
      // for (uint64_t i = 0; i < SECTOR_LEN; i++) {
      //   if (sector_buf[i] != tmp_buf[i]) {
      //     flag = true;
      //   }
      // }
      // assert(flag);
      // flag = false;
      // std::vector<AlignedRead> read_reqs;
      // read_reqs.emplace_back(sector_id * SECTOR_LEN, SECTOR_LEN, tmp_buf);
      // this->reader->read(read_reqs, upd->ctx);
      // for (uint64_t i = 0; i < SECTOR_LEN; i++) {
      //   if (sector_buf[i] != tmp_buf[i]) {
      //     flag = true;
      //   }
      // }
      // assert(!flag);
      // cur_node_id = (sector_id - 1) * this->nnodes_per_sector;
      // disk_nodes.clear();
      // for (uint32_t i = 0; i < this->nnodes_per_sector && cur_node_id < this->cur_max_id_; i++) {
      //   char *node_disk_buf = OFFSET_TO_NODE(tmp_buf, cur_node_id);
      //   disk_nodes.emplace_back(cur_node_id, 
      //                           OFFSET_TO_NODE_COORDS(node_disk_buf),
      //                           OFFSET_TO_NODE_NHOOD(node_disk_buf));
      //   last_hit[cur_node_id] = sector_id;
      //   cur_node_id++;
      // }
      // for (auto &disk_node : disk_nodes) {
      //   auto      id = disk_node.id;
      //   auto      nnbrs = disk_node.nnbrs;
      //   unsigned *nbrs = disk_node.nbrs;
      //   tsl::robin_set<_u32> nodes(nbrs, nbrs + nnbrs);
      //   assert(SetEqual(nodes, out_edges[id]));
      //   out_edges[id] = std::move(nodes);
      // }
    }
    {
      std::unique_lock<std::shared_mutex> guard(this->page_in_process_lock_);
      page_in_process_.erase(page_in_process_.find(sector_id));
    }
    /* Finish updating */
    auto pair = PagePair(sector_buf, tmp_buf);
    this->page_pairs_queue_.push(pair);
    this->page_pairs_queue_.push_notify_all();
    this->scratch_queue_.push(*upd);
    this->scratch_queue_.push_notify_all();
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::PushVisitedPage(_u64 sector_id, char *sector_buf) {
    /* copy buffers */
    /* Check if a page is already in processing */
    {
      std::unique_lock<std::shared_mutex> guard(this->page_in_process_lock_);
      if (page_in_process_.find(sector_id) != page_in_process_.end()) {
        return ;
      } else {
        page_in_process_.insert(sector_id);
      }
    }
    auto pair = this->page_pairs_queue_.pop();
    while (pair.first == nullptr) {
      this->page_pairs_queue_.wait_for_push_notify();
      pair = this->page_pairs_queue_.pop();
    }
    UpdateThreadData<T> data = this->scratch_queue_.pop();
    while (data.output_writer == nullptr) {
      this->scratch_queue_.wait_for_push_notify();
      data = this->scratch_queue_.pop();
    }
    // IOContext ctx = data.ctx;
    data.scratch.page_buf_ = pair.first;
    data.scratch.page_buf_copied_ = pair.second;
    memcpy(data.scratch.page_buf_, sector_buf, SECTOR_LEN);
        // std::cout << (pair.first == nullptr) << std::endl;
    _u32 cur_node_id = (sector_id - 1) * this->nnodes_per_sector;
    std::vector<DiskNode<T>> disk_nodes;
    for (uint32_t i = 0; i < this->nnodes_per_sector && cur_node_id < this->num_points; i++) {
      char *node_disk_buf = OFFSET_TO_NODE(data.scratch.page_buf_, cur_node_id);
      disk_nodes.emplace_back(cur_node_id, 
                              OFFSET_TO_NODE_COORDS(node_disk_buf),
                              OFFSET_TO_NODE_NHOOD(node_disk_buf));
      cur_node_id++;
    }
    Requests<T> new_req(sector_id, std::move(data));
    // std::cout << reqs_queue_.size() << std::endl;
    reqs_queue_.push(new_req);
    reqs_queue_.push_notify_all();
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::ProcessUpdateRequests(bool update_flag) {
    /* Get a request */
    auto req = this->reqs_queue_.pop();
    while (update_flag && req.sector_id == 0) {
      this->reqs_queue_.wait_for_push_notify();
      req = this->reqs_queue_.pop();
    }
    // std::cout << req.sector_id << std::endl;
    _u32 cur_node_id = (req.sector_id - 1) * this->nnodes_per_sector;
    std::vector<DiskNode<T>> disk_nodes;
    for (uint32_t i = 0; i < this->nnodes_per_sector && cur_node_id < this->num_points; i++) {
      char *node_disk_buf = OFFSET_TO_NODE(req.upd.scratch.page_buf_, cur_node_id);
      disk_nodes.emplace_back(cur_node_id, 
                              OFFSET_TO_NODE_COORDS(node_disk_buf),
                              OFFSET_TO_NODE_NHOOD(node_disk_buf));
      cur_node_id++;
    }
    if (!update_flag) {
      return ;
    }
    /* Process a page */
    // std::cout << "Into page processing ..." << std::endl;
    ProcessPage(req.sector_id, &req.upd);
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::PushDelete(_u32 external_id) { // need to set delete filter
    /* Read then process*/
    _u32 internal_id = this->inv_tags[external_id];
    /* Read */
    auto pair = this->page_pairs_queue_.pop();
    while (pair.first == nullptr) {
      this->page_pairs_queue_.wait_for_push_notify();
      pair = this->page_pairs_queue_.pop();
    }
    UpdateThreadData<T> data = this->scratch_queue_.pop();
    while (data.output_writer == nullptr) {
      this->scratch_queue_.wait_for_push_notify();
      data = this->scratch_queue_.pop();
    }
    IOContext ctx = data.ctx;
    // printf("Internal: %u\n", internal_id);
    {
      std::shared_lock<std::shared_mutex> guard(page_locks_read_[NODE_SECTOR_NO(((size_t) internal_id)) & (kPageLocks - 1)]);
      std::vector<AlignedRead> read_reqs;
      read_reqs.emplace_back(NODE_SECTOR_NO(((size_t) internal_id)) * SECTOR_LEN, SECTOR_LEN, pair.first);
      this->reader->read(read_reqs, ctx);
    }
    // printf("Exit1: %u\n", internal_id);
    char *node_disk_buf =
          OFFSET_TO_NODE(pair.first, internal_id);
    unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
    _u64      nnbrs = (_u64)(*node_buf);
    unsigned *node_nbrs = (node_buf + 1);
    std::vector<_u32> new_nbrs(node_nbrs, node_nbrs + nnbrs);
    tsl::robin_set<_u32> new_set(node_nbrs, node_nbrs + nnbrs);
    assert(new_set.size() == new_nbrs.size());
    {
      std::unique_lock<std::shared_mutex> guard(in_degree_lock_);

      for (_u64 i = 0; i < nnbrs; i++) {
        this->in_degree_[node_nbrs[i]]--;
        assert(in_nodes[node_nbrs[i]].find(internal_id) != in_nodes[node_nbrs[i]].end());
        in_nodes[node_nbrs[i]].erase(internal_id);
        assert(in_degree_[node_nbrs[i]] == in_nodes[node_nbrs[i]].size());
        CheckAndRecycle(node_nbrs[i]);
      }
      std::unique_lock<std::shared_mutex> insert_edges_guard(insert_edges_lock_);
      auto iter = insert_edges_.find(internal_id);
      if (iter != insert_edges_.end()) {
        for (auto id : iter->second) {
          this->in_degree_[id]--;
          CheckAndRecycle(id);
          in_nodes[id].erase(internal_id);
          new_nbrs.push_back(id);
        }
        insert_edges_.erase(iter);
      }
    }
    // printf("Node and in-degree: %u %u\n", internal_id, this->in_degree_[internal_id]);
    if (in_degree_[internal_id]) {
      std::unique_lock<std::shared_mutex> guard(delete_cache_lock_);
      // printf("Exit3: %u\n", internal_id);
      delete_cache_.insert(std::make_pair(internal_id, std::move(new_nbrs)));
      // printf("Exit4: %u\n", internal_id);
      this->del_filter_.set(internal_id);
      // printf("%u %lu\n", this->in_degree_[internal_id], delete_cache_.size());
    }
    // printf("Exit5: %u\n", internal_id);
    this->scratch_queue_.push(data);
    this->scratch_queue_.push_notify_all();
    this->page_pairs_queue_.push(pair);
    this->page_pairs_queue_.push_notify_all();
    // ProcessPage(NODE_SECTOR_NO(((size_t) internal_id)), &data);
    // PushVisitedPage(NODE_SECTOR_NO(((size_t) internal_id)), data.scratch.page_buf_);
    /* release free ids*/
    // RecycleId(internal_id);
  }
  template<typename T, typename TagT>
  void FreshPQFlashIndex<T, TagT>::PushInsert(_u32 external_id, T *point) {
    /* Assign an internal id & update tags */
    _u32 internal_id = GetInternalId();
    assert(1000 <= internal_id && internal_id < 10000);
    this->tags[internal_id] = external_id;
    this->inv_tags[external_id] = internal_id;
    std::vector<uint8_t> pq_coords = this->deflate_vector(point);
    uint64_t pq_offset = (uint64_t) internal_id * (uint64_t) this->n_chunks;
    memcpy(this->data + pq_offset, pq_coords.data(), this->n_chunks * sizeof(uint8_t));
    /* Search */
    std::vector<_u64> res(l_index); res.reserve(l_index);
    this->filtered_beam_search(point, l_index, l_index, res.data(), nullptr, beam_width);
    std::vector<_u32> insertions; insertions.reserve(l_index);
    for (auto id : res) {
      insertions.emplace_back(id);
    }
    // std::cout << "Insert size: " << insertions.size() << std::endl;
    /* Add edges */
    // std::unique_lock<std::shared_mutex> guard(insert_edges_lock_);
    // for (auto iter : insertions) {
    //   insert_edges_[iter.id].emplace_back(internal_id);
    // }
    // insert_edges_.insert(std::make_pair(internal_id, std::move(insertions)));

    /* Add edges */
    auto pair = this->page_pairs_queue_.pop();
    while (pair.first == nullptr) {
      this->page_pairs_queue_.wait_for_push_notify();
      pair = this->page_pairs_queue_.pop();
    }
    UpdateThreadData<T> data = this->scratch_queue_.pop();
    while (data.output_writer == nullptr) {
      this->scratch_queue_.wait_for_push_notify();
      data = this->scratch_queue_.pop();
    }
    IOContext ctx = data.ctx;
    T        *vec_buf;
    unsigned *node_buf;
    {
      std::unique_lock<std::mutex> write_guard(page_locks_write_[NODE_SECTOR_NO(((size_t) internal_id)) & (kPageLocks - 1)]);
      std::vector<AlignedRead> read_reqs;
      read_reqs.emplace_back(NODE_SECTOR_NO(((size_t) internal_id)) * SECTOR_LEN, SECTOR_LEN, pair.first);
      this->reader->read(read_reqs, ctx);
      char *node_disk_buf =
            OFFSET_TO_NODE(pair.first, internal_id);
      vec_buf = OFFSET_TO_NODE_COORDS(node_disk_buf);
      node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
      memcpy(vec_buf, point, sizeof(T) * this->data_dim);
      DiskNode<T> disk_node(internal_id, vec_buf, node_buf);
      PruneNeighbors(disk_node, insertions, data.scratch.scratch_, data.scratch.scratch_u16_);
      std::fstream &output_writer = *data.output_writer;
      std::unique_lock<std::shared_mutex> read_guard(page_locks_read_[NODE_SECTOR_NO(((size_t) internal_id)) & (kPageLocks - 1)]);
      this->dump_to_disk(output_writer, (NODE_SECTOR_NO(internal_id) - 1) * this->nnodes_per_sector, pair.first, 1);
      output_writer.flush();
    }
    {
      std::unique_lock<std::shared_mutex> insert_edges_guard(insert_edges_lock_);
      std::unique_lock<std::shared_mutex> in_degree_guard(in_degree_lock_);
      _u64      nnbrs = (_u64)(*node_buf);
      unsigned *node_nbrs = (node_buf + 1);
      in_degree_[internal_id] += nnbrs;
      for (_u64 i = 0; i < nnbrs; i++) {
        // std::cout << node_nbrs[i] << " ";
        insert_edges_[node_nbrs[i]].emplace_back(internal_id);
        in_degree_[node_nbrs[i]]++;
      }
      // std::cout << std::endl;
    }
    this->scratch_queue_.push(data);
    this->scratch_queue_.push_notify_all();
    this->page_pairs_queue_.push(pair);
    this->page_pairs_queue_.push_notify_all();
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
