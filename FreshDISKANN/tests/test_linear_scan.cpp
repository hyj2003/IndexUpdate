#include "pq_table.h"
#include "utils.h"
#include "pq_flash_index.h"
#include "aux_utils.h"
#include "timer.h"

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
void compute_recall(int query_num, std::string truthset_bin, int topk, std::vector<uint32_t> &query_result) {
    std::cout << "Computing recall ..." << std::endl;
    unsigned*         gt_ids = nullptr;
    uint32_t*         gt_tags_tmp = nullptr;
    float*            gt_dists = nullptr;
    size_t            gt_num, gt_dim;
    diskann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim,
                           &gt_tags_tmp);
    gt_dists = nullptr;
    auto recall = (float) diskann::calculate_recall(
            (_u32) query_num, gt_ids, gt_dists, (_u32) gt_dim,
            query_result.data(), (_u32) topk,
            (_u32) topk);
    std::cout << "Recall: " << recall << std::endl;
}
template<typename T>
void linear_scan(int argc, char **argv) {
    char *pq_prefix = argv[1];
    std::string pq_table_bin = std::string(pq_prefix) + "_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(pq_prefix) + "_compressed.bin";
    std::string origin_file = std::string(argv[2]);
    std::string gt_file = std::string(argv[3]);
    int topk = std::atoi(argv[4]);

    size_t pq_file_dim, pq_file_num_centroids;
    diskann::get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);

    if (pq_file_num_centroids != 256) {
      std::cout << "Error. Number of PQ centroids is not 256. Exitting."
                << std::endl;
      return ;
    }
    size_t npts_u64, nchunks_u64;
    _u8 *                pq_data = nullptr;
    diskann::load_bin<_u8>(pq_compressed_vectors, pq_data, npts_u64, nchunks_u64);
    std::cout << "Load bin finished." << " npts = " << npts_u64 << " nchunks = " << nchunks_u64 << std::endl;
    diskann::FixedChunkPQTable<T> pq_table;
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
    std::cout << "Load pq centroid finished." << std::endl;
    std::string query_bin = std::string(argv[2]);
    size_t            query_num, query_dim, query_aligned_dim;
    T *queries;
    diskann::load_aligned_bin<T>(query_bin, queries, query_num, query_dim,
                               query_aligned_dim);
    std::cout << query_aligned_dim << std::endl;
    query_num = 100;
    std::cout << "Load query finished." << std::endl;
    // diskann::ThreadData<T> *t_data = new diskann::ThreadData<T>(aligned_dim, 4096);
    std::cout << 32768 * 32 * sizeof(_u8) << " " << npts_u64 * sizeof(float) << std::endl;
    float *pq_dists = nullptr;
    diskann::alloc_aligned((void **) &pq_dists, 32768 * 32 * sizeof(_u8), 256);
    float *fp_dists = nullptr;
    diskann::alloc_aligned((void **) &fp_dists, npts_u64 * sizeof(float), 256);
    std::vector<uint32_t> query_result;
    std::cout << "Starting linear scan ..." << std::endl;
    diskann::Timer timer;
    for (size_t i = 0; i < query_num; i++) {
        // std::cout << i << "\n";
        T *query = queries + i * query_aligned_dim;
        pq_table.populate_chunk_distances(query, pq_dists);
        ::pq_dist_lookup(pq_data, npts_u64, nchunks_u64, pq_dists, fp_dists);
        std::priority_queue<std::pair<float, size_t>> q;
        for (size_t j = 0; j < npts_u64; j++) {
            // std::cout << fp_dists[j] << std::endl;
            if ((int)q.size() >= topk) {
                if (q.top().first > fp_dists[j]) {
                    q.pop();
                    q.emplace(fp_dists[j], j);
                }
            } else {
                q.emplace(fp_dists[j], j);
            }
        }
        while (!q.empty()) {
            query_result.push_back(q.top().second);
            q.pop();
        }
    }
    
    std::cout << "Search done in " << (float) timer.elapsed() / 1000 / 1000 << "s." << std::endl;
    compute_recall(query_num, gt_file, topk, query_result);

}
int main(int argc, char **argv) {
    if (argc != 6) {
        std::cout << "Usage: [pq_prefix] [origin_file] [gt_file] [topk] [data_type]" << std::endl;
        return 0;
    }
    std::string data_type = std::string(argv[5]);
    if (data_type == "float") {
        linear_scan<float>(argc, argv);
    } else if (data_type == "uint8") {
        linear_scan<uint8_t>(argc, argv);
    } else if (data_type == "int8") {
        linear_scan<int8_t>(argc, argv);
    } else {
        std::cout << "Data type doesn't support." << std::endl;
    }
    return 0;
}