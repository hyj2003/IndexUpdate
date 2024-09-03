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
                      const _u64 pq_nchunks, const uint8_t *pq_dists,
                      uint16_t *dists_out) {
    _mm_prefetch((char *) dists_out, _MM_HINT_T0);
    _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
    _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(uint16_t));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const uint8_t *chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
            _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
            _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
            dists_out[idx] += chunk_dists[pq_centerid];
        }
    }
}
void pq_dist_lookup_simd(const uint8_t* pq_ids, const size_t n_pts, 
                        const size_t pq_nchunks, const size_t group_size,
                        uint8_t*pq_dists, uint16_t* dists_out) {
    for(size_t i = 0; i < n_pts; i += group_size) {
        assert(group_size * sizeof(uint8_t) * 8 == 512);
        __m512i candidates_lo = _mm512_setzero_si512(); // 用于存储低32个uint16_t的累加结果
        __m512i candidates_hi = _mm512_setzero_si512(); // 用于存储高32个uint16_t的累加结果
        for(size_t j = 0; j < pq_nchunks; j++) {
            uint8_t* chunk_pq_dists = pq_dists + 256 * j;
            __m512i lut00 = _mm512_loadu_epi8(chunk_pq_dists);
            __m512i lut01 = _mm512_loadu_epi8(chunk_pq_dists + 64);
            __m512i lut10 = _mm512_loadu_epi8(chunk_pq_dists + 2 * 64);
            __m512i lut11 = _mm512_loadu_epi8(chunk_pq_dists + 3 * 64);

            const __m512i comps = _mm512_loadu_epi8(pq_ids + i * pq_nchunks + j * group_size);
            const __mmask64 bit8_m = _mm512_movepi8_mask(comps);

            const __m512i partial_0 = _mm512_permutex2var_epi8(lut00, comps, lut01);
            const __m512i partial_1 = _mm512_permutex2var_epi8(lut10, comps, lut11);
            const __m512i partial_sum = _mm512_mask_blend_epi8(bit8_m,partial_0,partial_1);


            // todo: pq_nchunks大于256还是会有问题
            // Extract the lower 32 uint8_t values and convert to uint16_t
            __m256i lower_256 = _mm512_extracti32x8_epi32(partial_sum, 0);
            __m512i partial_sum_lo = _mm512_cvtepu8_epi16(lower_256);

            // Extract the upper 32 uint8_t values and convert to uint16_t
            __m256i upper_256 = _mm512_extracti32x8_epi32(partial_sum, 1);
            __m512i partial_sum_hi = _mm512_cvtepu8_epi16(upper_256);

            candidates_lo = _mm512_adds_epu16(candidates_lo, partial_sum_lo);
            candidates_hi = _mm512_adds_epu16(candidates_hi, partial_sum_hi);
        }

        _mm512_storeu_epi16(dists_out + i, candidates_lo);
        _mm512_storeu_epi16(dists_out + i + 32, candidates_hi);
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
void linear_scan_simd(int argc, char **argv) {
    char *pq_prefix = argv[1];
    std::string pq_table_bin = std::string(pq_prefix) + "_pivots.bin";
    std::string pq_compressed_vectors =
        std::string(pq_prefix) + "_compressed.bin";
    std::string origin_file = std::string(argv[6]);
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
    _u8                 *pq_data = nullptr, *o_data = nullptr;
    diskann::load_bin<_u8>(pq_compressed_vectors, pq_data, npts_u64, nchunks_u64);
    o_data = new _u8[npts_u64 * nchunks_u64];
    memcpy(o_data, pq_data, npts_u64 * nchunks_u64);
    std::cout << "Load bin finished." << " npts = " << npts_u64 << " nchunks = " << nchunks_u64 << std::endl;
    diskann::SIMDPQTable<T> pq_table;
    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);
    const size_t gsize = 64;
    //
    {
        size_t npts, dim;
        T *data;
        diskann::load_bin<T>(origin_file, data, npts, dim);
        std::cout << "Load origin finished." << " npts = " << npts << " dim = " << dim << std::endl;
        std::vector<int> ids;
        ids.resize(npts);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 rnd(19260817);
        std::shuffle(ids.begin(), ids.end(), rnd);
        size_t n = std::min(size_t(10000), npts);
        std::vector<T> sample(dim * n);
        for (size_t i = 0; i < n; i++) {
            std::memcpy(sample.data() + i * dim, data + ids[i] * dim, dim * sizeof(T));
        }
        std::cout << "Sampling finished." << std::endl;
        pq_table.train(n, sample.data());
        pq_table.quantize_all_to_all();
        std::cout << "Training finished." << std::endl;
        diskann::transpose_pq_codes(pq_data, npts_u64, nchunks_u64, gsize);
    }
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
    float *fp_vec = nullptr;
    diskann::alloc_aligned((void **) &fp_vec, 256 * (size_t)MAX_PQ_CHUNKS * sizeof(float), 256);
    uint8_t *u8_vec = nullptr;
    diskann::alloc_aligned((void **) &u8_vec, 256 * (size_t)MAX_PQ_CHUNKS * sizeof(uint8_t), 256);
    uint8_t *u8_dists = nullptr;
    diskann::alloc_aligned((void **) &u8_dists, 256 * (size_t)MAX_PQ_CHUNKS * sizeof(uint8_t), 256);
    uint16_t *u16_dists = nullptr;
    diskann::alloc_aligned((void **) &u16_dists, npts_u64 * sizeof(uint16_t), 256);
    diskann::PQComputer<T> *computer = new diskann::PQComputer<T>(o_data, 
                                                                  pq_data, 
                                                                  npts_u64, 
                                                                  nchunks_u64, 
                                                                  u8_dists, 
                                                                  &pq_table);
    std::vector<uint32_t> query_result;
    std::cout << "Starting linear scan ..." << std::endl;
    diskann::Timer timer;
    for (size_t i = 0; i < query_num; i++) {
        // std::cout << i << "\n";
        T *query = queries + i * query_aligned_dim;
        for (size_t i = 0; i < query_aligned_dim; i++) {
            fp_vec[i] = (float) query[i];
        }
        pq_table.deflate_vec(fp_vec, u8_vec);
        // for (size_t i = 0; i < nchunks_u64; i++) {
        //     std::cout << (int)u8_vec[i] << " ";
        // }
        // std::cout << std::endl;
        computer->compute_batch_dists(u8_vec, 0, u16_dists);
        for (size_t j = 0; j < npts_u64; j++) {
            u16_dists[j] = computer->compute_dists(u8_vec, o_data + nchunks_u64 * j);
        }
        // computer->compute_pq_dists(u8_vec, o_data + nchunks_u64 * j);
        // ::pq_dist_lookup(pq_data, npts_u64, nchunks_u64, pq_dists, u16_dists);
        std::priority_queue<std::pair<uint16_t, size_t>> q;
        for (size_t j = 0; j < npts_u64; j++) {
            if ((int)q.size() >= topk) {
                if (q.top().first > u16_dists[j]) {
                    q.pop();
                    q.emplace(u16_dists[j], j);
                }
            } else {
                q.emplace(u16_dists[j], j);
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
    if (argc != 7) {
        std::cout << "Usage: [pq_prefix] [query_file] [gt_file] [topk] [data_type] [origin_file]" << std::endl;
        return 0;
    }
    std::string data_type = std::string(argv[5]);
    if (data_type == "float") {
        linear_scan_simd<float>(argc, argv);
    } else if (data_type == "uint8") {
        linear_scan_simd<uint8_t>(argc, argv);
    } else if (data_type == "int8") {
        linear_scan_simd<int8_t>(argc, argv);
    } else {
        std::cout << "Data type doesn't support." << std::endl;
    }
    return 0;
}