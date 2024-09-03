#pragma once
#include <immintrin.h>
#include <xmmintrin.h>

#include "utils.h"
#ifndef PQ_TABLE_H
#define PQ_TABLE_H
namespace diskann {
  struct QuantizerMAX_bolt  {
    float min_quant;
    float max_quant;
    int M;
    float delta;
    float inv_delta;
    uint8_t QMAX;

    QuantizerMAX_bolt(float min_q,float max_q, int M_) : min_quant(min_q), max_quant(max_q) {
      QMAX = std::numeric_limits<uint8_t>::max();
      delta = (max_quant - min_quant) / QMAX;
      inv_delta = 1.0f/delta;
    }

    void quantize_val(float val, uint8_t* qval, int m) const {
      float pval = std::min(255.0f,inv_delta*std::max(0.0f,val-min_quant));
      if(pval <= 0){
        *qval = 0;
      }else if(val >= max_quant) {
        *qval = QMAX;
      }else{
        *qval = static_cast<uint8_t>(pval);
      }
    }

    void quantize_sum(float val, uint8_t* qval) const {
      float pval = std::min(255.0f,inv_delta*std::max(0.0f,val-min_quant));
      *qval = static_cast<uint8_t>(pval);
    }

    float unquantize_sum(uint8_t qval) const {
      float fval=qval+0.5;
      return (fval*delta)+min_quant;
    }

    void quantize_sum16(float val, int16_t* qval) const {
      float pval = std::min(32767.0f,inv_delta*std::max(0.0f,val-M*min_quant));
      *qval = static_cast<int16_t>(pval);
    }

    float unquantize_sum16(int16_t qval) const {
      float fval=qval+0.5;
      return (fval*delta)+M*min_quant;
    }

    inline void quantize_val_simd(const float* val, uint8_t* qval, const int table_size, const int ksub, const int m) const {
      const __m256 min_r = _mm256_set1_ps(min_quant);
      const __m256 inv_delta_r = _mm256_set1_ps(inv_delta);
      const __m128i shuf_r = _mm_set_epi8(15,14,13,12, 7,6,5,4  ,11,10,9,8, 3,2,1,0);
      for(int i=0;i<ksub/16;i++){
          __m128i * t = (__m128i*)&qval[i*16];
          float * f1 = (float*)&val[i*16];
          float * f2 = (float*)&val[i*16+8];
          __m256 low = _mm256_loadu_ps(f1); // 8x32
          __m256 high = _mm256_loadu_ps(f2); // 8x32

          low = _mm256_sub_ps(low, min_r);
          high = _mm256_sub_ps(high, min_r);
          low = _mm256_mul_ps(low, inv_delta_r);
          high = _mm256_mul_ps(high, inv_delta_r);

          __m256i lowi = _mm256_cvtps_epi32(low);
          __m256i highi = _mm256_cvtps_epi32(high);
          __m256i packed16_interleaved4 = _mm256_packs_epi32(lowi, highi); // A B A B
          __m128i p16i_l = _mm256_extracti128_si256(packed16_interleaved4,0); // A B
          __m128i p16i_h = _mm256_extracti128_si256(packed16_interleaved4,1); // A B
          __m128i packed8_interleaved4 = _mm_packus_epi16(p16i_l, p16i_h);  // A B A B
          // Reorganize...
          __m128i packed8 = _mm_shuffle_epi8(packed8_interleaved4, shuf_r); // A A A A B B B B
          _mm_store_si128(t,packed8);
      }
      for(int i=ksub/16;i<table_size/16;i++){
          // Set to zero
          __m128i * t = (__m128i*)&qval[i*16];
          _mm_store_si128(t,  _mm_set1_epi8(0));
      }
    }
  };
  
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    float* tables =
        nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    //    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    _u64   ndims;  // ndims = chunk_size * n_chunks
    _u64   n_chunks;
    _u32*  chunk_offsets = nullptr;
    _u32*  rearrangement = nullptr;
    float* centroid = nullptr;
    float* tables_T = nullptr;  // same as pq_tables, but col-major
    float* all_to_all_dists = nullptr;
    uint8_t* all_to_all_dists_quanted = nullptr;
   public:
    FixedChunkPQTable() {
    }
    _u64 GetNChunks() const {
      return this->n_chunks;
    }
    _u64 GetNdims() const {
      return this->ndims;
    }
    virtual ~FixedChunkPQTable() {
      if (tables != nullptr)
        delete[] tables;
      if (tables_T != nullptr)
        delete[] tables_T;
      if (rearrangement != nullptr)
        delete[] rearrangement;
      if (chunk_offsets != nullptr)
        delete[] chunk_offsets;
      if (centroid != nullptr)
        delete[] centroid;
      if (all_to_all_dists != nullptr)
        delete[] all_to_all_dists;
      if (all_to_all_dists_quanted != nullptr)
        delete[] all_to_all_dists_quanted;
    }

    void load_pq_centroid_bin(const char* pq_table_file, size_t num_chunks) {
      std::string rearrangement_file =
          std::string(pq_table_file) + "_rearrangement_perm.bin";
      std::string chunk_offset_file =
          std::string(pq_table_file) + "_chunk_offsets.bin";
      std::string centroid_file = std::string(pq_table_file) + "_centroid.bin";

      // bin structure: [256][ndims][ndims(float)]
      uint64_t numr, numc;
      size_t   npts_u64, ndims_u64;
      diskann::load_bin<float>(pq_table_file, tables, npts_u64, ndims_u64);
      this->ndims = ndims_u64;

      if (file_exists(chunk_offset_file)) {
        diskann::load_bin<_u32>(rearrangement_file, rearrangement, numr, numc);
        if (numr != ndims_u64 || numc != 1) {
          std::cout << "Error loading rearrangement file" << std::endl;
          throw diskann::ANNException("Error loading rearrangement file", -1,
                                      __FUNCSIG__, __FILE__, __LINE__);
        }

        diskann::load_bin<_u32>(chunk_offset_file, chunk_offsets, numr, numc);
        if (numc != 1 || numr != num_chunks + 1) {
          std::cout << "Error loading chunk offsets file" << std::endl;
          throw diskann::ANNException("Error loading chunk offsets file", -1,
                                      __FUNCSIG__, __FILE__, __LINE__);
        }

        this->n_chunks = numr - 1;

        diskann::load_bin<float>(centroid_file, centroid, numr, numc);
        if (numc != 1 || numr != ndims_u64) {
          std::cout << "Error loading centroid file" << std::endl;
          throw diskann::ANNException("Error loading centroid file", -1,
                                      __FUNCSIG__, __FILE__, __LINE__);
        }
      } else {
        this->n_chunks = num_chunks;
        rearrangement = new uint32_t[ndims];

        uint64_t chunk_size = DIV_ROUND_UP(ndims, num_chunks);
        for (uint32_t d = 0; d < ndims; d++)
          rearrangement[d] = d;
        chunk_offsets = new uint32_t[num_chunks + 1];
        for (uint32_t d = 0; d <= num_chunks; d++)
          chunk_offsets[d] = (_u32)(std::min)(ndims, d * chunk_size);
        centroid = new float[ndims];
        std::memset(centroid, 0, ndims * sizeof(float));
      }

      //      std::cout << "PQ Pivots: #ctrs: " << npts_u64 << ", #dims: " <<
      //      ndims_u64
      //                << ", #chunks: " << n_chunks << std::endl;
      //      assert((_u64) ndims_u32 == n_chunks * chunk_size);
      // alloc and compute transpose
      tables_T = new float[256 * ndims_u64];
      for (_u64 i = 0; i < 256; i++) {
        for (_u64 j = 0; j < ndims_u64; j++) {
          tables_T[j * 256 + i] = tables[i * ndims_u64 + j];
        }
      }

      // ravi: added this for easy PQ-PQ squared-distance calculations
      all_to_all_dists = new float[256 * 256 * n_chunks];
      std::memset(all_to_all_dists, 0, 256 * 256 * n_chunks * sizeof(float));
      // should perhaps optimize later
      for (_u32 i = 0; i < 256; i++) {
        for (_u32 j = 0; j < 256; j++) {
          for (_u32 c = 0; c < n_chunks; c++) {
            for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
              float diff =
                  (tables[i * ndims_u64 + d] - tables[j * ndims_u64 + d]);
              all_to_all_dists[i * 256 * n_chunks + j * n_chunks + c] +=
                  diff * diff;
            }
          }
        }
      }
      // the quanted distance is inverted after quantization
      float min = std::numeric_limits<float>::max(), max = 0;
      all_to_all_dists_quanted = new uint8_t[n_chunks * 256 * 256];
      for (size_t k = 0; k < 256 * 256 * this->n_chunks; k++) {
        size_t i = k / 256 / this->n_chunks, j = k % (256 * this->n_chunks) / this->n_chunks;
        if (i == j) continue;
        float now = all_to_all_dists[k];
        min = std::min(min, now);
        max = std::max(max, now);
      }
      std::cout << "Quanted min, max: " << min << " " << max << std::endl;
      auto quant = QuantizerMAX_bolt(min, max, this->GetNChunks());
      for (size_t i = 0; i < 256 * 256 * this->GetNChunks(); i++) {
        // dis[i][j][c] -> dis[c][i][j]
        size_t new_pos = i % this->GetNChunks() * 256 * 256 + i / this->GetNChunks();
        quant.quantize_val(all_to_all_dists[i], all_to_all_dists_quanted + new_pos, 0);
      }
    }
    uint8_t* Symdistu8() { return all_to_all_dists_quanted; }
    float* Symdist() { return all_to_all_dists; }
    void populate_chunk_distances(const T* query_vec, float* dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float* chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64         permuted_dim_in_query = rearrangement[j];
          const float* centers_dim_vec = tables_T + (256 * j);
          for (_u64 idx = 0; idx < 256; idx++) {
            float diff = centers_dim_vec[idx] -
                         ((float) query_vec[permuted_dim_in_query] -
                          centroid[permuted_dim_in_query]);
            chunk_dists[idx] += (diff * diff);
          }
        }
      }
    }

    // computes PQ distance between comp_src and comp_dsts in efficient manner
    // comp_src: [nchunks]
    // comp_dsts: count * [nchunks]
    // dists: [count]
    // TODO (perf) :: re-order computation to get better locality
    void compute_distances(const _u8* comp_src, const _u8* comp_dsts,
                           float* dists, const _u32 count) {
      std::memset(dists, 0, count * sizeof(float));
      for (_u64 i = 0; i < count; i++) {
        for (_u64 c = 0; c < n_chunks; c++) {
          dists[i] +=
              all_to_all_dists[(_u64) comp_src[c] * 256 * n_chunks +
                               (_u64) comp_dsts[i * n_chunks + c] * n_chunks +
                               c];
        }
      }
    }

    // fp_vec: [ndims]
    // out_pq_vec : [nchunks]
    void deflate_vec(const float* fp_vec, _u8* out_pq_vec) {
      // permute the vector according to PQ rearrangement, compute all distances
      // to 256 centroids and choose the closest (for each chunk)
      for (_u32 c = 0; c < n_chunks; c++) {
        float closest_dist = std::numeric_limits<float>::max();
        for (_u32 i = 0; i < 256; i++) {
          float cur_dist = 0;
          for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
            float diff =
                (tables[i * ndims + d] - ((float) fp_vec[rearrangement[d]] -
                                          centroid[rearrangement[d]]));
            cur_dist += diff * diff;
          }
          if (cur_dist < closest_dist) {
            closest_dist = cur_dist;
            out_pq_vec[c] = i;
          }
        }
      }
    }
  };

  template<typename T>
  class SIMDPQTable : public FixedChunkPQTable<T> {
    public:
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
      void compute_multi_distance_tables(size_t nx, T* x, float* dis_tables) {
        for(size_t i = 0; i < nx; i++) {
          T* q = x + i * this->GetNdims();
          float* table = dis_tables + i * this->GetNChunks() * 256;
          this->populate_chunk_distances(q, table);
        }
      }
      void quantize_table(float* dis_tables, uint8_t* quant_table) {
        auto quant = QuantizerMAX_bolt(best_mindist, best_maxdist, this->GetNChunks());
    //    for(int i = 0; i < 256 * n_chunks; i++) {
    //        quant.quantize_val(dis_tables[i], quant_table + i, 0);
    //    }
        quant.quantize_val_simd(dis_tables, quant_table, 256 * this->GetNChunks(), 256 * this->GetNChunks(), 0);
      }
      void quantize_all_to_all() {
        float min = std::numeric_limits<float>::max(), max = 0;
        // for (_u32 i = 0; i < 256; i++) {
        //   for (_u32 j = 0; j < 256; j++) {
        //     if (i == j) continue;
        //     for (_u32 c = 0; this->GetNChunks(); c++) {
        //       float now = this->Symdist()[(i * 256 + j) * this->GetNChunks() + c];
        //       min = std::min(min, now);
        //       max = std::max(max, now);
        //     }
        //   }
        // }
        for (size_t k = 0; k < 256 * 256 * this->GetNChunks(); k++) {
          size_t i = k / 256 / this->GetNChunks(), j = k % (256 * this->GetNChunks()) / this->GetNChunks();
          if (i == j) continue;
          float now = this->Symdist()[k];
          min = std::min(min, now);
          max = std::max(max, now);
        }
        std::cout << min << " " << max << std::endl;
        auto quant = QuantizerMAX_bolt(min, max, this->GetNChunks());
        for (size_t i = 0; i < 256 * 256 * this->GetNChunks(); i++) {
          // dis[i][j][c] -> dis[c][i][j]
          size_t new_pos = i % this->GetNChunks() * 256 * 256 + i / this->GetNChunks();
          quant.quantize_val(this->Symdist()[i], this->Symdistu8() + new_pos, 0);
        }
      }
      void train(int n, T *x) {
        float best_loss = std::numeric_limits<float>::infinity();
        int ksub_total = this->GetNChunks() * 256;
        // Train the PQ Code
        //        AbstractVecProductQuantizer<TT_M,2,4,4,0,0,uint8_t,TT_TSCMM,TT_TSCMMXL>::train(n,x);

        // Train the distance quantizer
        // Pick up to 10000 vector as queries
        std::vector<int> perm(n);
        //        int seed=1234;
        //        rand_perm(perm.data (), n, seed);

        std::iota(perm.begin(), perm.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(perm.begin(), perm.end(), g);
        int nx = std::min(10000, n);
        std::vector<T> x_queries(nx * this->GetNdims());
        std::vector<float> true_dists(nx * ksub_total);
        for (int i = 0; i < nx; i++)
            memcpy (x_queries.data() + i * this->GetNdims(), x + perm[i] * this->GetNdims(), sizeof(T) * this->GetNdims());
        this->compute_multi_distance_tables(nx,x_queries.data(),true_dists.data()); //x = nx*d, nx*M*ksub
        std::sort(true_dists.begin(), true_dists.end());
        const std::vector<float> alphas ({.001, .002, .005, .01, .02, .05, .1});
        for(float a : alphas){
            std::cout << "a : " << a << std::endl;
            float lower_quantile=true_dists[static_cast<int>(std::floor(a*nx*ksub_total))];
            float upper_quantile=true_dists[static_cast<int>(std::ceil((1.0-a)*nx*ksub_total))];
            float scale=255.0/(upper_quantile-lower_quantile);
            float loss=0.0;
            for(int i=0;i<nx*ksub_total;i++){
                float quant_dist = std::min(255.0f,scale*std::max(0.0f,true_dists[i] - lower_quantile));
                loss += (true_dists[i]-quant_dist) * (true_dists[i]-quant_dist);
            }
            if(loss < best_loss){
                best_mindist=lower_quantile;
                best_maxdist=upper_quantile;
                best_loss=loss;
                std::cout << "alphas: " << a << std::endl;
            }
        }
        std::cout << "best_mindist=" << best_mindist << " best_maxdist=" << best_maxdist << " loss=" << best_loss << std::endl;
      }

    protected:
      float best_mindist;
      float best_maxdist;
  };
  
  // template<typename T>
  // class DeleteNeighbors {
  //     const uint32_t kSize = 64;
  //     void pq_dist_lookup_simd(const uint8_t* pq_ids, const size_t n_pts, 
  //                       const size_t pq_nchunks, const size_t group_size,
  //                       uint8_t*pq_dists, uint16_t* dists_out) {
  //       for(size_t i = 0; i < n_pts; i += group_size) {
  //           assert(group_size * sizeof(uint8_t) * 8 == 512);
  //           __m512i candidates_lo = _mm512_setzero_si512(); // 用于存储低32个uint16_t的累加结果
  //           __m512i candidates_hi = _mm512_setzero_si512(); // 用于存储高32个uint16_t的累加结果
  //           for(size_t j = 0; j < pq_nchunks; j++) {
  //               uint8_t* chunk_pq_dists = pq_dists + 256 * j;
  //               __m512i lut00 = _mm512_loadu_epi8(chunk_pq_dists);
  //               __m512i lut01 = _mm512_loadu_epi8(chunk_pq_dists + 64);
  //               __m512i lut10 = _mm512_loadu_epi8(chunk_pq_dists + 2 * 64);
  //               __m512i lut11 = _mm512_loadu_epi8(chunk_pq_dists + 3 * 64);

  //               const __m512i comps = _mm512_loadu_epi8(pq_ids + i * pq_nchunks + j * group_size);
  //               const __mmask64 bit8_m = _mm512_movepi8_mask(comps);

  //               const __m512i partial_0 = _mm512_permutex2var_epi8(lut00, comps, lut01);
  //               const __m512i partial_1 = _mm512_permutex2var_epi8(lut10, comps, lut11);
  //               const __m512i partial_sum = _mm512_mask_blend_epi8(bit8_m,partial_0,partial_1);


  //               // todo: pq_nchunks大于256还是会有问题
  //               // Extract the lower 32 uint8_t values and convert to uint16_t
  //               __m256i lower_256 = _mm512_extracti32x8_epi32(partial_sum, 0);
  //               __m512i partial_sum_lo = _mm512_cvtepu8_epi16(lower_256);

  //               // Extract the upper 32 uint8_t values and convert to uint16_t
  //               __m256i upper_256 = _mm512_extracti32x8_epi32(partial_sum, 1);
  //               __m512i partial_sum_hi = _mm512_cvtepu8_epi16(upper_256);

  //               candidates_lo = _mm512_adds_epu16(candidates_lo, partial_sum_lo);
  //               candidates_hi = _mm512_adds_epu16(candidates_hi, partial_sum_hi);
  //           }

  //           _mm512_storeu_epi16(dists_out + i, candidates_lo);
  //           _mm512_storeu_epi16(dists_out + i + 32, candidates_hi);
  //       }
  //     }
  //   public:
  //     _u8*                      pq_data_ = nullptr;
  //     SIMDPQTable*              pq_table_ = nullptr;
  //     uint64_t                  n_, d_, n_chunks_;
  //     _u8*                      nei_pq = nullptr;
  //     tsl::robin_map<uint32_t, uint64_t>  pos_;
  //     DeleteNeighbors(_u8* pq_data, SIMDPQTable* pq_table, uint64_t n, uint64_t max_degree) 
  //     : pq_data_(pq_data), pq_table_(pq_table), n_(n), d_((max_degree - 1) / kSize + 1) {
  //       this->n_chunks_ = this->pq_table_->GetNChunks();
  //       this->nei_pq = new _u8[n * d_ * n_chunks_];
  //       memset(this->nei_pq, 0, n * d_ * n_chunks_ * sizeof(_u8));
  //       this->pos.resize(n);
  //     }
  //     ~DeleteNeighbors() { delete[] this->nei_pq; }
  //     void build(const tsl::robin_map<uint32_t, std::vector<uint32_t>> &disk_deleted_nhoods) {
  //       uint64_t cnt = 0;
  //       for (auto &it : disk_deleted_nhoods) {
  //         auto id = it.first;
  //         auto &nei = it.second;
  //         _u8* pq = nei_pq + cnt * d_ * n_chunks_;
  //         for (size_t i = 0; i < nei.size(); i += kSize) {
  //           size_t len = std::min(nei.size() - i, kSize);
  //           for (size_t j = 0; j < len; j++) {
  //             for (size_t k = 0; k < n_chunks_; k++) {
  //               pq[k * kSize + j] = pq_data[n_chunks_ * nei[i + j] + k];
  //             }
  //           }
  //         }
  //         pos_[id] = cnt * d_ * n_chunks_;
  //         cnt++;
  //       }
  //     }
  //     void compute_dists(int id, uint8_t* pq_dists, uint16_t* dists_out) {
  //       auto it = pos_.find(id);
  //       if (it == pos_.end()) {

  //       } else {
  //         _u8* pq = nei_pq + it->second;
  //         pq_dist_lookup_simd(pq, this->d_, this->n_chunks_, kSize, pq_dists, dists_out);
  //       }
  //     }
  // };
  template<typename T>
  class PQComputer {
      const uint32_t kSize = 64;
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
      void pq_dist_lookup_simd_asym(const uint8_t* pq_ids, const size_t n_pts, 
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
      void pq_dist_lookup_simd_sym(const uint8_t* pq_ids, const size_t n_pts, 
                        const size_t pq_nchunks, const size_t group_size,
                        uint8_t*pq_cur, uint16_t* dists_out) {
        for(size_t i = 0; i < n_pts; i += group_size) {
            assert(group_size * sizeof(uint8_t) * 8 == 512);
            __m512i candidates_lo = _mm512_setzero_si512(); // 用于存储低32个uint16_t的累加结果
            __m512i candidates_hi = _mm512_setzero_si512(); // 用于存储高32个uint16_t的累加结果
            for(size_t j = 0; j < pq_nchunks; j++) {
                uint8_t* chunk_pq_dists = all_to_all_dists_quanted + 256 * 256 * j + 256 * pq_cur[j];
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
    public:
      _u8*                      pq_data_ = nullptr;
      _u8*                      pq_data_T_ = nullptr;
      const uint64_t            n_, n_chunks_;
      _u8*                      scratch_ = nullptr;
      SIMDPQTable<T>*           pq_table_ = nullptr;
      _u8*                      all_to_all_dists_quanted = nullptr;
      PQComputer(_u8* pq_data, 
                 _u8* pq_data_T, 
                 uint32_t n, 
                 uint32_t n_chunks, 
                 _u8* scratch,
                 SIMDPQTable<T>* pq_table) 
      : pq_data_(pq_data), 
        pq_data_T_(pq_data_T), 
        n_(n), 
        n_chunks_(n_chunks), 
        scratch_(scratch),
        pq_table_(pq_table),
        all_to_all_dists_quanted(pq_table->Symdistu8()) {}
      uint16_t compute_dists(_u8 *x, _u8 *y) {
        uint16_t res = 0;
        for (_u64 c = 0; c < n_chunks_; c++) {
          res += all_to_all_dists_quanted[(_u64)x[c] * 256 + (_u64)y[c] + c * 256 * 256];
        }
        return res;
      }
      void compute_pq_dists(uint32_t pos, uint32_t *ids, float *dists, uint32_t sz) {
        _u8 *x = pq_data_ + pos * n_chunks_;
        for (uint32_t i = 0; i < sz; i++) {
          dists[i] = static_cast<float>(compute_dists(x, pq_data_ + ids[i] * n_chunks_));
        }
      }
      // id -> [l, n_), l must be a multiple of 64
      // symmetric dists
      void compute_batch_dists(_u8 *pq, uint32_t l, uint16_t* dists_out) {
        // _u8 *pq_dists = scratch_;
        // for (_u64 c = 0; c < n_chunks_; c++) {
        //   uint8_t* chunk_dists = pq_dists + 256 * c;
        //   for (_u64 i = 0; i < 256; i++) {
        //     chunk_dists[i] = all_to_all_dists_quanted[(_u64) pq[c] * 256 * n_chunks_ + 
        //                                               (_u64) i * n_chunks_ + c];
        //   }
        // }
        pq_dist_lookup_simd_sym(pq_data_T_ + l * n_chunks_, 
                                n_ - l, 
                                n_chunks_, 
                                kSize, 
                                pq,
                                dists_out);
      }
  };
}  // namespace diskann
#endif