//
// Created by longxiang on 3/10/23.
//

#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include "mkl.h"
#include "utils/dist_header.h"

// #define likely(x)       __builtin_expect(!!(x), 1)
// #define unlikely(x)     __builtin_expect(!!(x), 0)
#if defined(_MSC_VER)
#define PRAGMA_IMPRECISE_LOOP
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    __pragma(float_control(precise, off, push))
#define PRAGMA_IMPRECISE_FUNCTION_END __pragma(float_control(pop))
#elif defined(__clang__)
#define PRAGMA_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize(enable) interleave(enable)")

// clang-format off

// the following ifdef is needed, because old versions of clang (prior to 14)
// do not generate FMAs on x86 unless this pragma is used. On the other hand,
// ARM does not support the following pragma flag.
// TODO: find out how to enable FMAs on clang 10 and earlier.
#if defined(__x86_64__) && (defined(__clang_major__) && (__clang_major__ > 10))
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define PRAGMA_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")
#else
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN
#define PRAGMA_IMPRECISE_FUNCTION_END
#endif
#elif defined(__GNUC__)
// Unfortunately, GCC does not provide a pragma for detecting it.
// So, we have to stick to GNUC, which is defined by MANY compilers.
// This is why clang/icc needs to be checked first.

// todo: add __INTEL_COMPILER check for the classic ICC
// todo: add __INTEL_LLVM_COMPILER for ICX

#define PRAGMA_IMPRECISE_LOOP
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("GCC push_options") \
    _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define PRAGMA_IMPRECISE_FUNCTION_END \
    _Pragma("GCC pop_options")
#else
#define PRAGMA_IMPRECISE_LOOP
#define PRAGMA_IMPRECISE_FUNCTION_BEGIN
#define PRAGMA_IMPRECISE_FUNCTION_END
#endif


namespace utils{

#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)

// Adapted from https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp
static inline __m128 MaskedReadFloat(const std::size_t dim, const float* data) {
    assert(0<=dim && dim < 4 );
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
    switch (dim) {
        case 3:
            buf[2] = data[2];
        case 2:
            buf[1] = data[1];
        case 1:
            buf[0] = data[0];
    }
    return _mm_load_ps(buf);
}

static inline __m128i MaskedReadInt(const std::size_t dim, const int* data) {
    assert(0<=dim && dim < 4 );
    ALIGNED(16) int buf[4] = {0, 0, 0, 0};
    switch (dim) {
        case 3:
            buf[2] = data[2];
        case 2:
            buf[1] = data[1];
        case 1:
            buf[0] = data[0];
    }
    return _mm_load_si128((__m128i *)buf);
}

// Adapted from https://stackoverflow.com/questions/60108658/fastest-method-to-calculate-sum-of-all-packed-32-bit-integers-using-avx512-or-av
static float HsumFloat128(__m128 x) {
//    __m128 h64 = _mm_unpackhi_ps(x, x);
    __m128 h64 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
    __m128 sum64 = _mm_add_ps(h64, x);
    __m128 h32 = _mm_shuffle_ps(sum64, sum64, _MM_SHUFFLE(0, 1, 2, 3));
    __m128 sum32 = _mm_add_ps(sum64, h32);
    return _mm_cvtss_f32(sum32);
}

static int HsumInt128(__m128i x) {
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

// Adapted from https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp
#if defined(USE_AVX512)
static float L2SqrFloatAVX512(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m512 mx512, my512, diff512;
    __m512 sum512 = _mm512_setzero_ps();

    while (dim >= 16) {
        mx512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        my512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        diff512 = _mm512_sub_ps(mx512, my512);
        sum512 = _mm512_fmadd_ps(diff512, diff512, sum512);
        dim -= 16;
    }
    __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512), _mm512_extractf32x8_ps(sum512, 1));

    if (dim >= 8) {
        __m256 mx256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 my256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        __m256 diff256 = _mm256_sub_ps(mx256, my256);
        sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 mx128, my128, diff128;

    if (dim >= 4) {
        mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        mx128 = MaskedReadFloat(dim, pVec1);
        my128 = MaskedReadFloat(dim, pVec2);
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
    }
    return HsumFloat128(sum128);
}
#endif // USE_AVX512

#if defined(USE_AVX)
static float L2SqrFloatAVX(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m256 sum256 = _mm256_setzero_ps();

    while (dim >= 8) {
        __m256 mx256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 my256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        __m256 diff256 = _mm256_sub_ps(mx256, my256);
        sum256 = _mm256_fmadd_ps(diff256, diff256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 mx128, my128, diff128;

    if (dim >= 4) {
        mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        mx128 = MaskedReadFloat(dim, pVec1);
        my128 = MaskedReadFloat(dim, pVec2);
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
    }
    return HsumFloat128(sum128);
}
#endif // USE_AVX

#if defined(USE_SSE)
static float L2SqrFloatSSE(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m128 sum128 = _mm_setzero_ps();
    __m128 mx128, my128, diff128;

    while (dim >= 4) {
        mx128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        my128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        mx128 = MaskedReadFloat(dim, pVec1);
        my128 = MaskedReadFloat(dim, pVec2);
        diff128 = _mm_sub_ps(mx128, my128);
        sum128  = _mm_fmadd_ps(diff128, diff128, sum128);
    }
    return HsumFloat128(sum128);
}
#endif // USE_SSE

#if defined(USE_AVX512)
static float InnerProductFloatAVX512(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m512 x512, y512, diff512;
    __m512 sum512 = _mm512_setzero_ps();

    while (dim >= 16) {
        x512 = _mm512_loadu_ps(pVec1); pVec1 += 16;
        y512 = _mm512_loadu_ps(pVec2); pVec2 += 16;
        sum512 = _mm512_fmadd_ps(x512, y512, sum512);
        dim -= 16;
    }
    __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(sum512), _mm512_extractf32x8_ps(sum512, 1));

    if (dim>=8){
        __m256 x256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 y256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        sum256 = _mm256_fmadd_ps(x256, y256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 x128, y128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        y128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pVec1);
        y128 = MaskedReadFloat(dim, pVec2);
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
    }
    return HsumFloat128(sum128);
}
#endif // USE_AVX512

#if defined(USE_AVX)
static float InnerProductFloatAVX(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m256 sum256 = _mm256_setzero_ps();

    while (dim>=8){
        __m256 x256 = _mm256_loadu_ps(pVec1); pVec1 += 8;
        __m256 y256 = _mm256_loadu_ps(pVec2); pVec2 += 8;
        sum256 = _mm256_fmadd_ps(x256, y256, sum256);
        dim -= 8;
    }
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
    __m128 x128, y128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        y128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pVec1);
        y128 = MaskedReadFloat(dim, pVec2);
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
    }
    return HsumFloat128(sum128);
}
#endif // USE_AVX

#if defined(USE_SSE)
static float InnerProductFloatSSE(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
    float *pVec1 = (float *) pVec1v;
    float *pVec2 = (float *) pVec2v;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m128 sum128 = _mm_setzero_ps();
    __m128 x128, y128;

    while (dim >= 4) {
        x128 = _mm_loadu_ps(pVec1); pVec1 += 4;
        y128 = _mm_loadu_ps(pVec2); pVec2 += 4;
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pVec1);
        y128 = MaskedReadFloat(dim, pVec2);
        sum128 = _mm_fmadd_ps(x128, y128, sum128);
    }
    return HsumFloat128(sum128);
}
#endif // USE_SSE

#if defined(USE_AVX512)
static float NormSqrFloatAVX512(const void *pVec, const void *dim_ptr) {
    float *pV = (float *) pVec;
    std::size_t dim = *((std::size_t *) dim_ptr);
    __m512 x512;
    __m512 res512 = _mm512_setzero_ps();

    while(dim >= 16) {
        x512 = _mm512_loadu_ps(pV); pV += 16;
        res512 = _mm512_fmadd_ps(x512, x512, res512);
        dim -= 16;
    }
    __m256 res256 = _mm256_add_ps(_mm512_castps512_ps256(res512), _mm512_extractf32x8_ps(res512, 1));

    if (dim >= 8) {
        __m256 x256 = _mm256_loadu_ps(pV); pV += 8;
        res256 = _mm256_fmadd_ps(x256, x256, res256);
        dim -= 8;
    }
    __m128 res128 = _mm_add_ps(_mm256_castps256_ps128(res256), _mm256_extractf128_ps(res256, 1));
    __m128 x128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pV); pV += 4;
        res128 = _mm_fmadd_ps(x128, x128, res128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pV);
        res128 = _mm_fmadd_ps(x128, x128, res128);
    }
    return HsumFloat128(res128);
}
#endif // USE_AVX512

// #if defined (USE_AVX512)
// static void SubFloatAVX512(const float *pVec1v, const float *pVec2v, const size_t kDim, float *result) {
//     __m512 mx512, my512, res;
//     size_t dim = kDim;

//     while (dim >= 16) {

//     }
// }
// #endif

#if defined(USE_AVX)
static float NormSqrFloatAVX(const void *pVec, const void *dim_ptr) {
    float *pV = (float *) pVec;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m256 res256 = _mm256_setzero_ps();
    while (dim >= 8) {
        __m256 x256 = _mm256_loadu_ps(pV); pV += 8;
        res256 = _mm256_fmadd_ps(x256, x256, res256);
        dim -= 8;
    }
    __m128 res128 = _mm_add_ps(_mm256_castps256_ps128(res256), _mm256_extractf128_ps(res256, 1));
    __m128 x128;

    if (dim >= 4) {
        x128 = _mm_loadu_ps(pV); pV += 4;
        res128 = _mm_fmadd_ps(x128, x128, res128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pV);
        res128 = _mm_fmadd_ps(x128, x128, res128);
    }
    return HsumFloat128(res128);
}
#endif // USE_AVX

#if defined(USE_SSE)
static float NormSqrFloatSSE(const void *pVec, const void *dim_ptr) {
    float *pV = (float *) pVec;
    std::size_t dim = *((std::size_t *) dim_ptr);

    __m128 res128 = _mm_setzero_ps();
    __m128 x128;

    while (dim >= 4) {
        x128 = _mm_loadu_ps(pV); pV += 4;
        res128 = _mm_fmadd_ps(x128, x128, res128);
        dim -= 4;
    }

    if (dim > 0) {
        x128 = MaskedReadFloat(dim, pV);
        res128 = _mm_fmadd_ps(x128, x128, res128);
    }
    return HsumFloat128(res128);
}
#endif // USE_SSE

#if defined(USE_AVX512)
static float L2SqrWithNormAVX512(const void *pVec1, const void *pVec2, const void *dim_ptr, const void *norm_ptr) {
    return *((float *)norm_ptr) - 2 * InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
}
#endif // USE_AVX512

#if defined(USE_AVX)
static float L2SqrWithNormAVX(const void *pVec1, const void *pVec2, const void *dim_ptr, const void *norm_ptr) {
    return *((float *)norm_ptr) - 2 * InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
}
#endif // USE_AVX

#if defined(USE_SSE)
static float L2SqrWithNormSSE(const void *pVec1, const void *pVec2, const void *dim_ptr, const void *norm_ptr) {
    return *((float *)norm_ptr) - 2 * InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
}
#endif // use_SSE

static float L2SqrSIMD(const void *pVec1, const void *pVec2, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return L2SqrFloatAVX512(pVec1, pVec2, dim_ptr);
    #elif defined(USE_AVX)
        return L2SqrFloatAVX(pVec1, pVec2, dim_ptr);
    #elif defined(USE_SSE)
        return L2SqrFloatSSE(pVec1, pVec2, dim_ptr);
    #endif
}

static float InverseL2SqrSIMD(const void *pVec1, const void *pVec2, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return -L2SqrFloatAVX512(pVec1, pVec2, dim_ptr);
    #elif defined(USE_AVX)
        return -L2SqrFloatAVX(pVec1, pVec2, dim_ptr);
    #elif defined(USE_SSE)
        return -L2SqrFloatSSE(pVec1, pVec2, dim_ptr);
    #endif
}

static float InnerProductSIMD(const void *pVec1, const void *pVec2, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
    #elif defined(USE_AVX)
        return InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
    #elif defined(USE_SSE)
        return InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
    #endif
}

static float InverseInnerProductSIMD(const void *pVec1, const void *pVec2, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return -InnerProductFloatAVX512(pVec1, pVec2, dim_ptr);
    #elif defined(USE_AVX)
        return -InnerProductFloatAVX(pVec1, pVec2, dim_ptr);
    #elif defined(USE_SSE)
        return -InnerProductFloatSSE(pVec1, pVec2, dim_ptr);
    #endif
}

static float AbsInnerProductSIMD(const void *pVec1, const void *pVec2, const void *dim_ptr) {
    #if defined(USE_AVX512)
        return fabs(InnerProductFloatAVX512(pVec1, pVec2, dim_ptr));
    #elif defined(USE_AVX)
        return fabs(InnerProductFloatAVX(pVec1, pVec2, dim_ptr));
    #elif defined(USE_SSE)
        return fabs(InnerProductFloatSSE(pVec1, pVec2, dim_ptr));
    #endif
}

#endif // defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)

    template<typename T>
    static T L2SqrNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        T *pVec1 = (T*) pVec1v;
        T *pVec2 = (T*) pVec2v;
        std::size_t dim = *((std::size_t *) dim_ptr);

        T diff, res=0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            diff = pVec1[idx] - pVec2[idx];
            res += diff * diff;
        }
        return res;
    }

    template<typename T>
    static float IPNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        T *pVec1 = (T*) pVec1v;
        T *pVec2 = (T*) pVec2v;
        std::size_t dim = *((std::size_t *) dim_ptr);

        T res=0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            res += pVec1[idx] * pVec2[idx];
        }
        return res;
    }

    template<typename T>
    static T NormSqr(const void *pVec, const void *dim_ptr) {
        T *pV = (T *) pVec;
        std::size_t dim =  *((std::size_t *) dim_ptr);

        T res = 0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            res += pV[idx] * pV[idx];
        }
        return res;
    }

    template<typename T>
    static T NormSqrT(const void *pVec, const void *dim_ptr) {
        T *pV = (T *) pVec;
        std::size_t dim =  *((std::size_t *) dim_ptr);

        T res = 0;
#pragma omp simd
        for (auto idx=0; idx<dim; ++idx) {
            res += pV[idx] * pV[idx];
        }
        return sqrt(res);
    }

    template<typename T>
    static void SubT(const void *pVec1v, const void *pVec2v, const size_t dim, void *result) {
        T *pV1 = (T *) pVec1v;
        T *pV2 = (T *) pVec2v;
        T *res = (T *)result;

#pragma omp simd
        for (std::size_t i=0; i<dim; ++i) {
            res[i] = pV1[i] - pV2[i];
        }
    }

    template<typename T>
    static void AddT(const void *pVec1v, const void *pVec2v, const size_t dim, void *result) {
        T *pV1 = (T *) pVec1v;
        T *pV2 = (T *) pVec2v;
        T *res = (T *)result;

#pragma omp simd
        for (size_t i=0; i<dim; ++i) {
            res[i] = pV1[i] + pV2[i];
        }
    }

    template<typename T>
    static void Add(void *pVec1v, void *pResult, const size_t dim) {
        T *pV1 = (T *) pVec1v;
        T *res = (T *)pResult;

#pragma omp simd
        for (size_t i=0; i<dim; ++i) {
            res[i] += pV1[i];
        }
    }

    template<typename T>
    static T P2HNaive(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        return fabs(IPNaive<T>(pVec1v, pVec2v, dim_ptr));
    }

    static float InnerProduct(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        float res = 0;
        #if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
            res = InnerProductSIMD(pVec1v, pVec2v, dim_ptr);
        #elif
            std::size_t dim = *((std::size_t *) dim_ptr);
            for (std::size_t i=0; i<dim; ++i) {
                res += ((float *) pVec1v)[i] * ((float *) pVec2v)[i];
            }
        #endif
        return res;
    }

    static float InverseInnerProduct(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        return -InnerProduct(pVec1v, pVec2v, dim_ptr);
    }

    static float AbsInnerProduct(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        return fabs(InnerProduct(pVec1v, pVec2v, dim_ptr));
    }

    static float InverseAbsInnerProduct(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        return -fabs(InnerProduct(pVec1v, pVec2v, dim_ptr));
    }

    static float L2Sqr(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        float res = 0;
        #if defined(USE_AVX512) || defined(USE_AVX512) || defined(USE_SSE)
            res = L2SqrSIMD(pVec1v, pVec2v, dim_ptr);
        #elif
            std::size_t dim = *((std::size_t *) dim_ptr);

            float diff=0;
            for (std::size_t i=0; i<dim; ++i) {
                diff = ((float *) pVec1v)[i] - ((float *) pVec2v)[i];
                res += diff * diff;
            }
        #endif

        return res;
    }

    static float L2SqrT(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        float res = 0;
        #if defined(USE_AVX512) || defined(USE_AVX512) || defined(USE_SSE)
            res = L2SqrSIMD(pVec1v, pVec2v, dim_ptr);
        #elif
            std::size_t dim = *((std::size_t *) dim_ptr);

            float diff=0;
            for (std::size_t i=0; i<dim; ++i) {
                diff = ((float *) pVec1v)[i] - ((float *) pVec2v)[i];
                res += diff * diff;
            }
        #endif

        return sqrt(res);
    }

    static float InverseL2Sqr(const void *pVec1v, const void *pVec2v, const void *dim_ptr) {
        return -L2Sqr(pVec1v, pVec2v, dim_ptr);
    }

    inline void gemv(float *vec, float *mat, float *res, const int dim, const int n) {
        cblas_sgemv(CblasRowMajor, 
                    CblasNoTrans,
                    n,
                    dim,
                    1.0,
                    mat,
                    dim,
                    vec, 
                    1, 
                    0.0, 
                    res, 
                    1
                );
    }

    inline void sgemm(float *mat1, const int dim1, const int n1, float *mat2, const int dim2, const int n2, float *res) {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasTrans,
                    n1,
                    n2,
                    dim1,
                    1.0,
                    mat1,
                    dim1,
                    mat2,
                    dim2,
                    0.0,
                    res,
                    n2
                );
    }

PRAGMA_IMPRECISE_FUNCTION_BEGIN
    inline void vec_mat_ip(const float *query,
                           const float *mat,
                           float *res,
                           const std::size_t dim,
                           const std::size_t n) {
        PRAGMA_IMPRECISE_LOOP
        for (std::size_t i=0; i<n; ++i) {
            res[i] = InnerProduct(query, mat+i*dim, &dim);
        }
    }
PRAGMA_IMPRECISE_FUNCTION_END

PRAGMA_IMPRECISE_FUNCTION_BEGIN
    inline void vec_mat_p2h(const float *query,
                            const float *mat,
                            float *res,
                            const std::size_t dim,
                            const std::size_t n) {
        PRAGMA_IMPRECISE_LOOP
        for (std::size_t i=0; i<n; ++i) {
            res[i] = std::fabs(InnerProduct(query, mat+i*dim, &dim) + query[dim]);
        }
    }
PRAGMA_IMPRECISE_FUNCTION_END

PRAGMA_IMPRECISE_FUNCTION_BEGIN
    inline void vec_mat_p2h(const float *query,
                            const float *mat,
                            std::vector<std::pair<float, unsigned>>& res,
                            const std::size_t dim,
                            const std::size_t n) {
        PRAGMA_IMPRECISE_LOOP
        for (std::size_t i=0; i<n; ++i) {
            res[i].first = std::fabs(InnerProduct(query, mat+i*dim, &dim) + query[dim]);
            res[i].second = i;
        }
    }
PRAGMA_IMPRECISE_FUNCTION_END



PRAGMA_IMPRECISE_FUNCTION_BEGIN
    inline void vec_mat_l2(const float *query,
                           const float *mat,
                           float *res,
                           const std::size_t dim,
                           const std::size_t n) {
        PRAGMA_IMPRECISE_LOOP
        for (std::size_t i=0; i<n; ++i) {
            res[i] = L2Sqr(query, mat+i*dim, &dim);
        }
    }
PRAGMA_IMPRECISE_FUNCTION_END


} // namespace utils
