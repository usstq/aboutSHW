// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef __AVX512F__
#define HAVE_AVX512F 1
#endif

#ifdef __AVX2__
#define HAVE_AVX2 1
#endif

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#if defined(HAVE_AVX512F)
// AVX512F
#define SIMDW 16
using SIMD_F32 = __m512;
using SIMD_I32 = __m512i;

inline SIMD_I32 simd_load_epu8_epi32(void const* ptr) {
    return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)));
}
inline SIMD_F32 simd_cvtepi32_ps(SIMD_I32 v) {
    return _mm512_cvtepi32_ps(v);
}
inline void simd_storeu_ps(float* ptr, __m512 v) {
    _mm512_storeu_ps(ptr, v);
}
inline SIMD_F32 simd_broadcast_ss(float const * mem_addr) {
    return _mm512_set1_ps(*mem_addr);
}
inline SIMD_F32 simd_loadu_ps(float const * mem_addr) {
    return _mm512_loadu_ps(mem_addr);
}
//static SIMD_F32 simd_loadu_ps(ov::float16* p) {
//    return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<__m256i const*>(p)));
//}
inline SIMD_F32 simd_add_ps(SIMD_F32 a, SIMD_F32 b) {
    return _mm512_add_ps(a, b);
}
inline SIMD_F32 simd_sub_ps(SIMD_F32 a, SIMD_F32 b) {
    return _mm512_sub_ps(a, b);
}
inline SIMD_F32 simd_mul_ps(SIMD_F32 a, SIMD_F32 b) {
    return _mm512_mul_ps(a, b);
}
inline SIMD_F32 simd_fmadd_ps(SIMD_F32 a, SIMD_F32 b, SIMD_F32 c) {
    return _mm512_fmadd_ps(a, b, c);
}
inline SIMD_F32 simd_setzero_ps(void) {
    return _mm512_setzero_ps();
}
inline SIMD_I32 simd_set1_epi32(int a) {
    return _mm512_set1_epi32(a);
}
inline SIMD_I32 simd_and_si(SIMD_I32 a, SIMD_I32 b) {
    return _mm512_and_si512(a, b);
}
inline SIMD_I32 simd_srli_epi32(SIMD_I32 a, unsigned int imm8) {
    return _mm512_srli_epi32(a, imm8);
}
inline void simd_prefetch(void const* p, int i) {
    _mm_prefetch(p, static_cast<_mm_hint>(i));
}
struct simd_mask {
    __mmask16 k;
    simd_mask(int num_tails) {
        k = _mm512_int2mask((1 << num_tails) - 1);
    }
    SIMD_F32 load(const float* mem_addr) {
        auto ret = simd_setzero_ps();
        return _mm512_mask_loadu_ps(ret, k, mem_addr);
    }
    //SIMD_F32 load(const ov::float16* p) {
    //    __m256i src = _mm256_set1_epi16(0);
    //    return _mm512_cvtph_ps(_mm256_mask_loadu_epi16(src, k, p));
    //}
    // todo
    // SIMD_F32 blend_load(SIMD_F32 src, void* mem_addr)
    void store(float* mem_addr, SIMD_F32 v) {
        _mm512_mask_storeu_ps(mem_addr, k, v);
    }
};
#elif defined(HAVE_AVX2)
// AVX2
#define SIMDW 8
using SIMD_F32 = __m256;
using SIMD_I32 = __m256i;
inline SIMD_I32 simd_load_epu8_epi32(void const* ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadu_si64(ptr));
}
inline SIMD_F32 simd_cvtepi32_ps(SIMD_I32 v) {
    return _mm256_cvtepi32_ps(v);
}
inline void simd_storeu_ps(float* ptr, SIMD_F32 v) {
    _mm256_storeu_ps(ptr, v);
}
inline SIMD_F32 simd_broadcast_ss(float const * mem_addr) {
    return _mm256_broadcast_ss(mem_addr);
}
inline SIMD_F32 simd_loadu_ps(float const * mem_addr) {
    return _mm256_loadu_ps(mem_addr);
}
//static SIMD_F32 simd_loadu_ps(ov::float16* p) {
//    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(p)));
//}
inline SIMD_F32 simd_add_ps(SIMD_F32 a, SIMD_F32 b) {
    return _mm256_add_ps(a, b);
}
inline SIMD_F32 simd_sub_ps(SIMD_F32 a, SIMD_F32 b) {
    return _mm256_sub_ps(a, b);
}
inline SIMD_F32 simd_mul_ps(SIMD_F32 a, SIMD_F32 b) {
    return _mm256_mul_ps(a, b);
}
inline SIMD_F32 simd_fmadd_ps(SIMD_F32 a, SIMD_F32 b, SIMD_F32 c) {
    return _mm256_fmadd_ps(a, b, c);
}
inline SIMD_F32 simd_setzero_ps(void) {
    return _mm256_setzero_ps();
}
inline SIMD_I32 simd_set1_epi32(int a) {
    return _mm256_set1_epi32(a);
}
inline SIMD_I32 simd_and_si(SIMD_I32 a, SIMD_I32 b) {
    return _mm256_and_si256(a, b);
}
inline SIMD_I32 simd_srli_epi32(SIMD_I32 a, int imm8) {
    return _mm256_srli_epi32(a, imm8);
}
inline void simd_prefetch(void const* p, int i) {
    _mm_prefetch(p, static_cast<_mm_hint>(i));
}
struct simd_mask {
    __m256i vmask;
    // assert(num_tails < SIMDW);
    simd_mask(int num_tails) {
        static int32_t mem_mask[] = {0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1};
        vmask = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&mem_mask[num_tails]));
    }
    SIMD_F32 load(const float* mem_addr) {
        return _mm256_maskload_ps(mem_addr, vmask);
    }
    //SIMD_F32 load(const ov::float16* p) {
        // AVX2:there is no mask in 16bit unit
    //    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(p)));
    //}
    void store(float* mem_addr, SIMD_F32 v) {
        _mm256_maskstore_ps(mem_addr, vmask, v);
    }
};
#else
// scalar
#define SIMDW 1
using SIMD_F32 = float;
using SIMD_I32 = int32_t;
inline SIMD_I32 simd_load_epu8_epi32(void const* ptr) {
    return *reinterpret_cast<const uint8_t*>(ptr);
}
float inline simd_cvtepi32_ps(SIMD_I32 v) {
    return v;
}
inline void simd_storeu_ps(float* ptr, SIMD_F32 v) {
    *ptr = v;
}
inline SIMD_F32 simd_broadcast_ss(float const * mem_addr) {
    return *(mem_addr);
}
inline SIMD_F32 simd_loadu_ps(float const * mem_addr) {
    return *(mem_addr);
}
//static SIMD_F32 simd_loadu_ps(ov::float16* p) {
//    return static_cast<float>(*p);
//}
inline SIMD_F32 simd_add_ps(SIMD_F32 a, SIMD_F32 b) {
    return a + b;
}
inline SIMD_F32 simd_sub_ps(SIMD_F32 a, SIMD_F32 b) {
    return a - b;
}
inline SIMD_F32 simd_mul_ps(SIMD_F32 a, SIMD_F32 b) {
    return a * b;
}
inline SIMD_F32 simd_fmadd_ps(SIMD_F32 a, SIMD_F32 b, SIMD_F32 c) {
    return a*b + c;
}
inline SIMD_F32 simd_setzero_ps(void) {
    return 0;
}
inline SIMD_I32 simd_set1_epi32(int a) {
    return a;
}
inline SIMD_I32 simd_and_si(SIMD_I32 a, SIMD_I32 b) {
    return a & b;
}
inline SIMD_I32 simd_srli_epi32(SIMD_I32 a, unsigned int imm8) {
    return a >> imm8;
}
inline void simd_prefetch(void const* p, int i) {
    //_mm_prefetch(p, i);
}
struct simd_mask {
    simd_mask(int num_tails) {
    }
    SIMD_F32 load(const float* mem_addr) {
        return *mem_addr;
    }
    //SIMD_F32 load(const ov::float16* p) {
    //    return static_cast<float>(*p);
    //}
    void store(float* mem_addr, SIMD_F32 v) {
        *mem_addr = v;
    }
};
#endif