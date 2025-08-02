/*
 * Copyright (c) 2020-2023, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>

#if defined(SHIM) || defined(CMRT_EMU)
#define ATTR
#define ATTR_BUF
#define CM_LOCAL_BARRIER 0x20
#include "emu/block2d.h"
#else
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
    matrix<T2, 8, 8> temp;
    temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
    temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
    temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
    temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
    temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
    temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
    temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
    temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

    out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
    out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
    out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
    out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
    out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
    out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
    out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
    out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
}
template <typename T, int N>
CM_INLINE void read_1d(vector_ref<T, N> out, svmptr_t base) {
    cm_ptr_block_read((T*)base, out);
}

template <typename T, int M, int N>
CM_INLINE void read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_ptr_block_read((T*)base, out.row(i));
    }
}

template <typename TSRC, int M, int N>
CM_INLINE void write_2d(matrix_ref<TSRC, M, N> out, svmptr_t base, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_ptr_block_write((TSRC*)base, out.row(i));
    }
}

template <typename TSRC, int M, int N>
CM_INLINE void write_2d(matrix_ref<TSRC, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, offset += pitch) {
        cm_store<int, N / (sizeof(int) / sizeof(TSRC)), DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(base, offset, out.row(i).format<int>());
    }
}

#if USE_KQ == 0
#if (BLOCK_SG_M == 64 && BLOCK_SG_N == 32)
// register tile: [8, 2] aka[(8*8,16), (16, 16*2)]
CM_INLINE void gemm_qk_8x2_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg

    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                               // --> 64*2 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    // N[0:32]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:32]xK[0:16]                                                                   --> 8+8 regs
    matrix<half, REG_N, BLOCK_REG_B> b0, b1;

    // M[0:64]xK[0:32]
    lsc::block_2d_desc<half, 2, 32, BLOCK_REG_K> desc_a0{ src_a, M - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    // M[32:64]xK[0:32]
    // prefetch A
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a, M - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        (STRIDE - 1) * HEAD_SIZE, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) };
    // 0~3 is M[:]xK[0:16], 4~7 is M[:]xK[16:32]                                        --> 32+32=64 regs
    matrix<half, 8, BLOCK_REG_A> a0, a1;

    // warmup
    // prefetch
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

    // load b: N[0:16]xK[0:16]
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
    desc_b0.set_block_x(desc_b0.get_block_x() + 8);
    cm_sbarrier(1);

    auto dot = [&](matrix_ref<half, 4, BLOCK_REG_A> A0, matrix_ref<half, 4, BLOCK_REG_A> A1, matrix_ref<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)((reg_m + 4) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 4) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A1.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
        desc_a0.set_block_x((STRIDE - 1 - s) * HEAD_SIZE);
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_a.set_block_x((s > 0 ? STRIDE - 1 - s - 1 : 0) * HEAD_SIZE);
            else
                desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load b: N[0:16]xK[16:32]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            // load a: M[0:32]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a1.format<half>(), desc_a0);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            // dot: (M[0:32]xK[0:16])*(N[0:16]xK[0:16])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(), a1.select<4, 1, BLOCK_REG_A, 1>(), b0.select_all());

            // load b: N[0:16]xK[32:48]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // dot: (M[0:32]xK[16:32])*(N[0:16]xK[16:32])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(4, 0), a1.select<4, 1, BLOCK_REG_A, 1>(4, 0), b1.select_all());
        
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
        
            // load b: N[0:16]xK[48:64]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // load a: M[0:32]xK[32:64]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a1.format<half>(), desc_a0);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            // dot: (M[0:32]xK[32:48])*(N[0:16]xK[32:48])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(), a1.select<4, 1, BLOCK_REG_A, 1>(), b0.select_all());

            // load b: N[0:16]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // dot: (M[0:32]xK[48:64])*(N[0:16]xK[48:64])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(4, 0), a1.select<4, 1, BLOCK_REG_A, 1>(4, 0), b1.select_all());
            cm_sbarrier(0);
            cm_sbarrier(1);
        }
    }

    cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, BLOCK_SG_N / 2> desc_c{ dst, M - 1, (uint)(N * sizeof(half) - 1), (uint)(ldc * sizeof(half) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<half, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n) * float{INV_S};
        }
    }
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(0 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(1 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(2 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(3 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 4>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(4 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 5>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(5 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 6>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(6 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 7>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(7 * BLOCK_REG_M, 0).format<int>());
}
#endif

#if (BLOCK_SG_M == 32 && BLOCK_SG_N == 32)
// register tile: [4, 2] aka[(4*8,16), (16, 16*2)]
CM_INLINE void gemm_qk_4x2_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg

    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                               // --> 32*2 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    // N[0:32]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:32]xK[0:16]                                                                   --> 8+8 regs
    matrix<half, REG_N, BLOCK_REG_B> b0, b1;

    // M[0:32]xK[0:32]
    lsc::block_2d_desc<half, 2, 32, BLOCK_REG_K> desc_a0{ src_a, M - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    // prefetch A
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a, M - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        (STRIDE - 1) * HEAD_SIZE, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) };
    // 0~3 is M[:]xK[0:16], 4~7 is M[:]xK[16:32]                                        --> 32 regs
    matrix<half, 8, BLOCK_REG_A> a0;

    // warmup
    // prefetch
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

    // load b: N[0:16]xK[0:16]
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
    desc_b0.set_block_x(desc_b0.get_block_x() + 8);
    cm_sbarrier(1);

    auto dot = [&](matrix_ref<half, 4, BLOCK_REG_A> A0, matrix_ref<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
        desc_a0.set_block_x((STRIDE - 1 - s) * HEAD_SIZE);
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_a.set_block_x((s > 0 ? STRIDE - 1 - s - 1 : 0) * HEAD_SIZE);
            else
                desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load b: N[0:16]xK[16:32]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            // load a: M[0:32]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            // dot: (M[0:32]xK[0:16])*(N[0:16]xK[0:16])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(), b0.select_all());

            // load b: N[0:16]xK[32:48]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // dot: (M[0:32]xK[16:32])*(N[0:16]xK[16:32])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(4, 0), b1.select_all());
        
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
        
            // load b: N[0:16]xK[48:64]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // load a: M[0:32]xK[32:64]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            // dot: (M[0:32]xK[32:48])*(N[0:16]xK[32:48])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(), b0.select_all());

            // load b: N[0:16]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // dot: (M[0:32]xK[48:64])*(N[0:16]xK[48:64])'
            dot(a0.select<4, 1, BLOCK_REG_A, 1>(4, 0), b1.select_all());
            cm_sbarrier(0);
            cm_sbarrier(1);
        }
    }

    cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, BLOCK_SG_N / 2> desc_c{ dst, M - 1, (uint)(N * sizeof(half) - 1), (uint)(ldc * sizeof(half) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<half, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n) * float{INV_S};
        }
    }
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(0 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(1 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(2 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(3 * BLOCK_REG_M, 0).format<int>());
}
#endif

#else

#if (BLOCK_SG_M == 32 && BLOCK_SG_N == 64)
// register tile: [4, 4] aka[(4*8,16), (16, 16*4)]
// src_a is key, src_b is query
CM_INLINE void gemm_kq_4x4_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg
#ifndef BLOCK_SG_M
    #define BLOCK_SG_M  32
    #define BLOCK_SG_N  64
    #define SG_M  4
    #define SG_N  2
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
#endif
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;
    static constexpr int KEY_LINES_PER_LOAD = KV_BLOCK_SIZE / STRIDE;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                              // --> 32*4 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 4, "block_2d_desc for b is manually unrolled by 4");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    static_assert(KV_BLOCK_SIZE == 256, "block size of key(src_a) should be 256");
    // N[0:16*4]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        (STRIDE - 1) * HEAD_SIZE, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:16]xK[0:16]                                                                  --> 8*4 regs
    matrix<half, REG_N, BLOCK_REG_B> b0;

    // M[0:16]xK[0:16]
    uint offset = (uint)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a0{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // M[16:32]xK[0:16]
    offset += KEY_LINES_PER_LOAD * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a1{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // prefetch A
    offset = (uint)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) * (uint)sizeof(half) * lda;
    static_assert(BLOCK_WG_M / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // 0~2 M[:]xK[0:16] 2~4 K[16:32]                                                     --> 32 * 2 regs
    matrix<half, 4, BLOCK_REG_A> a0, a1;

    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);

    auto dot = [&](matrix_ref<half, 2, BLOCK_REG_A> A0, matrix_ref<half, 2, BLOCK_REG_A> A1, matrix_ref<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 2) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 2) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A1.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
        desc_b0.set_block_x((STRIDE - 1 - s) * HEAD_SIZE / 2);
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            // load a: M[0:16*2]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a1.format<half>(), desc_a1);

            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_b.set_block_x((s > 0 ? STRIDE - 1 - s - 1 : 0) * HEAD_SIZE);
            else
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            desc_a1.set_block_x(desc_a1.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), a1.select<2, 1, BLOCK_REG_A, 1>(), b0);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), a1.select<2, 1, BLOCK_REG_A, 1>(2), b0);

            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            // load a: M[0:16*2]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a1.format<half>(), desc_a1);

            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            desc_a1.set_block_x(desc_a1.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), a1.select<2, 1, BLOCK_REG_A, 1>(), b0);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), a1.select<2, 1, BLOCK_REG_A, 1>(2), b0);
        }
    }

    // cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, 16> desc_c{ dst, M - 1, (uint)(N * sizeof(half) - 1), (uint)(ldc * sizeof(half) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<half, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n) * float{INV_S};
        }
    }
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0 * 16, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(0 * BLOCK_REG_M,  0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0 * 16, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(1 * BLOCK_REG_M,  0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0 * 16, 8 * 2>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(2 * BLOCK_REG_M,  0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0 * 16, 8 * 3>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(3 * BLOCK_REG_M,  0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 1 * 16, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(0 * BLOCK_REG_M, 32).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 1 * 16, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(1 * BLOCK_REG_M, 32).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 1 * 16, 8 * 2>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(2 * BLOCK_REG_M, 32).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 1 * 16, 8 * 3>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(3 * BLOCK_REG_M, 32).format<int>());
}
#endif

#if (BLOCK_SG_M == 16 && BLOCK_SG_N == 64)
// register tile: [2, 4] aka[(2*8,16), (16, 16*4)]
// src_a is key, src_b is query
CM_INLINE void gemm_kq_2x4_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg
#ifndef BLOCK_SG_M
    #define BLOCK_SG_M  16
    #define BLOCK_SG_N  64
    #define SG_M  8
    #define SG_N  2
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
#endif
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;
    static constexpr int KEY_LINES_PER_LOAD = KV_BLOCK_SIZE / STRIDE;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                              // --> 32*4 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 4, "block_2d_desc for b is manually unrolled by 4");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    static_assert(KV_BLOCK_SIZE == 256, "block size of key(src_a) should be 256");
    // N[0:16*4]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        (STRIDE - 1) * HEAD_SIZE, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:16]xK[0:16]                                                                  --> 8*4 regs
    matrix<half, REG_N, BLOCK_REG_B> b0;

    // M[0:16]xK[0:16]
    uint offset = (uint)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a0{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // prefetch A
    offset = (uint)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) * (uint)sizeof(half) * lda;
    static_assert(BLOCK_WG_M / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a + offset, BLOCK_WG_M / SG_MN - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // 0~2 M[:]xK[0:16] 2~4 K[16:32]                                                     --> 32 * 2 regs
    matrix<half, 4, BLOCK_REG_A> a0;

    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);

    auto dot = [&](matrix_ref<half, 2, BLOCK_REG_A> A0, matrix_ref<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
        desc_b0.set_block_x((STRIDE - 1 - s) * HEAD_SIZE / 2);
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            // load a: M[0:16]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);

            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_b.set_block_x((s > 0 ? STRIDE - 1 - s - 1 : 0) * HEAD_SIZE);
            else
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), b0);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), b0);

            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            // load a: M[0:16*2]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);

            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), b0);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b0.row(2).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b0.row(3).format<int>(), desc_b0);

            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), b0);
        }
    }

    // cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, 16> desc_c{ dst, M - 1, (uint)(N * sizeof(half) - 1), (uint)(ldc * sizeof(half) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<half, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n) * float{INV_S};
        }
    }
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0 * 16, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(0 * BLOCK_REG_M,  0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0 * 16, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(1 * BLOCK_REG_M,  0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 1 * 16, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(0 * BLOCK_REG_M, 32).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 1 * 16, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, 32, 1>(1 * BLOCK_REG_M, 32).format<int>());
}
#endif

#if (BLOCK_SG_M == 64 && BLOCK_SG_N == 32)
// register tile: [8, 2] aka[(8*8,16), (16, 16*2)]
// src_a is key, src_b is query
CM_INLINE void gemm_kq_8x2_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg
#ifndef BLOCK_SG_M
    #define BLOCK_SG_M  64
    #define BLOCK_SG_N  32
    #define SG_M  2
    #define SG_N  4
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
#endif
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;
    static constexpr int KEY_LINES_PER_LOAD = KV_BLOCK_SIZE / STRIDE;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                              // --> 64*2 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    static_assert(KV_BLOCK_SIZE == 256, "block size of key(src_a) should be 256");
    // N[0:16*2]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        (STRIDE - 1) * HEAD_SIZE / 2, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{src_b, N - 1, (uint)(K * sizeof(half) - 1), (uint)(ldb * sizeof(half) - 1),
        (STRIDE - 1) * HEAD_SIZE, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:16*2]xK[0:16]                                                                  --> 8+8 regs
    matrix<half, REG_N, BLOCK_REG_B> b0, b1;

    // M[0:16]xK[0:32]
    uint offset = (uint)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a0{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // M[16:32]xK[0:32]
    offset += KEY_LINES_PER_LOAD * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a1{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // M[32:48]xK[0:32]
    offset += KEY_LINES_PER_LOAD * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a2{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // M[48:64]xK[0:32]
    offset += KEY_LINES_PER_LOAD * (uint)sizeof(half) * lda;
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a3{ src_a + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // prefetch A
    offset = (uint)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) * (uint)sizeof(half) * lda;
    static_assert(BLOCK_WG_M / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a + offset, BLOCK_WG_M / SG_MN - 1, (uint)(K * sizeof(half) - 1), (uint)(lda * sizeof(half) - 1),
        0, 0 };
    // 0~2 M[:]xK[0:16] 2~4 K[16:32]                                                     --> 32 * 2 regs
    matrix<half, 4, BLOCK_REG_A> a0, a1, a2, a3;

    // warmup
    // prefetch
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

    // load b: N[0:16]xK[0:16]
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
    desc_b0.set_block_x(desc_b0.get_block_x() + 8);
    cm_sbarrier(1);

    auto dot = [&](matrix_ref<half, 2, BLOCK_REG_A> A0, matrix_ref<half, 2, BLOCK_REG_A> A1, matrix_ref<half, 2, BLOCK_REG_A> A2, matrix_ref<half, 2, BLOCK_REG_A> A3, matrix_ref<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 2) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 2) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A1.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 4) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 4) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A2.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 6) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 6) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A3.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_b.set_block_x((STRIDE - 1 - s - 1) * HEAD_SIZE);
            else
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);

            // load b: N[0:16*2]xK[16:32]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            // load a: M[0:16*4]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a1.format<half>(), desc_a1);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a2.format<half>(), desc_a2);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a3.format<half>(), desc_a3);

            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            desc_a1.set_block_x(desc_a1.get_block_x() + 32);
            desc_a2.set_block_x(desc_a2.get_block_x() + 32);
            desc_a3.set_block_x(desc_a3.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), a1.select<2, 1, BLOCK_REG_A, 1>(),
                a2.select<2, 1, BLOCK_REG_A, 1>(), a3.select<2, 1, BLOCK_REG_A, 1>(),
	    	    b0);

            // load b: N[0:16*2]xK[32:48]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), a1.select<2, 1, BLOCK_REG_A, 1>(2),
                a2.select<2, 1, BLOCK_REG_A, 1>(2), a3.select<2, 1, BLOCK_REG_A, 1>(2),
	            b1);

            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load b: N[0:16*2]xK[48:64]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_b0.set_block_x((STRIDE - 1 - s - 1) * HEAD_SIZE / 2);
            else
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // load a: M[0:32]xK[32:64]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a1.format<half>(), desc_a1);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a2.format<half>(), desc_a2);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a3.format<half>(), desc_a3);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            desc_a1.set_block_x(desc_a1.get_block_x() + 32);
            desc_a2.set_block_x(desc_a2.get_block_x() + 32);
            desc_a3.set_block_x(desc_a3.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), a1.select<2, 1, BLOCK_REG_A, 1>(),
                a2.select<2, 1, BLOCK_REG_A, 1>(), a3.select<2, 1, BLOCK_REG_A, 1>(),
	    	    b0);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), a1.select<2, 1, BLOCK_REG_A, 1>(2),
                a2.select<2, 1, BLOCK_REG_A, 1>(2), a3.select<2, 1, BLOCK_REG_A, 1>(2),
	            b1);
            cm_sbarrier(0);
            cm_sbarrier(1);
        }
    }

    cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, BLOCK_SG_N / 2> desc_c{ dst, M - 1, (uint)(N * sizeof(half) - 1), (uint)(ldc * sizeof(half) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<half, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n) * float{INV_S};
        }
    }
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(0 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(1 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(2 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(3 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 4>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(4 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 5>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(5 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 6>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(6 * BLOCK_REG_M, 0).format<int>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 7>(desc_c, tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(7 * BLOCK_REG_M, 0).format<int>());
}
#endif

#endif