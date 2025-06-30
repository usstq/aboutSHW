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

/*
 * one WG of C matrix[BLOCK_WG_M, BLOCK_WG_N]:
 *                          --------------------------------------------------------------------
 * EU_M=0                   | EU_N=0[BLOCK_SG_M, BLOCK_SG_N]  | EU_N=1 .... | EU_N=SG_M-1
 *                          --------------------------------------------------------------------
 * ...                      |
 *                          --------------------------------------------------------------------
 * EU_M=SG_N-1              |
 *                          --------------------------------------------------------------------
 *
 * one SG of C matrix[BLOCK_SG_M, BLOCK_SG_N]:
 *                          --------------------------------------------------------------------
 * ACC_M=0                  | ACC_N=0(float, 8, 8|16)  |  ACC_N=1...  |  ACC_N=REG_N-1
 *                          --------------------------------------------------------------------
 * ...                      |
 *                          --------------------------------------------------------------------
 * ACC_M=REG_M-1            |
 *                          --------------------------------------------------------------------
 *
 * one SG of A matrix[BLOCK_SG_M, BLOCK_WG_K]:
 *                          --------------------------------------------------------------------
 * A_M=0                    | A_N=0(half, 8, BLOCK_REG_K)  | A_N=1...| A_N=BLOCK_WG_K/BLOCK_REG_K-1
 *                          --------------------------------------------------------------------
 * ...                      |
 *                          --------------------------------------------------------------------
 * A_M=REG_M-1              |
 *                          --------------------------------------------------------------------
 *
 * one SG of B matrix[BLOCK_WG_K, BLOCK_SG_N]:
 *                          --------------------------------------------------------------------
 * B_M=0                    | B_N=0(half, BLOCK_REG_K, BLOCK_REG_N)  | B_N=1...| B_N=REG_N-1
 *                          --------------------------------------------------------------------
 * ...                      |
 *                          --------------------------------------------------------------------
 * B_M=BLOCK_WG_K/BLOCK_REG_K-1  |
 *                          --------------------------------------------------------------------
 * 
 * SLM for A[BLOCK_SG_M, BLOCK_WG_K]:
 *                          --------------------------------------------------------------------
 *                          | EU0: A_M=0(BLOCK_REG_M, BLOCK_REG_K)..., A_M=REG_M-1, next K ... | EU1: ...
 *
 * SLM for B[BLOCK_WG_K, BLOCK_SG_N]:
 *                          --------------------------------------------------------------------
 *                          | EU0: B_N=0(BLOCK_REG_K, BLOCK_REG_N)..., B_N=REG_N-1, next K ... | EU1: ...
 */
// global: [M/BLOCK_SG_M, N/BLOCK_SG_N]
// local:  [SG_M, SG_N]
// each EU will compute [BLOCK_SG_M, BLOCK_SG_N]
template<
    typename TYPE_SRC=half,
    typename TYPE_ACC=float,
    typename TYPE_DST=half,
    CmPrecisionType A_TYPE_PREC = CM_PRECISION_HF,
    CmPrecisionType B_TYPE_PREC = CM_PRECISION_HF,
    int SG_SIZE = 8,
    // subgroup blocking
    int BLOCK_SG_M = 16,
    int BLOCK_SG_N = 16,
    int SG_M = 8,
    int SG_N = 8,
    int BLOCK_WG_K = 64
>
CM_INLINE void gemm_xmx(uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    const int REPEAT = 8;
    const int DEPTH = 8;
    const int BLOCK_REG_M = REPEAT;
    const int BLOCK_REG_N = SG_SIZE;
    const int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    const int VNNI = sizeof(TYPE_SRC);
    const int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    const int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    const int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    const int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    const int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    const int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    const int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    const int REG_MN = REG_M * REG_N;
    const uint size_slm_a = BLOCK_WG_M * BLOCK_WG_K * sizeof(TYPE_SRC);
    const uint size_slm_b = BLOCK_WG_K * BLOCK_WG_N * sizeof(TYPE_SRC);

    matrix<TYPE_ACC, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;
    auto slm_b_offset = slm + size_slm_a;
    uint id_wg_m = cm_group_id(0);
    uint id_wg_n = cm_group_id(1);
    uint id_sg_m = cm_local_id(0);
    uint id_sg_n = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    int slm_offset_wg_a = id_sg_m * BLOCK_SG_M * BLOCK_WG_K * sizeof(TYPE_SRC);
    int slm_offset_wg_b = id_sg_n * BLOCK_SG_N * BLOCK_WG_K * sizeof(TYPE_SRC) + slm_b_offset;

    // for load slm
    const int BLOCK_REG_M_LOAD_A = BLOCK_WG_M / (SG_M * SG_N);
    const uint lda_bytes = lda * sizeof(TYPE_SRC);
    auto offset_sg_a = (id_wg_m * BLOCK_WG_M +                                          // workgroup
                        id_sg_mn * BLOCK_REG_M_LOAD_A) * lda_bytes;                     // subgroup
    const int BLOCK_REG_M_LOAD_B = BLOCK_WG_K * VNNI * BLOCK_WG_N / (SG_M * SG_N);
    uint offset_sg_b = id_wg_n * K / BLOCK_WG_K * BLOCK_WG_K * VNNI * BLOCK_WG_N +      // workgroup
                       id_sg_mn * BLOCK_REG_M_LOAD_B;                                   // subgroup
    auto offset_slm_sg_b = slm_b_offset + id_sg_mn * BLOCK_REG_M_LOAD_B;

    // compute K dimension inside one EU
    for (uint k = 0; k < K / BLOCK_WG_K; k++) {
        // src_a[BLOCK_SG_M, BLOCK_WG_K] -> slm
        // slm layout:
        // struct reg {                                 // --> one DPAS op
        //   int data[BLOCK_REG_M][DEPTH];              // BLOCK_REG_M==8, DEPTH==8, aka int[8,8]
        // };
        // struct sg_block {                            // --> one EU
        //   reg regs[BLOCK_WG_K/BLOCK_REG_K][REG_M];   // REG_M==2, BLOCK_WG_K/BLOCK_REG_K==4
        // };
        // struct wg_block {                            // --> one WG
        //   sg_block block[SG_M];
        // };
        {
            // TODO(tune): each EU will read serveral lines
            matrix<TYPE_SRC, BLOCK_REG_M_LOAD_A, BLOCK_WG_K> data;
            read_2d(data.format<int, BLOCK_REG_M_LOAD_A, BLOCK_WG_K * sizeof(TYPE_SRC) / sizeof(int)>(), (svmptr_t)src_a + offset_sg_a, lda_bytes);
            offset_sg_a += BLOCK_WG_K * sizeof(TYPE_SRC);
#pragma unroll
            for (uint m = 0; m < BLOCK_REG_M_LOAD_A; m++) {
                auto m_in_sg = id_sg_mn * BLOCK_REG_M_LOAD_A + m;
                auto sg_m = m_in_sg / BLOCK_SG_M;
                auto reg_m = m_in_sg % BLOCK_SG_M / BLOCK_REG_M;
                auto m_in_reg = m_in_sg % BLOCK_SG_M % BLOCK_REG_M;
                uint offset = sg_m * (BLOCK_WG_K / BLOCK_REG_K * REG_M * BLOCK_REG_A) +                     // block[x]
                              reg_m * BLOCK_REG_A +                                                         // regs[y][z]
                              m_in_reg * BLOCK_REG_K;                                                       // data[m]
                offset *= sizeof(TYPE_SRC);
#pragma unroll
                for (uint block_reg_k = 0; block_reg_k < BLOCK_WG_K; block_reg_k += BLOCK_REG_K) {
                    auto tmp = data.select<1, 1, BLOCK_REG_K, 1>(m, block_reg_k);
                    cm_slm_block_write(slm, offset, tmp.format<int>());
                    offset += REG_M * BLOCK_REG_A * sizeof(TYPE_SRC);                                       // regs[y]
                }
            }
        }

        // src_b -> slm, slm layout is same with `wg_block`
        //
        // src_b memory layout:
        // struct reg {                                 // --> one DPAS op
        //   int data[DEPTH][BLOCK_REG_N];              // DEPTH=8, BLOCK_REG_N=8, aka int[8,8]
        // };
        // struct sg_block {                            // --> one EU
        //   reg regs[BLOCK_WG_K/BLOCK_REG_K][REG_N];   // REG_N==2, BLOCK_WG_K/BLOCK_REG_K==4
        // };
        // struct wg_block {                            // --> one WG
        //   sg_block block[SG_N];
        // };
        // struct weight {                              // --> weight
        //   wg_block wg[WG_N][K/BLOCK_WG_K];
        // };
        {
            // because src_b is prepacked, only simple copy is needed
            // TODO(tune): each EU will read serveral lines
            vector<int, BLOCK_REG_M_LOAD_B / sizeof(int)> data;
            read_1d(data.select_all(), (svmptr_t)src_b + offset_sg_b);
            offset_sg_b += BLOCK_WG_K * VNNI * BLOCK_WG_N;
            cm_slm_block_write(slm, offset_slm_sg_b, data);
        }
        cm_barrier();

        matrix<TYPE_SRC, REG_M, BLOCK_REG_A> A;
        matrix<TYPE_SRC, REG_N, BLOCK_REG_B> B;
#pragma unroll
        for (uint reg_k = 0; reg_k < BLOCK_WG_K / BLOCK_REG_K; reg_k++) {
            int offset = slm_offset_wg_a + (reg_k * REG_M * BLOCK_REG_A) * sizeof(TYPE_SRC);
            cm_slm_block_read(slm, offset, A.format<int>());

            offset = slm_offset_wg_b + (reg_k * REG_N * BLOCK_REG_B) * sizeof(TYPE_SRC);
            cm_slm_block_read(slm, offset, B.format<int>());

#pragma unroll
            for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
#pragma unroll
                for (int reg_n = 0; reg_n < REG_N; reg_n++) {
                    acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                        B.row((ushort)reg_n).format<int>(), A.row((ushort)reg_m).format<int>());
                }
            }
        }
        cm_barrier();
    }
    // store
    auto ldc_bytes = ldc * sizeof(TYPE_DST);
    auto offset_c = id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N + 
                    (id_wg_m * BLOCK_WG_M +        // workgroup
                     id_sg_m * BLOCK_SG_M) * ldc;  // subgroup
    offset_c *= sizeof(TYPE_DST);
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        matrix<TYPE_DST, BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(0, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
        }
        write_2d(tmp.select_all(), (svmptr_t)dst + offset_c, ldc_bytes);
        offset_c += BLOCK_REG_M * ldc_bytes;
    }
}

template<
    int SG_SIZE = 8,
    // subgroup blocking
    int BLOCK_SG_M = 16,
    int BLOCK_SG_N = 16,
    int SG_M = 8,
    int SG_N = 8,
    int BLOCK_WG_K = 64
>
void repack_f16(half* src, half* dst, int K, int N) {
    const int DEPTH = 8;
    const int BLOCK_REG_M = 8;
    const int BLOCK_REG_N = SG_SIZE;
    // half is 2, u8 is 4
    const int VNNI = 2;
    const int BLOCK_REG_K = 32 / VNNI;
    const int BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const int BLOCK_WG_N = BLOCK_SG_N * SG_N;
    // register blocking
    const int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    const int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    // packed memory layout:
    struct reg {                                        // --> one DPAS op
        half data[BLOCK_REG_K / 2][BLOCK_REG_N * 2];    // BLOCK_REG_K=16, BLOCK_REG_N=8, aka half[8,16]
    };
    struct sg_block {                                   // --> one EU
        reg regs[BLOCK_WG_K / BLOCK_REG_K][REG_N];        // REG_N==2, BLOCK_WG_K/BLOCK_REG_K==4
    };
    struct wg_block {                                   // --> one WG
        sg_block block[SG_N];
    };
    //struct weight {                                   // --> weight
    //    wg_block wg[WG_N][K/BLOCK_WG_K];
    //};

    const int WG_N = N / BLOCK_WG_N;
    const int WG_K = K / BLOCK_WG_K;
    wg_block* wg_block_ptr = (wg_block*)dst;
    // for (int wg_n = 0; wg_n < WG_N; wg_n++) {
    //     for (int wg_k = 0; wg_k < WG_K; wg_k++) {
    int wg_n = cm_group_id(0);
    int wg_k = cm_group_id(1);
    wg_block& wg = *(wg_block_ptr + wg_n * WG_K + wg_k);
    half* src_wg_block = src + wg_k * BLOCK_WG_K * N + wg_n * BLOCK_WG_N;
    for (int sg_n = 0; sg_n < SG_N; sg_n++) {
        sg_block& sg = wg.block[sg_n];
        half* src_sg_block = src_wg_block + sg_n * BLOCK_SG_N;
        for (int reg_k = 0; reg_k < BLOCK_WG_K / BLOCK_REG_K; reg_k++) {
            for (int reg_n = 0; reg_n < REG_N; reg_n++) {
                reg& cur_reg = sg.regs[reg_k][reg_n];
                half* src_reg = src_sg_block + reg_k * BLOCK_REG_K * N + reg_n * BLOCK_REG_N;
                for (int k = 0; k < BLOCK_REG_K / 2; k++) {
                    vector<half, BLOCK_REG_N> data0, data1;
                    vector<half, 2 * BLOCK_REG_N> data;
                    cm_svm_block_read((svmptr_t)(src_reg + (k * 2 + 0) * N), data0);
                    cm_svm_block_read((svmptr_t)(src_reg + (k * 2 + 1) * N), data1);
                    data.select<BLOCK_REG_N, 2>(0) = data0;
                    data.select<BLOCK_REG_N, 2>(1) = data1;
                    cm_svm_block_write((svmptr_t)(&cur_reg.data[k][0]), data);
                }
            }
        }
    }
}

template<
    typename TYPE_SRC = half,
    typename TYPE_ACC = float,
    typename TYPE_DST = half,
    CmPrecisionType A_TYPE_PREC = CM_PRECISION_HF,
    CmPrecisionType B_TYPE_PREC = CM_PRECISION_HF,
    int SG_SIZE = 8,
    // subgroup blocking
    int BLOCK_SG_M = 64,
    int BLOCK_SG_N = 16,
    int SG_M = 2,
    int SG_N = 16,
    int BLOCK_WG_K = 64
>
CM_INLINE void gemm_xe1_f16(uint id_wg_m, uint id_wg_n, uint slm, SurfaceIndex src_a, SurfaceIndex src_b, SurfaceIndex dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(TYPE_SRC);
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
    static constexpr uint size_slm_a = BLOCK_WG_M * BLOCK_WG_K * sizeof(TYPE_SRC);

    matrix<TYPE_ACC, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    // A memory->reg
    static constexpr int LINE_REG_M_LOAD_A = BLOCK_WG_M / (SG_M * SG_N);
    // A memory address, step is 16*2
    matrix<uint32_t, LINE_REG_M_LOAD_A, SG_SIZE> mem_addr_a;                                    // ---> 4 regs (addr for A)
    {
        uint offset = (id_wg_m * BLOCK_WG_M + id_sg_mn * LINE_REG_M_LOAD_A) * lda * sizeof(TYPE_SRC);
#pragma unroll
        for (uint m = 0; m < LINE_REG_M_LOAD_A; m++) {
            mem_addr_a[m][0] = offset;
            offset += lda * sizeof(TYPE_SRC);
        }
    }
    // A reg->slm: slm address for store, step is (64*2)*16*2
    uint slm_addr_store_a = id_sg_m * (BLOCK_SG_M * BLOCK_WG_K * sizeof(TYPE_SRC)) +
        id_sg_n * (LINE_REG_M_LOAD_A * BLOCK_REG_K * sizeof(TYPE_SRC));                         // ----> 1 reg (addr for A)
    // A slm->reg: slm address for load, step is (64*2)*16*2
    matrix<uint, REG_M, SG_SIZE> slm_addr_load_a;                                               // ----> 8 regs (addr for A)
    uint slm_addr_load_a_sg = id_sg_m * (BLOCK_SG_M * BLOCK_WG_K * sizeof(TYPE_SRC)) + (id_sg_n & 1) * (BLOCK_REG_A / 2 * sizeof(TYPE_SRC));
    {
        uint offset = 0;
#pragma unroll
        for (uint m = 0; m < REG_M; m++) {
            slm_addr_load_a[m][0] = slm_addr_load_a_sg + offset;
            offset += BLOCK_REG_A * sizeof(TYPE_SRC);
        }
    }
    // B memory->reg: step is 16*2
    matrix<uint, REG_N, BLOCK_REG_N> mem_addr_b;                                                // ---> 2 regs (addr for B)
    {
        uint offset = (id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * ldb * sizeof(TYPE_SRC);
#pragma unroll
        for (uint reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint n = 0; n < BLOCK_REG_N; n++) {
                mem_addr_b[reg_n][n] = offset;
                offset += ldb * sizeof(TYPE_SRC);
            }
        }
    }

    matrix<TYPE_SRC, LINE_REG_M_LOAD_A, BLOCK_REG_K> a_copy0, a_copy1;                      // ---> 8 regs (load for A)
    auto load_a_mem = [&] (matrix_ref<TYPE_SRC, LINE_REG_M_LOAD_A, BLOCK_REG_K> mat_copy) {
#pragma unroll
        for (uint m = 0; m < LINE_REG_M_LOAD_A; m++) {
            mat_copy.row(m).format<int>() = cm_load<int, BLOCK_REG_K / (sizeof(int) / sizeof(TYPE_SRC)), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(src_a, mem_addr_a[m][0]);
        }
    };
    auto inc_load_a_mem = [&]() {
#pragma unroll
        for (uint m = 0; m < LINE_REG_M_LOAD_A; m++) {
            mem_addr_a[m][0] += BLOCK_REG_K * sizeof(TYPE_SRC);
        }
    };
    auto store_a_slm = [&] (matrix_ref<TYPE_SRC, LINE_REG_M_LOAD_A, BLOCK_REG_K> mat_copy) {
        cm_slm_block_write(slm, slm_addr_store_a, mat_copy.format<int>());
    };
    auto inc_store_a_slm = [&](const int k_idx) {
        if (k_idx == REG_K - 1) {
            slm_addr_store_a -= (REG_K - 1) * BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
        } else {
            slm_addr_store_a += BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
        }
    };

    matrix<TYPE_SRC, REG_M, BLOCK_REG_A / 2> a_regs;                                    // ---> 32 regs (dpasw for A)
    auto load_a_slm = [&](matrix_ref<TYPE_SRC, REG_M, BLOCK_REG_A / 2> mat_copy) {
#pragma unroll
        for (uint m = 0; m < REG_M; m++) {
            cm_slm_block_read(slm, slm_addr_load_a[m][0], mat_copy.row(m).format<int>());
        }
    };
    auto inc_load_a_slm = [&](const int k_idx) {
#pragma unroll
        for (uint m = 0; m < REG_M; m++) {
            if (k_idx == REG_K - 1) {
                slm_addr_load_a[m][0] -= (REG_K - 1) * BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
            } else {
                slm_addr_load_a[m][0] += BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
            }
        }
    };
    matrix<TYPE_SRC, REG_N, BLOCK_REG_B> b_regs0, b_regs1;                               // ---> 16*2 regs (dpasw for B)
    auto load_b_mem = [&] (matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> mat_copy) {
#pragma unroll
        for (uint n = 0; n < REG_N; n++) {
            mat_copy.row(n).format<int>() = cm_load<int, VectorSize::N8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(src_b, mem_addr_b.row(n));
        }
    };
    auto inc_load_b_mem = [&]() {
#pragma unroll
        for (uint n = 0; n < REG_N; n++) {
            mem_addr_b.row(n) += BLOCK_REG_K * sizeof(TYPE_SRC);
        }
    };

    auto dot = [&](matrix_ref<TYPE_SRC, REG_M, BLOCK_REG_A / 2> A, matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpasw<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A.row((ushort)reg_m).format<int>());
            }
        }
    };
    // warmup
    load_a_mem(a_copy0);            // k0 --> r0
    inc_load_a_mem();
    load_a_mem(a_copy1);            // k1 --> r1
    inc_load_a_mem();
    store_a_slm(a_copy0);           // r0 --> s0
    load_a_mem(a_copy0);            // k2 --> r0
    inc_store_a_slm(0);
    inc_load_a_mem();
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);
    store_a_slm(a_copy1);           // r1 --> s1
    load_a_mem(a_copy1);            // k3 --> r1
    load_b_mem(b_regs0);            // k0'--> b0
    inc_store_a_slm(1);
    inc_load_a_mem();
    inc_load_b_mem();
    cm_sbarrier(0);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);
    store_a_slm(a_copy0);           // r0 --> s2

    // main loop: K dimension inside one EU
    for (uint k = 0; k < K / BLOCK_WG_K - 1; k++) {
        load_a_mem(a_copy0);        // k0 --> r1
        load_b_mem(b_regs1);        // k1'--> b1
        load_a_slm(a_regs);         // s0 --> a
        inc_store_a_slm(2);
        inc_load_a_slm(0);
        inc_load_a_mem();
        inc_load_b_mem();
        dot(a_regs, b_regs0);       // compute k0
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
        store_a_slm(a_copy1);

        load_a_mem(a_copy1);        // k1 --> r0
        load_b_mem(b_regs0);        // k2'--> b0
        load_a_slm(a_regs);         // s1 --> a
        inc_store_a_slm(3);
        inc_load_a_slm(1);
        inc_load_a_mem();
        inc_load_b_mem();
        dot(a_regs, b_regs1);       // compute k1
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
        store_a_slm(a_copy0);

        load_a_mem(a_copy0);        // k2 --> r1
        load_b_mem(b_regs1);        // k3'--> b1
        load_a_slm(a_regs);         // s0 --> a
        inc_store_a_slm(0);
        inc_load_a_slm(2);
        inc_load_a_mem();
        inc_load_b_mem();
        dot(a_regs, b_regs0);       // compute k2
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
        store_a_slm(a_copy1);

        load_a_mem(a_copy1);        // k3 --> r0
        load_b_mem(b_regs0);        // k0'--> b0
        load_a_slm(a_regs);         // s1 --> a
        inc_store_a_slm(1);
        inc_load_a_slm(3);
        inc_load_a_mem();
        inc_load_b_mem();
        dot(a_regs, b_regs1);       // compute k3
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
        store_a_slm(a_copy0);
    }
    // last iteration
    {
        load_b_mem(b_regs1);        // k1'--> b1
        load_a_slm(a_regs);         // s0 --> a
        inc_store_a_slm(2);
        inc_load_a_slm(0);
        inc_load_b_mem();
        dot(a_regs, b_regs0);       // compute k0
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
        store_a_slm(a_copy1);

        load_b_mem(b_regs0);        // k2'--> b0
        load_a_slm(a_regs);         // s1 --> a
        inc_load_a_slm(1);
        inc_load_b_mem();
        dot(a_regs, b_regs1);       // compute k1
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);

        load_b_mem(b_regs1);        // k3'--> b1
        load_a_slm(a_regs);         // s0 --> a
        inc_load_a_slm(2);
        dot(a_regs, b_regs0);       // compute k2
        cm_sbarrier(0);

        load_a_slm(a_regs);         // s1 --> a
        dot(a_regs, b_regs1);       // compute k3
    }

    // store
    uint ldc_bytes = ldc * sizeof(TYPE_DST);
    uint offset_c = id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N +
        (id_wg_m * BLOCK_WG_M +        // workgroup
            id_sg_m * BLOCK_SG_M) * ldc;  // subgroup
    offset_c *= sizeof(TYPE_DST);
    matrix<TYPE_DST, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
        }
    }
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        write_2d(tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, 0), dst, offset_c, ldc_bytes);
        offset_c += BLOCK_REG_M * ldc_bytes;
    }
}

template<
    typename TYPE_SRC = half,
    typename TYPE_ACC = float,
    typename TYPE_DST = half,
    CmPrecisionType A_TYPE_PREC = CM_PRECISION_HF,
    CmPrecisionType B_TYPE_PREC = CM_PRECISION_HF,
    int SG_SIZE = 8,
    // subgroup blocking
    int BLOCK_SG_M = 64,
    int BLOCK_SG_N = 16,
    int SG_M = 2,
    int SG_N = 16,
    int BLOCK_WG_K = 128,
    int GROUP_SIZE = 128
>
CM_INLINE void gemm_a16w4_xe1(uint id_wg_m, uint id_wg_n, uint slm, SurfaceIndex src_a, SurfaceIndex src_b, SurfaceIndex src_b_scale, SurfaceIndex src_b_zp, SurfaceIndex dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(TYPE_SRC);
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
    static constexpr uint size_slm_a = BLOCK_WG_M * BLOCK_WG_K * sizeof(TYPE_SRC);

    matrix<TYPE_ACC, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    // A memory->reg
    static constexpr int LINE_REG_M_LOAD_A = BLOCK_WG_M / (SG_M * SG_N);
    // A memory address, step is 16*sizeof(half)*2, aka k step is 32(points)
    matrix<uint32_t, LINE_REG_M_LOAD_A, SG_SIZE> mem_addr_a;                                    // ---> 4 regs (addr for A)
    {
        uint offset = (id_wg_m * BLOCK_WG_M + id_sg_mn * LINE_REG_M_LOAD_A) * lda * sizeof(TYPE_SRC);
#pragma unroll
        for (uint m = 0; m < LINE_REG_M_LOAD_A; m++) {
            mem_addr_a[m][0] = offset;
            offset += lda * sizeof(TYPE_SRC);
        }
    }
    // A reg->slm: slm address for store, step is (64*2)*16*2=4096
    vector<uint, 8> slm_addr_store_a0, slm_addr_store_a1;                                       // ----> 2 reg (addr for A)
    slm_addr_store_a0[0] = id_sg_m * (BLOCK_SG_M * BLOCK_WG_K * sizeof(TYPE_SRC)) +
        id_sg_n * (LINE_REG_M_LOAD_A * BLOCK_REG_K * sizeof(TYPE_SRC));
    slm_addr_store_a1[0] = slm_addr_store_a0[0] + BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
    
    // A slm->reg: slm address for load, step is 64*16*2=2048
    matrix<uint, REG_M, SG_SIZE> slm_addr_load_a;                                               // ----> 8 regs (addr for A)
    uint slm_addr_load_a_sg = id_sg_m * (BLOCK_SG_M * BLOCK_WG_K * sizeof(TYPE_SRC)) + (id_sg_n & 1) * (BLOCK_REG_A / 2 * sizeof(TYPE_SRC));
    {
        uint offset = 0;
#pragma unroll
        for (uint m = 0; m < REG_M; m++) {
            slm_addr_load_a[m][0] = slm_addr_load_a_sg + offset;
            offset += BLOCK_REG_A * sizeof(TYPE_SRC);
        }
    }
    // B memory->reg: step is 16*sizeof(uchar)
    matrix<uint, REG_N, BLOCK_REG_N> mem_addr_b;                                                // ---> 2 regs (addr for B)
    {
        uint offset = (id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * ldb / 2;
#pragma unroll
        for (uint reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint n = 0; n < BLOCK_REG_N; n++) {
                mem_addr_b[reg_n][n] = offset;
                offset += ldb / 2;
            }
        }
    }

    matrix<TYPE_SRC, LINE_REG_M_LOAD_A, BLOCK_REG_K * 2> a_copy;                               // ---> 8 regs (load for A)
    matrix<TYPE_SRC, 2, 4 * BLOCK_REG_K> a_cvt;                                                // ---> 8 regs (cvt for A) ??? reuse a_regs?
    auto load_a_mem = [&] (matrix_ref<TYPE_SRC, LINE_REG_M_LOAD_A, BLOCK_REG_K * 2> mat_copy) {
#pragma unroll
        for (uint m = 0; m < LINE_REG_M_LOAD_A; m++) {
            mat_copy.row(m).format<int>() = cm_load<int, BLOCK_REG_K * 2 / (sizeof(int) / sizeof(TYPE_SRC)), DataSize::Default, CacheHint::Cached, CacheHint::Cached>(src_a, mem_addr_a[m][0]);
        }
    };
    auto inc_load_a_mem = [&]() {
#pragma unroll
        for (uint m = 0; m < LINE_REG_M_LOAD_A; m++) {
            mem_addr_a[m][0] += BLOCK_REG_K * 2 * sizeof(TYPE_SRC);
        }
    };
    auto store_a_slm = [&] (matrix<TYPE_SRC, 2, 4 * BLOCK_REG_K> mat_copy) {
        cm_slm_block_write(slm, slm_addr_store_a0[0], mat_copy.row(0).format<int>());
        cm_slm_block_write(slm, slm_addr_store_a1[0], mat_copy.row(1).format<int>());
    };
    auto inc_store_a_slm = [&](const int k_idx) {
        if (k_idx == 3) {
            slm_addr_store_a0[0] -= 3 * BLOCK_SG_M * BLOCK_REG_K * 2 * sizeof(TYPE_SRC);
            slm_addr_store_a1[0] -= 3 * BLOCK_SG_M * BLOCK_REG_K * 2 * sizeof(TYPE_SRC);
        } else {
            slm_addr_store_a0[0] += BLOCK_SG_M * BLOCK_REG_K * 2 * sizeof(TYPE_SRC);
            slm_addr_store_a1[0] += BLOCK_SG_M * BLOCK_REG_K * 2 * sizeof(TYPE_SRC);
        }
    };

    matrix<TYPE_SRC, REG_M, BLOCK_REG_A / 2> a_regs;                                    // ---> 32 regs (dpasw for A)
    auto load_a_slm = [&](matrix_ref<TYPE_SRC, REG_M, BLOCK_REG_A / 2> mat_copy) {
#pragma unroll
        for (uint m = 0; m < REG_M; m++) {
            cm_slm_block_read(slm, slm_addr_load_a[m][0], mat_copy.row(m).format<int>());
        }
    };
    auto inc_load_a_slm = [&](const int k_idx) {
#pragma unroll
        for (uint m = 0; m < REG_M; m++) {
            if (k_idx == 7) {
                slm_addr_load_a[m][0] -= 7 * BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
            } else {
                slm_addr_load_a[m][0] += BLOCK_SG_M * BLOCK_REG_K * sizeof(TYPE_SRC);
            }
        }
    };
    matrix<uint, REG_N, 8 * 4> b_copy0, b_copy1;                                         // ---> 16*2 regs (mem->reg for B)
    matrix<TYPE_SRC, REG_N, BLOCK_REG_B> b_regs;                                         // ---> 16*1 regs (cvt, dpasw for B)
    auto load_b_mem = [&] (matrix_ref<uint, REG_N, 8 * 4> mat_copy) {
#pragma unroll
        for (uint n = 0; n < REG_N; n++) {
            mat_copy.row(n).format<int>() = cm_load<int, VectorSize::N4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(src_b, mem_addr_b.row(n));
        }
    };
    auto inc_load_b_mem = [&]() {
#pragma unroll
        for (uint n = 0; n < REG_N; n++) {
            mem_addr_b.row(n) += 4 * sizeof(int);
        }
    };

    auto dot = [&](matrix_ref<TYPE_SRC, REG_M, BLOCK_REG_A / 2> A, matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpasw<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A.row((ushort)reg_m).format<int>());
            }
        }
    };

    auto deinterleave = [&] (matrix_ref<TYPE_SRC, LINE_REG_M_LOAD_A, BLOCK_REG_K * 2> src, matrix_ref<TYPE_SRC, 2, 4 * BLOCK_REG_K> dst) {
#pragma unroll
        for (uint i = 0; i < 2; i++) {
#pragma unroll
            for (uint j = 0; j < 4; j++) {
                dst.row(i).select<BLOCK_REG_K, 1>(j * BLOCK_REG_K).format<int>() = src.row(j).select<BLOCK_REG_K, 1>(i * BLOCK_REG_K).format<int>();
            }
        }
    };

    vector<uchar, 8> zp_copy;
    vector<TYPE_SRC, 32> zp_f;
    vector<TYPE_SRC, 16> scale_copy;
    vector<TYPE_SRC, 32> scale;
    uint zp_offset = (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2;
    uint scale_offset = (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * sizeof(TYPE_SRC);
    auto load_zp_scale = [&] () {
        zp_copy.format<int>() = cm_load<int, 2>(src_b_zp, zp_offset);
        scale_copy.format<int>() = cm_load<int, 8>(src_b_scale, scale_offset);
    };
    auto inc_load_zp_scale = [&] () {
        zp_offset += N / 2;
        scale_offset += N * sizeof(TYPE_SRC);
    };
    auto dec_zp = [&] (vector_ref<uchar, 8> z_src, vector_ref<TYPE_SRC, 32> z_f) {
        vector<ushort, 16> tmp;
        tmp.select<8, 2>(0) = (z_src & (ushort)0xf);
        tmp.select<8, 2>(1) = z_src >> (ushort)4;
        tmp.format<TYPE_SRC>() *= half{4096 * (4096 / 512)};
        z_f.select<16, 2>(0) = tmp.format<TYPE_SRC>();
        z_f.select<16, 2>(1) = tmp.format<TYPE_SRC>();
    };
    auto dec_wei = [&](vector_ref<int, 16> b_src0, vector_ref<int, 16> b_src1, matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> b_dst, vector_ref<TYPE_SRC, 32> s, vector_ref<TYPE_SRC, 32> z) {
#pragma unroll
        for (uint i = 0; i < REG_N; i++) {
            auto b_reg = b_dst.row(i).format<TYPE_SRC, BLOCK_REG_K / 2, BLOCK_REG_N * 2>();
#pragma unroll
            for (uint k = 0; k < BLOCK_REG_K / 2; k++) {
                auto tmp_f = b_reg.row(k);
                if (i == 0) {
                    auto tmp = b_src0.select<8, 1>(k / 4 * 8).format<uchar>().select<8, 4>(k % 4);
                    tmp_f.select<8, 2>(0).format<ushort>() = tmp & (ushort)0xf;
                    tmp_f.select<8, 2>(1).format<ushort>() = tmp >> 4;
                    //tmp_f.select<8, 2>(1).format<ushort>() = tmp;
                    //tmp_f.select<8, 2>(1).format<ushort>() /= 16;
                } else {
                    auto tmp = b_src1.select<8, 1>(k / 4 * 8).format<uchar>().select<8, 4>(k % 4);
                    tmp_f.select<8, 2>(0).format<ushort>() = tmp & (ushort)0xf;
                    tmp_f.select<8, 2>(1).format<ushort>() = tmp >> 4;
                    //tmp_f.select<8, 2>(1).format<ushort>() = tmp;
                    //tmp_f.select<8, 2>(1).format<ushort>() /= 16;
                }
            }
        }
#pragma unroll
        for (uint i = 0; i < REG_N; i++) {
            auto b_reg = b_dst.row(i).format<TYPE_SRC, BLOCK_REG_K / 2, BLOCK_REG_N * 2>();
#pragma unroll
            for (uint k = 0; k < BLOCK_REG_K / 2; k++) {
                auto tmp_f = b_reg.row(k);
                tmp_f = (tmp_f * half{32768.0} - z.select<16, 1>(i * 16));
            }
        }
#pragma unroll
        for (uint i = 0; i < REG_N; i++) {
            auto b_reg = b_dst.row(i).format<TYPE_SRC, BLOCK_REG_K / 2, BLOCK_REG_N * 2>();
#pragma unroll
            for (uint k = 0; k < BLOCK_REG_K / 2; k++) {
                auto tmp_f = b_reg.row(k);
                tmp_f *= half{512.0};
            }
        }
#pragma unroll
        for (uint i = 0; i < REG_N; i++) {
            auto b_reg = b_dst.row(i).format<TYPE_SRC, BLOCK_REG_K / 2, BLOCK_REG_N * 2>();
#pragma unroll
            for (uint k = 0; k < BLOCK_REG_K / 2; k++) {
                auto tmp_f = b_reg.row(k);
                tmp_f *= s.select<16, 1>(i * 16);
            }
        }
    };

    // warmup
    load_a_mem(a_copy);            // k0 --> r
    inc_load_a_mem();
    deinterleave(a_copy, a_cvt);
    store_a_slm(a_cvt);            // r  --> s0
    load_a_mem(a_copy);            // k1 --> r
    inc_store_a_slm(0);            // inc after s0
    inc_load_a_mem();
    deinterleave(a_copy, a_cvt);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);
    store_a_slm(a_cvt);            // r  --> s1
    load_a_mem(a_copy);            // k2 --> r
    load_b_mem(b_copy0);           // k0'--> r
    inc_store_a_slm(1);
    inc_load_a_mem();
    inc_load_b_mem();
    load_zp_scale();
    deinterleave(a_copy, a_cvt);
    cm_sbarrier(0);
    cm_slm_fence(CM_LOCAL_BARRIER);
    cm_sbarrier(1);
    store_a_slm(a_cvt);            // r  --> s2

    // main loop: K dimension inside one EU
    for (uint k = 0; k < K / BLOCK_WG_K - 1; k++) {
#pragma unroll
        for (uint i = 0; i < 4; i++) {
            load_a_mem(a_copy);            // k3 --> r
            load_b_mem((i & 1) ? b_copy0 : b_copy1);
                                           // k1'--> r
            load_a_slm(a_regs);            // s0(first half) --> r
            inc_store_a_slm((2 + i) % 4);
            if (i == 0) {
                inc_load_zp_scale();
            }
            inc_load_a_slm(2 * i + 0);
            if (i == 0) {
                dec_zp(zp_copy.select_all(), zp_f.select_all());
                scale.select<16, 2>(0) = scale_copy;
                scale.select<16, 2>(1) = scale_copy;
            }
            if (i & 1) {
                dec_wei(b_copy1.row(0).format<int>().select<16, 1>(), b_copy1.row(1).format<int>().select<16, 1>(),
                    b_regs, scale.select_all(), zp_f.select_all());
            } else {
                dec_wei(b_copy0.row(0).format<int>().select<16, 1>(), b_copy0.row(1).format<int>().select<16, 1>(),
                    b_regs, scale.select_all(), zp_f.select_all());
            }
            dot(a_regs, b_regs);
            load_a_slm(a_regs);            // s0(second half) --> r
            inc_load_a_mem();
            inc_load_b_mem();
            inc_load_a_slm(2 * i + 1);
            if (i & 1) {
                dec_wei(b_copy1.row(0).format<int>().select<16, 1>(16), b_copy1.row(1).format<int>().select<16, 1>(16),
                    b_regs, scale.select_all(), zp_f.select_all());
            } else {
                dec_wei(b_copy0.row(0).format<int>().select<16, 1>(16), b_copy0.row(1).format<int>().select<16, 1>(16),
                    b_regs, scale.select_all(), zp_f.select_all());
            }
            if (i == 3) {
                load_zp_scale();
            }
            dot(a_regs, b_regs);
            deinterleave(a_copy, a_cvt);
            cm_sbarrier(0);
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_sbarrier(1);
            store_a_slm(a_cvt);            // r  --> s3
        }
    }
    // tail
    {
        load_a_mem(a_copy);            // k3 --> r
        load_b_mem(b_copy1);           // k1'--> r
        load_a_slm(a_regs);            // s0(first half) --> r
        inc_store_a_slm(2);

        inc_load_a_slm(0);
        dec_zp(zp_copy.select_all(), zp_f.select_all());
        scale.select<16, 2>(0) = scale_copy;
        scale.select<16, 2>(1) = scale_copy;
        dec_wei(b_copy0.row(0).format<int>().select<16, 1>(), b_copy0.row(1).format<int>().select<16, 1>(),
            b_regs, scale.select_all(), zp_f.select_all());

        dot(a_regs, b_regs);
        load_a_slm(a_regs);            // s0(second half) --> r
        inc_load_b_mem();
        inc_load_a_slm(1);
        dec_wei(b_copy0.row(0).format<int>().select<16, 1>(16), b_copy0.row(1).format<int>().select<16, 1>(16),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
        deinterleave(a_copy, a_cvt);
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
        store_a_slm(a_cvt);            // r  --> s3
    }
    {
        load_b_mem(b_copy0);           // k2'--> r
        load_a_slm(a_regs);            // s1(first half) --> r
        inc_load_a_slm(2);
        dec_wei(b_copy1.row(0).format<int>().select<16, 1>(), b_copy1.row(1).format<int>().select<16, 1>(),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
        load_a_slm(a_regs);            // s1(second half) --> r
        inc_load_b_mem();
        inc_load_a_slm(3);
        dec_wei(b_copy1.row(0).format<int>().select<16, 1>(16), b_copy1.row(1).format<int>().select<16, 1>(16),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
        cm_sbarrier(0);
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_sbarrier(1);
    }
    {
        load_b_mem(b_copy1);            // k3' --> r
        load_a_slm(a_regs);             // s2(first half) --> r
        inc_load_a_slm(4);
        dec_wei(b_copy0.row(0).format<int>().select<16, 1>(), b_copy0.row(1).format<int>().select<16, 1>(),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
        load_a_slm(a_regs);            // s2(second half) --> r
        inc_load_a_slm(5);
        dec_wei(b_copy0.row(0).format<int>().select<16, 1>(16), b_copy0.row(1).format<int>().select<16, 1>(16),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
        cm_sbarrier(0);
    }
    {
        load_a_slm(a_regs);            // s3(first half) --> r
        inc_load_a_slm(6);
        dec_wei(b_copy1.row(0).format<int>().select<16, 1>(), b_copy1.row(1).format<int>().select<16, 1>(),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
        load_a_slm(a_regs);            // s3(second half) --> r
        dec_wei(b_copy1.row(0).format<int>().select<16, 1>(16), b_copy1.row(1).format<int>().select<16, 1>(16),
            b_regs, scale.select_all(), zp_f.select_all());
        dot(a_regs, b_regs);
    }

    // store
    uint ldc_bytes = ldc * sizeof(TYPE_DST);
    uint offset_c = id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N +
        (id_wg_m * BLOCK_WG_M +        // workgroup
            id_sg_m * BLOCK_SG_M) * ldc;  // subgroup
    offset_c *= sizeof(TYPE_DST);
    matrix<TYPE_DST, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
        }
    }
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        write_2d(tmp.select<BLOCK_REG_M, 1, REG_N * BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, 0), dst, offset_c, ldc_bytes);
        offset_c += BLOCK_REG_M * ldc_bytes;
    }
}

template<
    typename TYPE_SRC = half,
    typename TYPE_ACC = float,
    typename TYPE_DST = half,
    CmPrecisionType A_TYPE_PREC = CM_PRECISION_HF,
    CmPrecisionType B_TYPE_PREC = CM_PRECISION_HF,
    int SG_SIZE = 16,
    // subgroup blocking
    int BLOCK_SG_M = 64,
    int BLOCK_SG_N = 16,
    int SG_M = 2,
    int SG_N = 16,
    int BLOCK_WG_K = 64
>
CM_INLINE void gemm_xe2_f16(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(TYPE_SRC);
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
    static constexpr uint size_slm_a = BLOCK_WG_M * BLOCK_WG_K * sizeof(TYPE_SRC);

    matrix<TYPE_ACC, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    // N[0:32]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ src_b, N - 1, (uint)(K * sizeof(TYPE_SRC) - 1), (uint)(K * sizeof(TYPE_SRC) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<TYPE_SRC, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{src_b, N - 1, (uint)(K * sizeof(TYPE_SRC) - 1), (uint)(K * sizeof(TYPE_SRC) - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:32]xK[0:16]                                                                   --> 4 regs
    matrix<TYPE_SRC, REG_N, BLOCK_REG_B> b0, b1;

    // M[0:64]xK[0:32]
    lsc::block_2d_desc<TYPE_SRC, 2, 32, BLOCK_REG_K> desc_a0{ src_a, M - 1, (uint)(K * sizeof(TYPE_SRC) - 1), (uint)(K * sizeof(TYPE_SRC) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    // M[32:64]xK[0:32]
    // prefetch A
    lsc::block_2d_desc<TYPE_SRC, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a, M - 1, (uint)(K * sizeof(TYPE_SRC) - 1), (uint)(K * sizeof(TYPE_SRC) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) };
    // 0~3 is M[:]xK[0:16], 4~7 is M[:]xK[16:32]                                        --> 32+32=64 regs
    matrix<TYPE_SRC, 8, BLOCK_REG_A> a0, a1;

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
    //cm_sbarrier(1);

    auto dot = [&](matrix_ref<TYPE_SRC, 4, BLOCK_REG_A> A0, matrix_ref<TYPE_SRC, 4, BLOCK_REG_A> A1, matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)((reg_m + 4) * REG_N + reg_n)) = cm_dpas<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)((reg_m + 4) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A1.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint k = 0; k < K / BLOCK_WG_K; k++) {
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
        desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
        desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

        // load b: N[0:16]xK[16:32]
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1.row(0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
        desc_b0.set_block_x(desc_b0.get_block_x() + 8);

        // load a: M[0:32]xK[0:32]
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<TYPE_SRC>(), desc_a0);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a1.format<TYPE_SRC>(), desc_a0);
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
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<TYPE_SRC>(), desc_a0);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a1.format<TYPE_SRC>(), desc_a0);
        desc_a0.set_block_x(desc_a0.get_block_x() + 32);
        // dot: (M[0:32]xK[32:48])*(N[0:16]xK[32:48])'
        dot(a0.select<4, 1, BLOCK_REG_A, 1>(), a1.select<4, 1, BLOCK_REG_A, 1>(), b0.select_all());

        // load b: N[0:16]xK[0:16]
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
        desc_b0.set_block_x(desc_b0.get_block_x() + 8);
        // dot: (M[0:32]xK[48:64])*(N[0:16]xK[48:64])'
        dot(a0.select<4, 1, BLOCK_REG_A, 1>(4, 0), a1.select<4, 1, BLOCK_REG_A, 1>(4, 0), b1.select_all());
        //cm_sbarrier(0);
        //cm_sbarrier(1);
    }
    
    //cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, BLOCK_SG_N / 2> desc_c{ dst, M - 1, (uint)(N * sizeof(TYPE_SRC) - 1), (uint)(N * sizeof(TYPE_SRC) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<TYPE_DST, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
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

template<
    typename TYPE_SRC = half,
    typename TYPE_ACC = float,
    typename TYPE_DST = half,
    CmPrecisionType A_TYPE_PREC = CM_PRECISION_HF,
    CmPrecisionType B_TYPE_PREC = CM_PRECISION_HF,
    int SG_SIZE = 16,
    // subgroup blocking
    int BLOCK_SG_M = 64,
    int BLOCK_SG_N = 16,
    int SG_M = 2,
    int SG_N = 16,
    int BLOCK_WG_K = 64,
    int GROUP_SIZE = 128
>
CM_INLINE void gemm_a16w4_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, svmptr_t src_b_scale, SurfaceIndex src_b_zp, svmptr_t dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(TYPE_SRC);
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
    static constexpr uint size_slm_a = BLOCK_WG_M * BLOCK_WG_K * sizeof(TYPE_SRC);

    matrix<TYPE_ACC, REG_M* REG_N, BLOCK_DPAS_C> acc = 0;
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(BLOCK_WG_K == GROUP_SIZE, "group size must be equal to wg k size");
    // N[0:32]xK[0:32]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, 4> desc_b0{ src_b, N - 1, (uint)(K / 2 - 1), (uint)(K / 2 - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B(k step: 16)
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<uchar, 1, BLOCK_WG_N / SG_MN, 16> desc_prefetch_b{ src_b, N - 1, (uint)(K / 2 - 1), (uint)(K / 2 - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:32]xK[0:32]
    matrix<int, 16, 4> b0_copy, b1_copy;                                                 // --> 4+4 regs, b mem->regs
    matrix<TYPE_SRC, REG_N, BLOCK_REG_B> b0;                                             // --> 2*8 regs, b regs->decompressed

    // M[0:64]xK[0:16]
    lsc::block_2d_desc<TYPE_SRC, 1, 32, BLOCK_REG_K> desc_a0{ src_a, M - 1, (uint)(K * sizeof(TYPE_SRC) - 1), (uint)(K * sizeof(TYPE_SRC) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    // prefetch A(k step: 32)
    lsc::block_2d_desc<TYPE_SRC, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ src_a, M - 1, (uint)(K * sizeof(TYPE_SRC) - 1), (uint)(K * sizeof(TYPE_SRC) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) };
    // M[:]xK[0:16]                                                                         --> 16+16=32 regs
    matrix<TYPE_SRC, 4, BLOCK_REG_A> a0, a1;

    // warmup
    // prefetch
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 16);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 16);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

    //cm_sbarrier(1);
    // zp in u4, [0:32(=16*2)]
    vector<uchar, 16> zp_copy;
    lsc::block_2d_desc<TYPE_SRC, 1, 1, 32> desc_scale{ src_b_scale, K / GROUP_SIZE - 1, (uint)(N * sizeof(TYPE_SRC) - 1), (uint)(N * sizeof(TYPE_SRC) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N), 0 };
    vector<TYPE_SRC, 32> scale_copy;
    vector<TYPE_SRC, 64> scale;
    uint zp_offset = (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2;
    zp_copy.format<int>() = cm_load<int, 4>(src_b_zp, zp_offset);
    cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(scale_copy.select_all(), desc_scale);

    auto dot = [&](matrix_ref<TYPE_SRC, 4, BLOCK_REG_A> A0, matrix_ref<TYPE_SRC, 4, BLOCK_REG_A> A1, matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 4; reg_m++) {
                acc.row((ushort)((reg_m + 4) * REG_N + reg_n)) = cm_dpas<B_TYPE_PREC, A_TYPE_PREC, DEPTH, REPEAT>(acc.row((ushort)((reg_m + 4) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A1.row((ushort)reg_m).format<int>());
            }
        }
    };
    auto dec_wei = [&](vector_ref<int, 32> b_src0, vector_ref<int, 32> b_src1, matrix_ref<TYPE_SRC, REG_N, BLOCK_REG_B> b_dst, vector_ref<TYPE_SRC, 64> s, vector_ref<TYPE_SRC, 64> z) {
#pragma unroll
        for (uint i = 0; i < REG_N; i++) {
            auto b_reg = b_dst.row(i).format<TYPE_SRC, BLOCK_REG_K / 2, BLOCK_REG_N * 2>();
#pragma unroll
            for (uint k = 0; k < BLOCK_REG_K / 2; k++) {
                auto b_row = b_reg.row(k);
                vector<TYPE_SRC, 32> tmp_f;
                if (i == 0) {
                    auto tmp = b_src0.select<16, 1>(k / 4 * 16).format<uchar>().select<16, 4>(k % 4);
                    tmp_f.select<16, 2>(0).format<ushort>() = tmp & (ushort)0xf;
                    tmp_f.select<16, 2>(1).format<ushort>() = tmp >> 4;
                    //tmp_f.select<16, 2>(1).format<ushort>() = tmp;
                    //tmp_f.select<16, 2>(1).format<ushort>() /= 16;
                } else {
                    auto tmp = b_src1.select<16, 1>(k / 4 * 16).format<uchar>().select<16, 4>(k % 4);
                    tmp_f.select<16, 2>(0).format<ushort>() = tmp & (ushort)0xf;
                    tmp_f.select<16, 2>(1).format<ushort>() = tmp >> 4;
                    //tmp_f.select<16, 2>(1).format<ushort>() = tmp;
                    //tmp_f.select<16, 2>(1).format<ushort>() /= 16;
                }
                tmp_f = (tmp_f * half{32768.0} - z.select<32, 1>(i * 32));
                tmp_f *= half{512.0};
                b_row = tmp_f * s.select<32, 1>(i * 32);
            }
        }
    };
    auto dec_zp = [&] (vector_ref<uchar, 16> z_src, vector_ref<TYPE_SRC, 64> z_f) {
        vector<ushort, 16> tmp0 = (z_src & (ushort)0xf);
        tmp0.format<TYPE_SRC>() *= half{4096 * (4096 / 512)};
        //tmp0.format<TYPE_SRC>() *= half{4096 / 512};
        z_f.select<16, 4>(0) = tmp0.format<TYPE_SRC>();
        vector<ushort, 16> tmp1 = z_src >> (ushort)4;
        tmp1.format<TYPE_SRC>() *= half{4096 * (4096 / 512)};
        //tmp1.format<TYPE_SRC>() *= half{4096 / 512};
        z_f.select<16, 4>(2) = tmp1.format<TYPE_SRC>();
        z_f.select<16, 4>(1) = z_f.select<16, 4>(0);
        z_f.select<16, 4>(3) = z_f.select<16, 4>(2);
    };

    vector<TYPE_SRC, 64> zp_f;
    for (uint k = 0; k < K / BLOCK_WG_K; k++) {
        dec_zp(zp_copy.select_all(), zp_f.select_all());
        scale.select<32, 2>(0) = scale_copy;
        scale.select<32, 2>(1) = scale_copy;
#pragma unroll
        for (uint ks = 0; ks < BLOCK_WG_K / 32; ks++) {
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 16);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load b: N[0:16]xK[0:32]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_copy.format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1_copy.format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 4);

            // load a: M[0:32]xK[0:16]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<TYPE_SRC>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a1.format<TYPE_SRC>(), desc_a0);
            desc_a0.set_block_x(desc_a0.get_block_x() + 16);
            // decompress b: N[0:16]xK[0:16]
            dec_wei(b0_copy.format<int>().select<32, 1>(), b1_copy.format<int>().select<32, 1>(), b0, scale.select_all(), zp_f.select_all());

            // dot: (M[0:32]xK[0:16])*(N[0:16]xK[0:16])'
            dot(a0.select_all(), a1.select_all(), b0.select_all());

            // load a: M[0:32]xK[16:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<TYPE_SRC>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a1.format<TYPE_SRC>(), desc_a0);
            desc_a0.set_block_x(desc_a0.get_block_x() + 16);
            // decompress b: N[0:16]xK[0:16]
            dec_wei(b0_copy.format<int>().select<32, 1>(32), b1_copy.format<int>().select<32, 1>(32), b0, scale.select_all(), zp_f.select_all());

            // dot: (M[0:32]xK[0:16])*(N[0:16]xK[0:16])'
            dot(a0.select_all(), a1.select_all(), b0.select_all());
        }
        //cm_sbarrier(0);
        //cm_sbarrier(1);
        desc_scale.set_block_y(desc_scale.get_block_y() + 1);
        zp_offset += N / 2;
        zp_copy.format<int>() = cm_load<int, 4>(src_b_zp, zp_offset);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(scale_copy.select_all(), desc_scale);
    }

    //cm_sbarrier(0);

    // store
    lsc::block_2d_desc<int, 1, 8, BLOCK_SG_N / 2> desc_c{ dst, M - 1, (uint)(N * sizeof(TYPE_SRC) - 1), (uint)(N * sizeof(TYPE_SRC) - 1),
        (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / 2, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    matrix<TYPE_DST, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
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

template<
    typename TYPE_SRC = uchar,
    typename TYPE_ACC = int,
    typename TYPE_DST = half,
    CmPrecisionType A_TYPE_PREC = CM_PRECISION_S8,
    CmPrecisionType B_TYPE_PREC = CM_PRECISION_S4,
    int SG_SIZE = 16,
    // subgroup blocking
    int BLOCK_SG_M = 16,
    int BLOCK_SG_N = 64,
    int SG_M = 8,
    int SG_N = 2,
    int BLOCK_WG_K = 128,
    int GROUP_SIZE = 128
>
CM_INLINE void gemm_a8w4sym_xe2(uint id_wg_m, uint id_wg_n, uint slm, svmptr_t src_a, svmptr_t src_b, SurfaceIndex src_a_scale, SurfaceIndex src_b_scale, SurfaceIndex dst, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(uchar);
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

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 4, "block_2d_desc for b is manually unrolled by 4");
    static_assert(BLOCK_WG_K == GROUP_SIZE, "group size must be equal to wg k size");
    // scale, N[0:64]
    uint scale_offset = (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * (int)sizeof(half);
    vector<half, 64> scale_copy;
    vector<float, 64> scale;
    scale_copy.format<int>() = cm_load<int, 32>(src_b_scale, scale_offset);

    // prefetch B
    lsc::block_2d_desc<uchar, 1, 8, 64> desc_prefetch_b{ src_b, N - 1, (uint)(K / 2 - 1), (uint)(K / 2 - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (128 / 16)) };
    // prefetch A
    lsc::block_2d_desc<uchar, 1, 8, 64> desc_prefetch_a{ src_a, M - 1, (uint)(K * sizeof(uchar) - 1), (uint)(K * sizeof(uchar) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (128 / 16)) };
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached, 64>(desc_prefetch_a);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 64);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 128);

    // N[0:16]xK[0:64]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, 8> desc_b0{ src_b, N - 1, (uint)(K / 2 - 1), (uint)(K / 2 - 1),
        0, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // (4*N[0:16])xK[0:64]; for one dpas: 16(N) * 32(K) / 2(1char=2*int4) / 4(=sizeof(int)
    matrix<int, 4 * 2, 16 * 32 / 2 / 4> b;                                                           // --> 8 * 4 regs, b mem->regs
    // M[0:16]xK[0:64]
    lsc::block_2d_desc<uchar, 2, 2 * 8, BLOCK_REG_K> desc_a0{ src_a, M - 1, (uint)(K * sizeof(uchar) - 1), (uint)(K * sizeof(uchar) - 1),
        0, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    // M[0:16]x(K[0:64]*2)
    matrix<uchar, 2 * 2, 8 * 32> a0, a1;                                                             // --> 16+16 regs, a mem->regs

    auto dot = [&] (matrix_ref<uchar, 2, 8 * 32> A, matrix_ref<int, 4, 64> B, matrix_ref<int, 8, 8 * 16> C, bool is_first=false) {
#pragma unroll
        for (int n = 0; n < 4; n++) {
#pragma unroll
            for (uint m = 0; m < 2; m++) {
                if (is_first) {
                    C.row(m * 4 + n) = cm_dpas<CM_PRECISION_S4, CM_PRECISION_S8, 8, 8, int, int, int, 128>(0, B.row(n), A.row(m).format<int>());
                } else {
                    C.row(m * 4 + n) = cm_dpas<CM_PRECISION_S4, CM_PRECISION_S8, 8, 8>(C.row(m * 4 + n), B.row(n), A.row(m).format<int>());
                }
            }
        }
    };
    auto dequant = [&](matrix_ref<int, 8, 8 * 16> C, vector_ref<float, 64> w_scale, matrix_ref<float, 8, 8 * 16> acc) {
#pragma unroll
        for (int n = 0; n < 4; n++) {
#pragma unroll
            for (uint m = 0; m < 2; m++) {
                auto c_tile = C.row(m * 4 + n).format<int, 8, 16>();
                auto acc_tile = acc.row(m * 4 + n).format<float, 8, 16>();
                matrix<float, 8, 16> c_f = c_tile;
#pragma unroll
                for (uint i = 0; i < 8; i++) {
                    acc_tile.row(i) += c_f.row(i) * w_scale.select<16, 1>(n * 16);
                }
            }
        }
    };

    matrix<int, 8, 8 * 16> C;
    for (uint k = 0; k < K / BLOCK_WG_K; k++) {
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
        cm_prefetch<CacheHint::Cached, CacheHint::Cached, 64>(desc_prefetch_a);

        // load b: N[0:16]xK[0:64]
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b.select<2, 1, 64, 1>(0, 0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b.select<2, 1, 64, 1>(2, 0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b.select<2, 1, 64, 1>(4, 0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b.select<2, 1, 64, 1>(6, 0).format<int>(), desc_b0);

        // load a: M[0:16]xK[0:32]
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 0>(a0.format<uchar>(), desc_a0);
        cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 64,0>(a1.format<uchar>(), desc_a0);
        desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 64);
        desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 128);
        scale_offset += N * (int)sizeof(half);
        desc_b0.set_block_x(desc_b0.get_block_x() + 8);
        desc_a0.set_block_x(desc_a0.get_block_x() + 128);
        scale = scale_copy;
        scale_copy.format<int>() = cm_load<int, 32>(src_b_scale, scale_offset);

        dot(a0.select<2, 1, 8 * 32, 1>(0), b.select<4, 2, 64, 1>(0), C, true);
        dot(a0.select<2, 1, 8 * 32, 1>(2), b.select<4, 2, 64, 1>(1), C);

        // load b: N[0:16]xK[64:128]
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b.select<2, 1, 64, 1>(0, 0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b.select<2, 1, 64, 1>(2, 0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 32>(b.select<2, 1, 64, 1>(4, 0).format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 48>(b.select<2, 1, 64, 1>(6, 0).format<int>(), desc_b0);
        desc_b0.set_block_x(desc_b0.get_block_x() + 8);

        dot(a1.select<2, 1, 8 * 32, 1>(0), b.select<4, 2, 64, 1>(0), C);
        dot(a1.select<2, 1, 8 * 32, 1>(2), b.select<4, 2, 64, 1>(1), C);
        dequant(C, scale, acc);
    }

    //cm_sbarrier(0);
    vector<half, 16> scale_a;
    scale_offset = (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (int)sizeof(half);
    scale_a.format<int>() = cm_load<int, 8>(src_a_scale, scale_offset);
    // store
    matrix<TYPE_DST, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
        }
    }
#pragma unroll
    for (uint i = 0; i < 16; i++) {
        tmp.row(i) *= scale_a[i];
    }
    uint offset = (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (int)sizeof(half) * N + (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * (int)sizeof(half);
#pragma unroll
    for (uint i = 0; i < 16; i++) {
        cm_store<int, 16, DataSize::Default, CacheHint::Uncached, CacheHint::WriteBack>(dst, offset +  0, tmp.row(i).select<32, 1>(0).format<int>());
        cm_store<int, 16, DataSize::Default, CacheHint::Uncached, CacheHint::WriteBack>(dst, offset + 64, tmp.row(i).select<32, 1>(32).format<int>());
        offset += N * (int)sizeof(half);
    }
}
