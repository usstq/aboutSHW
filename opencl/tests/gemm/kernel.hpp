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
#else
#define ATTR [[type("svmptr_t")]]
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
    cm_svm_block_read(base, out);
}

template <typename T, int M, int N>
CM_INLINE void read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_svm_block_read(base, out.row(i));
    }
}

template <typename TSRC, int M, int N>
CM_INLINE void write_2d(matrix_ref<TSRC, M, N> out, svmptr_t base, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_svm_block_write(base, out.row(i));
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
    // compute K dimension inside one EU
    for (uint k = 0; k < K; k += BLOCK_WG_K) {
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
            const int BLOCK_REG_M_LOAD_A = BLOCK_WG_M / (SG_M * SG_N);
            matrix<TYPE_SRC, BLOCK_REG_M_LOAD_A, BLOCK_WG_K> data;
            uint offset_sg = (id_wg_m * BLOCK_WG_M +                   // workgroup
                              id_sg_mn * BLOCK_REG_M_LOAD_A) * lda;    // subgroup
            offset_sg += k;
            read_2d(data.select_all(), (svmptr_t)((TYPE_SRC*)src_a + offset_sg), lda * sizeof(TYPE_SRC));
#pragma unroll
            for (uint m = 0; m < BLOCK_REG_M_LOAD_A; m++) {
                auto m_in_sg = id_sg_mn * BLOCK_REG_M_LOAD_A + m;
                auto sg_m = m_in_sg / BLOCK_SG_M;
                auto reg_m = m_in_sg % BLOCK_SG_M / BLOCK_REG_M;
                auto m_in_reg = m_in_sg % BLOCK_SG_M % BLOCK_REG_M;
#pragma unroll
                for (uint block_reg_k = 0; block_reg_k < BLOCK_WG_K; block_reg_k += BLOCK_REG_K) {
                    auto tmp = data.select<1, 1, BLOCK_REG_K, 1>(m, block_reg_k);
                    auto reg_k = block_reg_k / BLOCK_REG_K;
                    uint offset = sg_m * BLOCK_WG_K / BLOCK_REG_K * REG_M * BLOCK_REG_A +                       // block[x]
                                  reg_k * REG_M * BLOCK_REG_A +                                                 // regs[y]
                                  reg_m * BLOCK_REG_A +                                                         // regs[y][z]
                                  m_in_reg * BLOCK_REG_K;                                                       // data[m]
                    offset *= sizeof(TYPE_SRC);
                    cm_slm_block_write(slm, offset, tmp.format<int>());
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
            const int BLOCK_REG_M_LOAD_B = BLOCK_WG_K * VNNI * BLOCK_WG_N / (SG_M * SG_N) / sizeof(int);
            vector<int, BLOCK_REG_M_LOAD_B> data;
            uint offset = (id_wg_n * K / BLOCK_WG_K + k / BLOCK_WG_K) * BLOCK_WG_K * VNNI * BLOCK_WG_N / sizeof(int) + // workgroup
                           id_sg_mn * BLOCK_REG_M_LOAD_B;    // subgroup
            read_1d(data.select_all(), (svmptr_t)((int*)src_b + offset));
            offset = slm_b_offset + id_sg_mn * BLOCK_REG_M_LOAD_B * sizeof(int);
            cm_slm_block_write(slm, offset, data);
        }
        cm_barrier();

        matrix<TYPE_SRC, REG_M, BLOCK_REG_A> A;
        matrix<TYPE_SRC, REG_N, BLOCK_REG_B> B;
#pragma unroll
        for (uint reg_k = 0; reg_k < BLOCK_WG_K / BLOCK_REG_K; reg_k++) {
            for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
                int offset = slm_offset_wg_a + (reg_k * REG_M * BLOCK_REG_A + reg_m * BLOCK_REG_A) * sizeof(TYPE_SRC);
                cm_slm_block_read(slm, offset, A.row(reg_m));
            }
#pragma unroll
            for (uint reg_n = 0; reg_n < REG_N; reg_n++) {
                int offset = slm_offset_wg_b + (reg_k * REG_N * BLOCK_REG_B + reg_n * BLOCK_REG_B) * sizeof(TYPE_SRC);
                cm_slm_block_read(slm, offset, B.row(reg_n));
            }
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
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
        matrix<TYPE_DST, BLOCK_REG_M, REG_N * BLOCK_REG_N> tmp;
        // TODO(TUNE): merge in N dimension
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            tmp.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(0, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n);
        }
        uint offset = (id_wg_m * BLOCK_WG_M +        // workgroup
                       id_sg_m * BLOCK_SG_M +        // subgroup
                       reg_m   * BLOCK_REG_M) * ldc; // reg-blocks
        offset += id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N;
        write_2d(tmp.select_all(), (svmptr_t)((TYPE_DST*)dst + offset), ldc * sizeof(TYPE_DST));
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
void repack_f16(int K, int N, half* src, half* dst) {
    const int DEPTH = 8;
    const int BLOCK_REG_M = 8;
    const int BLOCK_REG_N = SG_SIZE;
    // half is 2, u8 is 4
    const int VNNI = 2;
    const int BLOCK_REG_K = 32 / VNNI;
    const int BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const int BLOCK_WG_N = BLOCK_SG_N * SG_N;
    // register blocking
    const int REG_N = BLOCK_SG_M / BLOCK_REG_M;
    const int REG_M = BLOCK_SG_N / BLOCK_REG_N;
    // packed memory layout:
    struct reg {                                        // --> one DPAS op
        half data[BLOCK_REG_K / 2][BLOCK_REG_N * 2];        // BLOCK_REG_K=16, BLOCK_REG_N=8, aka half[8,16]
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