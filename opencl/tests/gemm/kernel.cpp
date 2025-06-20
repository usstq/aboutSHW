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

#ifdef SHIM
#include "shim_support.h"
#else
#define SHIM_API_EXPORT
#endif
extern "C" SHIM_API_EXPORT void gemm(svmptr_t, svmptr_t, svmptr_t, uint, uint, uint, uint, uint, uint);

#include "common.hpp"
#include "kernel.hpp"
#define ABS(x) (x) < 0 ? -(x) : (x)

CM_INLINE void get_mn(uint& id_wg_m, uint& id_wg_n, uint M, uint N, int slice_no, int slice, const int BLOCK_WG_M, const int BLOCK_WG_N) {
    uint id_wg_mn = cm_group_id(0);
    if (slice_no == 0) {
        if (slice == 0) {
            // loop M first, N is shared, total = N/256*M+N
            uint WG_MN = M / BLOCK_WG_M;
            id_wg_m = id_wg_mn % WG_MN;
            id_wg_n = id_wg_mn / WG_MN;
        } else {
            // loop N first, M is shared, total = M/128*N+M
            uint WG_MN = N / BLOCK_WG_N;
            id_wg_n = id_wg_mn % WG_MN;
            id_wg_m = id_wg_mn / WG_MN;
        }
    } else {
        uint wg_x = slice > 0 ? N / BLOCK_WG_N : M / BLOCK_WG_M;
        uint slice_no_abs = ABS(slice_no);
        uint slice_abs = ABS(slice);
        int id_wg_mn_in_reminder = (int)id_wg_mn - (int)(slice_no_abs * slice_abs * wg_x);
        uint slice_idx;
        // in [slice_no x slice]
        if (id_wg_mn_in_reminder < 0) {
            slice_idx = id_wg_mn / (slice_abs * wg_x);
            uint rem_in_slice = id_wg_mn % (slice_abs * wg_x);
            uint x = rem_in_slice % slice_abs;
            uint y = rem_in_slice / slice_abs;
            id_wg_m = slice > 0 ? x + slice_idx * slice_abs : y;
            id_wg_n = slice < 0 ? x + slice_idx * slice_abs : y;
        } else {
            uint slice_rem = slice_abs + (slice_no > 0 ? 1 : -1);
            slice_idx = id_wg_mn_in_reminder / (slice_rem * wg_x);
            uint rem_in_slice = id_wg_mn_in_reminder % (slice_rem * wg_x);
            uint x = rem_in_slice % slice_rem;
            uint y = rem_in_slice / slice_rem;
            id_wg_m = slice > 0 ? x + slice_idx * slice_rem + slice_no_abs * slice_abs : y;
            id_wg_n = slice < 0 ? x + slice_idx * slice_rem + slice_no_abs * slice_abs : y;
        }
    }
}

#if (CM_GENX >= 1270 && CM_GENX < 1280)

_GENX_MAIN_ void gemm(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    const uint BLOCK_WG_M = x::BLOCK_SG_M * x::SG_M;
    const uint BLOCK_WG_N = x::BLOCK_SG_N * x::SG_N;
    const uint size_slm_a = BLOCK_WG_M * x::BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = x::BLOCK_WG_K * BLOCK_WG_N * sizeof(half);
    cm_slm_init(size_slm_a + size_slm_b);
    auto slm = cm_slm_alloc(size_slm_a + size_slm_b);

    gemm_xmx<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        x::SG_SIZE, x::BLOCK_SG_M, x::BLOCK_SG_N, x::SG_M, x::SG_N>(slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}

_GENX_MAIN_ void repack_f16(half* src ATTR, half* dst ATTR, int K, int N) {
    repack_f16<x::SG_SIZE, x::BLOCK_SG_M, x::BLOCK_SG_N, x::SG_M, x::SG_N, x::BLOCK_WG_K>(src, dst, K, N);
}

// if slice_no == 0, slice is linear type, 0 for loop M first and 1 for loop N first
// if slice_no != 0, use wg blocking schema:
//    slice > 0 means the slice is in M dimension else is in N dimension
//    slice_no > 0 means the slice size of reminder will be slice+1 else be slice-1
_GENX_MAIN_ void gemm_nocopy_xe1(SurfaceIndex src_a ATTR_BUF, SurfaceIndex src_b ATTR_BUF, SurfaceIndex dst ATTR_BUF, uint M, uint N, uint K, uint lda, uint ldb, uint ldc,
    int slice_no, int slice) {
    const uint BLOCK_WG_M = v2::BLOCK_SG_M * v2::SG_M;
    const uint BLOCK_WG_N = v2::BLOCK_SG_N * v2::SG_N;
    const uint size_slm_a = BLOCK_WG_M * v2::BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = 0;
    cm_slm_init(size_slm_a + size_slm_b);
    auto slm = cm_slm_alloc(size_slm_a + size_slm_b);

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    gemm_xe1_f16<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        v2::SG_SIZE, v2::BLOCK_SG_M, v2::BLOCK_SG_N, v2::SG_M, v2::SG_N>(id_wg_m, id_wg_n, slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}
#endif

#if CM_GENX >= 1280

_GENX_MAIN_ void gemm_nocopy_xe2(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc,
    int slice_no, int slice) {
    const uint BLOCK_WG_M = v3::BLOCK_SG_M * v3::SG_M;
    const uint BLOCK_WG_N = v3::BLOCK_SG_N * v3::SG_N;
    const uint size_slm_a = BLOCK_WG_M * v3::BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = 0;
    auto slm = 0;

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    gemm_xe2_f16<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        v3::SG_SIZE, v3::BLOCK_SG_M, v3::BLOCK_SG_N, v3::SG_M, v3::SG_N>(id_wg_m, id_wg_n, slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}

_GENX_MAIN_ void gemm_a16w4_xe2(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t src_b_scale ATTR, SurfaceIndex src_b_zp ATTR_BUF, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc,
    int slice_no, int slice) {
    const uint BLOCK_WG_M = v4::BLOCK_SG_M * v4::SG_M;
    const uint BLOCK_WG_N = v4::BLOCK_SG_N * v4::SG_N;
    const uint size_slm_a = BLOCK_WG_M * v4::BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = 0;
    auto slm = 0;

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    gemm_a16w4_xe2<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        v4::SG_SIZE, v4::BLOCK_SG_M, v4::BLOCK_SG_N, v4::SG_M, v4::SG_N, v4::GROUP_SIZE>(id_wg_m, id_wg_n, slm, src_a, src_b, src_b_scale, src_b_zp, dst, M, N, K, lda, ldb, ldc);
}
#endif