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

// shim layer (CM kernel, OpenCL runtime, GPU)
#ifdef SHIM
EXPORT_SIGNATURE(gemm);
#endif

#include "common.hpp"
#include "kernel.hpp"

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

_GENX_MAIN_ void gemm_nocopy(SurfaceIndex src_a ATTR_BUF, SurfaceIndex src_b ATTR_BUF, SurfaceIndex dst ATTR_BUF, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    const uint BLOCK_WG_M = v2::BLOCK_SG_M * v2::SG_M;
    const uint BLOCK_WG_N = v2::BLOCK_SG_N * v2::SG_N;
    const uint size_slm_a = 0;
    const uint size_slm_b = v2::BLOCK_WG_K * BLOCK_WG_N * sizeof(half);
    cm_slm_init(size_slm_a + size_slm_b);
    auto slm = cm_slm_alloc(size_slm_a + size_slm_b);

    gemm_64x16_no_slmb<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        v2::SG_SIZE, v2::BLOCK_SG_M, v2::BLOCK_SG_N, v2::SG_M, v2::SG_N>(slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}