

#include "mxformat.hpp"

#define JIT_DEBUG
#include "../include/jit.h"

// convert MXFP4 weight (K/32, N, mxfp4) into (K, N, float)
// weight
static tensor2D<float> mxfp4_decompress(const tensor2D<mxformat::mxfp4>& weight) {
    tensor2D<float> ret(weight.shape[0]*32, weight.shape[1]);
    for (int bk = 0; bk < weight.shape[0]; bk ++) {
        for (int n = 0; n < weight.shape[1]; n++) {
            for (int ki = 0; ki < 32; ki++) {
                ret(bk*32 + ki, n) = weight(bk, n)[ki];
            }
        }
    }
    return ret;
}

// A: (BM, BK)
// weight: (BK/32, BN) in unit of mxfp4, each mxfp4 has 32 elements within same output channel
// C: (BM, BN)
template <typename AT>
void exec_reference(tensor2D<AT>& src, tensor2D<mxformat::mxfp4>& mxfp4_weight, tensor2D<float>& dst, bool verbose = false) {
    // decompress mxfp4 first
    int M = src.shape[0];
    int K = src.shape[1];
    ASSERT(K == mxfp4_weight.shape[0]*32);
    int N = mxfp4_weight.shape[1];
    ASSERT(M == dst.shape[0]);
    ASSERT(N == dst.shape[1]);

    // decompress weight into fp32 scratch buffer
    tensor2D<float> weight = mxfp4_decompress(mxfp4_weight);

    // do fp32 matmul
    for (int m = 0; m < M; m++) {
        const auto* A = src.ptr(m, 0);
        float* C = dst.ptr(m, 0);

        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[k] * weight(k, n);
            }
            C[n] = sum;
        }
    }

    if (verbose) {
        show_tensor2D("exec_reference_src", src);
        show_tensor2D("exec_reference_weight", weight);
        show_tensor2D("exec_reference_dst", dst);
    }
}