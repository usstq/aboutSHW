import numpy as np
import pycpp

@pycpp.clib("-march=core-avx2 -g")
def mylib():
    return r'''
#include "common.hpp"

template <int row>
void brgemm_4x3(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    SIMD_F32 c00, c01, c02;
    SIMD_F32 c10, c11, c12;
    SIMD_F32 c20, c21, c22;
    SIMD_F32 c30, c31, c32;

    if (is_accumulate_C) {
        c00 = simd_loadu_ps(C + SIMDW * 0);
        c01 = simd_loadu_ps(C + SIMDW * 1);
        c02 = simd_loadu_ps(C + SIMDW * 2);

        if (row > 1) {
            c10 = simd_loadu_ps(C + C_stride + SIMDW * 0);
            c11 = simd_loadu_ps(C + C_stride + SIMDW * 1);
            c12 = simd_loadu_ps(C + C_stride + SIMDW * 2);
        }
        if (row > 2) {
            c20 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 0);
            c21 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 1);
            c22 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 2);
        }
        if (row > 3) {
            c30 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 0);
            c31 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 1);
            c32 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 2);
        }
    } else {
        c00 = simd_setzero_ps();
        c01 = simd_setzero_ps();
        c02 = simd_setzero_ps();

        if (row > 1) {
            c10 = simd_setzero_ps();
            c11 = simd_setzero_ps();
            c12 = simd_setzero_ps();
        }
        if (row > 2) {
            c20 = simd_setzero_ps();
            c21 = simd_setzero_ps();
            c22 = simd_setzero_ps();
        }
        if (row > 3) {
            c30 = simd_setzero_ps();
            c31 = simd_setzero_ps();
            c32 = simd_setzero_ps();
        }
    }

    auto* prefetch_B = B + 64 / sizeof(float) * 10;

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++, prefetch_B += B_stride) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        auto b0 = simd_loadu_ps(B + SIMDW * 0);
        auto b1 = simd_loadu_ps(B + SIMDW * 1);
        auto b2 = simd_loadu_ps(B + SIMDW * 2);

        //_mm_prefetch(prefetch_B, _MM_HINT_T0);

        auto a = simd_broadcast_ss(A);
        c00 = simd_fmadd_ps(a, b0, c00);
        c01 = simd_fmadd_ps(a, b1, c01);
        c02 = simd_fmadd_ps(a, b2, c02);

        if (row > 1) {
            a = simd_broadcast_ss(A + A_stride);
            c10 = simd_fmadd_ps(a, b0, c10);
            c11 = simd_fmadd_ps(a, b1, c11);
            c12 = simd_fmadd_ps(a, b2, c12);
        }
        if (row > 2) {
            a = simd_broadcast_ss(A + 2 * A_stride);
            c20 = simd_fmadd_ps(a, b0, c20);
            c21 = simd_fmadd_ps(a, b1, c21);
            c22 = simd_fmadd_ps(a, b2, c22);
        }
        if (row > 3) {
            a = simd_broadcast_ss(A + 3 * A_stride);
            c30 = simd_fmadd_ps(a, b0, c30);
            c31 = simd_fmadd_ps(a, b1, c31);
            c32 = simd_fmadd_ps(a, b2, c32);
        }
    }

    // store C back
    simd_storeu_ps(C + SIMDW * 0, c00);
    simd_storeu_ps(C + SIMDW * 1, c01);
    simd_storeu_ps(C + SIMDW * 2, c02);
    if (row > 1) {
        simd_storeu_ps(C + C_stride + SIMDW * 0, c10);
        simd_storeu_ps(C + C_stride + SIMDW * 1, c11);
        simd_storeu_ps(C + C_stride + SIMDW * 2, c12);
    }
    if (row > 2) {
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 0, c20);
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 1, c21);
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 2, c22);
    }
    if (row > 3) {
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 0, c30);
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 1, c31);
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 2, c32);
    }
}

template <int row>
void brgemm_4x1(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    SIMD_F32 c00;
    SIMD_F32 c10;
    SIMD_F32 c20;
    SIMD_F32 c30;

    if (is_accumulate_C) {
        c00 = simd_loadu_ps(C + 8 * 0);
        if (row > 1) {
            c10 = simd_loadu_ps(C + C_stride + SIMDW * 0);
        }
        if (row > 2) {
            c20 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 0);
        }
        if (row > 3) {
            c30 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 0);
        }
    } else {
        c00 = simd_setzero_ps();
        if (row > 1) {
            c10 = simd_setzero_ps();
        }
        if (row > 2) {
            c20 = simd_setzero_ps();
        }
        if (row > 3) {
            c30 = simd_setzero_ps();
        }
    }

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        auto b0 = simd_loadu_ps(B + 8 * 0);
        auto a = simd_broadcast_ss(A);
        c00 = simd_fmadd_ps(a, b0, c00);
        if (row > 1) {
            a = simd_broadcast_ss(A + A_stride);
            c10 = simd_fmadd_ps(a, b0, c10);
        }
        if (row > 2) {
            a = simd_broadcast_ss(A + 2 * A_stride);
            c20 = simd_fmadd_ps(a, b0, c20);
        }
        if (row > 3) {
            a = simd_broadcast_ss(A + 3 * A_stride);
            c30 = simd_fmadd_ps(a, b0, c30);
        }
    }
    simd_storeu_ps(C + SIMDW * 0, c00);
    if (row > 1) {
        simd_storeu_ps(C + C_stride + SIMDW * 0, c10);
    }
    if (row > 2) {
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 0, c20);
    }
    if (row > 3) {
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 0, c30);
    }
}

template <class T>
static T* scratch_alloc(size_t cnt) {
    thread_local uint8_t scratch[1024 * 1024 * 2] __attribute__((aligned(4096)));
    // assert(cnt * sizeof(T) < sizeof(scratch));
    // DEBUG_LOG(reinterpret_cast<void*>(scratch));
    return reinterpret_cast<T*>(scratch);
}

void repack_weight_for_4x3(const uint8_t* W, int strideW, const float* scales, const float* zp, int K, int N, float* repacked_B_nx3, float* repacked_B_nx1) {
    //assert((N % 8) == 0);
#if 1
    for (int k = 0; k < K; k++) {
        int n0 = 0;
        auto* src = W + k*strideW;
        auto* dst = repacked_B_nx3 + k*SIMDW*3;
        auto dst_stride = K*SIMDW*3;
        for (n0 = 0; n0 + SIMDW*3 <= N; n0 += SIMDW*3, dst += dst_stride) {
            
            auto wi0 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
            auto wi1 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 1));
            auto wi2 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 2));

            auto zp0 = simd_loadu_ps(zp + n0 + SIMDW * 0);
            auto zp1 = simd_loadu_ps(zp + n0 + SIMDW * 1);
            auto zp2 = simd_loadu_ps(zp + n0 + SIMDW * 2);
            
            auto wf0 = simd_sub_ps(simd_cvtepi32_ps(wi0), (zp0));
            auto wf1 = simd_sub_ps(simd_cvtepi32_ps(wi1), (zp1));
            auto wf2 = simd_sub_ps(simd_cvtepi32_ps(wi2), (zp2));

            wf0 = simd_mul_ps(wf0, simd_loadu_ps(scales + n0 + SIMDW*0));
            wf1 = simd_mul_ps(wf1, simd_loadu_ps(scales + n0 + SIMDW*1));
            wf2 = simd_mul_ps(wf2, simd_loadu_ps(scales + n0 + SIMDW*2));
            
            simd_storeu_ps(dst + SIMDW*0, wf0);
            simd_storeu_ps(dst + SIMDW*1, wf1);
            simd_storeu_ps(dst + SIMDW*2, wf2);
        }

        dst = repacked_B_nx1 + k*SIMDW;
        dst_stride = K*SIMDW;
        for (; n0 < N; n0 += SIMDW, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW; n++) {
                auto wi0 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                auto zp0 = simd_loadu_ps(zp + n0 + SIMDW * 0);
                auto wf0 = simd_sub_ps(simd_cvtepi32_ps(wi0), (zp0));
                wf0 = simd_mul_ps(wf0, simd_loadu_ps(scales + n0 + SIMDW*0));
                simd_storeu_ps(dst + SIMDW*0, wf0);
            }
        }
    }
#else
    for (int k = 0; k < K; k++) {
        int n0 = 0;
        auto* src = W + k*strideW;
        auto* dst = repacked_B_nx3 + k*SIMDW*3;
        auto dst_stride = K*SIMDW*3;
        for (n0 = 0; n0 + SIMDW*3 <= N; n0 += SIMDW*3, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW*3; n++) {
                dst[n-n0] = (src[n] - zp[n]) * scales[n];
                //printf("%d,%d,%d  %d, %f, %f, =>  %f\n", k, n0, n, src[n], zp[n], scales[n], dst[n-n0]);
            }
        }
        dst = repacked_B_nx1 + k*SIMDW;
        dst_stride = K*SIMDW;
        for (; n0 < N; n0 += SIMDW, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW; n++) {
                dst[n-n0] = (src[n] - zp[n]) * scales[n];
            }
        }
    }
#endif
}

void MM_ComputeBounded_reuseB_i8(const float * A,
                                 float * C,
                                 const uint8_t* W,
                                 const uint8_t* zp,
                                 const float* scales,
                                 int M, int IC, int OC,
                                 int n0, int n1) {
    constexpr int BK = 512;
    constexpr int BN = 512;
    auto bN_SIMDWx3 = BN / (SIMDW*3) * (SIMDW*3);
    auto bN_SIMDWx1 = BN - bN_SIMDWx3;
    float* scratch = scratch_alloc<float>(BN * BK + BN);
    float* repacked_B_n24 = scratch;
    float* repacked_B_n8 = repacked_B_n24 + bN_SIMDWx3 * BK;
    float* zero_points = repacked_B_n8 + bN_SIMDWx1 * BK;

    const auto A_stride = IC;
    const auto B_stride = OC;
    const auto C_stride = OC;

    auto M_tails = M % 4;
    auto M_body = M - M_tails;
    
    for (int cur_n = n0; cur_n < n1; cur_n += BN) {
        int bN = std::min(n1 - cur_n, BN);
        const auto* pW = W + cur_n;

        // decompress zero-point
        {
            for (int n = 0; n < bN; n += SIMDW) {
                auto zp0 = simd_load_epu8_epi32(static_cast<void const*>(zp + cur_n + n));
                auto zpf32 = simd_cvtepi32_ps(zp0);
                simd_storeu_ps(zero_points + n, zpf32);
            }
        }

        for (int k0 = 0; k0 < IC; k0 += BK, pW += BK * B_stride) {
            int bK = std::min(IC - k0, BK);
            repack_weight_for_4x3(pW, B_stride,
                                  scales + cur_n,
                                  zero_points,
                                  bK, bN,
                                  repacked_B_n24,
                                  repacked_B_n8);
            bool is_accumulate_C = (k0 > 0);
            auto* pC = C + cur_n;
            int m;
            // re-use repacked B sub-matrix in L2 cache as long as we can.
            const auto* pA = A + k0;
            for (m = 0; m < M_body; m += 4, pA += 4 * A_stride, pC += 4 * C_stride) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    brgemm_4x3<4>(pA, A_stride, pB, SIMDW*3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    brgemm_4x1<4>(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
            // M tails
            for (; m < M; m++, pA += A_stride, pC += C_stride) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    brgemm_4x3<1>(pA, A_stride, pB, SIMDW*3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    brgemm_4x1<1>(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
        }
    }
}

extern "C" void MMi8(const float * A,
                     float * C,
                     const uint8_t* W,
                     const uint8_t* zp,
                     const float* scales,
                     int M, int IC, int OC) {
    MM_ComputeBounded_reuseB_i8(A, C, W, zp, scales, M, IC, OC, 0, OC);
}
'''

M = 102
IC = 512*4
OC = 4096
resolution = 8
perf = pycpp.perf(["HW_CPU_CYCLES", "SPLIT_LOADS=0xd0,0x41"])


np.random.seed(0)
input = np.random.rand(M, IC).astype(np.float32)
low = -resolution
high = resolution
Wi8 = np.random.randint(0, high, [IC, OC]).astype(np.uint8)
Wzp = np.random.randint(0, high, [1, OC]).astype(np.uint8)
Wscales = np.random.randint(0, 2, [1, OC]).astype(np.float32)
Wf32 = (Wi8.astype(np.float32) - Wzp.astype(np.float32)) * Wscales

C0 =  np.matmul(input, Wf32)

print("============ Wi8")
print(Wi8)
print("============ Wzp")
print(Wzp)
print("============ Wscales")
print(Wscales)
print("============ Wf32")
print(Wf32)
print("============ input")
print(input)

C1 = np.zeros([M, OC], dtype=np.float32)

ROUNDS = 1
k_loops = 1

for r in range(5):
    with perf.verbose(f"decomp{IC}x{OC}", ROUNDS, k_loops):
        mylib.MMi8(input, C1, Wi8, Wzp, Wscales, M, IC, OC)

#A2 = wq.fakequant

if not np.allclose(C1, C0):
    print("C0=================")
    print(C0)
    print("C1=================")
    print(C1)
    adiff = abs(C1 - C0)
    max_id = adiff.argmax()
    print(f" mean/max diff:{adiff.mean():.2f}/{adiff.max():.2f} @ {max_id}")
else:
    print("C1 ~= C0")
