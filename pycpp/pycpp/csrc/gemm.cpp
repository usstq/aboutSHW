
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "../include/linux_perf.hpp"
#include "omp.h"

#include "gemm.hpp"
#include "brg_reg_kernels.hpp"

#ifndef ASSERT
#    define ASSERT(cond)                                                     \
        if (!(cond)) {                                                       \
            std::stringstream ss;                                            \
            ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
            throw std::runtime_error(ss.str());                              \
        }
#endif

#define DEBUG0(...)        std::cout << "===" << __LINE__ << ":" << std::endl;
#define DEBUG1(x)          std::cout << "===" << __LINE__ << ":" << #x << "=" << x << std::endl;
#define DEBUG2(x1, x2)     std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << std::endl;
#define DEBUG3(x1, x2, x3) std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << std::endl;
#define DEBUG4(x1, x2, x3, x4) \
    std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 << std::endl;
#define DEBUG5(x1, x2, x3, x4, x5)                                                                                                                                        \
    std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 << "," << #x5 << "=" << x5 \
              << std::endl;
#define DEBUG6(x1, x2, x3, x4, x5, x6)                                                                                                                                           \
    std::cout << "===" << __LINE__ << ":" << #x1 << "=" << x1 << "," << #x2 << "=" << x2 << "," << #x3 << "=" << x3 << "," << #x4 << "=" << x4 << "," << #x5 << "=" << x5 << "," \
              << #x6 << "=" << x6 << std::endl;

#define GET_MACRO(_0, _1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define DEBUG_LOG(...)                                   GET_MACRO(_0 __VA_OPT__(, ) __VA_ARGS__, DEBUG6, DEBUG5, DEBUG4, DEBUG3, DEBUG2, DEBUG1, DEBUG0)(__VA_ARGS__)

/*
 g++ -march=core-avx2 -fopenmp -O2 -std=c++11 ./test.cpp
  [   brgemm_4x3<4>] 8.185 us CPU:3.71(GHz) CPI:0.35 CPK:5.9x5120
*/

//=================================================================================================================================================
#if 0
inline int64_t getenv(const char* var, int64_t default_value) {
    const char* p = std::getenv(var);
    if (p) {
        char str_value[256];
        int len = 0;
        while (p[len] >= '0' && p[len] <= '9') {
            str_value[len] = p[len];
            len++;
        }
        str_value[len] = 0;

        char unit = p[len];
        int64_t unit_value = 1;
        // has unit?
        if (unit == 'K' || unit == 'k')
            unit_value = 1024;
        if (unit == 'M' || unit == 'm')
            unit_value = 1024 * 1024;
        if (unit == 'G' || unit == 'g')
            unit_value = 1024 * 1024 * 1024;

        default_value = std::atoi(str_value) * unit_value;
    }
    printf("\e[32mENV:\t %s = %ld %s\e[0m\n", var, default_value, p ? "" : "(default)");

    return default_value;
}
struct MMtestCase {
    template <typename T>
    struct Tensor {
        std::shared_ptr<T> m_ptr;
        int m_count = 0;
        Tensor() = default;
        Tensor(int count, T init) : m_count(count) {
            m_ptr = std::shared_ptr<T>(reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))), [](void* p) {
                ::free(p);
            });
        }

        void reset(int count) {
            m_count = count;
            m_ptr = std::shared_ptr<T>(reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))), [](void* p) {
                ::free(p);
            });
        }

        T& operator[](int offset) const {
            return m_ptr.get()[offset];
        }
        T* begin() const {
            return m_ptr.get();
        }
        T* end() const {
            return m_ptr.get() + m_count;
        }
        T* data() const {
            return m_ptr.get();
        }
        void clr_cache() {
            static int CLFLUSH = getenv("CLFLUSH", 1);
            if (!CLFLUSH)
                return;
            auto* pi8 = reinterpret_cast<int8_t*>(m_ptr.get());
#    pragma omp parallel
            {
                _mm_mfence();
                for (size_t i = 0; i < m_count * sizeof(T); i += 64)
                    _mm_clflush(pi8 + i);
                _mm_mfence();
            }
        }
    };

    Tensor<float> A;      // [M, K]
    Tensor<float> B;      // [K, N]
    Tensor<float> C;      // [M, N]
    Tensor<float> C_ref;  // [M, N]
    int M;
    int K;
    int N;
    int stride_B;
    int stride_C;
    bool no_ref;
    MMtestCase(int M, int K, int N, int _stride_B = 0, bool is_accumulate_C = false, int resolution = 10)
        : M(M),
          K(K),
          N(N),
          stride_B(_stride_B),
          A(M * K, 0),
          C(M * N, 0),
          C_ref(M * N, 0) {
        if (stride_B == 0)
            stride_B = N;
        stride_C = N;
        B.reset(K * stride_B);

        /*
        std::cout << " MMtestCase ================== M = " << M << ", K = " << K << ", N = " << N << ",  stride_B = " << stride_B * sizeof(float) / 64.0 << " cacheline\n";
        std::cout << " MMtestCase ================== A size = " << M*K*sizeof(float)/1024.0 << " KB\n";
        std::cout << " MMtestCase ================== B size = " << N*K*sizeof(float)/1024.0 << " KB\n";
        std::cout << " MMtestCase ================== C size = " << M*N*sizeof(float)/1024.0 << " KB\n";
        */
        srand(0);
        float s = 1.0f / resolution;
        for (auto& v : A)
            v = (rand() % resolution) * s - 0.5f;
        for (auto& v : B)
            v = (rand() % resolution) * s - 0.5f;
        for (auto& v : C)
            v = (rand() % resolution) * s - 0.5f;

        no_ref = getenv("NOREF", 1);
        if (!no_ref) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = is_accumulate_C ? C[m * N + n] : 0;
                    const auto* srcA = &A[m * K];
                    const auto* srcB = &B[n];
                    for (int k = 0; k < K; k++) {
                        sum += srcA[k] * srcB[k * stride_B];
                    }
                    C_ref[m * N + n] = sum;
                }
            }
            std::cout << "ref is calculated\n";
        }
    }

    bool check(int rows = 9999, float atol = 0.01f, float rtol = 0.01f) {
        if (no_ref)
            return false;

        for (int m = 0; m < M && m < rows; m++) {
            for (int n = 0; n < N; n++) {
                auto base = C_ref[m * N + n];
                auto& val = C[m * N + n];
                auto delta = std::abs(val - base);
                if (delta > atol) {
                    std::cout << "@ (" << m << "," << n << ")  " << val << " vs " << base << "(base)  delta(" << delta << ") > atol(" << atol << ")" << std::endl;
                    return false;
                }
                if (delta > std::abs(base) * rtol) {
                    std::cout << "@ (" << m << "," << n << ")  " << val << " vs " << base << "(base)  delta(" << delta << ") > base*rtol (" << base * rtol << ")" << std::endl;
                    return false;
                }
                val = 0;
            }
        }
        return true;
    }

    void clr_cache() {
        A.clr_cache();
        B.clr_cache();
        C.clr_cache();
    }
};
#endif

//=================================================================================================================================================


//=================================================================================================================================================


template <class T>
static T* scratch_alloc(size_t cnt) {
    thread_local uint8_t scratch[1024 * 1024 * 2] __attribute__((aligned(4096)));
    assert(cnt * sizeof(T) < sizeof(scratch));
    // DEBUG_LOG(reinterpret_cast<void*>(scratch));
    return reinterpret_cast<T*>(scratch);
}


struct GemmArgs {
    const float* A;
    int A_stride;  // stride in number of element

    const float* B;
    int B_stride;  // stride in number of element
    //WeightArg* W;

    float* C;
    int C_stride;  // stride in number of element
    int M;
    int K;
    int N;
};

//===============================================================
// when M is big, problem is compute-bounded because we can keep A & B sub-matrix in cache
// to overcome memory bandwidth issue and reach the compute-bound.
//
// if we choose to put B sub-matrix [BK x BN] in cache
//
// then it can update [M x BN] results with [number of DDR accesses of A & C sub-matrix] = (M x (BK + BN))
// and after [K/BK] times, it can complete [M x BN] results,
//
//  we want (M x (BK + BN))*[K/BK] / [M x BN]  to be as small as possible, given [BK * BN] limited by L2 cache size
//
//   solve above eq we have BN == BK
//
// choose B sub-matrix [BK x BN] to put into L2 cache in packed (HW-prefetcher friendly)
// format, and accumulate all affected columns of C. B sub-matrix will be re-used [M/4] times.
// the bigger the M is, the better cache hit-rate of B will be.
//
//   this cache-blocking is very friendly to weight-compression, the repack
//   process can easily handle the weight-decompression process (from f16/int8/int4)
//   and computational kernel brgemm_4x3 can use float32 directly
//
void MM_ComputeBounded(GemmArgs& gemm_args, int n0, int n1) {
    static GemmRegBlocking jit4x3(4, 3, 0);
    static GemmRegBlocking jit4x1(4, 1, 0);
    static GemmRegBlocking jit1x3(1, 3, 0);
    static GemmRegBlocking jit1x1(1, 1, 0);

    constexpr int BK = 720;
    constexpr int BN = 512;
    float* scratch = scratch_alloc<float>(BN * BK);

    const auto A_stride = gemm_args.A_stride;
    const auto B_stride = gemm_args.B_stride;
    const auto C_stride = gemm_args.C_stride;
    auto M = gemm_args.M;
    auto K = gemm_args.K;

    GemmRegBlocking::CallArgs args;
    args.A_stride = A_stride * sizeof(gemm_args.A[0]);
    args.B_stride = B_stride * sizeof(gemm_args.B[0]);
    args.C_stride = C_stride * sizeof(gemm_args.C[0]);

    GemmRegBlocking* pjit_mtail = &jit1x3;
    auto M_tails = M % 4;
    auto M_body = M - M_tails;

    for (int n0 = n0; n0 < n1; n0 += BN) {
        int bN = std::min(n1 - n0, BN);
        const auto* A = gemm_args.A;
        const auto* B = gemm_args.B + n0;
        auto* C = gemm_args.C + n0;
        for (int k = 0; k < K; k += BK, A += BK, B += BK * B_stride) {
            int bK = std::min(K - k, BK);

            float* repacked_B_n24 = scratch;
            float* repacked_B_n8 = scratch + (bN / 24) * (24 * bK);
            {
                // scratch buffer is resident in L2, repack process will exploit
                // HW-prefetcher to access source data from DDR contingously.
                for (int k = 0; k < bK; k++) {
                    const auto* src = B + k * B_stride;
                    const auto* prefetch = B + k * B_stride + bK * B_stride;
                    if (k + BK >= K)
                        prefetch = src;
                    auto* dst = repacked_B_n24 + 24 * k;
                    int n = 0;
                    for (; n + 24 <= bN; n += 24, src += 24, prefetch += 24, dst += bK * 24) {
                        auto b0 = _mm256_loadu_ps(src + 8 * 0);
                        auto b1 = _mm256_loadu_ps(src + 8 * 1);
                        auto b2 = _mm256_loadu_ps(src + 8 * 2);

                        //_mm_prefetch(prefetch, _MM_HINT_T2);

                        _mm256_storeu_ps(dst + 8 * 0, b0);
                        _mm256_storeu_ps(dst + 8 * 1, b1);
                        _mm256_storeu_ps(dst + 8 * 2, b2);
                    }
                    // N tail part is packed in unit of 8-column
                    dst = repacked_B_n8 + 8 * k;
                    for (; n < bN; n += 8, src += 8, dst += bK * 8) {
                        auto b0 = _mm256_loadu_ps(src + 8 * 0);
                        _mm256_storeu_ps(dst, b0);
                    }
                }
            }
            bool is_accumulate_C = (k > 0);
            args.accumulate = is_accumulate_C;
            args.K = bK;
            args.A = A;
            auto* pC = C;
            int m;
            // re-use repacked B sub-matrix in L2 cache as long as we can.
            for (m = 0; m < M_body; m += 4, args.A += 4 * A_stride, pC += 4 * C_stride) {
                args.B = repacked_B_n24;
                args.B_stride = 3 * 8 * sizeof(float);
                int n = 0;
                for (; n + 24 <= bN; n += 24, args.B += 24 * bK) {
                    // brgemm_4x3<4>(pA, A_stride, pB, 24, pC + n, C_stride, bK, is_accumulate_C);
                    args.C = pC + n;
                    jit4x3(&args);
                }
                args.B_stride = 1 * 8 * sizeof(float);
                args.B = repacked_B_n8;
                for (; n < bN; n += 8, args.B += 8 * bK) {
                    // brgemm_4x1<4>(pA, A_stride, pB, 8, pC + n, C_stride, bK, is_accumulate_C);
                    args.C = pC + n;
                    jit4x1(&args);
                }
            }
            // M tails
            for (; m < M; m++, args.A += A_stride, pC += C_stride) {
                args.B = repacked_B_n24;
                args.B_stride = 3 * 8 * sizeof(float);
                int n = 0;
                for (; n + 24 <= bN; n += 24, args.B += 24 * bK) {
                    // brgemm_4x3<1>(pA, A_stride, pB, 24, pC + n, C_stride, bK, is_accumulate_C);
                    args.C = pC + n;
                    jit1x3(&args);
                }
                args.B_stride = 1 * 8 * sizeof(float);
                args.B = repacked_B_n8;
                for (; n < bN; n += 8, args.B += 8 * bK) {
                    // brgemm_4x1<1>(pA, A_stride, pB, 8, pC + n, C_stride, bK, is_accumulate_C);
                    args.C = pC + n;
                    jit1x1(&args);
                }
            }
        }
    }
}

static GemmRegBlocking jit6x2(6, 2, 64);
static GemmRegBlocking jit5x2(5, 2, 64);
static GemmRegBlocking jit4x2(4, 2, 64);
static GemmRegBlocking jit3x2(3, 2, 64);
static GemmRegBlocking jit2x2(2, 2, 64);
static GemmRegBlocking jit1x2(1, 2, 64);

/*
between compute-bound & mem-bound cases, M is not big enough to reuse B many times,
the cost of B sub-matrix repack is not worthy, we will access B w/o repacking
this requires re-use one column of B matrix from L1-cache.

W/o changing strides of B, one column of B matrix may overflow L2-cache due to the nature of set-associative cache.
so BK should be small enough to not overflow L2-cache-sets.

Is no-copy still the best choice when weight-compression (f16/int8/int4) is used?
no-copy is faster because:
   - it covers mem-latency better
   - it put A in cache and read B from DDR only once
*/
void MM_ComputeBounded_nocopy(GemmArgs& gemm_args, int n0, int n1) {
    constexpr int BK = 54;

    auto* A = gemm_args.A;
    auto* B = gemm_args.B;
    auto* C = gemm_args.C;
    const auto A_stride = gemm_args.A_stride;
    const auto B_stride = gemm_args.B_stride;
    const auto C_stride = gemm_args.C_stride;
    auto M = gemm_args.M;
    auto K = gemm_args.K;

    GemmRegBlocking::CallArgs args;
    args.A_stride = A_stride * sizeof(A[0]);
    args.B_stride = B_stride * sizeof(B[0]);
    args.C_stride = C_stride * sizeof(C[0]);
    args.accumulate = false;

    GemmRegBlocking* pjit_mtail = nullptr;
    auto M_tails = M % 6;
    auto M_body = M - M_tails;
    switch (M_tails) {
    case 5:
        pjit_mtail = &jit5x2;
        break;
    case 4:
        pjit_mtail = &jit4x2;
        break;
    case 3:
        pjit_mtail = &jit3x2;
        break;
    case 2:
        pjit_mtail = &jit2x2;
        break;
    case 1:
        pjit_mtail = &jit1x2;
        break;
    }

    for (int k = 0; k < K; k += BK, A += BK, B += BK * B_stride) {
        args.K = std::min(K - k, BK);
        args.accumulate = (k > 0);

        for (int n = n0; n + 2 * 8 <= n1; n += 2 * 8) {
            args.B = B + n;
            args.A = A;
            args.C = C + n;
            for (int m = 0; m < M_body; m += 6, args.A += 6 * A_stride, args.C += 6 * C_stride) {
                jit6x2(&args);
            }
            if (pjit_mtail)
                (*pjit_mtail)(&args);
        }
    }
}

// this version is slightly worse (-10%) than nocopy, but it maybe more suitable for weight-decompression
void MM_ComputeBounded_reuseA(GemmArgs& gemm_args, int n0, int n1) {
    constexpr int BK = 54;
    float* scratch = scratch_alloc<float>(16 * BK);

    auto* A = gemm_args.A;
    auto* B = gemm_args.B;
    auto* C = gemm_args.C;
    const auto A_stride = gemm_args.A_stride;
    const auto B_stride = gemm_args.B_stride;
    const auto C_stride = gemm_args.C_stride;
    int M = gemm_args.M;
    int K = gemm_args.K;

    GemmRegBlocking::CallArgs args;
    args.A_stride = A_stride * sizeof(A[0]);
    args.B_stride = B_stride * sizeof(B[0]);
    args.C_stride = C_stride * sizeof(C[0]);

    GemmRegBlocking* pjit_mtail = nullptr;
    auto M_tails = M % 6;
    auto M_body = M - M_tails;
    switch (M_tails) {
    case 5:
        pjit_mtail = &jit5x2;
        break;
    case 4:
        pjit_mtail = &jit4x2;
        break;
    case 3:
        pjit_mtail = &jit3x2;
        break;
    case 2:
        pjit_mtail = &jit2x2;
        break;
    case 1:
        pjit_mtail = &jit1x2;
        break;
    }

    for (int k = 0; k < K; k += BK, A += BK, B += BK * B_stride) {
        int bK = std::min(K - k, BK);
        args.K = bK;
        args.accumulate = (k > 0);

        for (int n = n0; n + 2 * 8 <= n1; n += 2 * 8) {
            // prepack [BK, 16] into scratch
            float* repacked_B = scratch;
            {
                const auto* src = B + n;
                auto* dst = repacked_B;
                for (int k = 0; k < bK; k++, dst += 2 * 8, src += B_stride) {
                    auto b0 = _mm256_loadu_ps(src + 8 * 0);
                    auto b1 = _mm256_loadu_ps(src + 8 * 1);

                    _mm_prefetch(src + 32, _MM_HINT_T0);

                    _mm256_storeu_ps(dst + 8 * 0, b0);
                    _mm256_storeu_ps(dst + 8 * 1, b1);
                }
            }
            args.B = repacked_B;
            args.B_stride = 2 * 8 * sizeof(B[0]);

            args.A = A;
            args.C = C + n;
            for (int m = 0; m < M_body; m += 6, args.A += 6 * A_stride, args.C += 6 * C_stride) {
                jit6x2(&args);
            }
            if (pjit_mtail)
                (*pjit_mtail)(&args);
        }
    }
}

//===============================================================
// when M is too small, problem becomes memory bounded because A & C are small
// and there is no opportunity to reuse the biggest matrix here : B
//
//  REF: https://abertschi.ch/blog/2022/prefetching/
//
// thus to reach the mem-bound, we need to make best use of HW-prefetcher by access data continously
// since L2 Hardware Prefetcher (Streamer) support multiple streams (32 at most), we can access many rows of B
// continously.
//
void MM_MemoryBounded(GemmArgs& gemm_args, int n0, int n1) {
    auto* A = gemm_args.A;
    auto* B = gemm_args.B;
    auto* C = gemm_args.C;
    const auto A_stride = gemm_args.A_stride;
    const auto B_stride = gemm_args.B_stride;
    const auto C_stride = gemm_args.C_stride;
    int M = gemm_args.M;
    int K = gemm_args.K;
    {
        auto c0 = _mm256_setzero_ps();
        auto* pC = C;
        for (int m = 0; m < M; m++, pC += C_stride) {
            for (int n = n0; n < n1; n += 8) {
                _mm256_storeu_ps(pC + n, c0);
            }
        }
    }
    ASSERT(((K % 4) == 0) && (((n1 - n0) % 8) == 0));

    for (int k = 0; k < K; k += 4) {
        const auto* pA = A;
        auto* pC = C;
        for (int m = 0; m < M; m++, pA += A_stride, pC += C_stride) {
            __m256 a0 = _mm256_broadcast_ss(pA + k + 0);
            __m256 a1 = _mm256_broadcast_ss(pA + k + 1);
            __m256 a2 = _mm256_broadcast_ss(pA + k + 2);
            __m256 a3 = _mm256_broadcast_ss(pA + k + 3);
            const auto* pB = B + k * B_stride;
            const auto* pB2 = pB + 2 * B_stride;
            int n = n0;
            for (; (n + 8 * 4) <= n1; n += 8 * 4, pB += 8 * 4, pB2 += 8 * 4) {
                __m256 c0 = _mm256_loadu_ps(pC + n + 8 * 0);
                __m256 c1 = _mm256_loadu_ps(pC + n + 8 * 1);
                __m256 c2 = _mm256_loadu_ps(pC + n + 8 * 2);
                __m256 c3 = _mm256_loadu_ps(pC + n + 8 * 3);

                c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(pB + 8 * 0), c0);
                c1 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(pB + 8 * 1), c1);
                c2 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(pB + 8 * 2), c2);
                c3 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(pB + 8 * 3), c3);

                c0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(pB + B_stride + 8 * 0), c0);
                c1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(pB + B_stride + 8 * 1), c1);
                c2 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(pB + B_stride + 8 * 2), c2);
                c3 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(pB + B_stride + 8 * 3), c3);

                c0 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(pB2 + 8 * 0), c0);
                c1 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(pB2 + 8 * 1), c1);
                c2 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(pB2 + 8 * 2), c2);
                c3 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(pB2 + 8 * 3), c3);

                c0 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(pB2 + B_stride + 8 * 0), c0);
                c1 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(pB2 + B_stride + 8 * 1), c1);
                c2 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(pB2 + B_stride + 8 * 2), c2);
                c3 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(pB2 + B_stride + 8 * 3), c3);

                _mm256_storeu_ps(pC + n + 8 * 0, c0);
                _mm256_storeu_ps(pC + n + 8 * 1, c1);
                _mm256_storeu_ps(pC + n + 8 * 2, c2);
                _mm256_storeu_ps(pC + n + 8 * 3, c3);
            }
            // n tails
            for (; n < n1; n += 8, pB += 8, pB2 += 8) {
                __m256 c0 = _mm256_loadu_ps(pC + n);
                c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(pB), c0);
                c0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(pB + B_stride), c0);
                c0 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(pB2), c0);
                c0 = _mm256_fmadd_ps(a3, _mm256_loadu_ps(pB2 + B_stride), c0);
                _mm256_storeu_ps(pC + n, c0);
            }
        }
    }
}

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

/*
    due to 2nd token's requirement, layout of B matrix is choosen to be plain [K, N].
    therefore accessing of B matrix will be strided, to prevent prefetching issue
    we need SW-prefetching it into cache.

    (BM * BN)*sizeof(float) + (BK*BN)*sizeof(float) < L2_cache_size
    choose BK = 128/256
    BN or BM=256/512
*/

void Gemm(GemmArgs& args) {
    int num_works = args.N / 8;
    auto nthr = omp_get_max_threads();
#pragma omp parallel
    {
        int n0, n1;
        int ithr = omp_get_thread_num();
        splitter(num_works, nthr, ithr, n0, n1);
        n0 *= 8;
        n1 *= 8;

        if (args.M < 6) {
            MM_MemoryBounded(args, n0, n1);
        } else if (args.M <= 60) {
            // empirical strategy selection
            // MM_ComputeBounded_nocopy(args, n0, n1);
            MM_ComputeBounded_reuseA(args, n0, n1);
        } else {
            MM_ComputeBounded(args, n0, n1);
        }
    }
}

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(gemm, m) {
    py::class_<GemmRegBlocking>(m, "GemmRegBlocking")
        .def(py::init<int, int>())  // GemmRegBlocking(int rows, int cols_in_avx2_regs)
        .def(
            "__call__",
            [](GemmRegBlocking& self, py::array_t<float> A, py::array_t<float> B, py::array_t<float> C, int ROUNDS) {
                py::buffer_info bufA = A.request();
                py::buffer_info bufB = B.request();
                py::buffer_info bufC = C.request();

                auto M = bufA.shape[0];
                auto K = bufA.shape[1];

                ASSERT(K == bufB.shape[0]);
                auto N = bufB.shape[1];

                ASSERT(M == bufC.shape[0]);
                ASSERT(N == bufC.shape[1]);

                ASSERT(M == self.m_rows);
                ASSERT(N == self.m_cols * 8);

                GemmRegBlocking::CallArgs args;
                args.A = static_cast<float*>(bufA.ptr);
                args.A_stride = bufA.strides[0];
                args.B = static_cast<float*>(bufB.ptr);
                args.B_stride = bufB.strides[0];
                args.C = static_cast<float*>(bufC.ptr);
                args.C_stride = bufC.strides[0];
                args.K = K;
                args.accumulate = false;

                for (int r = 0; r < ROUNDS; r++)
                    self(&args);
            },
            py::arg(),
            py::arg(),
            py::arg(),
            py::arg("ROUNDS") = 1);

    m.def("gemm", [](py::array_t<float> A, py::array_t<float> B, py::array_t<float> C) {
        py::buffer_info bufA = A.request();
        py::buffer_info bufB = B.request();
        py::buffer_info bufC = C.request();

        auto M = bufA.shape[0];
        auto K = bufA.shape[1];

        ASSERT(K == bufB.shape[0]);
        auto N = bufB.shape[1];

        ASSERT(M == bufC.shape[0]);
        ASSERT(N == bufC.shape[1]);

        GemmArgs args;
        args.A = static_cast<float*>(bufA.ptr);
        args.A_stride = bufA.strides[0] / bufA.itemsize;
        args.B = static_cast<float*>(bufB.ptr);
        args.B_stride = bufB.strides[0] / bufB.itemsize;
        args.C = static_cast<float*>(bufC.ptr);
        args.C_stride = bufC.strides[0] / bufC.itemsize;
        args.M = M;
        args.K = K;
        args.N = N;
        Gemm(args);
    });

    m.def("test_regblk", [](py::array_t<float> A, py::array_t<float> B, py::array_t<float> C, int ROUNDS) {
        py::buffer_info bufA = A.request();
        py::buffer_info bufB = B.request();
        py::buffer_info bufC = C.request();
        auto M = bufA.shape[0];
        auto K = bufA.shape[1];
        ASSERT(K == bufB.shape[0]);
        auto N = bufB.shape[1];

        ASSERT(M == bufC.shape[0]);
        ASSERT(N == bufC.shape[1]);

        // DEBUG_LOG(M, K, N);
        // DEBUG_LOG(bufA.strides[0], bufB.strides[0], bufC.strides[0]);

        if (M == 4 && N == 3 * 8) {
            for (int r = 0; r < ROUNDS; r++)
                brgemm_4x3<4>(static_cast<float*>(bufA.ptr),
                              bufA.strides[0] / bufA.itemsize,
                              static_cast<float*>(bufB.ptr),
                              bufB.strides[0] / bufB.itemsize,
                              static_cast<float*>(bufC.ptr),
                              bufC.strides[0] / bufC.itemsize,
                              K,
                              false);
        } else if (M == 6 && N == 2 * 8) {
            for (int r = 0; r < ROUNDS; r++)
                brgemm_6x2<6, 0>(static_cast<float*>(bufA.ptr),
                                 bufA.strides[0] / bufA.itemsize,
                                 static_cast<float*>(bufB.ptr),
                                 bufB.strides[0] / bufB.itemsize,
                                 static_cast<float*>(bufC.ptr),
                                 bufC.strides[0] / bufC.itemsize,
                                 K,
                                 false);
        } else if (M == 2 && N == 6 * 8) {
            for (int r = 0; r < ROUNDS; r++)
                brgemm_2x6(static_cast<float*>(bufA.ptr),
                           bufA.strides[0] / bufA.itemsize,
                           static_cast<float*>(bufB.ptr),
                           bufB.strides[0] / bufB.itemsize,
                           static_cast<float*>(bufC.ptr),
                           bufC.strides[0] / bufC.itemsize,
                           K,
                           false);
        } else {
            throw std::runtime_error("Unsupported M & N in test_regblk()");
        }
    });
}
