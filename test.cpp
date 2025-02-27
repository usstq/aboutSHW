
#include <cstdlib>
#include <memory>

#include "include/linux_perf.hpp"
#include "omp.h"



/*
 g++ -march=core-avx2 -fopenmp -O2 -std=c++11 ./test.cpp
  [   brgemm_4x3<4>] 8.185 us CPU:3.71(GHz) CPI:0.35 CPK:5.9x5120
*/

//=================================================================================================================================================
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
    template<typename T>
    struct Tensor {
        std::shared_ptr<T> m_ptr;
        int m_count = 0;
        Tensor() = default; 
        Tensor(int count, T init) : m_count(count) {
            m_ptr = std::shared_ptr<T>(reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))), [](void * p) { ::free(p); });
        }

        void reset(int count) {
            m_count = count;
            m_ptr = std::shared_ptr<T>(reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))), [](void * p) { ::free(p); });
        }

        T& operator [](int offset) const {return m_ptr.get()[offset];}
        T* begin() const { return m_ptr.get(); }
        T* end() const { return m_ptr.get() + m_count; }
        T* data() const {return m_ptr.get();}
        void clr_cache() {
            static int CLFLUSH = getenv("CLFLUSH", 1);
            if (!CLFLUSH) return;
            auto* pi8 = reinterpret_cast<int8_t*>(m_ptr.get());
            #pragma omp parallel
            {
                _mm_mfence();
                for (size_t i = 0; i < m_count*sizeof(T); i += 64) _mm_clflush(pi8 + i);
                _mm_mfence();
            }
        }
    };

    Tensor<float> A;      // [M, N]
    Tensor<float> B;      // [K, N]
    Tensor<float> C;      // [M, N]
    Tensor<float> C_ref;  // [M, N]
    int M;
    int K;
    int N;
    int stride_B;
    bool no_ref; 
    MMtestCase(int M, int K, int N, bool is_accumulate_C = false, int resolution = 10) : M(M), K(K), N(N), A(M * K, 0), C(M * N, 0), C_ref(M * N, 0) {
        stride_B = N + 16;
        B.reset(K * stride_B);

        std::cout << " MMtestCase ================== stride_B = " << stride_B * sizeof(float)/64.0 << " cacheline\n";
        srand(0);
        float s = 1.0f / resolution;
        for (auto& v : A)  v = (rand() % resolution) * s - 0.5f;
        for (auto& v : B)  v = (rand() % resolution) * s - 0.5f;
        for (auto& v : C)  v = (rand() % resolution) * s - 0.5f;

        no_ref = getenv("NOREF", 0);
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
        if (no_ref) return false;

        for (int m = 0; m < M && m < rows; m++) {
            for (int n = 0; n < N; n++) {
                auto base = C_ref[m * N + n];
                auto val = C[m * N + n];
                auto delta = std::abs(val - base);
                if (delta > atol) {
                    std::cout << "@ (" << m << "," << n << ")  " << val << " vs " << base << "(base)  delta(" << delta << ") > atol(" << atol << ")" << std::endl;
                    return false;
                }
                if (delta > std::abs(base) * rtol) {
                    std::cout << "@ (" << m << "," << n << ")  " << val << " vs " << base << "(base)  delta(" << delta << ") > base*rtol (" << base * rtol << ")" << std::endl;
                    return false;
                }
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

//=================================================================================================================================================

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
    __m256 c00, c01, c02;
    __m256 c10, c11, c12;
    __m256 c20, c21, c22;
    __m256 c30, c31, c32;

    if (is_accumulate_C) {
        c00 = _mm256_loadu_ps(C + 8 * 0);
        c01 = _mm256_loadu_ps(C + 8 * 1);
        c02 = _mm256_loadu_ps(C + 8 * 2);

        if (row > 1) {
            c10 = _mm256_loadu_ps(C + C_stride + 8 * 0);
            c11 = _mm256_loadu_ps(C + C_stride + 8 * 1);
            c12 = _mm256_loadu_ps(C + C_stride + 8 * 2);
        }
        if (row > 2) {
            c20 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 0);
            c21 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 1);
            c22 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 2);
        }
        if (row > 3) {
            c30 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 0);
            c31 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 1);
            c32 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 2);
        }
    } else {
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c02 = _mm256_setzero_ps();

        if (row > 1) {
            c10 = _mm256_setzero_ps();
            c11 = _mm256_setzero_ps();
            c12 = _mm256_setzero_ps();
        }
        if (row > 2) {
            c20 = _mm256_setzero_ps();
            c21 = _mm256_setzero_ps();
            c22 = _mm256_setzero_ps();
        }
        if (row > 3) {
            c30 = _mm256_setzero_ps();
            c31 = _mm256_setzero_ps();
            c32 = _mm256_setzero_ps();
        }
    }

    auto * prefetch_B = B + 8*3;

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++, prefetch_B += B_stride) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 b1 = _mm256_loadu_ps(B + 8 * 1);
        __m256 b2 = _mm256_loadu_ps(B + 8 * 2);
        
        //_mm_prefetch(prefetch_B, _MM_HINT_T1);
        //_mm_prefetch(B + 24*5, _MM_HINT_T2);

        __m256 a = _mm256_broadcast_ss(A);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);
        c02 = _mm256_fmadd_ps(a, b2, c02);
        if (row > 1) {
            a = _mm256_broadcast_ss(A + A_stride);
            c10 = _mm256_fmadd_ps(a, b0, c10);
            c11 = _mm256_fmadd_ps(a, b1, c11);
            c12 = _mm256_fmadd_ps(a, b2, c12);
        }
        if (row > 2) {
            a = _mm256_broadcast_ss(A + 2 * A_stride);
            c20 = _mm256_fmadd_ps(a, b0, c20);
            c21 = _mm256_fmadd_ps(a, b1, c21);
            c22 = _mm256_fmadd_ps(a, b2, c22);
        }
        if (row > 3) {
            a = _mm256_broadcast_ss(A + 3 * A_stride);
            c30 = _mm256_fmadd_ps(a, b0, c30);
            c31 = _mm256_fmadd_ps(a, b1, c31);
            c32 = _mm256_fmadd_ps(a, b2, c32);
        }
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c00);
    _mm256_storeu_ps(C + 8 * 1, c01);
    _mm256_storeu_ps(C + 8 * 2, c02);
    if (row > 1) {
        _mm256_storeu_ps(C + C_stride + 8 * 0, c10);
        _mm256_storeu_ps(C + C_stride + 8 * 1, c11);
        _mm256_storeu_ps(C + C_stride + 8 * 2, c12);
    }
    if (row > 2) {
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 0, c20);
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 1, c21);
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 2, c22);
    }
    if (row > 3) {
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 0, c30);
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 1, c31);
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 2, c32);
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
    __m256 c00;
    __m256 c10;
    __m256 c20;
    __m256 c30;

    if (is_accumulate_C) {
        c00 = _mm256_loadu_ps(C + 8 * 0);
        if (row > 1) {
            c10 = _mm256_loadu_ps(C + C_stride + 8 * 0);
        }
        if (row > 2) {
            c20 = _mm256_loadu_ps(C + 2 * C_stride + 8 * 0);
        }
        if (row > 3) {
            c30 = _mm256_loadu_ps(C + 3 * C_stride + 8 * 0);
        }
    } else {
        c00 = _mm256_setzero_ps();
        if (row > 1) {
            c10 = _mm256_setzero_ps();
        }
        if (row > 2) {
            c20 = _mm256_setzero_ps();
        }
        if (row > 3) {
            c30 = _mm256_setzero_ps();
        }
    }

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 a = _mm256_broadcast_ss(A);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        if (row > 1) {
            a = _mm256_broadcast_ss(A + A_stride);
            c10 = _mm256_fmadd_ps(a, b0, c10);
        }
        if (row > 2) {
            a = _mm256_broadcast_ss(A + 2 * A_stride);
            c20 = _mm256_fmadd_ps(a, b0, c20);
        }
        if (row > 3) {
            a = _mm256_broadcast_ss(A + 3 * A_stride);
            c30 = _mm256_fmadd_ps(a, b0, c30);
        }
    }
    _mm256_storeu_ps(C + 8 * 0, c00);
    if (row > 1) {
        _mm256_storeu_ps(C + C_stride + 8 * 0, c10);
    }
    if (row > 2) {
        _mm256_storeu_ps(C + 2 * C_stride + 8 * 0, c20);
    }
    if (row > 3) {
        _mm256_storeu_ps(C + 3 * C_stride + 8 * 0, c30);
    }
}

void brgemm_1x12(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    __m256 c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        c0 = _mm256_loadu_ps(C + 8 * 0);
        c1 = _mm256_loadu_ps(C + 8 * 1);
        c2 = _mm256_loadu_ps(C + 8 * 2);
        c3 = _mm256_loadu_ps(C + 8 * 3);
        c4 = _mm256_loadu_ps(C + 8 * 4);
        c5 = _mm256_loadu_ps(C + 8 * 5);
        c6 = _mm256_loadu_ps(C + 8 * 6);
        c7 = _mm256_loadu_ps(C + 8 * 7);
        c8 = _mm256_loadu_ps(C + 8 * 8);
        c9 = _mm256_loadu_ps(C + 8 * 9);
        ca = _mm256_loadu_ps(C + 8 * 10);
        cb = _mm256_loadu_ps(C + 8 * 11);
    } else {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
        c8 = _mm256_setzero_ps();
        c9 = _mm256_setzero_ps();
        ca = _mm256_setzero_ps();
        cb = _mm256_setzero_ps();
    }
    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b;
        __m256 a = _mm256_broadcast_ss(A);
        // read enough data along current cache fetching line
        b = _mm256_loadu_ps(B + 8 * 0); c0 = _mm256_fmadd_ps(a, b, c0);
        b = _mm256_loadu_ps(B + 8 * 1); c1 = _mm256_fmadd_ps(a, b, c1);
        b = _mm256_loadu_ps(B + 8 * 2); c2 = _mm256_fmadd_ps(a, b, c2);
        b = _mm256_loadu_ps(B + 8 * 3); c3 = _mm256_fmadd_ps(a, b, c3);
        b = _mm256_loadu_ps(B + 8 * 4); c4 = _mm256_fmadd_ps(a, b, c4);
        b = _mm256_loadu_ps(B + 8 * 5); c5 = _mm256_fmadd_ps(a, b, c5);
        b = _mm256_loadu_ps(B + 8 * 6); c6 = _mm256_fmadd_ps(a, b, c6);
        b = _mm256_loadu_ps(B + 8 * 7); c7 = _mm256_fmadd_ps(a, b, c7);
        b = _mm256_loadu_ps(B + 8 * 8); c8 = _mm256_fmadd_ps(a, b, c8);
        b = _mm256_loadu_ps(B + 8 * 9); c9 = _mm256_fmadd_ps(a, b, c9);
        b = _mm256_loadu_ps(B + 8 * 10); ca = _mm256_fmadd_ps(a, b, ca);
        b = _mm256_loadu_ps(B + 8 * 11); cb = _mm256_fmadd_ps(a, b, cb);
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c0);
    _mm256_storeu_ps(C + 8 * 1, c1);
    _mm256_storeu_ps(C + 8 * 2, c2);
    _mm256_storeu_ps(C + 8 * 3, c3);
    _mm256_storeu_ps(C + 8 * 4, c4);
    _mm256_storeu_ps(C + 8 * 5, c5);
    _mm256_storeu_ps(C + 8 * 6, c6);
    _mm256_storeu_ps(C + 8 * 7, c7);
    _mm256_storeu_ps(C + 8 * 8, c8);
    _mm256_storeu_ps(C + 8 * 9, c9);
    _mm256_storeu_ps(C + 8 * 10, ca);
    _mm256_storeu_ps(C + 8 * 11, cb);
}


void brgemm_2x6(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    __m256 c0, c1, c2, c3, c4, c5;
    __m256 c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        c0 = _mm256_loadu_ps(C + 8 * 0);
        c1 = _mm256_loadu_ps(C + 8 * 1);
        c2 = _mm256_loadu_ps(C + 8 * 2);
        c3 = _mm256_loadu_ps(C + 8 * 3);
        c4 = _mm256_loadu_ps(C + 8 * 4);
        c5 = _mm256_loadu_ps(C + 8 * 5);

        c6 = _mm256_loadu_ps(C + C_stride + 8 * 0);
        c7 = _mm256_loadu_ps(C + C_stride + 8 * 1);
        c8 = _mm256_loadu_ps(C + C_stride + 8 * 2);
        c9 = _mm256_loadu_ps(C + C_stride + 8 * 3);
        ca = _mm256_loadu_ps(C + C_stride + 8 * 4);
        cb = _mm256_loadu_ps(C + C_stride + 8 * 5);
    } else {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
        c8 = _mm256_setzero_ps();
        c9 = _mm256_setzero_ps();
        ca = _mm256_setzero_ps();
        cb = _mm256_setzero_ps();
    }
    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b;
        __m256 a0 = _mm256_broadcast_ss(A);
        __m256 a1 = _mm256_broadcast_ss(A + A_stride);
        // read enough data along current cache fetching line
        b = _mm256_loadu_ps(B + 8 * 0); c0 = _mm256_fmadd_ps(a0, b, c0); c6 = _mm256_fmadd_ps(a1, b, c6);
        b = _mm256_loadu_ps(B + 8 * 1); c1 = _mm256_fmadd_ps(a0, b, c1); c7 = _mm256_fmadd_ps(a1, b, c7);
        b = _mm256_loadu_ps(B + 8 * 2); c2 = _mm256_fmadd_ps(a0, b, c2); c8 = _mm256_fmadd_ps(a1, b, c8);
        b = _mm256_loadu_ps(B + 8 * 3); c3 = _mm256_fmadd_ps(a0, b, c3); c9 = _mm256_fmadd_ps(a1, b, c9);
        b = _mm256_loadu_ps(B + 8 * 4); c4 = _mm256_fmadd_ps(a0, b, c4); ca = _mm256_fmadd_ps(a1, b, ca);
        b = _mm256_loadu_ps(B + 8 * 5); c5 = _mm256_fmadd_ps(a0, b, c5); cb = _mm256_fmadd_ps(a1, b, cb);
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c0);
    _mm256_storeu_ps(C + 8 * 1, c1);
    _mm256_storeu_ps(C + 8 * 2, c2);
    _mm256_storeu_ps(C + 8 * 3, c3);
    _mm256_storeu_ps(C + 8 * 4, c4);
    _mm256_storeu_ps(C + 8 * 5, c5);
    _mm256_storeu_ps(C + C_stride + 8 * 0, c6);
    _mm256_storeu_ps(C + C_stride + 8 * 1, c7);
    _mm256_storeu_ps(C + C_stride + 8 * 2, c8);
    _mm256_storeu_ps(C + C_stride + 8 * 3, c9);
    _mm256_storeu_ps(C + C_stride + 8 * 4, ca);
    _mm256_storeu_ps(C + C_stride + 8 * 5, cb);
}

/*
  brgemm_4x3 can get 98% peak GFLOPS if both A & B are in cache
  so the computational kernel is OK.

  brgemm_1x12 & brgemm_2x6 are worse (60~70% only)
*/
int test_basic(LinuxPerf::PerfEventGroup pevg) {
    int M = 4;
    int K = 256;
    int N = 24;
    MMtestCase mtc(M, K, N);

#define REPEATS 20
    auto custom_log = [&](uint64_t _duration_ns, uint64_t* pmc, char*& log) {
        double cpu_freq_GHz = 1.0 * pmc[0] / _duration_ns;
        double fma_peak_gflops = cpu_freq_GHz * 2 * 8 * 2;  // SIMD-8 x 0.5 tput x 2FLOPS/fma
        double duration_ns = _duration_ns * 1.0 / REPEATS;
        float bw = ((M * K) * sizeof(float) + (K * N) * sizeof(float) + (M * N) * sizeof(float)) / duration_ns;
        float gflops = (M * N * K * 2) / duration_ns;
        log += sprintf(log, "   %.3f (GFlop/s) / %.2f(GB/s)   compute-bound:%.1f %%", gflops, bw, gflops * 100 / fma_peak_gflops);
    };

    for (int r = 0; r < 10; r++) {
        auto pmc = pevg.rdpmc(
            [&]() {
                // repeat
                for (int repeat = 0; repeat < REPEATS; repeat++) {
                    brgemm_4x3<1>(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.K, false);
                }
            },
            "brgemm_4x3<1>",
            REPEATS * K,
            custom_log);
    }
    std::cout << "==========================================================" << mtc.check(1) << std::endl;

    for (int r = 0; r < 10; r++) {
        auto pmc = pevg.rdpmc(
            [&]() {
                // repeat
                for (int repeat = 0; repeat < REPEATS; repeat++) {
                    brgemm_4x3<2>(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.K, false);
                }
            },
            "brgemm_4x3<2>",
            REPEATS * K,
            custom_log);
    }
    std::cout << "==========================================================" << mtc.check(2) << std::endl;
    for (int r = 0; r < 10; r++) {
        auto pmc = pevg.rdpmc(
            [&]() {
                // repeat
                for (int repeat = 0; repeat < REPEATS; repeat++) {
                    brgemm_4x3<3>(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.K, false);
                }
            },
            "brgemm_4x3<3>",
            REPEATS * K,
            custom_log);
    }
    std::cout << "==========================================================" << mtc.check(3) << std::endl;

    for (int r = 0; r < 10; r++) {
        auto pmc = pevg.rdpmc(
            [&]() {
                // repeat
                for (int repeat = 0; repeat < REPEATS; repeat++) {
                    brgemm_4x3<4>(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.K, false);
                }
            },
            "brgemm_4x3<4>",
            REPEATS * K,
            custom_log);
    }
    std::cout << "==========================================================" << mtc.check(4) << std::endl;
    {
        M = 1;
        N = 12*8;
        MMtestCase mtc(M, K, N);
        for (int r = 0; r < 10; r++) {
            auto pmc = pevg.rdpmc(
                [&]() {
                    // repeat
                    for (int repeat = 0; repeat < REPEATS; repeat++) {
                        brgemm_1x12(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.K, false);
                    }
                },
                "brgemm_1x12",
                REPEATS * K,
                custom_log);
        }
        std::cout << "==========================================================" << mtc.check() << std::endl;
    }
    {
        M = 2;
        N = 6*8;
        MMtestCase mtc(M, K, N);
        for (int r = 0; r < 10; r++) {
            auto pmc = pevg.rdpmc(
                [&]() {
                    // repeat
                    for (int repeat = 0; repeat < REPEATS; repeat++) {
                        brgemm_2x6(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.K, false);
                    }
                },
                "brgemm_2x6",
                REPEATS * K,
                custom_log);
        }
        std::cout << "==========================================================" << mtc.check() << std::endl;
    }    
    return 0;
}

//===============================================================
void MM_CacheBlock(const float* A,
                   int A_stride,  // stride in number of element
                   const float* B,
                   int B_stride,  // stride in number of element
                   float* C,
                   int C_stride,  // stride in number of element
                   int M,
                   int K,
                   int N,
                   int BK) {
    for (int k = 0; k < K; k += BK, A += BK, B += BK * B_stride) {
        auto* pA = A;
        auto* pC = C;
        int m;
        bool is_accumulate_C = (k > 0);
        // prefetch next B block
        auto* prefetch_B = B + (BK * B_stride);
        int prefetch_n = 0;

        // reuse B sub-matrix [BK x N]   [M/4] times
        // loading B sub-matrix [BK x N] into cache is mem-bound, bigger M can better reuse B sub-mat
        //
        for (m = 0; m + 4 <= M; m += 4, pA += 4 * A_stride, pC += 4 * C_stride) {
            auto* prefetch_A = pA + 4 * A_stride;
            auto* prefetch_A_end = prefetch_A + BK;

            const auto * pB = B;
            int n = 0;
            for (; n + 24 <= N; n += 24, pB += 24*BK) {
                //brgemm_4x3<4>(pA, A_stride, pB, 24, pC + n, C_stride, BK, is_accumulate_C);
                brgemm_4x3<4>(pA, A_stride, B + n, B_stride, pC + n, C_stride, BK, is_accumulate_C);
            }
            // N tails
            for (; n < N; n += 8) {
                brgemm_4x1<4>(pA, A_stride, B + n, B_stride, pC + n, C_stride, BK, is_accumulate_C);
            }
        }
        // M tails
        for (; m < M; m++, pA += A_stride, pC += C_stride) {
            int n = 0;
            for (; n + 24 <= N; n += 24) {
                brgemm_4x3<1>(pA, A_stride, B + n, B_stride, pC + n, C_stride, BK, is_accumulate_C);
            }
            for (; n < N; n += 8) {
                brgemm_4x1<4>(pA, A_stride, B + n, B_stride, pC + n, C_stride, BK, is_accumulate_C);
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

// [4,1024,24] can get to 96% GFLOPS
void MM_brgemm(const float* A,
               int A_stride,  // stride in number of element
               const float* B,
               int B_stride,  // stride in number of element
               float* C,
               int C_stride,  // stride in number of element
               int M,
               int K,
               int N,
               bool is_accumulate_C) {
    /*
        due to 2nd token's requirement, layout of B matrix is choosen to be plain [K, N].
        therefore accessing of B matrix will be strided, to prevent prefetching issue
        we need SW-prefetching it into cache.

        (BM * BN)*sizeof(float) + (BK*BN)*sizeof(float) < L2_cache_size
        choose BK = 128/256
        BN or BM=256/512
    */
    int num_works = N / 8;
    auto nthr = omp_get_max_threads();
#pragma omp parallel
    {
        int n0, n1;
        int ithr = omp_get_thread_num();
        splitter(num_works, nthr, ithr, n0, n1);
        n0 *= 8;
        n1 *= 8;
        int BN = 240;
        for (int n = n0; n < n1; n += BN) {
            int bN = std::min(n1 - n, BN);
            MM_CacheBlock(A, A_stride, B + n, B_stride, C + n, C_stride, M, K, bN, 256);
            // MM_CacheBlock<256>(A, A_stride, B + n*K, bN, C + n, C_stride, M, K, bN);
        }
    }
}

int test(LinuxPerf::PerfEventGroup& pevg, int64_t M, int64_t K, int64_t N) {
    MMtestCase mtc(M, K, N);
    std::stringstream ss;
    ss << "MM_brgemm(" << M << "," << K << "," << N << ")";
    int repeat = 10;
    auto nthr = omp_get_max_threads();
    for (int r = 0; r < repeat; r++) {
        // clear cache
        mtc.clr_cache();
        // profile once
        auto pmc = pevg.rdpmc(
            [&]() {
                MM_brgemm(mtc.A.data(), mtc.K, mtc.B.data(), mtc.stride_B, mtc.C.data(), mtc.N, mtc.M, mtc.K, mtc.N, false);
            },
            ss.str(),
            1 * K,
            [&](uint64_t _duration_ns, uint64_t* pmc, char*& log) {
                // printf(" MemBW:%.1f(GB/s)", (K*N/2)*1*1.0/duration_ns);
                double cpu_freq_GHz = 1.0 * pmc[0] / _duration_ns;
                double fma_peak_gflops = cpu_freq_GHz * nthr * 2 * 8 * 2;  // SIMD-8 x 0.5 tput x 2FLOPS/fma
                double duration_ns = _duration_ns * 1.0;
                float mem_size = (K * N) * sizeof(float) + (M * N) * sizeof(float);
                float bw = ((M * K) * sizeof(float) + (K * N) * sizeof(float) + (M * N) * sizeof(float)) / duration_ns;
                float gflops = (M * N * K * 2) / duration_ns;
                log += sprintf(log, "  %.1f MB   %.3f (GFlop/s) / %.2f(GB/s)   compute-bound(%%):%.1f ", mem_size / (1024 * 1024), gflops, bw, gflops * 100 / fma_peak_gflops);
                if (r == repeat - 1)
                    log += sprintf(log, " accuracy: %s", (mtc.check()) ? "OK" : "failed");
            });
    }
    return 0;
}

int main() {
    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x04, 0x04), "STALLS_TOTAL"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x06, 0x06), "STALLS_L3_MISS"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x05, 0x05), "STALLS_L2_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x40, 0x02),"BOUND_ON_STORES"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05), "BOUND_ON_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00), "ALL_LOADS"},

        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00), "SPLIT_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00), "SPLIT_STORES"},
    });
    pevg.show_header();
    //test_basic(pevg); return 0;

    int M = getenv("M", 4);
    int K = getenv("K", 256);
    int N = getenv("N", 24);
    test(pevg, M, K, N); return 0;

    test(pevg, 4*1000, 1024, 120);
    //test(pevg, 128, 1024, 120 * 5);
    //test(pevg, 128, 1024, 120 * 10);
    //test(pevg, 32, 1024, 4080);
    //test(pevg, 32, 4096, 4080);
    std::cout << ".............................................................." << std::endl;
    // test(pevg, 32, 4096, 4096);
    // test(pevg, 32, 4096, 11008);
    // test(pevg, 32, 11008, 4096);

    // test(pevg, 128, 1024, 256);
    // test(pevg, 128, 1024, 512);
    // test(pevg, 128, 1024, 1024);
    std::cout << ".............................................................." << std::endl;
    // test(pevg, M, K, N);
    return 0;

    return 0;
}
