
#include <cstdlib>
#include <memory>
#include <stdexcept>

#define JIT_DEBUG 1
#include "../include/jit.h"
#include "../include/linux_perf.hpp"
#include "omp.h"

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
#endif

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
#pragma omp parallel
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

//=================================================================================================================================================
class GemmRegBlocking : public jit_generator {
public:
    const int m_rows;
    const int m_cols;
    const bool m_preload_b;
    GemmRegBlocking(int rows, int cols) : m_rows(rows), m_cols(cols), m_preload_b(rows >= cols) {
        if (m_preload_b) {
            ASSERT(rows * cols + cols + 1 <= 16);
        } else {
            ASSERT(rows * cols + rows + 1 <= 16);
        }
        create_kernel("GemmRegBlocking");
    }

    struct CallArgs {
        const float* A;
        int64_t A_stride;  // stride in number of bytes
        const float* B;
        int64_t B_stride;  // stride in number of bytes
        float* C;
        int64_t C_stride;  // stride in number of bytes
        int64_t K;
        int64_t accumulate;
    };

    Xbyak::Ymm ymmC(int row, int col) {
        return Xbyak::Ymm(row * m_cols + col);
    }
    Xbyak::Ymm ymmB(int col) {
        if (m_preload_b)
            return Xbyak::Ymm(m_rows * m_cols + col);
        else
            return Xbyak::Ymm(m_rows * m_cols);
    }
    Xbyak::Ymm ymmA(int row) {
        if (m_preload_b)
            return Xbyak::Ymm(m_rows * m_cols + m_cols);
        else
            return Xbyak::Ymm(m_rows * m_cols + 1 + row);
    }

    void generate() override {
        auto A_ptr = abi_param2;
        auto A_stride = abi_param3;
        auto B_ptr = abi_param4;
        auto B_stride = abi_param5;

        mov(A_ptr, ptr[abi_param1 + offsetof(CallArgs, A)]);
        mov(A_stride, ptr[abi_param1 + offsetof(CallArgs, A_stride)]);
        mov(B_ptr, ptr[abi_param1 + offsetof(CallArgs, B)]);
        mov(B_stride, ptr[abi_param1 + offsetof(CallArgs, B_stride)]);

        // initilaize C
        {
            Xbyak::Label skip_load;
            auto reg_tmp = rax;
            for (int r = 0; r < m_rows; r++)
                for (int c = 0; c < m_cols; c++) {
                    auto ymm = ymmC(r, c);
                    vxorps(ymm, ymm, ymm);
                }

            mov(reg_tmp, ptr[abi_param1 + offsetof(CallArgs, accumulate)]);
            and_(reg_tmp, 1);
            jz(skip_load);
            {
                auto dst_ptr = r10;
                auto dst_stride = r11;
                mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, C)]);
                mov(dst_stride, ptr[abi_param1 + offsetof(CallArgs, C_stride)]);

                // load subC[m_rows, m_cols]
                for (int r = 0; r < m_rows; r++) {
                    for (int c = 0; c < m_cols; c++) {
                        vmovups(ymmC(r, c), ptr[dst_ptr + c * 32]);
                    }
                    add(dst_ptr, dst_stride);
                }
            }
            L(skip_load);
        }

        // loop over K
        //            B:    1 x cols regs
        // A : 1 regs C: rows x cols regs
        {
            Xbyak::Label loop_over_k;
            auto reg_k = r10;
            auto A_ptr3 = r11;

            auto loadA = [&](int r) {
                switch (r) {
                case 0:
                    vbroadcastss(ymmA(r), ptr[A_ptr]);
                    break;
                case 1:
                    vbroadcastss(ymmA(r), ptr[A_ptr + A_stride]);
                    break;
                case 2:
                    vbroadcastss(ymmA(r), ptr[A_ptr + 2 * A_stride]);
                    break;
                case 3:
                    vbroadcastss(ymmA(r), ptr[A_ptr3]);
                    break;
                case 4:
                    vbroadcastss(ymmA(r), ptr[A_ptr3 + A_stride]);
                    break;
                case 5:
                    vbroadcastss(ymmA(r), ptr[A_ptr3 + 2 * A_stride]);
                    break;
                default:
                    throw std::runtime_error("number of reg-blocking rows is not supported");
                }
            };

            if (m_rows > 3) {
                lea(A_ptr3, ptr[A_ptr + 2 * A_stride]);
                lea(A_ptr3, ptr[A_ptr3 + A_stride]);
            }
            mov(reg_k, ptr[abi_param1 + offsetof(CallArgs, K)]);

            align(64, false);
            L(loop_over_k);
            if (m_preload_b) {
                // preload B regs
                for (int c = 0; c < m_cols; c++)
                    vmovups(ymmB(c), ptr[B_ptr + c * 32]);

                lea(B_ptr, ptr[B_ptr + B_stride]);
                for (int r = 0; r < m_rows; r++) {
                    loadA(r);
                    for (int c = 0; c < m_cols; c++)
                        vfmadd231ps(ymmC(r, c), ymmA(r), ymmB(c));
                }

                lea(A_ptr, ptr[A_ptr + 4]);
                if (m_rows > 3) lea(A_ptr3, ptr[A_ptr3 + 4]);
            } else {
                // preload A regs
                for (int r = 0; r < m_rows; r++)
                    loadA(r);

                for (int c = 0; c < m_cols; c++) {
                    vmovups(ymmB(c), ptr[B_ptr + c * 32]);
                    for (int r = 0; r < m_rows; r++)
                        vfmadd231ps(ymmC(r, c), ymmA(r), ymmB(c));
                }

                lea(B_ptr, ptr[B_ptr + B_stride]);
                lea(A_ptr, ptr[A_ptr + 4]);
                if (m_rows > 3) lea(A_ptr3, ptr[A_ptr3 + 4]);
            }
            dec(reg_k);
            jnz(loop_over_k, T_NEAR);
        }

        // save C
        {
            auto dst_ptr = r10;
            auto dst_stride = r11;
            mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, C)]);
            mov(dst_stride, ptr[abi_param1 + offsetof(CallArgs, C_stride)]);
            for (int r = 0; r < m_rows; r++) {
                for (int c = 0; c < m_cols; c++) {
                    vmovups(ptr[dst_ptr + c * 32], ymmC(r, c));
                }
                add(dst_ptr, dst_stride);
            }
        }
        ret();
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

    auto* prefetch_B = B + 64 / sizeof(float) * 10;

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++, prefetch_B += B_stride) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 b1 = _mm256_loadu_ps(B + 8 * 1);
        __m256 b2 = _mm256_loadu_ps(B + 8 * 2);

        //_mm_prefetch(prefetch_B, _MM_HINT_T0);

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
        b = _mm256_loadu_ps(B + 8 * 0);
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c6 = _mm256_fmadd_ps(a1, b, c6);
        b = _mm256_loadu_ps(B + 8 * 1);
        c1 = _mm256_fmadd_ps(a0, b, c1);
        c7 = _mm256_fmadd_ps(a1, b, c7);
        b = _mm256_loadu_ps(B + 8 * 2);
        c2 = _mm256_fmadd_ps(a0, b, c2);
        c8 = _mm256_fmadd_ps(a1, b, c8);
        b = _mm256_loadu_ps(B + 8 * 3);
        c3 = _mm256_fmadd_ps(a0, b, c3);
        c9 = _mm256_fmadd_ps(a1, b, c9);
        b = _mm256_loadu_ps(B + 8 * 4);
        c4 = _mm256_fmadd_ps(a0, b, c4);
        ca = _mm256_fmadd_ps(a1, b, ca);
        b = _mm256_loadu_ps(B + 8 * 5);
        c5 = _mm256_fmadd_ps(a0, b, c5);
        cb = _mm256_fmadd_ps(a1, b, cb);
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

static int SN = getenv("SN", 16);

template <int rows, int prefetch_v = 16>
void brgemm_6x2(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    __m256 c0, c1, c2, c3, c4, c5;
    __m256 c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        auto* src = C;
        c0 = _mm256_loadu_ps(src + 8 * 0);
        c1 = _mm256_loadu_ps(src + 8 * 1);
        if (rows > 1) {
            src += C_stride;
            c2 = _mm256_loadu_ps(src + 8 * 0);
            c3 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 2) {
            src += C_stride;
            c4 = _mm256_loadu_ps(src + 8 * 0);
            c5 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 3) {
            src += C_stride;
            c6 = _mm256_loadu_ps(src + 8 * 0);
            c7 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 4) {
            src += C_stride;
            c8 = _mm256_loadu_ps(src + 8 * 0);
            c9 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 5) {
            src += C_stride;
            ca = _mm256_loadu_ps(src + 8 * 0);
            cb = _mm256_loadu_ps(src + 8 * 1);
        }
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

    const auto* pA3 = A + 3 * A_stride;
    const auto prefetch_stride = B_stride * prefetch_v;
    int k;
    for (k = 0; k < K; k++, B += B_stride, A++, pA3++) {
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 b1 = _mm256_loadu_ps(B + 8 * 1);

        _mm_prefetch(B + 16, _MM_HINT_T0);
        if (prefetch_v)
            _mm_prefetch(B + prefetch_stride, _MM_HINT_T0);

        __m256 a0 = _mm256_broadcast_ss(A);
        c0 = _mm256_fmadd_ps(a0, b0, c0);
        c1 = _mm256_fmadd_ps(a0, b1, c1);
        if (rows > 1) {
            a0 = _mm256_broadcast_ss(A + A_stride);
            c2 = _mm256_fmadd_ps(a0, b0, c2);
            c3 = _mm256_fmadd_ps(a0, b1, c3);
        }
        if (rows > 2) {
            a0 = _mm256_broadcast_ss(A + 2 * A_stride);
            c4 = _mm256_fmadd_ps(a0, b0, c4);
            c5 = _mm256_fmadd_ps(a0, b1, c5);
        }

        if (rows > 3) {
            a0 = _mm256_broadcast_ss(pA3);
            c6 = _mm256_fmadd_ps(a0, b0, c6);
            c7 = _mm256_fmadd_ps(a0, b1, c7);
        }
        if (rows > 4) {
            a0 = _mm256_broadcast_ss(pA3 + A_stride);
            c8 = _mm256_fmadd_ps(a0, b0, c8);
            c9 = _mm256_fmadd_ps(a0, b1, c9);
        }
        if (rows > 5) {
            a0 = _mm256_broadcast_ss(pA3 + 2 * A_stride);
            ca = _mm256_fmadd_ps(a0, b0, ca);
            cb = _mm256_fmadd_ps(a0, b1, cb);
        }
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c0);
    _mm256_storeu_ps(C + 8 * 1, c1);
    if (rows > 1) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c2);
        _mm256_storeu_ps(C + 8 * 1, c3);
    }
    if (rows > 2) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c4);
        _mm256_storeu_ps(C + 8 * 1, c5);
    }
    if (rows > 3) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c6);
        _mm256_storeu_ps(C + 8 * 1, c7);
    }
    if (rows > 4) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c8);
        _mm256_storeu_ps(C + 8 * 1, c9);
    }
    if (rows > 5) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, ca);
        _mm256_storeu_ps(C + 8 * 1, cb);
    }
}

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
void MM_ComputeBounded(const float* _A,
                       int A_stride,  // stride in number of element
                       const float* _B,
                       int B_stride,  // stride in number of element
                       float* _C,
                       int C_stride,  // stride in number of element
                       int M,
                       int K,
                       int N) {
    thread_local float scratch[1024 * 1024];
    constexpr int BK = 512;
    constexpr int BN = 480;
    for (int n0 = 0; n0 < N; n0 += BN) {
        int bN = std::min(N - n0, BN);
        const auto* A = _A;
        const auto* B = _B + n0;
        auto* C = _C + n0;
        for (int k = 0; k < K; k += BK, A += BK, B += BK * B_stride) {
            int bK = std::min(K - k, BK);
            auto* pA = A;
            auto* pC = C;
            int m;
            bool is_accumulate_C = (k > 0);

            float* repacked_B_n24 = scratch;
            float* repacked_B_n8 = scratch + (bN / 24) * (24 * bK);
            {
                // scratch buffer is resident in L2, repack process will exploit
                // HW-prefetcher to access source data from DDR contingously.
                for (int k = 0; k < bK; k++) {
                    const auto* src = B + k * B_stride;
                    auto* dst = repacked_B_n24 + 24 * k;
                    int n = 0;
                    for (; n + 24 <= bN; n += 24, src += 24, dst += bK * 24) {
                        auto b0 = _mm256_loadu_ps(src + 8 * 0);
                        auto b1 = _mm256_loadu_ps(src + 8 * 1);
                        auto b2 = _mm256_loadu_ps(src + 8 * 2);

                        _mm_prefetch(src + 16 * 10, _MM_HINT_T0);

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
            // re-use repacked B sub-matrix in L2 cache as long as we can.
            for (m = 0; m + 4 <= M; m += 4, pA += 4 * A_stride, pC += 4 * C_stride) {
                const auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + 24 <= bN; n += 24, pB += 24 * bK) {
                    brgemm_4x3<4>(pA, A_stride, pB, 24, pC + n, C_stride, bK, is_accumulate_C);
                }
                pB = repacked_B_n8;
                for (; n < bN; n += 8, pB += 8 * bK) {
                    brgemm_4x1<4>(pA, A_stride, pB, 8, pC + n, C_stride, bK, is_accumulate_C);
                }
            }
            // M tails
            for (; m < M; m++, pA += A_stride, pC += C_stride) {
                const auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + 24 <= bN; n += 24, pB += 24 * bK) {
                    brgemm_4x3<1>(pA, A_stride, pB, 24, pC + n, C_stride, bK, is_accumulate_C);
                }
                pB = repacked_B_n8;
                for (; n < bN; n += 8, pB += 8 * bK) {
                    brgemm_4x1<1>(pA, A_stride, pB, 8, pC + n, C_stride, bK, is_accumulate_C);
                }
            }
        }
    }
}
/*
between compute-bound & mem-bound cases, M is not big enough to reuse B many times,
the cost of B sub-matrix repack is not worthy, we will access B w/o repacking
this requires re-use one column of B matrix from L1-cache.

W/o changing strides of B, one column of B matrix may overflow L2-cache due to the nature of set-associative cache.
so BK should be small enough to not overflow L2-cache-sets.
*/

void MM_ComputeBounded_nocopy(const float* A,
                              int A_stride,  // stride in number of element
                              const float* B,
                              int B_stride,  // stride in number of element
                              float* C,
                              int C_stride,  // stride in number of element
                              int M,
                              int K,
                              int N) {
#define PREFETCH_V 16
    static int BK = getenv("BK", 54);
    for (int k = 0; k < K; k += BK, A += BK, B += BK * B_stride) {
        int bK = std::min(K - k, BK);
        for (int n = 0; n + 16 <= N; n += 16) {
            auto* pA = A;
            auto* pC = C;
            int m;
            if (n == 0) {
                // first column will prefetch the next column together with next few rows in same column
                for (m = 0; m + 6 <= M; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
                    brgemm_6x2<6, PREFETCH_V>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                }
                if (m < M) {
                    switch (M - m) {
                    case 5:
                        brgemm_6x2<5, PREFETCH_V>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 4:
                        brgemm_6x2<4, PREFETCH_V>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 3:
                        brgemm_6x2<3, PREFETCH_V>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 2:
                        brgemm_6x2<2, PREFETCH_V>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 1:
                        brgemm_6x2<1, PREFETCH_V>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    }
                }
            } else {
                // the rest columns only prefetch the next columns
                for (m = 0; m + 6 <= M; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
                    brgemm_6x2<6, 0>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                }
                if (m < M) {
                    switch (M - m) {
                    case 5:
                        brgemm_6x2<5, 0>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 4:
                        brgemm_6x2<4, 0>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 3:
                        brgemm_6x2<3, 0>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 2:
                        brgemm_6x2<2, 0>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    case 1:
                        brgemm_6x2<1, 0>(pA, A_stride, B + n, B_stride, pC + n, C_stride, bK, k > 0);
                        break;
                    }
                }
            }
        }
    }
#undef PREFETCH_V
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
void MM_MemoryBounded(const float* A,
                      int A_stride,  // stride in number of element
                      const float* B,
                      int B_stride,  // stride in number of element
                      float* C,
                      int C_stride,  // stride in number of element
                      int M,
                      int K,
                      int N) {
    //
    {
        auto c0 = _mm256_setzero_ps();
        auto* pC = C;
        for (int m = 0; m < M; m++, pC += C_stride) {
            for (int n = 0; n < N; n += 8) {
                _mm256_storeu_ps(pC + n, c0);
            }
        }
    }
    ASSERT(((K % 4) == 0) && ((N % 8) == 0));

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
            int n = 0;
            for (; (n + 8 * 4) <= N; n += 8 * 4, pB += 8 * 4, pB2 += 8 * 4) {
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
            for (; n < N; n += 8, pB += 8, pB2 += 8) {
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

        if (M < 6) {
            MM_MemoryBounded(A, A_stride, B + n0, B_stride, C + n0, C_stride, M, K, n1 - n0);
        } else if (M <= 60) {
            MM_ComputeBounded_nocopy(A, A_stride, B + n0, B_stride, C + n0, C_stride, M, K, n1 - n0);
        } else {
            MM_ComputeBounded(A, A_stride, B + n0, B_stride, C + n0, C_stride, M, K, n1 - n0);
        }
    }
}

int test(LinuxPerf::PerfEventGroup* pevg, int64_t M, int64_t K, int64_t N) {
    MMtestCase mtc(M, K, N);
    std::stringstream ss;
    ss << "MM_brgemm(" << M << "," << K << "," << N << ")";
    int repeat = 10;
    auto nthr = omp_get_max_threads();
    for (int r = 0; r < repeat; r++) {
        // clear cache
        mtc.clr_cache();
        // profile once
        auto pmc = pevg->rdpmc(
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(gemm, m) {
    py::class_<GemmRegBlocking>(m, "GemmRegBlocking")
        .def(py::init<int, int>())  // GemmRegBlocking(int rows, int cols_in_avx2_regs)
        .def("__call__", [](GemmRegBlocking& self, py::array_t<float> A, py::array_t<float> B, py::array_t<float> C, int ROUNDS) {
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

            for(int r = 0; r < ROUNDS; r++)
                self(&args);
        }, py::arg(), py::arg(), py::arg(), py::arg("ROUNDS") = 1);

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

        MM_brgemm(static_cast<float*>(bufA.ptr), K, static_cast<float*>(bufB.ptr), N, static_cast<float*>(bufC.ptr), N, M, K, N, false);
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

        //DEBUG_LOG(M, K, N);
        //DEBUG_LOG(bufA.strides[0], bufB.strides[0], bufC.strides[0]);

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
