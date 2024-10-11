#include "../include/linux_perf.hpp"
#include "../include/misc.hpp"
#include "omp.h"

/*

$ lscpu
  Model name:            Intel(R) Core(TM) i9-14900K
  NUMA node0 CPU(s):     0-31

P-cores has hyper-threading 0,2,4,6,8,10,12,14,

$ lscpu --all --extended
CPU NODE SOCKET CORE L1d:L1i:L2:L3 ONLINE    MAXMHZ   MINMHZ      MHZ
  0    0      0    0 0:0:0:0          yes 5700.0000 800.0000  800.000
  1    0      0    0 0:0:0:0          yes 5700.0000 800.0000  800.000
  2    0      0    1 4:4:1:0          yes 5700.0000 800.0000 4731.219
  3    0      0    1 4:4:1:0          yes 5700.0000 800.0000  800.000
  4    0      0    2 8:8:2:0          yes 5700.0000 800.0000  800.000
  5    0      0    2 8:8:2:0          yes 5700.0000 800.0000  800.000
  6    0      0    3 12:12:3:0        yes 5700.0000 800.0000  800.000
  7    0      0    3 12:12:3:0        yes 5700.0000 800.0000  800.000
  8    0      0    4 16:16:4:0        yes 6000.0000 800.0000  800.000
  9    0      0    4 16:16:4:0        yes 6000.0000 800.0000  800.000
 10    0      0    5 20:20:5:0        yes 6000.0000 800.0000  800.000
 11    0      0    5 20:20:5:0        yes 6000.0000 800.0000  800.000
 12    0      0    6 24:24:6:0        yes 5700.0000 800.0000 5699.759
 13    0      0    6 24:24:6:0        yes 5700.0000 800.0000  800.000
 14    0      0    7 28:28:7:0        yes 5700.0000 800.0000 5700.236
 15    0      0    7 28:28:7:0        yes 5700.0000 800.0000  800.000

 16    0      0    8 32:32:8:0        yes 4400.0000 800.0000  800.000
 17    0      0    9 33:33:8:0        yes 4400.0000 800.0000 4399.160
 18    0      0   10 34:34:8:0        yes 4400.0000 800.0000  800.000
 19    0      0   11 35:35:8:0        yes 4400.0000 800.0000  800.000
 20    0      0   12 36:36:9:0        yes 4400.0000 800.0000 4392.290
 21    0      0   13 37:37:9:0        yes 4400.0000 800.0000  800.000
 22    0      0   14 38:38:9:0        yes 4400.0000 800.0000  800.000
 23    0      0   15 39:39:9:0        yes 4400.0000 800.0000  800.000
 24    0      0   16 40:40:10:0       yes 4400.0000 800.0000  800.000
 25    0      0   17 41:41:10:0       yes 4400.0000 800.0000  800.000
 26    0      0   18 42:42:10:0       yes 4400.0000 800.0000  800.000
 27    0      0   19 43:43:10:0       yes 4400.0000 800.0000  800.000
 28    0      0   20 44:44:11:0       yes 4400.0000 800.0000  800.000
 29    0      0   21 45:45:11:0       yes 4400.0000 800.0000  800.000
 30    0      0   22 46:46:11:0       yes 4400.0000 800.0000  800.000
 31    0      0   23 47:47:11:0       yes 4400.0000 800.0000  800.000


$ g++ -fopenmp -O2 -march=native ./test.cpp

$ numactl -C0,2,4,6,8,10,12,14 ./a.out
(    1.00 GB buff    11.10 GB/s x 8 threads) =    88.83 GB/s



 SM=1 CLC=1  numactl -C0,2,4,6,8,10,12,14 ./a.out 
ENV:     SM = 1 
ENV:     CLC = 1 
ENV:     M = 1 (default)
ENV:     N = 32768 (default)
ENV:     K = 4096 (default)
 +0.000 ms test.cpp:262 main()  nthr=8
 +138.341 ms test.cpp:273 main()  A matrix size = 0.016384 MB(1,000,000 bytes)
 +0.009 ms test.cpp:274 main()  C matrix size = 0.131072 MB(1,000,000 bytes)
 +0.002 ms test.cpp:275 main()  Bf32 matrix size = 536.871 MB(1,000,000 bytes)
 +0.002 ms test.cpp:276 main()  Bi8 matrix size = 134.218 MB(1,000,000 bytes)
[LINUX_PERF:239]  LINUX_PERF is unset, example: LINUX_PERF=dump,switch-cpu,L2_MISS=0x10d1
#0:HW_CPU_CYCLES, HW_INSTRUCTIONS, 
============== round 0 ===================
    106699463,        14708990,  [mm_avx2_f32_reg_x1] 18799.756 us CPU:5.68(GHz) CPI:7.25 CPK:50.9x2097152 MemBW: 28.56(GB/s)
     31654605,        16789231,  [mm_avx2_f32_reg_x1s] 5607.244 us CPU:5.65(GHz) CPI:1.89 CPK:15.1x2097152 MemBW: 95.75(GB/s)
     10542639,        18884878,  [mm_avx2_i8_reg_x1s] 1928.471 us CPU:5.47(GHz) CPI:0.56 CPK:5.0x2097152 MemBW: 69.60(GB/s)
      9308857,        11541426,  [mm_avx2_i8_reg_x4s] 1726.776 us CPU:5.39(GHz) CPI:0.81 CPK:4.4x2097152 MemBW: 77.73(GB/s)
============== round 1 ===================
    102877517,        15231751,  [mm_avx2_f32_reg_x1] 18181.175 us CPU:5.66(GHz) CPI:6.75 CPK:49.1x2097152 MemBW: 29.53(GB/s)
     31864302,        16791526,  [mm_avx2_f32_reg_x1s] 5611.021 us CPU:5.68(GHz) CPI:1.90 CPK:15.2x2097152 MemBW: 95.68(GB/s)
     10701120,        18885931,  [mm_avx2_i8_reg_x1s] 1931.835 us CPU:5.54(GHz) CPI:0.57 CPK:5.1x2097152 MemBW: 69.48(GB/s)
      9032548,        11538058,  [mm_avx2_i8_reg_x4s] 1668.182 us CPU:5.41(GHz) CPI:0.78 CPK:4.3x2097152 MemBW: 80.46(GB/s)


*/

// AVX2: 16x 256bit register = 8xfp32
#define SIMD_WIDTH 8

thread_local LinuxPerf::PerfEventGroup pevg({
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},

    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x04, 0x04),"STALLS_TOTAL"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x06, 0x06),"STALLS_L3_MISS"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x05, 0x05),"STALLS_L2_MISS"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x40, 0x02),"BOUND_ON_STORES"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05), "BOUND_ON_LOADS"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00), "ALL_LOADS"},
    
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x01, 0x00), "XSNP_MISS"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x02, 0x00), "XSNP_NO_FWD"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x04, 0x00), "XSNP_FWD"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x08, 0x00), "XSNP_NONE"},
    
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00), "SPLIT_LOADS"},
    //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00), "SPLIT_STORES"},
});

auto SM = getenv("SM", 1);

// A: MxN   B: KxN  C: MxN
void mm_avx2_f32_reg_x1(tensorND<float> A, tensorND<float> B, tensorND<float> C) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = C.size(1);
    auto nthr = omp_get_max_threads();

    auto reg_blk_N = SIMD_WIDTH;

    ASSERT(((N/reg_blk_N) % nthr) == 0);
    auto BN = ((N/reg_blk_N) / nthr) * reg_blk_N;

    auto B_size_per_thr = K * BN * sizeof(float);

    auto* pA0 = A.ptr();
    auto* pB0 = B.ptr();
    auto* pC0 = C.ptr();
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        auto n0 = BN * ithr;
        auto n1 = n0 + BN;
        auto stride_B = B.strides()[0] * SM;
        for(int n = n0; n < n1; n += reg_blk_N) {
            auto * pB = pB0 + n;
            auto acc0 = _mm256_setzero_ps();

            // sub-block matmul: [1 x K] x [K, SIMD_WIDTH] => [1 x SIMD_WIDTH]
            for(int k = 0; k < K; k++, pB += stride_B) {
                // compute ~ 4.7 cycles (FMA latency 4)
                // +load_from DDR ~50 cycles
                auto b0 = _mm256_loadu_ps(pB);
                auto a0 = _mm256_broadcast_ss(pA0 + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                //acc0 = _mm256_xor_ps(acc0, b0);
            }

            _mm256_storeu_ps(pC0 + n, acc0);
        }
    }
}

void mm_avx2_f32_reg_x1s(tensorND<float> A, tensorND<float> B, tensorND<float> C) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = C.size(1);
    auto nthr = omp_get_max_threads();

    auto reg_blk_N = SIMD_WIDTH;

    ASSERT(((N/reg_blk_N) % nthr) == 0);
    auto BN = ((N/reg_blk_N) / nthr) * reg_blk_N;

    auto* pA0 = A.ptr();
    auto* pC0 = C.ptr();
    auto stride_B = SIMD_WIDTH * SM;
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        auto n0 = BN * ithr;
        auto n1 = n0 + BN;
        auto* pB = B.ptr() + ithr * (K * BN);
        for(int n = n0; n < n1; n += reg_blk_N) {
            auto acc0 = _mm256_setzero_ps();

            // sub-block matmul: [1 x K] x [K, SIMD_WIDTH] => [1 x SIMD_WIDTH]
            for(int k = 0; k < K; k++) {
                // compute ~4 cycles (FMA latency 4)
                // 
                auto b0 = _mm256_loadu_ps(pB);
                pB += stride_B;
                auto a0 = _mm256_broadcast_ss(pA0 + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                //acc0 = _mm256_xor_ps(acc0, b0);
            }

            _mm256_storeu_ps(pC0 + n, acc0);
        }
    }
}

void mm_avx2_i8_reg_x1s(tensorND<float> A, tensorND<int8_t> B, tensorND<float> C) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = C.size(1);
    auto nthr = omp_get_max_threads();

    auto reg_blk_N = SIMD_WIDTH;

    ASSERT(((N/reg_blk_N) % nthr) == 0);
    auto BN = ((N/reg_blk_N) / nthr) * reg_blk_N;

    auto* pA0 = A.ptr();
    auto* pC0 = C.ptr();
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        auto n0 = BN * ithr;
        auto n1 = n0 + BN;
        auto* pB = B.ptr() + ithr*(K * BN);
        auto stride_B = SIMD_WIDTH * SM;
        for(int n = n0; n < n1; n += reg_blk_N) {
            auto acc0 = _mm256_setzero_ps();

            // sub-block matmul: [1 x K] x [K, SIMD_WIDTH * 4] => [1 x SIMD_WIDTH * 4]
            for(int k = 0; k < K; k++, pB += stride_B) {
                auto b0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(*reinterpret_cast<int64_t*>(pB))));

                auto a0 = _mm256_broadcast_ss(pA0 + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
            }

            _mm256_storeu_ps(pC0 + n, acc0);
        }
    }
}

template<int PREFETCH_ADV = 0>
void mm_avx2_i8_reg_x4s(tensorND<float> A, tensorND<int8_t> B, tensorND<float> C) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = C.size(1);
    auto nthr = omp_get_max_threads();

    auto reg_blk_N = SIMD_WIDTH * 4;

    ASSERT(((N/reg_blk_N) % nthr) == 0);
    auto BN = ((N/reg_blk_N) / nthr) * reg_blk_N;

    auto* pA0 = A.ptr();
    auto* pC0 = C.ptr();
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        auto n0 = BN * ithr;
        auto n1 = n0 + BN;
        auto* pB = B.ptr() + ithr*(K * BN);
        auto stride_B = SIMD_WIDTH * 4 * SM;
        for(int n = n0; n < n1; n += reg_blk_N) {
            auto acc0 = _mm256_setzero_ps();
            auto acc1 = _mm256_setzero_ps();
            auto acc2 = _mm256_setzero_ps();
            auto acc3 = _mm256_setzero_ps();

            // sub-block matmul: [1 x K] x [K, SIMD_WIDTH * 4] => [1 x SIMD_WIDTH * 4]
            for(int k = 0; k < K; k++, pB += stride_B) {

                auto bi80 = _mm_loadu_si128((const __m128i_u*)pB);
                auto bi81 = _mm_loadu_si128((const __m128i_u*)(pB + SIMD_WIDTH));
                auto bi82 = _mm_loadu_si128((const __m128i_u*)(pB + SIMD_WIDTH*2));
                auto bi83 = _mm_loadu_si128((const __m128i_u*)(pB + SIMD_WIDTH*3));

                auto b0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bi80));
                auto b1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bi81));
                auto b2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bi82));
                auto b3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bi83));

                auto a0 = _mm256_broadcast_ss(pA0 + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a0, b1, acc1);
                acc2 = _mm256_fmadd_ps(a0, b2, acc2);
                acc3 = _mm256_fmadd_ps(a0, b3, acc3);

                if (PREFETCH_ADV > 0)
                    _mm_prefetch(pB + PREFETCH_ADV, _MM_HINT_T2);
            }

            _mm256_storeu_ps(pC0 + n, acc0);
            _mm256_storeu_ps(pC0 + n + SIMD_WIDTH, acc1);
            _mm256_storeu_ps(pC0 + n + SIMD_WIDTH*2, acc2);
            _mm256_storeu_ps(pC0 + n + SIMD_WIDTH*3, acc3);
        }
    }
}


template<int PREFETCH_ADV = 0>
void mm_avx2_i4_reg_x4s(tensorND<float> A, tensorND<int8_t> B, tensorND<float> C) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = C.size(1);
    auto nthr = omp_get_max_threads();

    auto reg_blk_N = SIMD_WIDTH * 4;

    ASSERT(((N/reg_blk_N) % nthr) == 0);
    auto BN = ((N/reg_blk_N) / nthr) * reg_blk_N;

    auto* pA0 = A.ptr();
    auto* pC0 = C.ptr();
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        auto n0 = BN * ithr;
        auto n1 = n0 + BN;
        auto* pB = B.ptr() + ithr*(K * BN / 2);
        auto stride_B = SIMD_WIDTH * 4 * SM / 2;
        for(int n = n0; n < n1; n += reg_blk_N) {
            auto acc0 = _mm256_setzero_ps();
            auto acc1 = _mm256_setzero_ps();
            auto acc2 = _mm256_setzero_ps();
            auto acc3 = _mm256_setzero_ps();

            // sub-block matmul: [1 x K] x [K, SIMD_WIDTH * 4] => [1 x SIMD_WIDTH * 4]
            auto vmask = _mm256_set1_epi32(0xf0);
            for(int k = 0; k < K; k++, pB += stride_B) {
                auto bi80 = _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i_u*)pB));
                auto bi81 = _mm256_cvtepu8_epi32(_mm_loadu_si128((const __m128i_u*)(pB + SIMD_WIDTH)));

                auto b0 = _mm256_cvtepi32_ps(_mm256_srli_epi32(bi80, 4));
                auto b1 = _mm256_cvtepi32_ps(_mm256_srli_epi32(bi81, 4));
                auto b2 = _mm256_cvtepi32_ps(_mm256_and_si256(bi80, vmask));
                auto b3 = _mm256_cvtepi32_ps(_mm256_and_si256(bi81, vmask));

                auto a0 = _mm256_broadcast_ss(pA0 + k);
                acc0 = _mm256_fmadd_ps(a0, b0, acc0);
                acc1 = _mm256_fmadd_ps(a0, b1, acc1);
                acc2 = _mm256_fmadd_ps(a0, b2, acc2);
                acc3 = _mm256_fmadd_ps(a0, b3, acc3);

                if (PREFETCH_ADV > 0)
                    _mm_prefetch(pB + PREFETCH_ADV, _MM_HINT_T2);
            }

            _mm256_storeu_ps(pC0 + n, acc0);
            _mm256_storeu_ps(pC0 + n + SIMD_WIDTH, acc1);
            _mm256_storeu_ps(pC0 + n + SIMD_WIDTH*2, acc2);
            _mm256_storeu_ps(pC0 + n + SIMD_WIDTH*3, acc3);
        }
    }
}

int main() {
    
    auto CLC = getenv("CLC", 1);
    size_t M = getenv("M", 1);
    size_t N = getenv("N", 4096*8);
    size_t K = getenv("K", 4096);
    auto nthr = omp_get_max_threads();
    ECOUT("nthr=", nthr);
    tensorND<float> A({M, K}, 0.0f);
    tensorND<float> Bf32({K, N}, 0.0f);
    tensorND<float> C({M, N}, 0.0f);
    tensorND<int8_t> Bi8({K, N}, 0);
    tensorND<int8_t> Bi4({K, N/2}, 0);

    auto A_bytes = A.numel() * sizeof(float);
    auto C_bytes = C.numel() * sizeof(float);
    auto B_bytes = Bf32.numel() * sizeof(float);
    auto Bi8_bytes = Bi8.numel() * sizeof(int8_t);
    auto Bi4_bytes = Bi4.numel() * sizeof(int8_t);

    ECOUT("====================================");
    ECOUT("   A matrix size = ", A_bytes * 1e-6, " MB(1,000,000 bytes)");
    ECOUT("   C matrix size = ", C_bytes * 1e-6, " MB(1,000,000 bytes)");
    ECOUT("Bf32 matrix size = ", B_bytes * 1e-6, " MB(1,000,000 bytes)");
    ECOUT(" Bi8 matrix size = ", Bi8_bytes * 1e-6, " MB(1,000,000 bytes)");
    ECOUT(" Bi4 matrix size = ", Bi4_bytes * 1e-6, " MB(1,000,000 bytes)");

    auto clr_cache = [&](){
        if (!CLC) return;
        #pragma omp parallel
        {
            auto* src = reinterpret_cast<int8_t*>(Bf32.ptr());
            for(size_t i = 0; i < B_bytes; i += 64) {
                _mm_clflush(src + i);
            }
            src = reinterpret_cast<int8_t*>(Bi8.ptr());
            for(size_t i = 0; i < Bi8_bytes; i += 64) {
                _mm_clflush(src + i);
            }
        }
    };

    pevg.show_header();
    for(int round = 0; round < 2; round++) {
        std::cout << "============== round " << round << " ===================" << std::endl;
        clr_cache();
        pevg.rdpmc(
            [&]() { mm_avx2_f32_reg_x1(A, Bf32, C); },
            "mm_avx2_f32_reg_x1",
            K*(N/SIMD_WIDTH)/nthr,
            [&](uint64_t duration_ns, uint64_t* pmc, char*& log){
                auto bw = B_bytes * 1.0/duration_ns;
                log += sprintf(log, " MemBW: %.2f(GB/s)", bw);
            });

        clr_cache();
        pevg.rdpmc(
            [&]() { mm_avx2_f32_reg_x1s(A, Bf32, C); },
            "mm_avx2_f32_reg_x1s",
            K*(N/SIMD_WIDTH)/nthr,
            [&](uint64_t duration_ns, uint64_t* pmc, char*& log){
                auto bw = B_bytes * 1.0/duration_ns;
                log += sprintf(log, " MemBW: %.2f(GB/s)  \t+ exploit HW-pefetcher", bw);
            });

        clr_cache();
        pevg.rdpmc(
            [&]() { mm_avx2_i8_reg_x1s(A, Bi8, C); },
            "mm_avx2_i8_reg_x1s",
            K*(N/SIMD_WIDTH)/nthr,
            [&](uint64_t duration_ns, uint64_t* pmc, char*& log){
                auto bw = Bi8_bytes * 1.0/duration_ns;
                log += sprintf(log, " MemBW: %.2f(GB/s)  \t+ Weight-Compressed INT8", bw);
            });

        clr_cache();
        pevg.rdpmc(
            [&]() { mm_avx2_i8_reg_x4s<0>(A, Bi8, C); },
            "mm_avx2_i8_x4s<0>",
            K*(N/(SIMD_WIDTH*4))/nthr,
            [&](uint64_t duration_ns, uint64_t* pmc, char*& log){
                auto bw = Bi8_bytes * 1.0/duration_ns;
                log += sprintf(log, " MemBW: %.2f(GB/s)  \t+ register-blocking", bw);
            });

        // prefetch is helpful
        clr_cache();
        pevg.rdpmc(
            [&]() { mm_avx2_i8_reg_x4s<4096>(A, Bi8, C); },
            "mm_avx2_i8_x4s<4096>",
            K*(N/(SIMD_WIDTH*4))/nthr,
            [&](uint64_t duration_ns, uint64_t* pmc, char*& log){
                auto bw = Bi8_bytes * 1.0/duration_ns;
                log += sprintf(log, " MemBW: %.2f(GB/s)  \t+ SW-prefetch across page boundary ", bw);
            });

        clr_cache();
        pevg.rdpmc(
            [&]() { mm_avx2_i4_reg_x4s<4096>(A, Bi4, C); },
            "mm_avx2_i4_x4s<4096>",
            K*(N/(SIMD_WIDTH*4))/nthr,
            [&](uint64_t duration_ns, uint64_t* pmc, char*& log){
                auto bw = Bi4_bytes * 1.0/duration_ns;
                log += sprintf(log, " MemBW: %.2f(GB/s)  \t+ Weight-Compressed INT4", bw);
            });

    }
    
    return 0;
}