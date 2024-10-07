#include "cuda_utils.h"
/*
nvcc --generate-line-info --keep --keep-dir build -O2 max_gflops.cu

################## 91% peak performance is reached ##################
## because we use (32,32) as thread block size
## we need M >= 32*(num_SM) = 512 to fill all SM work to do and reach peak performance




$ M=1024 K=4096000 N=32 ./a.exe
 +0.000 ms cuda_utils.h:20 CUDADevice()  cudaSetDevice(0)
 cudaGetDeviceProperties(..., 0 ) :
         totalGlobalMem     : 8589672448 (8191 MB)
         sharedMemPerBlock  : 49152
         regsPerBlock       : 65536
         warpSize           : 32
         memPitch           : 2147483647
         maxThreadsPerBlock : 1024
         totalConstMem      : 65536
         major          : 6
         minor          : 1
         clockRate              : 1645000(KHz)
         multiProcessorCount    : 16 (each SM has 128 CUDA-cores)
         kernelExecTimeoutEnabled: 1
         integrated         : 0
         canMapHostMemory   : 1
         computeMode        : 0
         ... peak performance        : 6.738(TFLOP/s)
ENV:     M = 1024
ENV:     K = 4096000
ENV:     N = 32
 +103.462 ms max_gflops.cu:225 main()  test_max_gflops(1024,32,4096000)
gridDim=dim3{1, 32, 1} blockDim=dim3{32, 32, 1}
ENV:     CUDATIMEIT =
cuda_timeit #0 test_max_gflops:119 sgemm_max_gflops x 3  0(bytes) 4294967296000(flops)
 [AutoCUDATimer # 0] @host   5.750 ms | @device (+  0.000 us) 734.235 ms   5.850 TFLOP/s          sgemm_max_gflops (test_max_gflops:0) 0.000 MB 4294.967 Gflops
 [AutoCUDATimer # 0] @host   3.900 us | @device (+  3.072 us) 699.881 ms   6.137 TFLOP/s          sgemm_max_gflops (test_max_gflops:0) 0.000 MB 4294.967 Gflops
 [AutoCUDATimer # 0] @host   1.800 us | @device (+  2.048 us) 699.818 ms   6.137 TFLOP/s          sgemm_max_gflops (test_max_gflops:0) 0.000 MB 4294.967 Gflops
 +2.136 sec max_gflops.cu:142 test_max_gflops()  ==========SM statistics:==========
 +0.022 ms max_gflops.cu:149 test_max_gflops()  SM 0 clock_range = 439668072864~440811145023 duration: 1143072159 elements: 2048 FMA:117 (FMA/cycle)
 +0.013 ms max_gflops.cu:149 test_max_gflops()  SM 1 clock_range = 441001454700~442144564657 duration: 1143109957 elements: 2048 FMA:117 (FMA/cycle)
 +0.009 ms max_gflops.cu:149 test_max_gflops()  SM 2 clock_range = 440748883407~441891746127 duration: 1142862720 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 3 clock_range = 440444968141~441588077414 duration: 1143109273 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 4 clock_range = 439668072844~440811101779 duration: 1143028935 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 5 clock_range = 441001454690~442144563310 duration: 1143108620 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 6 clock_range = 440748883389~441891895607 duration: 1143012218 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 7 clock_range = 440444968133~441588097121 duration: 1143128988 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 8 clock_range = 439668072867~440811221853 duration: 1143148986 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 9 clock_range = 441001454712~442144533160 duration: 1143078448 elements: 2048 FMA:117 (FMA/cycle)
 +0.008 ms max_gflops.cu:149 test_max_gflops()  SM 10 clock_range = 440748883420~441891767179 duration: 1142883759 elements: 2048 FMA:117 (FMA/cycle)
 +0.007 ms max_gflops.cu:149 test_max_gflops()  SM 11 clock_range = 440444968163~441588103576 duration: 1143135413 elements: 2048 FMA:117 (FMA/cycle)
 +0.007 ms max_gflops.cu:149 test_max_gflops()  SM 12 clock_range = 439668072851~440811098886 duration: 1143026035 elements: 2048 FMA:117 (FMA/cycle)
 +0.007 ms max_gflops.cu:149 test_max_gflops()  SM 13 clock_range = 441001454711~442144599108 duration: 1143144397 elements: 2048 FMA:117 (FMA/cycle)
 +0.007 ms max_gflops.cu:149 test_max_gflops()  SM 14 clock_range = 440748883410~441891787516 duration: 1142904106 elements: 2048 FMA:117 (FMA/cycle)
 +0.007 ms max_gflops.cu:149 test_max_gflops()  SM 15 clock_range = 440444968157~441588130951 duration: 1143162794 elements: 2048 FMA:117 (FMA/cycle)
 +0.125 ms max_gflops.cu:151 test_max_gflops()   clock_overall_dur = 1143162794
 +0.015 ms max_gflops.cu:152 test_max_gflops()   GPU_avg_frequency = 1.60712 (GHz)
 +0.008 ms max_gflops.cu:153 test_max_gflops()   average FMA = 117.409 (FMA/SM/cycle)
 +0.007 ms max_gflops.cu:154 test_max_gflops()   average FMA = 3019.05 (GFMA/s)
 +11.174 ms misc.hpp:535 ~ChromeTraceDumpper()  dumpped 169.52(KB) to ct.json
 +0.519 ms cuda_utils.h:46 ~CUDADevice()  cudaDeviceReset()

*/

__forceinline__ __device__ unsigned get_warpid() {
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned get_smid() {
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

#define FMACNT 32
// mapping between threadIdx & (x,y) are transposed
__global__ void sgemm_max_gflops(uint64_t * C, size_t M, size_t N, size_t K) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N) {
        auto* pdata = C + (y*N + x)*8;
        pdata[0] = blockIdx.x;
        pdata[1] = blockIdx.y;
        pdata[2] = get_smid();

        float tmp[FMACNT] = {0};
        float a = y;
        float b = x;
        pdata[3] = clock64();
        for (int k = 0; k < K; ++k) {
            for(int c = 0; c < FMACNT; c++)
                tmp[c] = fma(a, b, tmp[c]);
        }
        pdata[4] = clock64() - pdata[3];

        pdata[5] = threadIdx.x;
        pdata[6] = threadIdx.y;
        pdata[7] = get_warpid();

        // to prevent optimization
        if (tmp[0] == 1.2134f) {
            float csum = 0;
            for(int c = 0; c < FMACNT; c++)
                csum += tmp[c];
            printf("%f", csum);
        }
    }
}

void test_max_gflops(size_t M, size_t N, size_t K) {
    tensorND<uint64_t> C({M, N, 8}, 0);

    dim3 gridDim(CEIL_DIV(N, WRAP_SIZE), CEIL_DIV(M, WRAP_SIZE));
    dim3 blockDim(WRAP_SIZE, WRAP_SIZE);

    std::cout << "gridDim=" << gridDim << " blockDim=" << blockDim << std::endl;

    auto flops = M*N*K*2.0*FMACNT;
    auto avg_dur_ns = cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_max_gflops<<<gridDim, blockDim>>>(C.to_gpu(), M, N, K);
    }, __func__, __LINE__, "sgemm_max_gflops", 0, flops, 3);

    auto* pdata = C.to_cpu();

    // calibrate all clocks from different SMs
    std::vector<uint64_t> clock_min(128, std::numeric_limits<uint64_t>::max());
    std::vector<uint64_t> clock_max(128, std::numeric_limits<uint64_t>::min());
    std::vector<uint64_t> element_cnt(128, 0);
    uint64_t sm_cnt = 0;
    for(int i = 0; i < M*N; i++, pdata += 8) {
        auto& blk_x = pdata[0];
        auto& blk_y = pdata[1];
        auto& smid = pdata[2];
        auto& clk_start = pdata[3];
        auto& clk_dur = pdata[4];
        sm_cnt = std::max(sm_cnt, smid+1);
        if (clk_dur > 0) {
            clock_min[smid] = std::min(clock_min[smid], clk_start);
            clock_max[smid] = std::max(clock_max[smid], clk_start + clk_dur);
            element_cnt[smid] ++;
        }
        //ECOUT("SM ", smid , blk_x, ",", blk_y, " clock_start= ", clk_start, ", ", clk_dur);
    }
    ECOUT("==========SM statistics:==========");
    uint64_t clock_overall_dur = std::numeric_limits<uint64_t>::min();
    for(uint64_t smid = 0; smid < sm_cnt; smid++) {
        auto clock_dur = clock_max[smid] - clock_min[smid];
        clock_overall_dur = std::max(clock_overall_dur, clock_dur);
        ECOUT("SM ", smid , " clock_range = ", clock_min[smid], "~", clock_max[smid],
              " duration: ", clock_dur, " elements: ", element_cnt[smid],
              " FMA:", (FMACNT * K * element_cnt[smid])/clock_dur, " (FMA/cycle)");
    }
    ECOUT(" clock_overall_dur = ", clock_overall_dur);
    ECOUT(" GPU_avg_frequency = ", clock_overall_dur / avg_dur_ns, " (GHz)");
    ECOUT(" average FMA = ", double(FMACNT)*K*M*N/sm_cnt/clock_overall_dur, " (FMA/SM/cycle)");
    ECOUT(" average FMA = ", double(FMACNT)*K*M*N/avg_dur_ns, " (GFMA/s)");

    pdata = C.to_cpu();
    for(int i = 0; i < M*N; i++, pdata += 8) {
        auto& smid = pdata[2];
        auto& clk_start = pdata[3];
        clk_start -= clock_min[smid];
    }

    ChromeTraceDumpper dumpper("ct.json");
    pdata = C.to_cpu();
    // 32 threads from same warp (if block-size in X direction is larger than 32)
    struct warp_info {
        int64_t blk_x = -1;
        int64_t blk_y = -1;
        uint64_t smid = 0;
        uint64_t clk_start = 0;
        uint64_t clk_dur = 0;
        uint64_t thr_x0 = 0;
        uint64_t thr_y0 = 0;
        uint64_t warpid = 0;

        uint64_t thr_cnt = 0;
    } warp;
    auto dump = [&](warp_info* pw) {
        if (pw->thr_cnt > 0) {
            std::stringstream ss;
            std::stringstream ss_tid;
            ss << "block(" << pw->blk_x <<"," << pw->blk_y << ")";
            //ss_tid << "thr(" << blk_x <<"," << blk_y << ")(" << thr_x0 << "+" << thr_cnt << "," << thr_y0 << ")";
            //auto end = tsc.tsc_to_usec(d.tsc_end);
            dumpper.phX(ss.str(), "", 
                std::string("SM_") + std::to_string(pw->smid),
                std::string("warp_") + std::to_string(pw->warpid),
                pw->clk_start, pw->clk_dur,
                {
                    {"thr_x0",std::to_string(pw->thr_x0) + "+" + std::to_string(pw->thr_cnt)},
                    {"thr_y0",std::to_string(pw->thr_y0)},
                    {"OPS/cycle", std::to_string(double(FMACNT * K)/pw->clk_dur)}
                });
        }
    };
    for (int n = 0; n < M*N; n++, pdata += 8) {
        auto* pw = reinterpret_cast<warp_info*>(pdata);
        if (warp.blk_x == pw->blk_x
            && warp.blk_y == pw->blk_y
            && warp.smid == pw->smid
            && warp.clk_start == pw->clk_start && warp.clk_dur == pw->clk_dur
            && warp.warpid == pw->warpid
            && warp.thr_y0 == pw->thr_y0) {
            warp.thr_cnt++;
            //if (warp.thr_cnt == WRAP_SIZE) {
            //    warp.dump(dumpper);
            //    warp.thr_cnt = 0;
            //}
        } else {
            dump(&warp);
            warp = *pw;
            warp.thr_cnt = 1;
        }
    }
    dump(&warp);
}

int main() {
    CUDADevice dev(0);

    auto M = getenv("M", 4096);
    auto K = getenv("K", 4096);
    auto N = getenv("N", 4096);

    ECOUT("test_max_gflops(", M, ",", N, ",", K, ")");
    test_max_gflops(M, N, K);
    return 0;
}

