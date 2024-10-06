#include "cuda_utils.h"
/*
nvcc --generate-line-info --keep --keep-dir build -O2 max_gflops.cu
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

#define FMACNT 16
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

    auto flops = M*N*K*2*FMACNT;
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

