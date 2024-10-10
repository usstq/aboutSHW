#include "cuda_utils.h"

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

/*
 elementwise inplace operation: C[y, x] += 1;

 each thread should do more work, so mapping between work-item & thread-block-grid
 becomes complex & important:
  - thread-warp-block-grid config from blockIdx/threadIdx
  - work-item config from input parameter

  each warp do BN*WRAP_SIZE of work in inner-most dimension(N),
  so [M, N] is blocked as [M, N/BN, BN]
*/

struct thread_info {
    int64_t blk_x = -1;
    int64_t blk_y = -1;
    uint64_t thr_x0 = 0;
    uint64_t thr_y0 = 0;
    uint64_t smid = 0;
    uint64_t warpid = 0;
    uint64_t clk_start = 0;
    uint64_t clk_dur = 0;
};

__global__ void sgemm_max_membw(thread_info * tinfo, float * C, size_t M, size_t N, size_t BM, size_t BN) {
    const int x = blockIdx.x * (blockDim.x * BN) + threadIdx.x;
    const int y = blockIdx.y * (blockDim.y * BM) + threadIdx.y;

    if (y < M && x < N) {
        auto linear_id_x = blockIdx.x * blockDim.x + threadIdx.x;
        auto linear_id_y = blockIdx.y * blockDim.y + threadIdx.y;
        auto* pt = tinfo + linear_id_y * (gridDim.x * blockDim.x) + linear_id_x;
        pt->blk_x = blockIdx.x;
        pt->blk_y = blockIdx.y;
        pt->thr_x0 = threadIdx.x;
        pt->thr_y0 = threadIdx.y;
        pt->smid = get_smid();
        pt->warpid = get_warpid();

        auto* pdata = C + (y*N + x);
        float sum = 0;

        pt->clk_start = clock64();
        for(int bm = 0; bm < BM; bm++, pdata += WRAP_SIZE*N) {
            auto* ptr = pdata;
            for(int bn = 0; bn < BN; bn++, ptr += WRAP_SIZE) {
                //ptr[0] += 1;
                sum += ptr[0];
            }
        }
        pt->clk_dur = clock64() - pt->clk_start;

        if (sum == 1.0f) {
            printf("impossible, just to prevent optimization of sum");
        }
    }
}

void test_max_membw(size_t M, size_t N, size_t BM, size_t BN) {
    tensorND<float> C({M, N}, 0);
    tensorND<thread_info> T({N/BN, M/BM}, thread_info());

    dim3 gridDim(CEIL_DIV(N, BN*WRAP_SIZE), CEIL_DIV(M, BM*WRAP_SIZE));
    dim3 blockDim(WRAP_SIZE, WRAP_SIZE);

    ECOUT("test_max_membw(", M, ",", N, ",", BM, ",", BN, ")");
    std::cout << "gridDim=" << gridDim << " blockDim=" << blockDim << std::endl;

    auto bytes = M*N*sizeof(float);
    auto flops = 0;
    auto repeat = 3;
    
    // warmp-up
    sgemm_max_membw<<<gridDim, blockDim>>>(T.to_gpu(), C.to_gpu(), M, N, BM, BN);

    auto avg_dur_ns = cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_max_membw<<<gridDim, blockDim>>>(T.to_gpu(), C.to_gpu(), M, N, BM, BN);
    }, __func__, __LINE__, "sgemm_max_membw", bytes, flops, repeat);

    C.to_cpu();
    tensorND<float> Cref({M, N}, repeat);
   
    //ASSERT(C == Cref);

    auto* ptinfo = T.to_cpu();
    // calibrate all clocks from different SMs
    std::vector<uint64_t> clock_min(128, std::numeric_limits<uint64_t>::max());
    std::vector<uint64_t> clock_max(128, std::numeric_limits<uint64_t>::min());
    std::vector<uint64_t> element_cnt(128, 0);
    uint64_t sm_cnt = 0;
    for(int i = 0; i < T.numel(); i++, ptinfo ++) {
        sm_cnt = std::max(sm_cnt, ptinfo->smid+1);
        if (ptinfo->clk_dur > 0) {
            clock_min[ptinfo->smid] = std::min(clock_min[ptinfo->smid], ptinfo->clk_start);
            clock_max[ptinfo->smid] = std::max(clock_max[ptinfo->smid], ptinfo->clk_start + ptinfo->clk_dur);
            element_cnt[ptinfo->smid] += BM*BN;
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
              " bandwith:", (element_cnt[smid]*sizeof(float))/clock_dur, " (bytes/cycle)");
    }
    ECOUT(" clock_overall_dur = ", clock_overall_dur);
    ECOUT(" avg_dur_ns = ", avg_dur_ns, "(ns)");
    ECOUT(" GPU_avg_frequency = ", clock_overall_dur / avg_dur_ns, " (GHz)");
    //ECOUT(" average FMA = ", double(FMACNT)*K*M*N/sm_cnt/clock_overall_dur, " (FMA/SM/cycle)");
    //ECOUT(" average FMA = ", double(FMACNT)*K*M*N/avg_dur_ns, " (GFMA/s)");

    ptinfo = T.to_cpu();
    for(int i = 0; i < T.numel(); i++, ptinfo ++) {
        ptinfo->clk_start -= clock_min[ptinfo->smid];
    }

    ChromeTraceDumpper dumpper("ct.json");
    ptinfo = T.to_cpu();
    // 32 threads from same warp (if block-size in X direction is larger than 32)
    struct warp_info : public thread_info {
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
                    {"bytes/cycle", std::to_string(double(BM*BN*sizeof(float)*pw->thr_cnt)/pw->clk_dur)}
                });
        }
    };
    for (int n = 0; n < T.numel(); n++, ptinfo ++) {
        if (warp.blk_x == ptinfo->blk_x
            && warp.blk_y == ptinfo->blk_y
            && warp.smid == ptinfo->smid
            && warp.clk_start == ptinfo->clk_start && warp.clk_dur == ptinfo->clk_dur
            && warp.warpid == ptinfo->warpid
            && warp.thr_y0 == ptinfo->thr_y0) {
            warp.thr_cnt++;
            //if (warp.thr_cnt == WRAP_SIZE) {
            //    warp.dump(dumpper);
            //    warp.thr_cnt = 0;
            //}
        } else {
            dump(&warp);
            memcpy(&warp, ptinfo, sizeof(*ptinfo));
            warp.thr_cnt = 1;
        }
    }
    dump(&warp);
}

int main() {
    CUDADevice dev(0);

    auto M = getenv("M", 4096);
    auto BM = getenv("BM", 4);
    auto BN = getenv("BN", 4);
    auto N = getenv("N", 4096);

    test_max_membw(M, N, BM, BN);
    return 0;
}

