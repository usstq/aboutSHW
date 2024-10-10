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

    thread_info::dump(T.to_cpu(), T.numel(), avg_dur_ns, BM*BN*sizeof(float));

    C.to_cpu();
    tensorND<float> Cref({M, N}, repeat);
    ASSERT(C == Cref);
}

int main() {
    CUDADevice dev(0);

    auto M = getenv("M", 16*1024);
    auto BM = getenv("BM", 32);
    auto BN = getenv("BN", 32);
    auto N = getenv("N", 16*1024);

    test_max_membw(M, N, BM, BN);
    return 0;
}

