#include "cuda_utils.h"
#include "Windows.h"

// https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers

// The first problem - as @Patwie suggests - is that %warp_id does not give you what you actually want 
//  it's not the index of the warp in the context of the grid, but rather in the context of the physical SM
//  (which can hold so many warps resident at a time), and those two are not the same.
//
// thus it's actually warp scheduler id instead of warp id.
//     that's why we can observe over-subcribing using this example:
//
__forceinline__ __device__ unsigned laneid()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warpid()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}
__forceinline__ __device__ unsigned nwarpid()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %nwarpid;" : "=r"(ret));
    return ret;
}

struct thread_info {
    unsigned blockIdx_x;
    unsigned threadIdx_x;
    unsigned warpid;
    unsigned nwarpid;
    unsigned laneid;
};

__global__ void kernel(thread_info * tinfo, int val, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tinfo[i].blockIdx_x = blockIdx.x;   // PTX:  ctaid.x
    tinfo[i].threadIdx_x = threadIdx.x; // PTX:  tid.x
    tinfo[i].warpid = warpid();         // PTX:  warpid
    tinfo[i].nwarpid = nwarpid();       // PTX:  nwarpid
    tinfo[i].laneid = laneid();         // PTX:  laneid
    /*
    printf("src=%p,    blockDim(%d,%d,%d) blockIdx.threadIdx(%d.%d,%d.%d,%d.%d)  i=%d  wrap.lane=%u.%u\n", src,
            blockDim.x, blockDim.y, blockDim.z,
            blockIdx.x, threadIdx.x,
            blockIdx.y, threadIdx.y,
            blockIdx.z, threadIdx.z, i,
            warpid(), laneid());
    */
}

bool is_bad_read_ptr(void* src) {
    // use SEH to detect bad Host pointer
    // https://learn.microsoft.com/en-us/cpp/cpp/try-except-statement?view=msvc-170
    __try{
        *reinterpret_cast<char*>(src) = 1;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::cerr << "\tSEH happens on accessing int @ 0x" << std::hex << src << " from host!" << std::endl;
        return true;
    }
    return false;
}

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    const int N = 16*(64*32);
    const int sz = N*sizeof(thread_info);  // 4MB
    void *tinfo;
    int val = 0;
    cudaMalloc(&tinfo, sz);
    std::cout << "cudaMalloc " << sz << " bytes @ 0x" << std::hex << tinfo << " is_bad_read_ptr()=" << is_bad_read_ptr(tinfo) << std::dec << std::endl;
    cudaMemset(tinfo, 0, sz);

    kernel << <16*2, 32*32>> > (reinterpret_cast<thread_info*>(tinfo), val, N);

    ASSERT(cudaDeviceSynchronize() == cudaSuccess);

    auto* ptinfo = new thread_info[N];
    ASSERT(cudaMemcpy(ptinfo, tinfo, sz, cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(tinfo);

    std::cout << "nwarpid (Wraps per SM): " << ptinfo[0].nwarpid << std::endl;
    for(int i = 0; i < N; i+=32) {
        std::cout << "[" << i << "]: ";        
        std::cout << ptinfo[i].blockIdx_x << "." << std::fixed << ptinfo[i].threadIdx_x;
        std::cout << " @ \t";

        bool laneId_expected = true;
        for(int k = i; k < i+32; k++) {
            if (ptinfo[k].laneid != k-i) {
                laneId_expected = false;
            }
        }
        if (laneId_expected) {
            std::cout << ptinfo[i].warpid << ": 0~31\n";
        } else {
            for(int k = i; k < i+32; k++) {
                std::cout << ptinfo[k].warpid << "." << ptinfo[k].laneid << " ";
            }
            std::cout << std::endl;
        }
        
    }

    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
