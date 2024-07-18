#include "cuda_utils.h"

__global__ void kernel(int* src, int val, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    src[i] = val;
}

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    const int N = 1024*1024;  // 1M
    const int sz = N*sizeof(int);  // 4MB
    int *src;
    cudaMalloc(&src, sz*10);
    cudaMemset(src, 1, sz);
    kernel << <160, 1024>> > (src, 2, N);

    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
