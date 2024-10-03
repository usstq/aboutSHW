// Demystifying GPU Microarchitecture through Microbenchmarking
#include "cuda_utils.h"


#define UNROLL 1
__global__ void compute_test(int start, int step, int loop) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    auto sum = start + m;
    step += m;

    // warm-up
    auto t0 = clock();
    for(int l = 0; l < loop; l++)
        for(int i  = 0; i < UNROLL; i++)
            sum += i;
    auto t1 = clock();

    if (sum != 0 && m == 0) {
        size_t total_ops = loop * UNROLL;
        printf("OPs: %llu latency=%.2f(cycles/op) tput=%.2f(bytes/cycle)\n",
                total_ops,
                double(t1 - t0)/total_ops,
                double(total_ops)/(t1-t0));
    }
}

int main() {
    compute_test<<<{1},{32}>>>(0, 1, 1000);
}