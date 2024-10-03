// Demystifying GPU Microarchitecture through Microbenchmarking
#include "cuda_utils.h"

__global__ void basic_test() {
    printf("sizeof(int)=%llu\n", sizeof(int));
    printf("sizeof(uintptr_t)=%llu\n", sizeof(uintptr_t));
    printf("sizeof(int*)=%llu\n", sizeof(int*));
}

__global__ void cache_test(uintptr_t* device_array, int array_size, int cnt, uint64_t *clocks) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    uintptr_t* p = device_array + m;
    
    // warm-up
    for(int i = 0; i < cnt; i++) p = *(uintptr_t**)p;
    if (p == 0) printf("Impossible\n");

    p = device_array + m;
    auto t0 = clock();
    // dependent loads cannot be paralleled, better for latency profiling
    // and for cache latency test
    for(int i = 0; i < cnt; i+=32) {
        for(int unroll = 0; unroll < 32; unroll++)
            p = *(uintptr_t**)p;
    }
    auto t1 = clock();
    if (clocks)
        clocks[m] = t1 - t0;
}

void test(int array_size) {
    tensor2D<uintptr_t> array(1, array_size);
    uintptr_t* phost = array.ptr_host.get();
    uintptr_t* pdev = array.ptr_dev.get();

    for(int i = 0; i < array_size; i++) {
        int t = i + 32;
        if (t >= array_size) t -= array_size;
        phost[i] = reinterpret_cast<uintptr_t>(pdev + t);
    }
    //std::cout << array << std::endl;
    array.to_dev();
    //cudaFuncSetCacheConfig(cache_test, cudaFuncCachePreferL1);

    tensor2D<uint64_t> clocks(1, 32, true);

    // test latency
    cache_test<<<{1},{32}>>>(array, array_size, 128*1024, clocks);
    clocks.to_host();

    std::cout << "clocks: " << carray(clocks) << std::end;

/*
    cuda_timeit([&](int i, std::stringstream& ss){
        cache_test<<<{1},{32}>>>(array, array_size, 128*1024);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm_naive", bytes_accessed, flops);
*/
    
}

int main() {
    basic_test<<<{1},{1}>>>();
    auto U = getenv("U", 8*1024);
    auto L = getenv("L", 8*1024);
    auto H = getenv("H", 128*1024);
    // in step of 128KB, upto 16MB
    int unit_size = U/sizeof(uintptr_t);
    for(int array_size = L/sizeof(uintptr_t); array_size < H/sizeof(uintptr_t); array_size += unit_size)
        test(array_size);
}