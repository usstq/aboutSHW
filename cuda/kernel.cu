
#include "cuda_utils.h"
#include <immintrin.h>

__global__ void TensorAddKernel(int* c, int* a, int* b, int d0, int d1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// OP dispatcher
template<typename T>
void tensor_add(tensor2D<T>& c, tensor2D<T>& a, tensor2D<T>& b) {
    ASSERT(c.shape[0] == a.shape[0] && c.shape[1] == a.shape[1]);
    ASSERT(c.shape[0] == b.shape[0] && c.shape[1] == b.shape[1]);
    ASSERT(a.on_device == b.on_device);

    // x is inner-most dimension
    if (a.on_device) {
        c.to_dev(false);
        dim3 blockDim(32, 32);
        dim3 gridDim((c.shape[1] + 31) / 32, (c.shape[0] + 31) / 32);
        TIMEIT(
            TensorAddKernel << <gridDim, blockDim >> > (c.ptr_dev, a.ptr_dev, b.ptr_dev, a.shape[0], a.shape[1]);
        );
    }
    if (!a.on_device) {
        // CPU side
        c.to_host(false);
        TIMEIT_BEGIN();
        #pragma omp parallel for
        for (int i = 0; i < a.size; i+=8) {
            __m256i ra = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a.ptr_host + i));
            __m256i rb = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b.ptr_host + i));
            __m256i rc = _mm256_add_epi32(ra, rb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(c.ptr_host + i), rc);
        }
        TIMEIT_END();
    }
}



int addWithCuda2() {

    tensor2D<int> c(256, 4096);
    tensor2D<int> a(256, 4096);
    tensor2D<int> b(256, 4096);
    for (int i = 0; i < a.size; i++) {
        a.ptr_host[i] = i;
        b.ptr_host[i] = i;
    }
    
    tensor_add(c, a, b);
    tensor_add(c, a, b);
    tensor_add(c, a, b);
    tensor_add(c, a, b);
    std::cout << "c=" << c << std::endl;
    a.to_dev();
    b.to_dev();
    tensor_add(c, a, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    tensor_add(c, c, b);
    c.to_host();
    std::cout << "c=" << c << std::endl;
    return 0;
}

// https://www.youtube.com/watch?v=6kT7vVHCZIc
__device__ float result = 0.1f;
__global__ void reduceAtomicGlobal(const float* input, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N)
        atomicAdd(&result, input[id]);
}

// this version actually is slower than reduceAtomicGlobal
// since in each thread it brings too much overheads before calling `atomicAdd`
// it only faster if partial sum is used more than once
__global__ void reduceAtomicShared(const float* input, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory is allocated per thread block, so all threads in the block
    // have access to the same shared memory.
    __shared__ float partial_sum;
    // thus only the first thread clears it
    if (threadIdx.x == 0) partial_sum = 0;
    __syncthreads();

    if (id < N) atomicAdd(&partial_sum, input[id]);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(&result, partial_sum);
}

#define BLOCKSIZE 1024
__global__ void reduceParallelShared(const float* input, int N) {
    __shared__ float data[BLOCKSIZE];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // load input into shared data for current thread-block
    data[threadIdx.x] = (id < N ? input[id] : 0);
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        __syncthreads();
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }
    if (threadIdx.x == 0)
        atomicAdd(&result, data[0]);
}

__global__ void reduceParallelSharedShfl(const float* input, int N) {
    __shared__ float data[BLOCKSIZE];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // load input into shared data for current thread-block
    data[threadIdx.x] = (id < N ? input[id] : 0);
    for (int s = blockDim.x / 2; s > 16; s /= 2) {
        __syncthreads();
        // SIMD horizontal add
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }
    float x = data[threadIdx.x];
    if (threadIdx.x < 32) {
        // SIMD horizontal reduce
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }
    if (threadIdx.x == 0)
        atomicAdd(&result, x);
}

__global__ void reduceParallelSharedShfl2(const float* input, int N) {
    __shared__ float data[BLOCKSIZE];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    data[threadIdx.x] = (id < N ? input[id] : 0);
    
    int thread_offset = 32 * (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;

    // horizontal reduce
    float x = data[threadIdx.x];

    // SIMD horizontal reduce
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
    x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
    x += __shfl_sync(0xFFFFFFFF, x, 1);

    if (lane_id == 0)
        atomicAdd(&result, x);
}


__global__ void reduceParallelMemBoundx8(const float* input, int subN) {
    __shared__ float data[BLOCKSIZE][8];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < subN) {
        int threadOffset = (32 * (id / 32)) * 8;
        int id0 = threadOffset + (threadIdx.x % 32);
        int id1 = id0;
        for (int i = 0; i < 8; i++) {
            data[threadIdx.x][i] = input[id1];
            id1 += 32;
        }
        __syncthreads();
        float x = 0;
        for (int i = 0; i < 8; i++) {
            x += data[threadIdx.x][i];
        }
        //data[threadIdx.x] = x;
        if (x > 16000) {
            // cannot happen, just to prevent optimization
            atomicAdd(&result, x);
        }
    }
}
/*
According to:
https://www.anandtech.com/show/10325/the-nvidia-geforce-gtx-1080-and-1070-founders-edition-review/4

1070 has 8GB VRAM with 256GB/s bandwidth, 16 SMs with 1.645GHz, assume each SM can load 8*4 u32 data, 
and following kernel loaded 4bytes into R2 within 18 instructions (9cycles if 2-dispatch units works well)

1.645*16*32/9 = 93.58 GB/s, actuall observed bandwidth is higher (140GB/s), but still not reach 256GB/s.
major problem is CUDA cores are busy executing non-memory load instructions, so it didn't issue enough LDG instructions.
for example, CUDA core need to issue one f32/int32 read on every (1.645*16*4*8*4)/256 ~= 13 cycles to get 256GB/s.

_Z22reduceParallelMemBoundPKfi:
		MOV R1, c[0x0][0x20] ;
		S2R R0, SR_TID.X ;
		S2R R2, SR_CTAID.X ;
		XMAD R0, R2.reuse, c[0x0] [0x8], R0 ;
		XMAD.MRG R3, R2.reuse, c[0x0] [0x8].H1, RZ ;
		XMAD.PSL.CBCC R2, R2.H1, R3.H1, R0 ;
		ISETP.GE.AND P0, PT, R2, c[0x0][0x148], PT ;
		NOP ;
		@P0 EXIT ; <================================================ if (id < N)
		SHR R0, R2.reuse, 0x1e ;
		ISCADD R2.CC, R2, c[0x0][0x140], 0x2 ;
		IADD.X R3, R0, c[0x0][0x144] ;
		LDG.E R0, [R2] ; <========================================== Load 
		FSETP.GT.AND P0, PT, R0, 160, PT ;
		NOP ;
		NOP ;
		NOP ;
		@!P0 EXIT ; <=============================================== if (x > 160) all threads exit here
		MOV R2, c[0x4][0x0] ;
		MOV R3, c[0x4][0x4] ;
		RED.E.ADD.F32.FTZ.RN [R2], R0 ;
		NOP ;
		NOP ;
		NOP ;
		EXIT ;
.L_x_0:
		BRA `(.L_x_0) ;
		NOP;
		NOP;
		NOP;
		NOP;
.L_x_1:
*/


template<typename T>
T tensor_reduce(tensor2D<T>& c) {
    if (c.on_device) {
        dim3 gridSize((c.size + 1023) / 1024);
        dim3 blockSize(1024);
        float Y = 0.0f;
        uint64_t bytes = c.size * sizeof(T);
        uint64_t flops = c.size;

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceAtomicGlobal", bytes, flops);
        reduceAtomicGlobal << <gridSize, blockSize>> > (c.ptr_dev, c.size); // 3ms
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceAtomicShared", bytes, flops);
        reduceAtomicShared << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 20ms
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceParallelShared", bytes, flops);
        reduceParallelShared << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 100us
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceParallelSharedShfl", bytes, flops);
        reduceParallelSharedShfl << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 64us
        TIMEIT_END();

        CUDA_CALL(cudaMemcpyToSymbol(result, &Y, sizeof(Y)));
        TIMEIT_BEGIN("reduceParallelSharedShfl2", bytes, flops);
        reduceParallelSharedShfl2 << <gridSize, blockSize >> > (c.ptr_dev, c.size); // 64us
        TIMEIT_END();

        dim3 gridSize2(c.size / 1024/8);
        TIMEIT_BEGIN("reduceParallelMemBoundx8", bytes, flops);
        reduceParallelMemBoundx8 << <gridSize2, blockSize >> > (c.ptr_dev, c.size/8);
        TIMEIT_END();
        CUDA_CALL(cudaMemcpyFromSymbol(&Y, result, sizeof(result)));
        return Y;
    }
    else {
        TIMEIT_BEGIN("tensor_reduce_CPU");
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < c.size; i++)
            sum += c.ptr_host[i];
        TIMEIT_END();
        return sum;
    }
}



void testReduce() {
    tensor2D<float> c(1024*16, 1024);
    for (int i = 0; i < c.size; i++)
        c.ptr_host[i] = (i % 16) - 8;
    auto s0 = tensor_reduce(c);
    c.to_dev();
    auto s1 = tensor_reduce(c);
    printf("s0(host)=%f, s1(device)=%f\n", s0, s1);

    c.to_host();
    tensor_reduce(c);
    tensor_reduce(c);
    tensor_reduce(c);
    c.to_dev();
    tensor_reduce(c);
    tensor_reduce(c);
    tensor_reduce(c);
}


__global__ void testCpy(int* in, int* out, int N) {
    int block_offset = blockIdx.x * blockDim.x;
    int warp_offset = 32 * (threadIdx.x / 32);
    int lane_id = threadIdx.x % 32;
    int id = (block_offset + warp_offset + lane_id) % N;
    out[id] = in[id];
}

void test_cpy() {
    tensor2D<int> c(10240, 1024);
    tensor2D<int> d(10240, 1024);
    for (int i = 0; i < c.size; i++)
        c.ptr_host[i] = (i % 16) - 8;
    c.to_dev();
    d.to_dev(false);
    dim3 gridSize((c.size + 1023) / 1024);
    dim3 blockSize(1024);
    uint64_t bytes = c.size * sizeof(int);
    for (int i = 0; i < 5; i++) {
        TIMEIT_BEGIN("testCpy", bytes);
        testCpy << <gridSize, blockSize >> > (c.ptr_dev, d.ptr_dev, c.size);
        TIMEIT_END();
    }
}

int main()
{
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    //test_cpy();
    testReduce();
    //addWithCuda2();

    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
