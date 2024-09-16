
// https://siboehm.com/articles/22/CUDA-MMM

#include "cuda_utils.h"
#include "Windows.h"

#define CEIL_DIV(x, a) ((x + a - 1)/a)

/*

what a wrap would access memory?
is it coalescing memory access of all 32 threads within it?

consider blockDim={32, 32}, so each wrap shares same y among 32 threads,
but with different & consecutive x for each thread, so:
 - A[x * K + i] cannot be coalesced, 32 float32-loads from non-continous memory addresses are required.
 - B[i * N + y] is a single element broadcased to all 32 theads within a wrap
 - C[x * N + y] is just like A, 32 float32-stores are required.

*/
__global__ void sgemm_naive(int M, int N, int K,
                            const float *A, const float *B, float *C) {
  // compute position in C that this thread is responsible for
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = A@B
    C[x * N + y] = tmp;
  }
}

#define WRAP_SIZE 32

__global__ void sgemm_coalescing(int M, int N, int K,
                                 const float *A, const float *B, float *C) {

/*
    wrap0 ~ wrap31 will share B sub-matrix memory-access through L3 spatially
*/
    const int m = blockIdx.x * WRAP_SIZE + (threadIdx.x / WRAP_SIZE);
    const int n = blockIdx.y * WRAP_SIZE + (threadIdx.x % WRAP_SIZE);

    if (m < M && n < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = tmp;
    }
}


// mapping between threadIdx & (x,y) are transposed
__global__ void sgemm_coalescing2(int M, int N, int K,
                                 const float *A, const float *B, float *C) {
    const int m = blockIdx.x * blockDim.x + threadIdx.y;
    const int n = blockIdx.y * blockDim.y + threadIdx.x;

    if (m < M && n < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = tmp;
    }
}


__global__ void sgemm_shared_mem_block(int M, int N, int K,
                                       const float *A, const float *B, float *C) {
  // the output block that we want to compute in this threadblock
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[WRAP_SIZE * WRAP_SIZE];
    __shared__ float Bs[WRAP_SIZE * WRAP_SIZE];

    // the inner row & col that we're accessing in this thread
    const int threadCol = threadIdx.x % WRAP_SIZE;
    const int threadRow = threadIdx.x / WRAP_SIZE;

    // advance pointers to the starting positions
    A += cRow * WRAP_SIZE * K;                    // row=cRow, col=0
    B += cCol * WRAP_SIZE;                        // row=0, col=cCol
    C += cRow * WRAP_SIZE * N + cCol * WRAP_SIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += WRAP_SIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * WRAP_SIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * WRAP_SIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += WRAP_SIZE;
        B += WRAP_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < WRAP_SIZE; ++dotIdx) {
            tmp += As[threadRow * WRAP_SIZE + dotIdx] *
                    Bs[dotIdx * WRAP_SIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    const int m = cRow * WRAP_SIZE + threadRow;
    const int n = cCol * WRAP_SIZE + threadCol;
    if (m >= M || n >= N) return;
    C[threadRow * N + threadCol] = tmp;
}

tensor2D<float> rand_tensor(int d0, int d1, int levels = 5) {
    tensor2D<float> t(d0, d1, false);
    for(int i = 0; i < t.size; i++) {
        t.ptr_host.get()[i]  = (rand() % levels) * 1.0f / (levels-1) - 0.5f;
    }
    t.to_dev();
    return t;
}

tensor2D<float> const_tensor(int d0, int d1, float val = 1.0f) {
    tensor2D<float> t(d0, d1, false);
    auto* p = t.ptr_host.get();
    for(int i = 0; i < t.size; i++) {
        p[i]  = val;
    }
    t.to_dev();
    return t;
}

tensor2D<float> get_ref(tensor2D<float>& A, tensor2D<float>& B) {
    auto M = A.shape[0];
    auto K = A.shape[1];
    assert(K == B.shape[0]);
    auto N = B.shape[1];
    tensor2D<float> C(M, N, false);
    for (int m = 0; m < M; m++) {
        auto* pA = A.ptr_host.get() + m*K;
        auto* pC = C.ptr_host.get() + m*N;
        for (int n = 0; n < N; n++) {
            float fsum = 0;
            auto* pB = B.ptr_host.get() + n;
            for (int k = 0; k < K; k++, pB += N) {
                fsum += pA[k] * (*pB);
            }
            pC[n] = fsum;
        }
    }
    return C;
}

void test_sgemm_naive_correctness(int64_t M, int64_t N, int64_t K) {
    tensor2D<float> A = rand_tensor(M, K);
    tensor2D<float> B = rand_tensor(K, N);
    tensor2D<float> C = rand_tensor(M, N);
    tensor2D<float> C_ref = get_ref(A, B);
    sgemm_naive<<<dim3(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1), dim3(32, 32, 1)>>>(M, N, K, A.ptr_dev.get(), B.ptr_dev.get(), C.ptr_dev.get());

    C.to_host();
    if (!(C_ref == C))
        throw std::runtime_error("test_sgemm_naive_correctness failed!");
}

void test_gemm(int64_t M, int64_t N, int64_t K) {
    tensor2D<float> A(M, K, true);
    tensor2D<float> B(K, N, true);
    tensor2D<float> C(M, N, true);
    tensor2D<float> C_ref(M, N, true);

    A.rand();
    B.rand();
    dim3 gridDim(CEIL_DIV(M, WRAP_SIZE), CEIL_DIV(N, WRAP_SIZE), 1); // create as many blocks as necessary to map all of C
    dim3 blockDim(WRAP_SIZE, WRAP_SIZE, 1); // 32 * 32 = 1024 thread per block

    auto bytes_accessed = A.size_bytes + B.size_bytes + C.size_bytes;
    auto flops = M*N*K*2;

    auto check_accuracy = [&](){
        C.to_host();
        std::string ret = (C_ref == C ? "...OK":"...failed");
        C.zero();
        C.to_dev();
        cuda_timeit_last_ps() << ret;
    };

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C_ref);
    C_ref.to_host();

    cuda_timeit([&](){
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    }, __func__, __LINE__, "sgemm_naive", bytes_accessed, flops);
    check_accuracy();

    cuda_timeit([&](){
        sgemm_coalescing<<<gridDim, dim3(WRAP_SIZE*WRAP_SIZE)>>>(M, N, K, A, B, C);
    }, __func__, __LINE__, "sgemm_coalescing", bytes_accessed, flops);
    check_accuracy();

    cuda_timeit([&](){
        sgemm_coalescing2<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    }, __func__, __LINE__, "sgemm_coalescing2", bytes_accessed, flops);
    check_accuracy();

    cuda_timeit([&](){
        sgemm_shared_mem_block<<<gridDim, dim3(WRAP_SIZE*WRAP_SIZE)>>>(M, N, K, A, B, C);
    }, __func__, __LINE__, "sgemm_shared_mem_block", bytes_accessed, flops);
    check_accuracy();
}


int main() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    // sgemm_naive will be used as a reference in following tests, so itself must be correct
    test_sgemm_naive_correctness(15, 211, 133);

    auto M = getenv("M", 4092);
    auto K = getenv("K", 4092);
    auto N = getenv("N", 4092);
    test_gemm(M, N, K);

    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
