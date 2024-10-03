
// https://siboehm.com/articles/22/CUDA-MMM

#include "cuda_utils.h"
#include "Windows.h"
#include <thread>

#define CEIL_DIV(x, a) (((x) + (a) - 1)/(a))

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

/******************************************************************
Memory Access Coalescing:
 threads execution are actually happens in unit of warp, coalesced memory access
 simplified warp's 

 */

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
    // to my surprise: the assembly machine codes are almost identical if we swap threadIdx.y & threadIdx.x
    // in following two statements. so compiler dosen't know/care about whether memory coalescing happens
    // (it's also very difficult for compiler to deduce/infer whether memory coalescing happens,
    //  since the address/offset(m, n) may go through a complex algorithm and there is no easy way
    // to know whether memory coalescing happens, so it's not feasible to rely on compiler to detect that and generate
    // different machine code (broadcast or continuous mem-loads or individual/independent 32 loads)
    // 
    // So memory coalescing happens in HW level in real-time, compiler just generate 32 independent addresses
    // in 32 registers, and HW logic (cache or DDR controller) will detect memory coalescing in real-time and make it
    // happen faster. (that's why we saw individual LD/ST unit in SM which only loads 1 register each).
    //   https://forums.developer.nvidia.com/t/what-is-the-functionality-of-ld-st-units-in-sm/290706
    //   https://forums.developer.nvidia.com/t/coalesced-access-and-hardware-load-store-units/51309
    //
    const int m = blockIdx.x * WRAP_SIZE + threadIdx.y;
    const int n = blockIdx.y * WRAP_SIZE + threadIdx.x;

    if (m < M && n < N) {
        float tmp = 0.0;
        for (int k = 0; k < K; ++k) {
            tmp += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = tmp;
    }
}


/***************************************************************************************
when shared memory is used, we need to care-about/coordinate all warps in a block.

 */

__global__ void sgemm_shared_mem_block(int M, int N, int K,
                                       const float *A, const float *B, float *C) {
  // the output block that we want to compute in this threadblock
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[WRAP_SIZE * WRAP_SIZE]; // 4 KiB
    __shared__ float Bs[WRAP_SIZE * WRAP_SIZE]; // 4 KiB

    // the inner row & col that we're accessing in this thread
    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;

    // advance pointers to the starting positions
    A += cRow * WRAP_SIZE * K;                    // row=cRow, col=0
    B += cCol * WRAP_SIZE;                        // row=0, col=cCol
    C += cRow * WRAP_SIZE * N + cCol * WRAP_SIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    auto A_Offset = threadRow * K + threadCol;
    auto B_Offset = threadRow * N + threadCol;
    auto SMEM_Offset = threadRow * WRAP_SIZE + threadCol;
    for (int bkIdx = 0; bkIdx < K; bkIdx += WRAP_SIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[SMEM_Offset] = A[A_Offset];
        Bs[SMEM_Offset] = B[B_Offset];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += WRAP_SIZE;
        B += WRAP_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < WRAP_SIZE; ++dotIdx) {
            tmp += As[threadRow * WRAP_SIZE + dotIdx] *
                    Bs[dotIdx * WRAP_SIZE + threadCol];
            /*
            ld.shared.f32   %f91, [%r8+3456];
            ld.shared.f32   %f92, [%r7+108];
            fma.rn.f32      %f93, %f92, %f91, %f90;

            warp cannot take full usage of CUDA cores due to waitting/blocking on LDS (load from shared mem)

            so shared memory solved global memory access latency issue (just like cache)
            but the access latency of itself becomes next bottleneck, register blocking can help.

            but we need each thread to caculate more than 1 result for register blocking to work
            (to reuse the register content).

            so load from shared memory into register and reuse it multiple-times for multiple results
            is how register blocking works.
            */
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

template<int BM_SIZE, int BN_SIZE, int BK_SIZE>
__global__ void sgemm_shared_mem_block2(int M, int N, int K,
                                       const float *A, const float *B, float *C) {
  // the output block that we want to compute in this threadblock
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BM_SIZE * BK_SIZE]; // 4 KiB
    __shared__ float Bs[BK_SIZE * BN_SIZE]; // 4 KiB

    // the inner row & col that we're accessing in this thread
    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;

    // advance pointers to the starting positions
    A += cRow * BM_SIZE * K;                    // row=cRow, col=0
    B += cCol * BN_SIZE;                        // row=0, col=cCol

    // threadRow 32 threadCol 32
    auto A_row_off_src = (threadRow*2)*K;
    auto A_row_off_dst = (threadRow*2)*BK_SIZE;
    auto A_col = (threadCol*2);

    float tmp[4] = {0};
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK_SIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[A_row_off_dst + A_col] = A[A_row_off_src + A_col];
        As[A_row_off_dst + A_col + 1] = A[A_row_off_src + A_col + 1];
        As[A_row_off_dst + BK_SIZE + A_col] = A[A_row_off_src + K + A_col];
        As[A_row_off_dst + BK_SIZE + A_col + 1] = A[A_row_off_src + K + A_col + 1];

        Bs[A_row_off_dst + A_col] = B[A_row_off_src + A_col];
        Bs[A_row_off_dst + A_col + 1] = B[A_row_off_src + A_col + 1];
        Bs[A_row_off_dst + BK_SIZE + A_col] = B[A_row_off_src + K + A_col];
        Bs[A_row_off_dst + BK_SIZE + A_col + 1] = B[A_row_off_src + K + A_col + 1];

        // block threads in this block until cache is fully populated
        __syncthreads();

        A += BM_SIZE;
        B += BN_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BK_SIZE; ++dotIdx) {
            auto As_off = threadRow * 2 * BK_SIZE + dotIdx;
            auto a0 = As[As_off];
            auto a1 = As[As_off + BK_SIZE];
            auto Bs_off = dotIdx * BN_SIZE + threadCol*2;
            auto b0 = Bs[Bs_off];
            auto b1 = Bs[Bs_off + 1];
            tmp[0] +=  a0*b0;
            tmp[1] +=  a0*b1;
            tmp[2] +=  a1*b0;
            tmp[3] +=  a1*b1;
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();

    }

    const int m = cRow * BM_SIZE + threadRow*2;
    const int n = cCol * BN_SIZE + threadCol*2;
    if (m < M && n < N) {
        C[m * N + n] = tmp[0];
        C[m * N + n + 1] = tmp[1];
        C[(m+1) * N + n] = tmp[2];
        C[(m+1) * N + n + 1] = tmp[3];
    }
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, 
                                   const float *A, const float *B, float *C) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint32_t cRow = blockIdx.y;
  const uint32_t cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  // this allocation is per-block
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const int innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const int innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const int innerRowB = threadIdx.x / BN;

  auto A_src_off = innerRowA * K + innerColA;
  auto A_dst_off = innerRowA * BK + innerColA;
  auto B_src_off = innerRowB * N + innerColB;
  auto B_dst_off = innerRowB * BN + innerColB;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[A_dst_off] = A[A_src_off];
    Bs[B_dst_off] = B[B_src_off];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    const int m = threadRow * TM + resIdx;
    const int n = threadCol;
    if (m < M && n < N) {
        C[m * N + n] = threadResults[resIdx];
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, 
                       const float *A, const float *B, float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / BK;
  const int innerRowB = threadIdx.x / BN;
  const int innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
    }
  }
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
    static auto ROUNDS = getenv("ROUNDS", 1);

    ECOUT("test_gemm");
    tensor2D<float> A(M, K, true);
    tensor2D<float> B(K, N, true);
    tensor2D<float> C(M, N, true);
    tensor2D<float> C_ref(M, N, true);

    ECOUT("A.rand");
    A.rand();
    ECOUT("B.rand");
    B.rand();
    ECOUT("start");
    dim3 gridDim(CEIL_DIV(M, WRAP_SIZE), CEIL_DIV(N, WRAP_SIZE), 1); // create as many blocks as necessary to map all of C
    dim3 blockDim(WRAP_SIZE, WRAP_SIZE, 1); // 32 * 32 = 1024 thread per block

    auto bytes_accessed = A.size_bytes + B.size_bytes + C.size_bytes;
    auto flops = M*N*K*2;

    auto check_accuracy = [&](std::stringstream& ss){
        using namespace std::chrono_literals;
        C.to_host();
        std::string ret = (C_ref == C ? "...OK":"...failed");
        C.zero();
        C.to_dev();
        ss << ret;
        std::this_thread::sleep_for(10000ms);
    };

    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C_ref);
    C_ref.to_host();

    cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm_naive", bytes_accessed, flops);


    cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_coalescing<<<gridDim, dim3(WRAP_SIZE*WRAP_SIZE)>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm_coalescing", bytes_accessed, flops, ROUNDS);

    cuda_timeit([&](int i, std::stringstream& ss){
        dim3 gridDim(CEIL_DIV(M, WRAP_SIZE), CEIL_DIV(N, WRAP_SIZE));
        dim3 blockDim(WRAP_SIZE, WRAP_SIZE);
        sgemm_coalescing2<<<gridDim, blockDim>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm_coalescing2", bytes_accessed, flops, ROUNDS);

    cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_shared_mem_block<<<gridDim, blockDim>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm_shared_mem_block", bytes_accessed, flops, ROUNDS);

    cuda_timeit([&](int i, std::stringstream& ss){
        dim3 gridDim2(CEIL_DIV(M, WRAP_SIZE*2), CEIL_DIV(N, WRAP_SIZE*2), 1);
        sgemm_shared_mem_block2<WRAP_SIZE*2, WRAP_SIZE*2, WRAP_SIZE*2><<<gridDim2, blockDim>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm_shared_mem_block2", bytes_accessed, flops, ROUNDS);

    cuda_timeit([&](int i, std::stringstream& ss){
        const uint32_t BM = 64;
        const uint32_t BN = 64;
        const uint32_t BK = 8;
        const uint32_t TM = 8;
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / TM);
        sgemm1DBlocktiling<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm1DBlocktiling", bytes_accessed, flops, ROUNDS);

    cuda_timeit([&](int i, std::stringstream& ss){
        const uint32_t BK = 8;
        const uint32_t TM = 8;
        const uint32_t TN = 8;
        const uint32_t BM = 128;
        const uint32_t BN = 128;            
        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim((BM * BN) / (TM * TN));
        sgemm2DBlocktiling<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, A, B, C);
        if (i == 0) check_accuracy(ss);
    }, __func__, __LINE__, "sgemm2DBlocktiling", bytes_accessed, flops, ROUNDS);
}

int main() {
    // Choose which GPU to run on, change this on a multi-GPU system.
    ECOUT("cudaSetDevice(0)");
    ASSERT(cudaSetDevice(0) == cudaSuccess);

    auto M = getenv("M", 4096);
    auto K = getenv("K", 4096);
    auto N = getenv("N", 4096);

    ECOUT("test_max_gflops(", M, ",", N, ",", K, ")");
    test_max_gflops(M, N, K);
    return 0;


    ECOUT("test_sgemm_naive_correctness");
    // sgemm_naive will be used as a reference in following tests, so itself must be correct
    test_sgemm_naive_correctness(15, 211, 133);


    ECOUT("test_gemm(", M, ",", N, ",", K, ")");
    test_gemm(M, N, K);

    ECOUT("finished");
    TIMEIT_FINISH();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    ASSERT(cudaDeviceReset() == cudaSuccess);

    return 0;
}
