#include "cuda_utils.h"
/*
nvcc --generate-line-info --keep --keep-dir build -O2 max_gflops.cu

using only 

M=1 K=409600 N=32 ./a.exe
    thread 0 : CArray<__int64,32>{8295715...x32} 5854890.664(ns) GPU_freq:1.417(GHz)? FMA:1.119(GFMA/s/CUDA-core)?
M=1 K=409600 N=2048 ./a.exe
    thread 0 : CArray<__int64,32>{8295695...x32} 11787946.701(ns) GPU_freq:0.704(GHz)? FMA:0.556(GFMA/s/CUDA-core)?
    the clocks almost doubled, because of over-subscription

M=32 K=409600 N=1024 ./a.exe
    this can keep clocks small, but M=33 cannot, why?

*/
#include <fstream>

void chrome_trace_dump(const char * dump_file_name,
                       uint32_t* blk_x,
                       uint32_t* blk_y,
                       uint32_t* smid,
                       uint64_t* clk_start,
                       uint64_t* clk_dur,
                       int N) {
    std::ofstream fw;
    // start dump
    fw.open(dump_file_name, std::ios::out);
    fw << "{\n";
    fw << "\"schemaVersion\": 1,\n";
    fw << "\"traceEvents\": [\n";
    fw.flush();

    auto pid = 0;
    auto cat = "gpu";

    for (int n = 0; n < N; n+=32) {
        std::stringstream ss;
        ss << "kernel_block_" << 
        auto duration = clk_dur[n];
        auto start = clk_start[n];
        auto sm = smid[n];
        //auto end = tsc.tsc_to_usec(d.tsc_end);

        fw << "{\"ph\": \"X\", \"name\": \"" << name << "\", \"cat\":\"" << cat << "\","
            << "\"pid\": " << sm << ", \"tid\": \"thr" << n <<  "~" << n+31 << "\","
            << "\"ts\": " << std::setprecision (15) << start << ", \"dur\": " << duration << "},\n";
    }
    fw << R"({
        "name": "Profiler End",
        "ph": "i",
        "s": "g",
        "pid": "Traces",
        "tid": "Trace OV Profiler",
        "ts":)"
        << 0 << "}",
        fw << "]\n";
    fw << "}\n";
    auto total_size = fw.tellp();
    fw.close();

}

__forceinline__ __device__ unsigned get_smid()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret; 
    asm volatile ("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

#define FMACNT 16
// mapping between threadIdx & (x,y) are transposed
__global__ void sgemm_max_gflops(float * C,
        uint32_t * blk_x,
        uint32_t * blk_y,
        uint32_t * smid,
        uint64_t * cycles0,
        uint64_t * cycles1,
        int M, int N, int K) {
    const int m = blockIdx.x * WRAP_SIZE + threadIdx.y;
    const int n = blockIdx.y * WRAP_SIZE + threadIdx.x;

    if (m < M && n < N) {
        blk_x[m*N + n] = blockIdx.x;
        blk_y[m*N + n] = blockIdx.y;
        float tmp[FMACNT] = {0};
        float a = m;
        float b = n;
        cycles0[m * N + n] = clock64();
        for (int k = 0; k < K; ++k) {
            for(int c = 0; c < FMACNT; c++)
                tmp[c] = fma(a, b, tmp[c]);
        }
        cycles1[m * N + n] = clock64() - cycles0[m * N + n];
        smid[m * N + n] = get_smid();
        if (tmp[0] == 0.0f) {
            float csum = 0;
            for(int c = 0; c < FMACNT; c++)
                csum += tmp[c];
            C[m * N + n] = csum;
        }
    }
}

std::ostream& operator<<(std::ostream& os, const dim3& d) {
    os << "dim3{" << d.x << ", " << d.y << ", " << d.z << "}";
    return os;
}

void test_max_gflops(int64_t M, int64_t N, int64_t K) {
    tensor2D<float> C(M, N, true);
    tensor2D<uint64_t> clock0(M, N, true);
    tensor2D<uint64_t> clock1(M, N, true);
    tensor2D<uint32_t> smid(M, N, true);
    tensor2D<uint32_t> blk_x(M, N, true);
    tensor2D<uint32_t> blk_y(M, N, true);

    dim3 gridDim(CEIL_DIV(M, WRAP_SIZE), CEIL_DIV(N, WRAP_SIZE));
    dim3 blockDim(WRAP_SIZE, WRAP_SIZE);

    std::cout << "gridDim=" << gridDim << " blockDim=" << blockDim << std::endl;

    auto flops = M*N*K*2*FMACNT;
    auto avg_dur_ns = cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_max_gflops<<<gridDim, blockDim>>>(C, blk_x, blk_y, smid, clock0, clock1, M, N, K);
    }, __func__, __LINE__, "sgemm_max_gflops", 0, flops, 3);


    clock0.to_host();
    clock1.to_host();
    smid.to_host();
    auto* pclock0 = clock0.ptr_host.get();
    auto* pclocks = clock1.ptr_host.get();

    // calibrate all clocks from different SMs
    std::vector<uint64_t> t0(128, std::numeric_limits<uint64_t>::max());
    for(int i = 0; i < M*N; i++) {
        t0[smid[i]] = std::min(t0[smid[i]], clock0[i]);
    }
    for(int i = 0; i < M*N; i++) {
        pclock0[i] -= t0[smid[i]];
        pclocks[i] = pclocks[i];
    }

    auto blocksize = WRAP_SIZE*WRAP_SIZE;

    for (int i = 0; i <(M*N); i+=32, pclocks+=32, pclock0+=32) {
        if ((i % blocksize) == 0)
            std::cout << "============== thread block ============\n";
        std::cout << "thread " << i << " : " << carray(pclock0, 32) << carray(pclocks, 32)
                  << " " << avg_dur_ns << "(ns)"
                  << " GPU_freq:" << pclocks[0]/avg_dur_ns << "(GHz)?"
                  << " FMA:" << FMACNT*K/avg_dur_ns << "(GFMA/s/CUDA-core)?"
                  << std::endl;
    }
    chrome_trace_dump("ct.json", smid, clock0, clock1, (M*N));
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