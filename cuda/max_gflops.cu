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


void chrome_trace_dump(const char * dump_file_name, tensorND<uint64_t>& C) {

    // C : [M, N, 8]
    std::ofstream fw;
    // start dump
    fw.open(dump_file_name, std::ios::out);
    fw << "{\n";
    fw << "\"schemaVersion\": 1,\n";
    fw << "\"traceEvents\": [\n";
    fw.flush();

    auto cat = "gpu";
    auto* pdata = C.to_cpu();
    auto sz_last = C.size(-1);
    auto N = C.numel()/sz_last;
    // 32 threads from same warp (if block-size in X direction is larger than 32)
    for (int n = 0; n < N; n+=32, pdata += sz_last*32) {
        auto& blk_x = pdata[0];
        auto& blk_y = pdata[1];
        auto& smid = pdata[2];
        auto& clk_start = pdata[3];
        auto& clk_dur = pdata[4];
        std::stringstream ss;
        ss << "kernel(" << blk_x << "," << blk_y << ")";
        //auto end = tsc.tsc_to_usec(d.tsc_end);

        fw << "{\"ph\": \"X\", \"name\": \"" << ss.str() << "\", \"cat\":\"" << cat << "\","
            << "\"pid\": " << smid << ", \"tid\": \"thr" << n <<  "~" << n+31 << "\","
            << "\"ts\": " << std::setprecision (15) << clk_start << ", \"dur\": " << clk_dur << "},\n";
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
__global__ void sgemm_max_gflops(uint64_t * C, size_t M, size_t N, size_t K) {
    const int m = blockIdx.x * WRAP_SIZE + threadIdx.y;
    const int n = blockIdx.y * WRAP_SIZE + threadIdx.x;

    if (m < M && n < N) {
        auto* pdata = C + (m*N + n)*8;
        pdata[0] = blockIdx.x;
        pdata[1] = blockIdx.y;
        pdata[2] = get_smid();

        float tmp[FMACNT] = {0};
        float a = m;
        float b = n;
        pdata[3] = clock64();
        for (int k = 0; k < K; ++k) {
            for(int c = 0; c < FMACNT; c++)
                tmp[c] = fma(a, b, tmp[c]);
        }
        pdata[4] = clock64() - pdata[3];

        // to prevent optimization
        if (tmp[0] == 1.2134f) {
            float csum = 0;
            for(int c = 0; c < FMACNT; c++)
                csum += tmp[c];
            printf("%f", csum);
        }
    }
}

std::ostream& operator<<(std::ostream& os, const dim3& d) {
    os << "dim3{" << d.x << ", " << d.y << ", " << d.z << "}";
    return os;
}

void test_max_gflops(size_t M, size_t N, size_t K) {
    tensorND<uint64_t> C({M, N, 8});

    dim3 gridDim(CEIL_DIV(M, WRAP_SIZE), CEIL_DIV(N, WRAP_SIZE));
    dim3 blockDim(WRAP_SIZE, WRAP_SIZE);

    std::cout << "gridDim=" << gridDim << " blockDim=" << blockDim << std::endl;

    auto flops = M*N*K*2*FMACNT;
    auto avg_dur_ns = cuda_timeit([&](int i, std::stringstream& ss){
        sgemm_max_gflops<<<gridDim, blockDim>>>(C.to_gpu(), M, N, K);
    }, __func__, __LINE__, "sgemm_max_gflops", 0, flops, 3);

    auto* pdata = C.to_cpu();

    // calibrate all clocks from different SMs
    std::vector<uint64_t> t0(128, std::numeric_limits<uint64_t>::max());
    for(int i = 0; i < M*N; i++, pdata += 8) {
        auto& blk_x = pdata[0];
        auto& blk_y = pdata[1];
        auto& smid = pdata[2];
        auto& clk_start = pdata[3];
        auto& clk_dur = pdata[4];
        
        t0[smid] = std::min(t0[smid], clk_start);
    }
    pdata = C.to_cpu();
    for(int i = 0; i < M*N; i++, pdata += 8) {
        auto& smid = pdata[2];
        auto& clk_start = pdata[3];
        clk_start -= t0[smid];
    }
    /*
    auto blocksize = WRAP_SIZE*WRAP_SIZE;
    for (int i = 0; i <(M*N); i+=32, pclocks+=32, pclock0+=32) {
        if ((i % blocksize) == 0)
            std::cout << "============== thread block ============\n";
        std::cout << "thread " << i << " : " << carray(pclock0, 32) << carray(pclocks, 32)
                  << " " << avg_dur_ns << "(ns)"
                  << " GPU_freq:" << pclocks[0]/avg_dur_ns << "(GHz)?"
                  << " FMA:" << FMACNT*K/avg_dur_ns << "(GFMA/s/CUDA-core)?"
                  << std::endl;
    }*/
    
    chrome_trace_dump("ct.json", C);
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

