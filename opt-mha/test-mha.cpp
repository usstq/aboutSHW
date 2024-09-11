/*

    [B=16, H=32, b=4, S=128, L=2016]
        work-amount B*H

        Q*K   (b*S) x (S*L)
*/

#include <iostream>
#include <sstream>

#include <vector>

#include "../mxfp4/bf16.hpp"
#include "../include/linux_perf.hpp"
#define JIT_DEBUG
#include "../include/jit.h"

#include "omp.h"

static auto B = getenv("B", 16);
static auto H = getenv("H", 32);
static auto S = getenv("S", 128);
static auto L = getenv("L", 2016);
static auto nb = getenv("nb", 4);
static auto NL = getenv("NL", 32);
static int PREFETCH_ADV = getenv("PREFETCH_ADV", 4096);

class AMXMatmul : public jit_generator {
public:
    int m_Ktiles;
    int m_Ntiles;
    int m_M;
    TileConfig m_tile_cfg;

    AMXMatmul(int M, int Ktiles, int Ntiles) : m_M(M), m_Ktiles(Ktiles), m_Ntiles(Ntiles) {
        create_kernel("AMXMatmul");
        tile_config_M(m_tile_cfg, M);
    }

    void tile_config_M(TileConfig& tile_cfg, int M) {
        auto rows0 = 16;
        auto rows1 = 16;
        if (M < 32) {
            // kernel is for processing Mtails
            if (M > 16) {
                rows0 = 16;
                rows1 = M - 16;
            } else {
                //  both A0 & A1 load from same memory, to avoid code-regeneration
                rows0 = rows1 = M;
            }
        }
        tile_cfg.reset(1,
                       0,
                       {
                           {rows0, 64},  // C00:0
                           {rows0, 64},  // C01:1
                           {rows1, 64},  // C10:2
                           {rows1, 64},  // C11:3
                           {rows0, 64},  // A0:4
                           {rows1, 64},  // A1:5
                           {16, 64},     // B0:6
                           {16, 64},     // B1:7
                       });
    }

    struct CallArgs {
        const ov::bfloat16* src_ptr;
        const ov::bfloat16* wei_ptr;
        int64_t src_stride;  // in bytes
        float* dst_ptr;
        int64_t dst_stride;  // in bytes
    };

    void generate() override {
        //  abi_param1 ~ abi_param6, RAX, R10, R11
        // 
        auto src_ptr = abi_param2;
        auto wei_ptr = abi_param3;
        auto src_stride = abi_param4;
        auto dst_ptr = abi_param5;
        auto dst_stride = abi_param6;

        Xbyak::Tmm tmmC00 = tmm0;
        Xbyak::Tmm tmmC01 = tmm1;
        Xbyak::Tmm tmmC10 = tmm2;
        Xbyak::Tmm tmmC11 = tmm3;
        Xbyak::Tmm tmmA0 = tmm4;
        Xbyak::Tmm tmmA1 = tmm5;
        Xbyak::Tmm tmmB0 = tmm6;
        Xbyak::Tmm tmmB1 = tmm7;

        mov(wei_ptr, ptr[abi_param1 + offsetof(CallArgs, wei_ptr)]);
        mov(src_stride, ptr[abi_param1 + offsetof(CallArgs, src_stride)]);
        mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, dst_ptr)]);
        mov(dst_stride, ptr[abi_param1 + offsetof(CallArgs, dst_stride)]);

        auto n_tiles = rax;
        auto wei_stride = r10;
    
        mov(n_tiles, m_Ntiles);
        mov(wei_stride, 64);
        Xbyak::Label loop_begin;

        align(64, false);
        L(loop_begin);
        {
            mov(src_ptr, ptr[abi_param1 + offsetof(CallArgs, src_ptr)]);
            tilezero(tmmC00);
            for(int k_tile = 0; k_tile < m_Ktiles; k_tile++) {
                tileloadd(tmmA0, ptr[src_ptr + src_stride + k_tile * 64]);
                tileloadd(tmmB0, ptr[wei_ptr + wei_stride]);
                lea(wei_ptr, ptr[wei_ptr + 1024]);
                tdpbf16ps(tmmC00, tmmA0, tmmB0);
            }
            tilestored(ptr[dst_ptr + dst_stride], tmmC00);
            lea(dst_ptr, ptr[dst_ptr + 64]);
        }
        dec(n_tiles);
        jnz(loop_begin, T_NEAR);
        ret();
    }

    void run(ov::bfloat16* q, int64_t q_stride_bytes,
             ov::bfloat16* k,
             float* dst, int64_t dst_stride_bytes) {
        // q: [M, K]
        // k: [N/16, S/32*(32*16)]
        CallArgs args;
        args.src_ptr = q;
        args.src_stride = q_stride_bytes;
        args.wei_ptr = k;
        args.dst_ptr = dst;
        args.dst_stride = dst_stride_bytes;
        (*this)(&args);
    }
};

struct Work {
    ov::bfloat16 * q;
    int64_t q_stride_bytes;
    tensor2D<ov::bfloat16> promptKeys;
    Work() = default;
};

std::vector<std::pair<int, int>> split_works(int work_amount, int nthr) {
    std::vector<std::pair<int, int>> ret(nthr, {0, 0});
    // evenly distribute works
    for(int i = 0; i < work_amount; i++) {
        ret[i % nthr].second ++;
    }
    // start-end
    for(int i = 0, wid = 0; i < nthr; i++) {
        ret[i].first = wid;     // start
        wid += ret[i].second;
        ret[i].second = wid;    // end
    }
    return ret;
}

struct Layer {
    tensor2D<ov::bfloat16> m_query;
    std::vector<Work> m_works;
    std::vector<std::pair<int, int>> m_plan;

    std::vector<tensor2D<float>> & get_scratch_C() {
        static std::vector<tensor2D<float>> C;
        return C;
    }

    void setup(int nthr) {

        bool init_data = true;
        int Ktiles = S/32;
        m_works.resize(B*H);
        
        m_plan = split_works(B*H, nthr);

        auto& Cs = get_scratch_C();
        Cs.resize(B*H);

        // B, nb, H, S
        m_query = tensor2D<ov::bfloat16>(B*nb*H, S, init_data);

        for(int b = 0, i = 0; b < B; b++) {
            for(int h = 0; h < H; h++, i++) {
                auto& w = m_works[i];
                w.q = m_query.ptr(b*nb*H + h, 0);
                w.q_stride_bytes = H*S*sizeof(ov::bfloat16);
                w.promptKeys = tensor2D<ov::bfloat16>(L/16, Ktiles * (32*16), init_data);
                Cs[i] = tensor2D<float>(nb, L, init_data);
            }
        }
    }

    void run(AMXMatmul& amxmm) {
        auto& Cs = get_scratch_C();
        #pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            int work0 = m_plan[ithr].first;
            int work1 = m_plan[ithr].second;
            if (work1 > work0) {
                //printf("amxmm.run [%d] %d~%d\n", ithr, work0, work1);
                TileConfigScope tcfg(amxmm.m_tile_cfg);
                for(int wid = work0; wid < work1; wid++) {
                    auto& w = m_works[wid];
                    //printf("amxmm.run [%d] %d/%d\n", ithr, wid, work1);
                    amxmm.run(w.q, w.q_stride_bytes, w.promptKeys.ptr(),
                              Cs[wid].ptr(),
                              Cs[wid].stride_bytes);
                }
            }
        }
    }
};





void test() {
    auto nthr = omp_get_max_threads();
    int Ktiles = S/32;

    AMXMatmul amxmm(nb, Ktiles, L/16);

    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x04, 0x04),"STALLS_TOTAL"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x06, 0x06),"STALLS_L3_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x05, 0x05),"STALLS_L2_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x40, 0x02),"BOUND_ON_STORES"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05), "BOUND_ON_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00), "ALL_LOADS"},

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00), "SPLIT_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00), "SPLIT_STORES"},
    });

    std::vector<Layer> layers(NL);
    for(int n = 0; n < NL; n++) {
        layers[n].setup(nthr);
    }

    pevg.show_header();
    for(int i = 0; i < 10; i++) {
            pevg.rdpmc([&]() {
                int ithr = 0;//omp_get_thread_num();

                for(int n = 0; n < NL; n++) {
                    layers[n].run(amxmm);
                }
            }, "amxmm", 0,
            [&](uint64_t duration_ns, uint64_t * pmu, char*& log) {
                double memSize = NL*(B*H*(S*L + nb*S))*sizeof(ov::bfloat16);
                log += sprintf(log, " MemSize:%.2f(GB) MemBW:%.3f(GB/s)", memSize*1e-9, memSize/duration_ns);
            });
    }
}


int main() {
    printf("omp_get_max_threads()=%d\n", omp_get_max_threads());
    bool initAMX = initXTILE();
    test();
    printf("==========test finshed\n");
    return 0;
}