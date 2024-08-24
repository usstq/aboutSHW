#include <iostream>
#include <sstream>

#include "bf16.hpp"
#include "mxformat.hpp"
using namespace mxformat;

#include <vector>

#include "../include/linux_perf.hpp"

#include "utils.hpp"

#include "omp.h"

static int AMX_DEBUG = getenv("AMX_DEBUG", 12);
static int ADDR_ADV32 = getenv("ADDR_ADV32", 32);
static int PREFETCH_ADV = getenv("PREFETCH_ADV", 4096);

class MXFP4BrgemmAMX : public jit_generator {
public:
    int BM = 0;
    typedef ov::bfloat16 act_type;
    TileConfig m_tile_cfg;

    MXFP4BrgemmAMX(int BM = 1) : BM(BM) {
        log_reset();
        create_kernel("MXFP4BrgemmAMX");
        tile_config_M(m_tile_cfg, BM);
    }

    // M can change by ldtilecfg w/o code-regeneration
    // with the help of :
    //  - m_BM_hint controls dynamic behaviour of the kernel
    //  - tile config controls behaviour of tileload & TMUL
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
        const act_type* src_ptr;
        int64_t src_stride;         // in bytes
        const uint8_t* scale_ptr;   // E8M0
        const uint8_t* weight_ptr;  // E2M1
        float* dst_ptr;
        int64_t dst_stride;  // in bytes
        int64_t mxblock_cnt;
        ov::bfloat16* scratch_buff;  // B0|B1 ; B2|B3
    };

    void generate() override {
        //  abi_param1 ~ abi_param6, RAX, R10, R11
        auto src_ptr = abi_param2;
        auto src_stride = abi_param3;
        auto scale_ptr = abi_param4;
        auto weight_ptr = abi_param5;
        auto dst_ptr = abi_param6;
        auto scratch_ptr = r10;
        auto mxblock_cnt = r11;

        Xbyak::Tmm tmmC00 = tmm0;
        Xbyak::Tmm tmmC01 = tmm1;
        Xbyak::Tmm tmmC10 = tmm2;
        Xbyak::Tmm tmmC11 = tmm3;
        Xbyak::Tmm tmmA0 = tmm4;
        Xbyak::Tmm tmmA1 = tmm5;
        Xbyak::Tmm tmmB0 = tmm6;
        Xbyak::Tmm tmmB1 = tmm7;

        mov(src_ptr, ptr[abi_param1 + offsetof(CallArgs, src_ptr)]);
        mov(src_stride, ptr[abi_param1 + offsetof(CallArgs, src_stride)]);
        mov(scale_ptr, ptr[abi_param1 + offsetof(CallArgs, scale_ptr)]);
        mov(weight_ptr, ptr[abi_param1 + offsetof(CallArgs, weight_ptr)]);
        mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, dst_ptr)]);
        mov(mxblock_cnt, ptr[abi_param1 + offsetof(CallArgs, mxblock_cnt)]);
        mov(scratch_ptr, ptr[abi_param1 + offsetof(CallArgs, scratch_buff)]);

        auto zmm_zero = zmm0;
        auto zmm_e2m1_lut = zmm1;
        auto zmm_exponent_bias = zmm2;
        auto zmm_scales = zmm3;

        Xbyak::Label kx32_loop_end;
        Xbyak::Label kx32_loop_begin;

        // zmm_e2m1_lut is look-up table maps E2M1 into BF16/FP16 values 32 entries
        // with high-16 entries duplicates low-16 entries so 5bits index for vpermw
        // can works irrespect of the value of the highest bit.
        // clang-format off
        static ov::bfloat16 lut_e2m1_bf16[] = {
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
             0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
            // duplicate
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
             0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        // clang-format on
        vpxorq(zmm_zero, zmm_zero, zmm_zero);
        mov(rax, reinterpret_cast<uintptr_t>(lut_e2m1_bf16));
        vmovdqu32(zmm_e2m1_lut, ptr[rax]);
        mov(eax, 0x007f);
        vpbroadcastw(zmm_exponent_bias, ax);

        tilezero(tmmC00);
        tilezero(tmmC01);
        tilezero(tmmC10);
        tilezero(tmmC11);

        {
            auto AMX_DEBUG_Decomp = AMX_DEBUG & 4;
            auto AMX_DEBUG_Tmuls = AMX_DEBUG & 8;
            auto AMX_DEBUG_MOV = AMX_DEBUG & 0x10;
            Xbyak::Reg64 double_buff1 = scratch_ptr;
            Xbyak::Reg64 double_buff2 = r13;
            push(double_buff2);
            lea(double_buff2, ptr[double_buff1 + 2048]);

            auto decompress_scale = [&](int Btile_id) {
                if (!AMX_DEBUG_Decomp) return;
                if (Btile_id == 0) {
                    vmovdqu8(Xbyak::Ymm(zmm_scales.getIdx()), ptr[scale_ptr]);         // 32x E8M0
                    vpmovzxbw(zmm_scales, Xbyak::Ymm(zmm_scales.getIdx()));                  // 32x 16bit
                    vpsubw(zmm_scales, zmm_scales, zmm_exponent_bias);  // subtract bias 127
                    vpsllw(zmm_scales, zmm_scales, 0x7);                // shift-left into bf16/fp16 exponent: 32x scales
                    vpunpcklwd(zmm5, zmm_scales, zmm_scales);
                    prefetcht2(ptr[scale_ptr + 1024]);
                    add(scale_ptr, ADDR_ADV32);
                } else {
                    vpunpckhwd(zmm5, zmm_scales, zmm_scales);
                }
            };

            auto decompress_Btile = [&](int dst_offset, int i0, int i1) {
                if (AMX_DEBUG_MOV) {
                    for (int i = i0; i < i1; i++) {
                        vmovdqu8(ymm6, ptr[weight_ptr]);
                        prefetcht2(ptr[weight_ptr + PREFETCH_ADV]);
                        add(weight_ptr, ADDR_ADV32);
                    }
                    return;
                }
                if (!AMX_DEBUG_Decomp) return;
                for (int i = i0; i < i1; i++) {
                    vmovdqu8(ymm6, ptr[weight_ptr]);   // load 4x16 e2m1 = 32bytes = 256bits
                    vpmovzxbw(zmm6, ymm6);             //
                    vpsrld(zmm7, zmm6, 0x4);           // zmm6 : 32x (0x00, even-nibble) zmm7 : 32x (0x00, odd-nibble)
                    vpermw(zmm6, zmm6, zmm_e2m1_lut);  // e2m1 nibble to bf16
                    vpermw(zmm7, zmm7, zmm_e2m1_lut);  // e2m1 nibble to bf16
                    vpcmpw(k1, zmm6, zmm_zero, 4);
                    vpcmpw(k2, zmm7, zmm_zero, 4);
                    vpaddw(zmm6|k1, zmm6, zmm5);       // bf16 exponent-addition with zero-guard to apply MX-scale
                    vpaddw(zmm7|k2, zmm7, zmm5);       // bf16 exponent-addition with zero-guard to apply MX-scale
                    vmovdqu32(ptr[double_buff1 + i * 128 + dst_offset], zmm6);
                    vmovdqu32(ptr[double_buff1 + i * 128 + dst_offset + 64], zmm7);

                    prefetcht2(ptr[weight_ptr + PREFETCH_ADV]);
                    add(weight_ptr, ADDR_ADV32);
                }
            };

            auto reg_B_stride = rax;
            mov(reg_B_stride, 64);

            Xbyak::Reg64 src_ptr1 = r12;
            if (BM > 16) {
                push(src_ptr1);
                lea(src_ptr1, ptr[src_ptr + src_stride * 8]);
                lea(src_ptr1, ptr[src_ptr1 + src_stride * 8]);
            }

            // decode B0|B1
            decompress_scale(0);
            decompress_Btile(0, 0, 8);
            decompress_scale(1);
            decompress_Btile(1024, 0, 8);

            dec(mxblock_cnt);
            jz(kx32_loop_end, T_NEAR);

            // Loop-body:
            //  load & decode mxblocks    [1, 2, 3, ... , mxblock_cnt-2, mxblock_cnt-1]
            //  TMUL mxblocks          [0, 1, 2, 3, ... , mxblock_cnt-2]
            align(64, false);
            L(kx32_loop_begin);
            {
                // exchange double-buffer
                // still decode into buff1, load from buff2
                xchg(double_buff1, double_buff2);

                decompress_scale(0);

                if (AMX_DEBUG_Tmuls) tileloadd(tmmA0, ptr[src_ptr + src_stride]);

                if (AMX_DEBUG_Tmuls) tileloadd(tmmB0, ptr[double_buff2 + reg_B_stride + 0]);

                if (AMX_DEBUG_Tmuls) tdpbf16ps(tmmC00, tmmA0, tmmB0);
                decompress_Btile(0, 0, 4);

                if (AMX_DEBUG_Tmuls && BM > 16)  tileloadd(tmmA1, ptr[src_ptr1 + src_stride]);

                if (AMX_DEBUG_Tmuls) tileloadd(tmmB1, ptr[double_buff2 + reg_B_stride + 1024]);

                if (AMX_DEBUG_Tmuls && BM > 16) tdpbf16ps(tmmC10, tmmA1, tmmB0);
                decompress_Btile(0, 4, 8);

                decompress_scale(1);

                if (AMX_DEBUG_Tmuls) tdpbf16ps(tmmC01, tmmA0, tmmB1);
                decompress_Btile(1024, 0, 4);

                if (AMX_DEBUG_Tmuls && BM > 16) tdpbf16ps(tmmC11, tmmA1, tmmB1);
                decompress_Btile(1024, 4, 8);

                // move to next MX-block(32 along K dimension)
                // Btile is automatically moved
                lea(src_ptr, ptr[src_ptr + 64]);
                if (BM > 16)
                    lea(src_ptr1, ptr[src_ptr1 + 64]);
            }
            dec(mxblock_cnt);
            jnz(kx32_loop_begin, T_NEAR);

            // last TMUL : [mxblock_cnt-1]
            L(kx32_loop_end);
            xchg(double_buff1, double_buff2);
            if (AMX_DEBUG_Tmuls) {
                tileloadd(tmmA0, ptr[src_ptr + src_stride]);
                tileloadd(tmmB0, ptr[double_buff2 + reg_B_stride + 0]);
                tdpbf16ps(tmmC00, tmmA0, tmmB0);
                if (BM > 16)
                    tileloadd(tmmA1, ptr[src_ptr1 + src_stride]);
                tileloadd(tmmB1, ptr[double_buff2 + reg_B_stride + 1024]);
                if (BM > 16)
                    tdpbf16ps(tmmC10, tmmA1, tmmB0);
                tdpbf16ps(tmmC01, tmmA0, tmmB1);
                if (BM > 16)
                    tdpbf16ps(tmmC11, tmmA1, tmmB1);
            }

            if (BM > 16) {
                pop(src_ptr1);
            }
            pop(double_buff2);
        }

        // store accumulator C0~3 to dst
        auto dst_stride = rax;
        mov(dst_stride, ptr[abi_param1 + offsetof(CallArgs, dst_stride)]);
        tilestored(ptr[dst_ptr + dst_stride], tmmC00);
        tilestored(ptr[dst_ptr + dst_stride + 64], tmmC01);
        if (BM > 16) {
            lea(dst_ptr, ptr[dst_ptr + dst_stride * 8]);
            lea(dst_ptr, ptr[dst_ptr + dst_stride * 8]);
            tilestored(ptr[dst_ptr + dst_stride], tmmC10);
            tilestored(ptr[dst_ptr + dst_stride + 64], tmmC11);
        }

        ret();
    }

    struct e2m1_32x32 {
        uint8_t data[32*32/2];
    };
    struct e8m0_1x32 {
        uint8_t data[32];
    };
    struct PackedWeight {
        tensor2D<e8m0_1x32> scales;
        tensor2D<e2m1_32x32> elements;
        PackedWeight() = default;
        PackedWeight(int K, int N) : elements(N/32, K/32), scales(N/32, K/32) {
            ASSERT(K % 32 == 0);
            ASSERT(N % 32 == 0);
        }
        uint8_t * scales_ptr() {
            return scales[0].data;
        }
        uint8_t * elements_ptr() {
            return elements[0].data;
        }
    };
    // t_mxfp4_weight: (K / 32, N)
    PackedWeight reorder_weights(tensor2D<mxfp4>& t_mxfp4_weight) {
        int num_k_blocks = t_mxfp4_weight.shape[0];
        int K = num_k_blocks * 32;
        int N = t_mxfp4_weight.shape[1];
        int num_n_blocks = N / 32;
        static int nmap_b1[16] = {4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31};
        static int nmap_b0[16] = {0,1,2,3, 8,9,10,11,   16,17,18,19, 24,25,26,27};
        PackedWeight pw(K, N);
        for(int nb = 0; nb < num_n_blocks; nb++) {
            for(int kb = 0; kb < num_k_blocks; kb++) {
                auto* p_scales = pw.scales(nb, kb).data;
                auto* p_elements = pw.elements(nb, kb).data;
                auto* mxfp4_weight = t_mxfp4_weight.ptr(kb, nb*32);
                //============ tile B0 ==================
                for (int m = 0; m < 32; m += 4) {
                    for (int n = 0; n < 16; n++) {
                        auto a = mxfp4_weight[n].get_e2m1(m);
                        auto b = mxfp4_weight[n].get_e2m1(m + 1);
                        auto c = mxfp4_weight[n].get_e2m1(m + 2);
                        auto d = mxfp4_weight[n].get_e2m1(m + 3);
                        p_elements[2 * n + 0] = (c << 4) | a;
                        p_elements[2 * n + 1] = (d << 4) | b;
                    }
                    p_elements += 32;
                }
                for (int n = 0; n < 16; n++) {
                    // to adapt VPUNPCKLWD/VPUNPCKHWD 
                    // if scale_e8m0 is zero, all elements should be zero too;
                    ASSERT(mxfp4_weight[n].scale_e8m0 != 0);
                    p_scales[nmap_b0[n]] = mxfp4_weight[n].scale_e8m0;
                }

                //============ tile B1 ==================
                for (int m = 0; m < 32; m += 4) {
                    for (int n = 0; n < 16; n++) {
                        auto a = mxfp4_weight[16 + n].get_e2m1(m);
                        auto b = mxfp4_weight[16 + n].get_e2m1(m + 1);
                        auto c = mxfp4_weight[16 + n].get_e2m1(m + 2);
                        auto d = mxfp4_weight[16 + n].get_e2m1(m + 3);
                        p_elements[2 * n + 0] = (c << 4) | a;
                        p_elements[2 * n + 1] = (d << 4) | b;
                    }
                    p_elements += 32;
                }
                for (int n = 0; n < 16; n++) {
                    // if scale_e8m0 is zero, all elements should be zero too;
                    ASSERT(mxfp4_weight[16 + n].scale_e8m0 != 0);
                    p_scales[nmap_b1[n]] = mxfp4_weight[16 + n].scale_e8m0;
                }
            }
        }
        return pw;
    }

    tensor2D<ov::bfloat16> get_scratch() {
        return tensor2D<ov::bfloat16>({4, 1024});
    }

    void run(tensor2D<ov::bfloat16>& src, PackedWeight& pw, tensor2D<float>& dst, tensor2D<ov::bfloat16>& scratch, CallArgs& args) {
        TileConfigScope tcfg(m_tile_cfg);

        auto MXBLOCKS = src.shape[1] / 32;
        auto num_n_blocks = pw.scales.shape[0];

        args.src_ptr = src.ptr();
        args.src_stride = src.stride_bytes;
        args.dst_stride = dst.stride_bytes;
        args.mxblock_cnt = MXBLOCKS;
        args.scratch_buff = scratch.ptr();

        for(int nb = 0; nb < num_n_blocks; nb++) {
            args.scale_ptr = pw.scales(nb, 0).data;
            args.weight_ptr = pw.elements(nb, 0).data;
            args.dst_ptr = dst.ptr(0, nb*32);
            (*this)(&args);
        }
    }    
};




static auto REF_VERBOSE = getenv("REF_VERBOSE", 0);
static int CLR_CACHE = getenv("CLR_CACHE", 0);

void test(int BM, int K, int N) {
    int MXBLOCKS = K / 32;
    MXFP4BrgemmAMX jit_kernel(BM);
    using ActType = typename MXFP4BrgemmAMX::act_type;

    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},

        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x04, 0x04),"STALLS_TOTAL"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x06, 0x06),"STALLS_L3_MISS"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x05, 0x05),"STALLS_L2_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x40, 0x02),"BOUND_ON_STORES"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05), "BOUND_ON_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00), "ALL_LOADS"},

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00), "SPLIT_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00), "SPLIT_STORES"},
    });

    tensor2D<ActType> src(BM, K);
    tensor2D<mxfp4> mxfp4_weight(K / 32, N);
    tensor2D<float> ref(BM, N);

    // random weights & src
    float weights[32] = {0};
    for (int i = 0; i < (K / 32) * N; i++) {
        weights[i % 32] = ((rand() % 100) - 50) * 0.1f;
        if (i == 0) {
            // for(int k=0; k<32; k++) weights[k] = k*0.1;
        }
        mxfp4_weight[i].assign(weights);
        if (i == 0) {
            for (int k = 0; k < 32; k++)
                weights[k] = 0;
        }
    }

    for (int m = 0; m < BM; m++) {
        for (int k = 0; k < K; k++) {
            src(m, k) = ((k + m) % 3) - 1;
        }
    }
    src[4] = 1.0f;

    // check accuracy
    exec_reference(src, mxfp4_weight, ref, REF_VERBOSE > 1);

    // reorder weights
    auto pw0 = jit_kernel.reorder_weights(mxfp4_weight);

    struct Work {
        MXFP4BrgemmAMX::PackedWeight weight;
        MXFP4BrgemmAMX::CallArgs args;
        tensor2D<ov::bfloat16> scratch;
        tensor2D<float> dst;
        Work() = default;
    };

    std::vector<Work> works(omp_get_max_threads());

    for(int i = 0; i < omp_get_max_threads(); i++) {
        works[i].scratch = jit_kernel.get_scratch();
        works[i].weight.scales = pw0.scales.clone();
        works[i].weight.elements = pw0.elements.clone();
        works[i].dst = tensor2D<float>(BM, N);
    }

    jit_kernel.run(src, works[0].weight, works[0].dst, works[0].scratch, works[0].args);

    auto all_equal = compare(ref, works[0].dst, REF_VERBOSE);

    printf("%s[%s] ACCURACY:%s\e[0m \n", all_equal ? "\e[32m" : "\e[31m", jit_kernel.name(), all_equal ? "PASSED!" : "FAILED!");
    pevg.show_header();
    // jit_kernel.log_show();

    double avg_cycles = 0;
    double avg_instructions = 0;
    double avg_cpi0 = 0;
    double avg_cpi1 = 0;
    constexpr int REPEATS = 1;

    // warm-up
    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        CLflush::run(works[ithr].weight.elements_ptr(), K*N/2);
    }
    
    for (int i = 0; i < 12; i++) {
        if ((i & 1) == 1) {
            #pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                CLflush::run(works[ithr].weight.elements_ptr(), K*N/2);
            }
            //std::this_thread::sleep_for(std::chrono::seconds(1));
            printf("============\n");
        }

        auto pmc = pevg.rdpmc(
            [&]() {
                // repeat
                #pragma omp parallel
                {
                    int ithr = omp_get_thread_num();
                    jit_kernel.run(src, works[ithr].weight, works[ithr].dst, works[ithr].scratch, works[ithr].args);
                }
            },
            jit_kernel.name(),
            (N/32) * (K/32),
            [&](uint64_t duration_ns, uint64_t* pmc){
                auto bw = (K*N/2 + K*N/32)*1.0/duration_ns;
                printf(" MemBW:%.1f(GB/s) x %d = %.1f(GB/s)", bw, omp_get_max_threads(), bw*omp_get_max_threads());
            });
    }
}

static int BM = getenv("BM", 1);
static int K = getenv("K", 4096);
static int N = getenv("N", 128);


int main() {
    ASSERT(K % 32 == 0);
    ASSERT(N % 32 == 0);

    printf("omp_get_max_threads()=%d\n", omp_get_max_threads());
    bool initAMX = initXTILE();
    test(BM, K, N);
    return 0;
}
