#include <iostream>
#include <sstream>

#include "bf16.hpp"
#include "mxformat.hpp"
using namespace mxformat;

#include <vector>

#include "../perf/linux_perf.hpp"

#define JIT_DEBUG
#include "jit.h"


// decompress MXFP4 weights on-the-fly and do matmul in brgemm style
// jit kernel does a small block sub-matrix matmul:
//    C=A @ B
//
//  A: BM x BK  float32 (or bf16 in AMX case)
//  B: BK x BN  MXFP4 
//  C: BM x BN  float32
//
// C is accumulation register block which limits the size of BM x BN
//
// BK=32 which is also MX-scaling block size, so MX-scales can be shared within jit kernel
// since avx512 FMA is used , 512/32=16 fp32 can be saved in zmm, so BN is multiple of 16
// 
// decompression of MXFP4 is very expensive, but it needs to be done on every BMxBN sub-block
// so the on-the-fly mode only works well when BM==M, which also means M is small.
// 
// in case M is big, decompressed weights must be kept resident in cache and reused across blocks
//
// let's focus on latency case in which M is just 1 (or 4).
//

class MXFP4BrgemmFMA : public jit_generator {
 public:
  int BM = 0;
  MXFP4BrgemmFMA(int BM=1) : BM(BM) { log_reset(); create_kernel("MXFP4BrgemmFMA"); }
    struct CallArgs {
        const float* src_ptr;
        int64_t src_stride;   // in bytes
        const uint8_t* scale_ptr;  // bfloat16
        const uint8_t* weight_ptr;  // float32
        float * dst_ptr;
        int64_t dst_stride;    // in bytes
        int64_t mxblock_cnt;
    };

  void generate() override {
    //  abi_param1 ~ abi_param6, RAX, R10, R11
    auto src_ptr = abi_param2;
    auto src_stride = abi_param3;
    auto scale_ptr = abi_param4;
    auto weight_ptr = abi_param5;
    auto dst_ptr = abi_param6;
    auto mx_size = r10;
    auto mxblock_cnt = r11;

    mov(src_ptr, ptr[abi_param1 + offsetof(CallArgs, src_ptr)]);
    mov(src_stride, ptr[abi_param1 + offsetof(CallArgs, src_stride)]);
    mov(scale_ptr, ptr[abi_param1 + offsetof(CallArgs, scale_ptr)]);
    mov(weight_ptr, ptr[abi_param1 + offsetof(CallArgs, weight_ptr)]);
    mov(dst_ptr, ptr[abi_param1 + offsetof(CallArgs, dst_ptr)]);
    mov(mxblock_cnt, ptr[abi_param1 + offsetof(CallArgs, mxblock_cnt)]);

    auto zmm_zero = zmm0;
    auto zmm_e2m1_lut = zmm1;
    auto zmm_exponent_bias = zmm2;

    Xbyak::Label loop_begin;
    Xbyak::Label kx32_loop_begin;

    // zmm_e2m1_lut is look-up table maps E2M1 into BF16/FP16 values 32 entries with high-16 entries duplicates low-16 entries
    // so 5bits index for vpermw can works irrespect of the value of the highest bit.
    static ov::bfloat16 lut_e2m1_bf16[] ={
        0.0f,   0.5f,  1.0f,   1.5f,   2.0f,   3.0f,  4.0f,   6.0f,
        -0.0f,  -0.5f,-1.0f,  -1.5f,  -2.0f,  -3.0f, -4.0f,  -6.0f,
        // duplicate
        0.0f,   0.5f,  1.0f,   1.5f,   2.0f,   3.0f,  4.0f,   6.0f,
        -0.0f,  -0.5f,-1.0f,  -1.5f,  -2.0f,  -3.0f, -4.0f,  -6.0f,
    };

    vpxorq(zmm_zero, zmm_zero, zmm_zero);

    // accumulation registers
    auto acc_zmm = [&](int m, int n) {
        return Xbyak::Zmm(11 + 2*m + n);
    };
    for(int m = 0; m < BM; m++) {
        auto zmmA = acc_zmm(m, 0);
        auto zmmB = acc_zmm(m, 1);
        vpxorq(zmmA, zmmA, zmmA);
        vpxorq(zmmB, zmmB, zmmB);
    }

    mov(eax, reinterpret_cast<uintptr_t>(lut_e2m1_bf16));
    vmovdqu32(zmm_e2m1_lut, ptr[eax]);

    mov(eax, 0x007f);
    vpbroadcastw(zmm_exponent_bias, ax);

    //
    L(kx32_loop_begin);

    // zmm5 : 32x MX scales
    vmovdqu8(ymm5, ptr[scale_ptr]);  // 32x E8M0
    vpmovzxbw(zmm5, ymm5);           // 32x 16bit
    vpcmpw(k1, zmm5, zmm_zero, 0);   // -0.0f would fail the logic here, so MXFP4 must replace -0.0f with +0.0f
    vpsubw(zmm5, zmm5, zmm_exponent_bias);   // subtract bias 127
    vpsllw(zmm5, zmm5, 0x7);         // shift-left into bf16/fp16 exponent: 32x scales

    mov(mx_size, 16);
    align(64, false);
    L(loop_begin);
    {
        // global : zmm_e2m1_lut, zmm_zero
        // input  : zmm5, k1, weight_ptr
        // output : zmm6/zmm7
        // scratch: k2, k3
        vmovdqu8(ymm6, ptr[weight_ptr]);    // load 64x FP4 or 32x(nibble-pair) = 256bits
        vpmovzxbw(zmm6, ymm6);              //      32x (0x00, nibble-pair)

        //log_zmm("u16", __LINE__, zmm6);

        vpsrld(zmm7, zmm6, 0x4);            // zmm6 : 32x (0x00, even-nibble) zmm7 : 32x (0x00, odd-nibble)
        vpermw(zmm6, zmm6, zmm_e2m1_lut);   // e2m1 nibble to float32

        vpcmpw(k2, zmm6, zmm_zero, 0);      // -0.0f would fail the logic here, so MXFP4 must replace -0.0f with +0.0f
        kord(k2, k1, k2);
        vpaddw(zmm6, zmm6, zmm5);
        vpblendmw(zmm6|k2, zmm6, zmm_zero); // 32x bf16/fp16 even weights: w0, w2, w4, ... 

        vpermw(zmm7, zmm7, zmm_e2m1_lut);
        vpcmpw(k3, zmm7, zmm_zero, 0);
        kord(k3, k1, k3);
        vpaddw(zmm7, zmm7, zmm5);
        vpblendmw(zmm7|k3, zmm7, zmm_zero); // 32x bf16/fp16 odd weights: w1, w3, w5, ...

        //log_zmm("bf16", __LINE__, zmm6);
        //log_zmm("bf16", __LINE__, zmm7);

        // the even/odd part share same scales, thus they come from 2 adjacent rows of weight-matrix(belong to same MX block)
        vpunpcklwd(zmm9, zmm_zero, zmm6);
        vpunpckhwd(zmm10, zmm_zero, zmm6);
        mov(rax, src_ptr);
        for(int m = 0; m < BM; m++) {
            vpbroadcastd(zmm8, ptr[rax]);    // load & broadcast a fp32 from source
            vfmadd231ps(acc_zmm(m, 0), zmm8, zmm9);
            vfmadd231ps(acc_zmm(m, 1), zmm8, zmm10);
            add(rax, src_stride);
        }

        vpunpcklwd(zmm9, zmm_zero, zmm7);
        vpunpckhwd(zmm10, zmm_zero, zmm7);
        mov(rax, src_ptr);
        for(int m = 0; m < BM; m++) {
            vpbroadcastd(zmm8, ptr[rax + 4]);    // load & broadcast a fp32 from source
            vfmadd231ps(acc_zmm(m, 0), zmm8, zmm9);
            vfmadd231ps(acc_zmm(m, 1), zmm8, zmm10);
            add(rax, src_stride);
        }
    }
    add(src_ptr, 8);
    add(weight_ptr, 32);
    dec(mx_size);
    jnz(loop_begin, T_NEAR);

    add(scale_ptr, 32);
    dec(mxblock_cnt);
    jnz(kx32_loop_begin, T_NEAR);
    
    //log_zmm("f32", __LINE__, acc_zmm(0, 0));
    //log_zmm("f32", __LINE__, acc_zmm(0, 1));
    vmovdqu32(ptr[dst_ptr], acc_zmm(0, 0));
    vmovdqu32(ptr[dst_ptr + 64], acc_zmm(0, 1));

    if (BM > 0) {
        mov(rax, ptr[abi_param1 + offsetof(CallArgs, dst_stride)]);
        for(int m = 1; m < BM; m++) {
            add(dst_ptr, rax);
            vmovdqu32(ptr[dst_ptr], acc_zmm(m, 0));
            vmovdqu32(ptr[dst_ptr + 64], acc_zmm(m, 1));
        }
    }
    ret();
  }

  struct PackedWeight {
    std::vector<uint8_t> scales;
    std::vector<uint8_t> elements;
  };
  // weight: mxblocks x (32 x mxfp4)
  PackedWeight reorder_weights(mxfp4 * mxfp4_weight, int mxblocks) {
    PackedWeight pw;
    pw.scales.resize(mxblocks*32, 0);
    pw.elements.resize(mxblocks*32*32/2, 0);
    
    const int nmap[] = {
        0,  1,  2,  3,  16,17,18,19,
        4,  5,  6,  7,  20,21,22,23,
        8,  9, 10, 11,  24,25,26,27,
        12,13, 14, 15,  28,29,30,31
    };

    uint8_t * p_scales = &pw.scales[0];
    uint8_t * p_elements = &pw.elements[0];
    for(int mblk = 0; mblk < mxblocks; mblk++) {
        for(int m = 0; m < 32; m+=2) {
            for(int n = 0; n < 32; n++) {
                auto idx = nmap[n];
                p_scales[n] = mxfp4_weight[idx].scale_e8m0;

                auto e2m1_even = mxfp4_weight[idx].get_e2m1(m);
                auto e2m1_odd = mxfp4_weight[idx].get_e2m1(m + 1);
                p_elements[n] = (e2m1_odd << 4) | e2m1_even;
            }
            p_elements += 32;
        }
        p_scales += 32;
        mxfp4_weight += 32;
    }
    return pw;
  }

  // convert FP4 into required layout
  // A: BM x (32*mxblocks)
  // weight: mxblocks x (32 x mxfp4) (each mxfp4 has 32 elements)
  // C: BM x 32
  void exec_reference(const float *src, int64_t src_stride,
                     mxfp4 * mxfp4_weight,
                     float *dst, int64_t dst_stride,
                     int BM, int mxblocks, bool verbose=false) {
    // decompress mxfp4 first
    int BK = mxblocks*32; // 32-elements per MX-block
    int BN = 32;
    std::vector<float> weight(BN * BK, 0);

    auto getW = [&](int k, int n) -> float& {
        return weight[n*BK + k];
    };
    for(int k = 0; k < BK; k+=32) {
        for(int n = 0; n < BN; n++) {
            for(int ki=0; ki < 32; ki++) {
                getW(k + ki, n) = mxfp4_weight[n][ki];
            }
        }
        mxfp4_weight += BN;
    }
    for(int m = 0; m < BM; m++) {
        const float* A = src + m*src_stride;
        float* C = dst + m*dst_stride;

        for(int n = 0; n < BN; n++) {
            float sum = 0;
            for(int k = 0; k < BK; k++) {
                sum += A[k] * getW(k, n);
            }
            C[n] = sum;
        }
    }

    if (verbose) {
        printf("exec_reference src:\n");
        for(int k=0; k<BK; k++) printf("%4.1f,", src[k]);
        printf("\n");
        printf("exec_reference weight:\n");
        for(int k=0; k<BK; k++) {
            for(int n=0; n<BN; n++) {
                printf("%4.1f,", getW(k, n));
            }
            printf("\n");
        }
        printf("exec_reference dst:\n");
        for(int n=0; n<BN; n++) printf("%4.1f,", dst[n]);
        printf("\n");
    }
  }

  void self_test() {
    CallArgs args;

    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    });
    #define MXBLOCKS 128

    std::vector<float>src(BM*32*MXBLOCKS, 0);
    std::vector<mxfp4>mxfp4_weight(MXBLOCKS*32);
    std::vector<float> dst(BM*32, 0);
    std::vector<float> ref(BM*32, 0);

    // random weights & src
    float weights[32] = {0};
    for(int i=0; i<MXBLOCKS*32; i++) {
        auto mask = (rand() % 100) - 50;
        weights[i % 32] = mask * 0.1;
        mxfp4_weight[i].assign(weights);
    }
    for(int m = 0; m < BM; m++) {
        for(int i=0; i<MXBLOCKS*32; i++)
            src[m*MXBLOCKS*32 + i] = ((i + m)% 3)-1;
    }
    // check accuracy
    exec_reference(src.data(), 32*MXBLOCKS, mxfp4_weight.data(), ref.data(), 32, BM, MXBLOCKS, false);

    // reorder weights
    PackedWeight pw = reorder_weights(mxfp4_weight.data(), MXBLOCKS);

    args.src_ptr = src.data();
    args.src_stride = sizeof(float)*32*MXBLOCKS;
    args.scale_ptr = &pw.scales[0];
    args.weight_ptr = &pw.elements[0];
    args.dst_ptr = dst.data();
    args.dst_stride = sizeof(float)*32;
    args.mxblock_cnt = MXBLOCKS;

    (*this)(&args);
    bool all_equal = true;
    for(int m = 0; m < BM; m++) {
        printf("\nref[%d]:", m); for(int n=0; n<32; n++) printf("%4.1f,", ref[m*32 + n]);
        printf("\njit[%d]:", m); for(int n=0; n<32; n++) {
            bool equal = dst[m*32 + n] == ref[m*32 + n];
            if (!equal) printf("\e[33m");
            printf("%4.1f,", dst[m*32 + n]);
            if (!equal) printf("\e[0m");
            all_equal = all_equal && equal;
        }
    }
    printf("\n[ACCURACY] %s\n", all_equal ? "PASSED!" : "FAILED!");

    log_show();

    constexpr int REPEATS = 100;
    for(int i = 0; i < 10; i++) {
        auto pmc = pevg.rdpmc([&]() {
            // repeat
            for(int repeat = 0; repeat < REPEATS; repeat++)
                (*this)(&args);
        }, true);
        printf("CPI: %.2f  cycles-per-iteration: %.2f\n", (double)(pmc[0])/pmc[1], double(pmc[0])/(REPEATS*16*MXBLOCKS));
    }
  }
};

int main(int argc, const char * argv[]) {
  int BM = argc > 1 ? atoi(argv[1]): 1;
  MXFP4BrgemmFMA brg(BM);
  brg.self_test();
  return 0;
}