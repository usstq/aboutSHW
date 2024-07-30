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
//
class MXFP4Brgemm : public jit_generator {
 public:
  MXFP4Brgemm() { log_reset(); create_kernel("MXFP4Brgemm"); }

  void generate() override {
    //  abi_param1 ~ abi_param6, RAX, R10, R11
    auto src_ptr = abi_param1;
    auto scale_ptr = abi_param2;
    auto weight_ptr = abi_param3;
    auto dst_ptr = abi_param4;

    auto zmm_zero = zmm0;
    auto zmm_e2m1_lut = zmm1;
    auto zmm_exponent_bias = zmm2;

    Xbyak::Label loop_begin;

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

    vpxorq(zmm16, zmm16, zmm16);
    vpxorq(zmm17, zmm17, zmm17);

    mov(eax, reinterpret_cast<uintptr_t>(lut_e2m1_bf16));
    vmovdqu32(zmm_e2m1_lut, ptr[eax]);

    mov(eax, 0x007f);
    vpbroadcastw(zmm_exponent_bias, ax);

    // zmm5 : 
    vmovdqu8(ymm5, ptr[scale_ptr]);  // 32x E8M0
    vpmovzxbw(zmm5, ymm5);           // 32x 16bit
    vpcmpw(k1, zmm5, zmm_zero, 0);   // -0.0f would fail the logic here, so MXFP4 must replace -0.0f with +0.0f
    vpsubw(zmm5, zmm5, zmm_exponent_bias);   // subtract bias 127
    vpsllw(zmm5, zmm5, 0x7);    // shift-left into bf16/fp16 exponent: 32x scales

    mov(r10, 32);
    align(64, false);
    L(loop_begin);
    {
        // we should not waste any computation resource
        // load 64x FP4 = 256bits
        // the even/odd 32xFP4 shares same scales, thus they comes from same column
        
        
        // global : zmm_e2m1_lut, zmm_zero
        // input  : zmm5, k1, weight_ptr
        // output : zmm6/zmm7
        // scratch: k2, k3
        vmovdqu8(ymm6, ptr[weight_ptr]);
        vpmovzxbw(zmm6, ymm6);

        vpsrld(zmm7, zmm6, 0x4);
        vpermw(zmm6, zmm6, zmm_e2m1_lut);

        vpcmpw(k2, zmm6, zmm_zero, 0);      // -0.0f would fail the logic here, so MXFP4 must replace -0.0f with +0.0f
        kord(k2, k1, k2);
        vpaddw(zmm6, zmm6, zmm5);
        vpblendmw(zmm6|k2, zmm6, zmm_zero); // 32x bf16/fp16 even weights

        //vmovdqu64(zmm8, zmm6);
        vpermw(zmm7, zmm7, zmm_e2m1_lut);
        vpcmpw(k3, zmm7, zmm_zero, 0);
        kord(k3, k1, k3);
        vpaddw(zmm7, zmm7, zmm5);
        vpblendmw(zmm7|k3, zmm7, zmm_zero); // 32x bf16/fp16 odd weights

        // above decompression process consumes 8.1 cycles per 32bytes-DDR-read
        // for further optimization we can check it in https://uica.uops.info/
        // so 2.8GHz CPU can consume 2.8/8*32=11.2 GB/s memory BW
        // 

        // if we use AVX512-BF16 for computation, additional 4 cycles will be consumed
        // so 2.8GHz CPU can only consume 2.8/12.3*32=7.28 GB/s memory BW

        //vdpbf16ps(zmm18, zmm6, zmm16);
        //vdpbf16ps(zmm19, zmm7, zmm17);

        // if we use AVX512-FMA for computation, additional 5 cycles will be consumed
        // due to vpunpcklwd instructions used for conversion from BF16 to FP32

        // but if we have M>1: multiple rows reuse uncompressed weights using FMA, then
        // FMA would have advantage over AVX512-BF16.

        // there is also another optimization chance, which is to apply MX-scales after FMA
        // this could be much faster since one VPERMD may convert FP4-E2M1 into float32
        // and FMA can run immediately after that, after 32 MX-block size has been accumulated
        // we can apply MX-scales at once, then accumulate into real accumulation buffer

        // and if we have M>BM which means register blocking is not enough to hold all activations
        // we need to save the extracted weights to cache and reuse it.
        // in this case, we also need to consider using ping-pong buffer to interleave weight-decompression
        // with FMA/AMX

        // load & broadcast 2xbf16 or 1xfp32 from source
        vpbroadcastd(zmm8, ptr[src_ptr]);

        // FMA path:
        //   zmm6 contains 1xrow 32 weights
        //   zmm7 contains 1xrow 32 weights
        vpunpcklwd(zmm9, zmm_zero, zmm6);
        vpunpcklwd(zmm10, zmm_zero, zmm7);

        vfmadd231ps(zmm16, zmm8, zmm9);
        vfmadd231ps(zmm16, zmm8, zmm10);

        vpunpckhwd(zmm9, zmm_zero, zmm6);
        vpunpckhwd(zmm10, zmm_zero, zmm7);

        vfmadd231ps(zmm17, zmm8, zmm9);
        vfmadd231ps(zmm17, zmm8, zmm10);

        //vmovdqu32(ptr[dst_ptr], zmm6);
        //vmovdqu32(ptr[dst_ptr + 64], zmm7);

        // store 2x32 bf16 decompressed weights to scratch-buffer
        // if these weights are supposed to be used by AMX-BF16 or AVX512-BF16
        // so they should be in repacked format, and each 512-bits register should contain 2-row 16-OC-channels
        //
        // but if these weights are to be converted to FP32 and used by FMA, then depending on the conversion method:
        //  normal AVX512  (VPMOVZXWD + vpslld/vpunpcklwd):        each 512-bits register should contain 1-row 16-OC-channels.
        //  AVX-NE-CONVERT (VCVTNEOBF162PS + vpslld):   each 512-bits register should contain 1-row 32-OC-channels.
        //

        //log_zmm("bf16", zmm6);
        //log_zmm("bf16", zmm7);
    }
    add(src_ptr, 4);
    add(weight_ptr, 32);
    dec(r10);
    jne(loop_begin, T_NEAR);

    // vpbroadcastd _mm512_set1_epi32
    // vdpbf16ps    _mm512_dpbf16_ps
    //vmovdqu16(ptr[dst_ptr], zmm6);
    //vmovdqu16(ptr[dst_ptr + 64], zmm7);

    ret();
  }

  void self_test_perf() {
    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    });
    uint8_t scales[32];
    uint8_t element[32*32];
    uint8_t scratch[32][64] = {0};

    for(int i=0; i < 32*32; i+=8) {
        element[i] = 0x12;
        element[i+1] = 0x32;
        element[i+2] = 0x56;
        element[i+3] = 0x76;
        element[i+4] = 0x9A;
        element[i+5] = 0xBC;
        element[i+6] = 0xDE;
        element[i+7] = 0xF0;
    }
    constexpr int REPEATS = 100;
    for(int i = 0; i < 10; i++) {
        auto pmc = pevg.rdpmc([&]() {
            // repeat
            for(int repeat = 0; repeat < REPEATS; repeat++)
                (*this)(scales, element, scratch);
        }, true);
        printf("CPI: %.2f  cycles-per-iteration: %.2f\n", (double)(pmc[0])/pmc[1], double(pmc[0])/(32*REPEATS));
    }
  }

  void self_test() {
    // generate scales & elements
    uint8_t scales[32];
    // totally 256bits 64xE2M1 elements (32bytes)
    // two rows of 32 elements are interleaved:
    //     - even 4bit-nibbles contain first row
    //     - odd 4bit-nibbles contain second row
    // two rows share same 32 scales
    // in bf16, each row can fit a AVX512 register
    uint8_t element[] = {0x12, 0x34, 0x56, 0x76, 0x9A, 0xBC, 0xDE, 0xF0,
                         0x12, 0x34, 0x56, 0x76, 0x9A, 0xBC, 0xDE, 0xF0,
                         0x12, 0x34, 0x56, 0x76, 0x9A, 0xBC, 0xDE, 0xF0,
                         0x12, 0x34, 0x56, 0x76, 0x9A, 0xBC, 0xDE, 0xF0,
                         };
    uint8_t scratch[32][64] = {0};

    for(int i = 0; i < 32; i++) scales[i] = 127 + (i%8) - 4;

    // AVX512 will treat even
    // show reference results
    for(int row = 0; row < 2; row++) {
        for(int i = 0; i < 32; i++) {
            auto fscale = e8m0_to_float(scales[i]);
            auto e2m1 = (row & 1) ? (element[i] >> 4) : (element[i] & 0xF);
            auto fp4_value = e2m1_to_float(e2m1);
            printf("(%d,%2d) scale: %f(0x%x)  element: %f(0x%x)  =>  %f\n",
                    row, i,
                    fscale, scales[i],
                    fp4_value, e2m1,
                    fp4_value * fscale);
        }
    }

    (*this)(scales, element, scratch);
    log_show();
  }
};



class MXFP4BrgemmFMA : public jit_generator {
 public:
  int BM = 0;
  MXFP4BrgemmFMA(int BM=1) : BM(BM) { log_reset(); create_kernel("MXFP4BrgemmFMA"); }

  void generate() override {
    //  abi_param1 ~ abi_param6, RAX, R10, R11
    auto src_ptr = abi_param1;
    auto src_stride = abi_param2;
    auto scale_ptr = abi_param3;
    auto weight_ptr = abi_param4;
    auto dst_ptr = abi_param5;
    auto mxblock_cnt = abi_param6;

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
        return Xbyak::Zmm(16 + 2*m + n);
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

    mov(r10, 16);
    align(64, false);
    L(loop_begin);
    {
        // global : zmm_e2m1_lut, zmm_zero
        // input  : zmm5, k1, weight_ptr
        // output : zmm6/zmm7
        // scratch: k2, k3
        vmovdqu8(ymm6, ptr[weight_ptr]);    // load 64x FP4 or 32x(nibble-pair) = 256bits
        vpmovzxbw(zmm6, ymm6);              //      32x (0x00, nibble-pair)

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
    dec(r10);
    jnz(loop_begin, T_NEAR);

    add(scale_ptr, 32);
    dec(mxblock_cnt);
    jnz(kx32_loop_begin, T_NEAR);
    
    log_zmm("bf16", zmm16);
    log_zmm("bf16", zmm17);

    // 
    for(int m = 0; m < BM; m++) {
        vmovdqu32(ptr[dst_ptr], acc_zmm(m, 0));
        vmovdqu32(ptr[dst_ptr + 64], acc_zmm(m, 1));
        add(dst_ptr, src_stride);
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
    pw.scales.resize(mxblocks, 0);
    pw.elements.resize(mxblocks*32*32/2, 0);
    
    return pw;
  }

  // convert FP4 into required layout
  // A: BM x (32*mxblocks)
  // weight: mxblocks x (32 x mxfp4) (each mxfp4 has 32 elements)
  // C: BM x 32
  void exec_reference(float *src, int src_stride,
                      mxfp4 * mxfp4_weight,
                      float * dst, int dst_stride,
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
        float* A = src + m*src_stride;
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
    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    });
    #define MXBLOCKS 2

    float src[32*MXBLOCKS] = {0};
    mxfp4 mxfp4_weight[MXBLOCKS*32];
    float dst[32*MXBLOCKS] = {0};
    float ref[32*MXBLOCKS] = {0};

    float weights[32] = {0};
    for(int i=0; i<MXBLOCKS*32; i++) {
        auto mask = (rand() % 100) - 50;
        weights[i % 32] = mask * 0.1;
        mxfp4_weight[i].assign(weights);
    }

    src[3] = 2.0f;
    // check accuracy
    exec_reference(src, 0, mxfp4_weight, ref, 0, 1, MXBLOCKS, true);

    // reorder weights
    PackedWeight pw = reorder_weights(mxfp4_weight, MXBLOCKS);
    (*this)(src, 0, &pw.scales[0], &pw.elements[0], dst, MXBLOCKS);


    constexpr int REPEATS = 100;
    for(int i = 0; i < 10; i++) {
        auto pmc = pevg.rdpmc([&]() {
            // repeat
            for(int repeat = 0; repeat < REPEATS; repeat++)
                (*this)(src, 0, &pw.scales[0], &pw.elements[0], dst, MXBLOCKS);
        }, true);
        printf("CPI: %.2f  cycles-per-iteration: %.2f\n", (double)(pmc[0])/pmc[1], double(pmc[0])/(REPEATS*16*MXBLOCKS));
    }
  }
};

int main() {
  //test_CPI();
  MXFP4BrgemmFMA brg;
  //brg.self_test();
  brg.self_test();
  return 0;
}