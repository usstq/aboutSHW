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
// BK=32 which is also MX-scaling block size, so MX-scales can be shared within
// jit kernel since avx512 FMA is used , 512/32=16 fp32 can be saved in zmm, so
// BN is multiple of 16
//
// decompression of MXFP4 is very expensive, but it needs to be done on every
// BMxBN sub-block so the on-the-fly mode only works well when BM==M, which also
// means M is small.
//
// in case M is big, decompressed weights must be kept resident in cache and
// reused across blocks
//
// let's focus on latency case in which M is just 1 (or 4).
//

template <bool is_vmul_scale>
class MXFP4BrgemmFMA : public jit_generator {
public:
    int BM = 0;
    typedef float act_type;

    MXFP4BrgemmFMA(int BM = 1) : BM(BM) {
        log_reset();
        create_kernel(is_vmul_scale ? "MXFP4BrgemmFMA:vmul" : "MXFP4BrgemmFMA:add");
    }
    struct CallArgs {
        const act_type* src_ptr;
        int64_t src_stride;         // in bytes
        const uint8_t* scale_ptr;   // E8M0
        const uint8_t* weight_ptr;  // E2M1
        float* dst_ptr;
        int64_t dst_stride;  // in bytes
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

        // zmm_e2m1_lut is look-up table maps E2M1 into BF16/FP16 values 32 entries
        // with high-16 entries duplicates low-16 entries so 5bits index for vpermw
        // can works irrespect of the value of the highest bit.
        // clang-format off
        static ov::bfloat16 lut_e2m1_bf16[] = {
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
            // duplicate
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        // clang-format on
        vpxorq(zmm_zero, zmm_zero, zmm_zero);

        // accumulation registers
        auto acc_zmm = [&](int m, int n) {
            return Xbyak::Zmm(11 + 2 * m + n);
        };
        for (int m = 0; m < BM; m++) {
            auto zmmA = acc_zmm(m, 0);
            auto zmmB = acc_zmm(m, 1);
            vpxorq(zmmA, zmmA, zmmA);
            vpxorq(zmmB, zmmB, zmmB);
        }

        mov(rax, reinterpret_cast<uintptr_t>(lut_e2m1_bf16));
        vmovdqu32(zmm_e2m1_lut, ptr[rax]);

        mov(eax, 0x007f);
        vpbroadcastw(zmm_exponent_bias, ax);

        //
        L(kx32_loop_begin);

        // zmm5 : 32x MX scales

        if (!is_vmul_scale) {
            vmovdqu8(ymm5, ptr[scale_ptr]);         // 32x E8M0
            vpmovzxbw(zmm5, ymm5);                  // 32x 16bit
            vpcmpw(k1, zmm5, zmm_zero, 0);          // -0.0f would fail the logic here, so
                                                    // MXFP4 must replace -0.0f with +0.0f
            vpsubw(zmm5, zmm5, zmm_exponent_bias);  // subtract bias 127
            vpsllw(zmm5, zmm5,
                   0x7);  // shift-left into bf16/fp16 exponent: 32x scales
        } else {
            vmovdqu8(ymm5, ptr[scale_ptr]);  // 32x E8M0
            vpmovzxbw(zmm5, ymm5);           // 32x 16bit
            vpsllw(zmm5, zmm5, 0x7);         // bf16
            vpunpcklwd(zmm3, zmm_zero, zmm5);
            vpunpckhwd(zmm4, zmm_zero, zmm5);
        }

        mov(mx_size, 16);
        align(64, false);
        L(loop_begin);
        {
            // global : zmm_e2m1_lut, zmm_zero
            // input  : zmm5, k1, weight_ptr
            // output : zmm6/zmm7
            // scratch: k2, k3
            vmovdqu8(ymm6,
                     ptr[weight_ptr]);  // load 64x FP4 or 32x(nibble-pair) = 256bits
            vpmovzxbw(zmm6, ymm6);      //      32x (0x00, nibble-pair)

            // log_zmm("u16", __LINE__, zmm6);

            vpsrld(zmm7, zmm6,
                   0x4);                       // zmm6 : 32x (0x00, even-nibble) zmm7 : 32x (0x00, odd-nibble)
            vpermw(zmm6, zmm6, zmm_e2m1_lut);  // e2m1 nibble to float32
            vpermw(zmm7, zmm7, zmm_e2m1_lut);

            if (!is_vmul_scale) {
                // use exponent-add to apply MX-scale
                vpcmpw(k2, zmm6, zmm_zero, 0);  // -0.0f would fail the logic here, so
                                                // MXFP4 must replace -0.0f with +0.0f
                vpcmpw(k3, zmm7, zmm_zero, 0);

                kord(k2, k1, k2);
                kord(k3, k1, k3);

                vpaddw(zmm6, zmm6, zmm5);
                vpaddw(zmm7, zmm7, zmm5);

                vpblendmw(zmm6 | k2, zmm6,
                          zmm_zero);  // 32x bf16/fp16 even weights: w0, w2, w4, ...
                vpblendmw(zmm7 | k3, zmm7,
                          zmm_zero);  // 32x bf16/fp16 odd weights: w1, w3, w5, ...
            }

            // log_zmm("bf16", __LINE__, zmm6);
            // log_zmm("bf16", __LINE__, zmm7);

            // the even/odd part share same scales, thus they come from 2 adjacent
            // rows of weight-matrix(belong to same MX block)
            vpunpcklwd(zmm9, zmm_zero, zmm6);
            vpunpckhwd(zmm10, zmm_zero, zmm6);

            if (is_vmul_scale) {
                // use vmulps to apply MX-scales
                vmulps(zmm9, zmm9, zmm3);
                vmulps(zmm10, zmm10, zmm4);
            }

            mov(rax, src_ptr);
            for (int m = 0; m < BM; m++) {
                vpbroadcastd(zmm8, ptr[rax]);  // load & broadcast a fp32 from source
                vfmadd231ps(acc_zmm(m, 0), zmm8, zmm9);
                vfmadd231ps(acc_zmm(m, 1), zmm8, zmm10);
                add(rax, src_stride);
            }

            vpunpcklwd(zmm9, zmm_zero, zmm7);
            vpunpckhwd(zmm10, zmm_zero, zmm7);
            if (is_vmul_scale) {
                // use vmulps to apply MX-scales
                vmulps(zmm9, zmm9, zmm3);
                vmulps(zmm10, zmm10, zmm4);
            }

            mov(rax, src_ptr);
            for (int m = 0; m < BM; m++) {
                vpbroadcastd(zmm8,
                             ptr[rax + 4]);  // load & broadcast a fp32 from source
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

        // log_zmm("f32", __LINE__, acc_zmm(0, 0));
        // log_zmm("f32", __LINE__, acc_zmm(0, 1));
        vmovdqu32(ptr[dst_ptr], acc_zmm(0, 0));
        vmovdqu32(ptr[dst_ptr + 64], acc_zmm(0, 1));

        if (BM > 0) {
            mov(rax, ptr[abi_param1 + offsetof(CallArgs, dst_stride)]);
            for (int m = 1; m < BM; m++) {
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
    PackedWeight reorder_weights(mxfp4* mxfp4_weight, int mxblocks) {
        PackedWeight pw;
        pw.scales.resize(mxblocks * 32, 0);
        pw.elements.resize(mxblocks * 32 * 32 / 2, 0);

        const int nmap[] = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};

        uint8_t* p_scales = &pw.scales[0];
        uint8_t* p_elements = &pw.elements[0];
        for (int mblk = 0; mblk < mxblocks; mblk++) {
            for (int m = 0; m < 32; m += 2) {
                for (int n = 0; n < 32; n++) {
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

    void run(CallArgs* args_ptr) {
        (*this)(args_ptr);
    }
};

class MXFP4BrgemmAVX512_BF16 : public jit_generator {
public:
    int BM = 0;
    typedef ov::bfloat16 act_type;

    MXFP4BrgemmAVX512_BF16(int BM = 1) : BM(BM) {
        log_reset();
        create_kernel("MXFP4BrgemmAVX512_BF16");
    }
    struct CallArgs {
        const act_type* src_ptr;
        int64_t src_stride;         // in bytes
        const uint8_t* scale_ptr;   // E8M0
        const uint8_t* weight_ptr;  // E2M1
        float* dst_ptr;
        int64_t dst_stride;  // in bytes
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

        // zmm_e2m1_lut is look-up table maps E2M1 into BF16/FP16 values 32 entries
        // with high-16 entries duplicates low-16 entries so 5bits index for vpermw
        // can works irrespect of the value of the highest bit.
        // clang-format off
        static ov::bfloat16 lut_e2m1_bf16[] = {
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
            // duplicate
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
        };
        // clang-format on
        vpxorq(zmm_zero, zmm_zero, zmm_zero);

        // accumulation registers
        auto acc_zmm = [&](int m, int n) {
            return Xbyak::Zmm(11 + 2 * m + n);
        };
        for (int m = 0; m < BM; m++) {
            auto zmmA = acc_zmm(m, 0);
            auto zmmB = acc_zmm(m, 1);
            vpxorq(zmmA, zmmA, zmmA);
            vpxorq(zmmB, zmmB, zmmB);
        }

        mov(rax, reinterpret_cast<uintptr_t>(lut_e2m1_bf16));
        vmovdqu32(zmm_e2m1_lut, ptr[rax]);

        mov(eax, 0x007f);
        vpbroadcastw(zmm_exponent_bias, ax);

        //
        L(kx32_loop_begin);

        // zmm5 : 32x MX scales
        vmovdqu8(ymm5, ptr[scale_ptr]);  // 32x E8M0
        vpmovzxbw(zmm5, ymm5);           // 32x 16bit
        vpunpcklwd(zmm3, zmm5, zmm5);
        vpunpckhwd(zmm4, zmm5, zmm5);

        vpcmpw(k3, zmm3, zmm_zero, 0);          // -0.0f would fail the logic here!
        vpsubw(zmm3, zmm3, zmm_exponent_bias);  // subtract bias 127
        vpsllw(zmm3, zmm3, 0x7);                // shift-left into bf16/fp16 exponent: 32x scales

        vpcmpw(k4, zmm4, zmm_zero, 0);          // -0.0f would fail the logic here!
        vpsubw(zmm4, zmm4, zmm_exponent_bias);  // subtract bias 127
        vpsllw(zmm4, zmm4, 0x7);                // shift-left into bf16/fp16 exponent: 32x scales

        mov(mx_size, 16);
        align(64, false);
        L(loop_begin);
        {
            // global : zmm_e2m1_lut, zmm_zero
            // input  : zmm5, k1, weight_ptr
            // output : zmm6/zmm7
            // scratch: k2, k3
            vmovdqu8(ymm6, ptr[weight_ptr]);  // load 64x FP4 or 32x(nibble-pair) = 256bits
            vpmovzxbw(zmm6, ymm6);            //      32x (0x00, nibble-pair)

            // log_zmm("u16", __LINE__, zmm6);

            vpsrld(zmm7, zmm6, 0x4);           // zmm6 : 32x (0x00, even-nibble) zmm7 : 32x (0x00, odd-nibble)
            vpermw(zmm6, zmm6, zmm_e2m1_lut);  // e2m1 nibble to float32
            vpermw(zmm7, zmm7, zmm_e2m1_lut);

            // use exponent-add to apply MX-scale
            vpcmpw(k1, zmm6, zmm_zero, 0);  // -0.0f would fail the logic here!
            vpcmpw(k2, zmm7, zmm_zero, 0);

            kord(k1, k3, k1);
            kord(k2, k4, k2);

            vpaddw(zmm6, zmm6, zmm3);
            vpaddw(zmm7, zmm7, zmm4);

            vpblendmw(zmm6 | k1, zmm6, zmm_zero);  // 32x bf16/fp16 even weights: w0, w2, w4, ...
            vpblendmw(zmm7 | k2, zmm7, zmm_zero);  // 32x bf16/fp16 odd weights: w1, w3, w5, ...

            mov(rax, src_ptr);
            for (int m = 0; m < BM; m++) {
                vpbroadcastd(zmm8, ptr[rax]);  // load & broadcast 2 x bf16 from source
                vdpbf16ps(acc_zmm(m, 0), zmm8, zmm6);
                vdpbf16ps(acc_zmm(m, 1), zmm8, zmm7);
                add(rax, src_stride);
            }
            // log_zmm("bf16", __LINE__, zmm6);
            // log_zmm("bf16", __LINE__, zmm7);
        }
        add(src_ptr, 4);
        add(weight_ptr, 32);
        dec(mx_size);
        jnz(loop_begin, T_NEAR);

        add(scale_ptr, 32);
        dec(mxblock_cnt);
        jnz(kx32_loop_begin, T_NEAR);

        // log_zmm("f32", __LINE__, acc_zmm(0, 0));
        // log_zmm("f32", __LINE__, acc_zmm(0, 1));
        vmovdqu32(ptr[dst_ptr], acc_zmm(0, 0));
        vmovdqu32(ptr[dst_ptr + 64], acc_zmm(0, 1));

        if (BM > 0) {
            mov(rax, ptr[abi_param1 + offsetof(CallArgs, dst_stride)]);
            for (int m = 1; m < BM; m++) {
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

    // suppose zmm0 is load from mem:
    //   vpunpcklwd(zmm1, zmm0, zmm0);
    //   vpunpckhwd(zmm2, zmm0, zmm0);
    // zmm1 & zmm2 are [0,0,1,1,2,2,.....]
    int reverse_self_vpunpcklhwd(int n) {
        const int map[] = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
        return map[n];
    }

    // weight: mxblocks x (32 x mxfp4)
    PackedWeight reorder_weights(mxfp4* mxfp4_weight, int mxblocks) {
        PackedWeight pw;
        pw.scales.resize(mxblocks * 32, 0);
        pw.elements.resize(mxblocks * 32 * 32 / 2, 0);
        uint8_t* p_scales = &pw.scales[0];
        uint8_t* p_elements = &pw.elements[0];
        for (int mblk = 0; mblk < mxblocks; mblk++) {
            for (int m = 0; m < 32; m += 2) {
                int i0 = 0;   //  0~15
                int i1 = 16;  // 16~31
                for (int n = 0; n < 32; n += 2, i0++, i1++) {
                    auto e2m1_even = mxfp4_weight[i0].get_e2m1(m);
                    auto e2m1_odd = mxfp4_weight[i1].get_e2m1(m);
                    p_elements[n] = (e2m1_odd << 4) | e2m1_even;

                    e2m1_even = mxfp4_weight[i0].get_e2m1(m + 1);
                    e2m1_odd = mxfp4_weight[i1].get_e2m1(m + 1);
                    p_elements[n + 1] = (e2m1_odd << 4) | e2m1_even;
                }
                p_elements += 32;
            }

            for (int n = 0; n < 32; n++) {
                p_scales[n] = mxfp4_weight[reverse_self_vpunpcklhwd(n)].scale_e8m0;
            }
            p_scales += 32;

            mxfp4_weight += 32;
        }
        return pw;
    }

    void run(CallArgs* args_ptr) {
        (*this)(args_ptr);
    }
};

static int AMX_DEBUG = getenv("AMX_DEBUG", 3);
/*
    32x32 weights:     2-tiles B0 | B1
    ping-pong buffer:
*/
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
        ov::bfloat16 *scratch_buff;  // B0|B1 ; B2|B3

        CallArgs() {
            // cache line align
            scratch_buff = reinterpret_cast<ov::bfloat16 *>(::aligned_alloc(64, 4096*4));
            if (scratch_buff == nullptr) {
                printf("::aligned_alloc failed\n");
                abort();
            }
        }
        ~CallArgs() {
            ::free(scratch_buff);
        }
        CallArgs( const CallArgs& ) = delete; // non construction-copyable
        CallArgs& operator=( const CallArgs& ) = delete; // non copyable
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

        Xbyak::Label kx32_loop_end;
        Xbyak::Label kx32_loop_begin;

        // zmm_e2m1_lut is look-up table maps E2M1 into BF16/FP16 values 32 entries
        // with high-16 entries duplicates low-16 entries so 5bits index for vpermw
        // can works irrespect of the value of the highest bit.
        // clang-format off
        static ov::bfloat16 lut_e2m1_bf16[] = {
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
            // duplicate
             0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
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

        auto decompress_Btile = [&](int dst_offset) {
            // unroll into code segment which decompresses a B-tile into scratch buffer
            // input: weight_ptr/scratch_ptr
            //        dst_offset determines which of 4 Btile buffer it decompresses into
            //
            // zmm5 : 32x MX scales
            vmovdqu8(ymm5, ptr[scale_ptr]);         // 32x E8M0
            vpmovzxbw(zmm5, ymm5);                  // 32x 16bit
            vpcmpw(k5, zmm5, zmm_zero, 0);          // -0.0f would fail the logic here!
            vpsubw(zmm5, zmm5, zmm_exponent_bias);  // subtract bias 127
            vpsllw(zmm5, zmm5, 0x7);                // shift-left into bf16/fp16 exponent: 32x scales
            add(scale_ptr, 32);

            for (int i = 0; i < 8; i++) {
                vmovdqu8(ymm6, ptr[weight_ptr]);   // load 4x16 e2m1 = 32bytes = 256bits
                vpmovzxbw(zmm6, ymm6);             //
                vpsrld(zmm7, zmm6, 0x4);           // zmm6 : 32x (0x00, even-nibble) zmm7 : 32x (0x00, odd-nibble)
                vpermw(zmm6, zmm6, zmm_e2m1_lut);  // e2m1 nibble to bf16
                vpermw(zmm7, zmm7, zmm_e2m1_lut);  // e2m1 nibble to bf16

                vpcmpw(k1, zmm6, zmm_zero, 0);  // -0.0f would fail the logic here!
                vpcmpw(k2, zmm7, zmm_zero, 0);
                // zmm6 & zmm7 from same group of 16-mx-blocks (belong to same 32x16 B-tile)
                kord(k1, k5, k1);
                kord(k2, k5, k2);
                vpaddw(zmm6, zmm6, zmm5);              // use exponent-add to apply MX-scale
                vpaddw(zmm7, zmm7, zmm5);              // use exponent-add to apply MX-scale
                vpblendmw(zmm6 | k1, zmm6, zmm_zero);  // 16x2 bf16/fp16 weights from row 0,1
                vpblendmw(zmm7 | k2, zmm7, zmm_zero);  // 16x2 bf16/fp16 weights from row 2,3

                vmovdqu32(ptr[scratch_ptr + i * 128 + dst_offset], zmm6);
                vmovdqu32(ptr[scratch_ptr + i * 128 + dst_offset + 64], zmm7);

                // is 32x16 weight correct?
                // log_zmm("bf16", __LINE__, zmm6);
                // log_zmm("bf16", __LINE__, zmm7);

                add(weight_ptr, 32);
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

        align(64, false);
        L(kx32_loop_begin);
        {
            // decompress B0|B1 (~118 cycles)
            if (AMX_DEBUG & 1) {
                decompress_Btile(0);
                decompress_Btile(1024);
            }
            if (AMX_DEBUG & 2) {
                // load B0|B1
                // load A0|A1
                // TMUL into C0,C1,C2,C3
                tileloadd(tmmA0, ptr[src_ptr + src_stride]);
                if (BM > 16) tileloadd(tmmA1, ptr[src_ptr1 + src_stride]);

                tileloadd(tmmB0, ptr[scratch_ptr + reg_B_stride + 0]);
                tileloadd(tmmB1, ptr[scratch_ptr + reg_B_stride + 1024]);

                tdpbf16ps(tmmC00, tmmA0, tmmB0);
                if (BM > 16) tdpbf16ps(tmmC10, tmmA1, tmmB0);
                tdpbf16ps(tmmC01, tmmA0, tmmB1);
                if (BM > 16) tdpbf16ps(tmmC11, tmmA1, tmmB1);
            }
            // move to next MX-block(32 along K dimension)
            // Btile is automatically moved
            lea(src_ptr, ptr[src_ptr + 64]);
            if (BM > 16) lea(src_ptr1, ptr[src_ptr1 + 64]);
        }
        dec(mxblock_cnt);
        jnz(kx32_loop_begin, T_NEAR);

        if (BM > 16) {
            pop(src_ptr1);
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

    struct PackedWeight {
        std::vector<uint8_t> scales;
        std::vector<uint8_t> elements;
    };
    //
    // AMX weight is special, it stores in unit of 32x16-repacked B-tile format
    // which is row of 2-tiles by row.
    //
    // weight: mxblocks x (32 x mxfp4)
    PackedWeight reorder_weights(mxfp4* mxfp4_weight, int mxblocks) {
        PackedWeight pw;
        pw.scales.resize(mxblocks * 32 * 2, 0);
        pw.elements.resize(mxblocks * 32 * 32 / 2, 0);

        uint8_t* p_scales = &pw.scales[0];
        uint8_t* p_elements = &pw.elements[0];
        for (int mblk = 0; mblk < mxblocks; mblk++) {
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
                p_scales[2 * n + 0] = mxfp4_weight[n].scale_e8m0;
                p_scales[2 * n + 1] = mxfp4_weight[n].scale_e8m0;
            }
            p_scales += 32;

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
                p_scales[2 * n + 0] = mxfp4_weight[16 + n].scale_e8m0;
                p_scales[2 * n + 1] = mxfp4_weight[16 + n].scale_e8m0;
            }
            p_scales += 32;

            //=================================
            mxfp4_weight += 32;
        }
        return pw;
    }
    void run(CallArgs* args_ptr) {
        TileConfigScope tcfg(m_tile_cfg);
        //printf("src: %p   buff: %p\n", args_ptr->src_ptr, args_ptr->scratch_buff);
        (*this)(args_ptr);
    }
#if 0
    void run(const ov::bfloat16* src, int64_t src_stride, PackedWeight* weight_mxfp4, float* dst, int64_t dst_stride, int BM, int mxblocks) {
        CallArgs args;
        args.src_ptr = src;
        args.src_stride = sizeof(ov::bfloat16) * 32 * mxblocks;
        args.scale_ptr = &weight_mxfp4->scales[0];
        args.weight_ptr = &weight_mxfp4->elements[0];
        args.dst_ptr = dst;
        args.dst_stride = sizeof(float) * 32;
        args.mxblock_cnt = mxblocks;

        TileConfigScope tcfg(m_tile_cfg);
        (*this)(&args);

        // loop order
        int M = A.dims[0];
        int K = A.dims[1];
        // ASSERT((M%32) == 0); ASSERT(K == m_K); ASSERT(C.dims[0] == M); ASSERT(C.dims[1] == m_N);

        auto strideA = A.stride;
        auto strideC = C.stride;
        auto* pA = reinterpret_cast<uint8_t*>(&A[0]);
        auto* pC = reinterpret_cast<uint8_t*>(&C[0]);
        for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            auto* pB = reinterpret_cast<uint8_t*>(&m_Weight[0]);
            for (int n = 0; n < m_N; n += 32, pB += m_ktiles * 2048) {
                (*this)(pA, strideA, pB, pC + n * sizeof(float), strideC);
            }
        }
    }
#endif
};

// convert FP4 into required layout
// A: BM x (32*mxblocks)
// weight: mxblocks x (32 x mxfp4) (each mxfp4 has 32 elements)
// C: BM x 32
template <typename AT>
void exec_reference(const AT* src, int64_t src_stride, mxfp4* mxfp4_weight, float* dst, int64_t dst_stride, int BM, int mxblocks, bool verbose = false) {
    // decompress mxfp4 first
    int BK = mxblocks * 32;  // 32-elements per MX-block
    int BN = 32;
    std::vector<float> weight(BN * BK, 0);

    auto getW = [&](int k, int n) -> float& {
        return weight[n * BK + k];
    };
    for (int k = 0; k < BK; k += 32) {
        for (int n = 0; n < BN; n++) {
            for (int ki = 0; ki < 32; ki++) {
                getW(k + ki, n) = mxfp4_weight[n][ki];
            }
        }
        mxfp4_weight += BN;
    }
    for (int m = 0; m < BM; m++) {
        const auto* A = src + m * src_stride;
        float* C = dst + m * dst_stride;

        for (int n = 0; n < BN; n++) {
            float sum = 0;
            for (int k = 0; k < BK; k++) {
                sum += A[k] * getW(k, n);
            }
            C[n] = sum;
        }
    }

    if (verbose) {
        printf("exec_reference src:\n");
        for (int k = 0; k < BK; k++)
            printf("%4.1f,", static_cast<float>(src[k]));
        printf("\n");
        printf("exec_reference weight:\n");
        for (int k = 0; k < BK; k++) {
            for (int n = 0; n < BN; n++) {
                printf("%4.1f,", getW(k, n));
            }
            printf("\n");
        }
        printf("exec_reference dst:\n");
        for (int n = 0; n < BN; n++)
            printf("%4.1f,", dst[n]);
        printf("\n");
    }
}

static auto REF_VERBOSE = getenv("REF_VERBOSE", 0);

template <class MXFP4Gemm>
void test(int BM, int MXBLOCKS) {
    MXFP4Gemm jit_kernel(BM);
    typename MXFP4Gemm::CallArgs args;
    using ActType = typename MXFP4Gemm::act_type;

    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x04, 0x04),"STALLS_TOTAL"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x06, 0x06),"STALLS_L3_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x05, 0x05),"STALLS_L2_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x40, 0x02),"BOUND_ON_STORES"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05),"BOUND_ON_LOADS"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00),"ALL_LOADS"},

        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00),"SPLIT_LOADS"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00),"SPLIT_STORES"},
    });

    //std::vector<ActType> src(BM * 32 * MXBLOCKS, 0);
    std::vector<mxfp4> mxfp4_weight(MXBLOCKS * 32);
    std::vector<float> dst(BM * 32, 0);
    std::vector<float> ref(BM * 32, 0);

    // cache-line aligned loads
    auto src = std::shared_ptr<ActType>(
                reinterpret_cast<ActType*>(aligned_alloc(64, BM * 32 * MXBLOCKS * sizeof(ActType))),
                [](void * p) { ::free(p); });

    // random weights & src
    float weights[32] = {0};
    for (int i = 0; i < MXBLOCKS * 32; i++) {
        auto mask = (rand() % 100) - 50;
        weights[i % 32] = mask * 0.1;
        if (i == 0) {
            // for(int k=0; k<32; k++) weights[k] = k*0.1;
        }
        mxfp4_weight[i].assign(weights);
        if (i == 0) {
            for (int k = 0; k < 32; k++)
                weights[k] = 0;
        }
    }

    auto* psrc = src.get();
    psrc[3] = 1;
    for (int m = 0; m < BM; m++) {
        for (int i = 0; i < MXBLOCKS * 32; i++) {
            psrc[m * MXBLOCKS * 32 + i] = ((i + m) % 3) - 1;
        }
    }

    // check accuracy
    exec_reference(src.get(), 32 * MXBLOCKS, mxfp4_weight.data(), ref.data(), 32, BM, MXBLOCKS, REF_VERBOSE);

    // reorder weights
    auto pw = jit_kernel.reorder_weights(mxfp4_weight.data(), MXBLOCKS);

    args.src_ptr = src.get();
    args.src_stride = sizeof(ActType) * 32 * MXBLOCKS;
    args.scale_ptr = &pw.scales[0];
    args.weight_ptr = &pw.elements[0];
    args.dst_ptr = dst.data();
    args.dst_stride = sizeof(float) * 32;
    args.mxblock_cnt = MXBLOCKS;

    jit_kernel.run(&args);
    bool all_equal = true;
    for (int m = 0; m < BM; m++) {
        bool equal = true;
        for (int n = 0; n < 32; n++)
            equal = equal && (ref[m * 32 + n] == dst[m * 32 + n]);
        if (equal)
            continue;

        all_equal = false;

        if (REF_VERBOSE) {
            printf("\nref[%d]:", m);
            for (int n = 0; n < 32; n++)
                printf("%4.1f,", ref[m * 32 + n]);
            printf("\njit[%d]:", m);
            for (int n = 0; n < 32; n++) {
                bool equal = dst[m * 32 + n] == ref[m * 32 + n];
                if (!equal)
                    printf("\e[31m");
                printf("%4.1f,", dst[m * 32 + n]);
                if (!equal)
                    printf("\e[0m");
            }
        }
    }
    printf("\n");

    printf("%s[%s] ACCURACY:%s\e[0m \n", all_equal ? "\e[32m" : "\e[31m", jit_kernel.name(), all_equal ? "PASSED!" : "FAILED!");
    pevg.show_header();
    //jit_kernel.log_show();
    
    double avg_cycles = 0;
    double avg_instructions = 0;
    double avg_cpi0 = 0;
    double avg_cpi1 = 0;
    constexpr int REPEATS = 100;

    // warm-up
    for (int repeat = 0; repeat < REPEATS; repeat++)
        jit_kernel.run(&args);

    for (int i = 0; i < 10; i++) {
        auto pmc = pevg.rdpmc(
            [&]() {
                // repeat
                for (int repeat = 0; repeat < REPEATS; repeat++)
                    jit_kernel.run(&args);
            },
            jit_kernel.name(),
            REPEATS*MXBLOCKS);
    }
}

int main(int argc, const char* argv[]) {
    int BM = argc > 1 ? atoi(argv[1]) : 1;
    int MXBLOCKS = argc > 2 ? atoi(argv[2]) : 1;

    bool initAMX = initXTILE();
    if (BM <= 10) {
        test<MXFP4BrgemmFMA<false>>(BM, MXBLOCKS);
        test<MXFP4BrgemmFMA<true>>(BM, MXBLOCKS);
        test<MXFP4BrgemmAVX512_BF16>(BM, MXBLOCKS);
    }
    test<MXFP4BrgemmAMX>(BM, MXBLOCKS);
    return 0;
}