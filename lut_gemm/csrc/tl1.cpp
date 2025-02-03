#include "common.hpp"

/*
TL1 Kernel preprocesses every two full-precision weights by packing them into 4-bit index (see
Table 2), and pre-computes their corresponding activations into 32 = 9 values. The index-value pairs
are stored in a lookup table to perform LUT computation [PPK+22, WCC+24]. GEMV processing
is performed using an int16 LUT and accumulation through addition. We recommend using it with a
limited number of threads when serving large models.

16-bit LUT:

*/

// preprocess activation into LUT, each 4 activations
//     a[k:k+4] =>
//                     16 bytes + 16 bytes   = 32 bytes
//   LUT-low-8bits  : a[k:k+2] | a[k+2:k+4]
//   LUT-high-8bits : a[k:k+2] | a[k+2:k+4]
//
//
static int8_t TL1_plus1[32] = {
    0, 0,  // 0
    0, 1,  // 1
    0, 2,  // 2
    1, 0,  // 3
    1, 1,  // 4
    1, 2,  // 5
    2, 0,  // 6
    2, 1,  // 7
    2, 2,  // 8
    1, 1,  // padding (0,0)
    1, 1,  // padding (0,0)
    1, 1,  // padding (0,0)
    1, 1,  // padding (0,0)
    1, 1,  // padding (0,0)
    1, 1,  // padding (0,0)
    1, 1,  // padding (0,0)
};

// pack weights, convert 4x32 sub-weight into LUT index
py::array_t<uint8_t> pack_tl1(py::array_t<int8_t> ternary_w) {
    ASSERT(ternary_w.ndim() == 2);
    auto K = ternary_w.shape(0);
    auto N = ternary_w.shape(1);
    ASSERT((N % 32) == 0);
    ASSERT((K % 4) == 0);

    auto pack_4x32_to_32 = [&](const int8_t* src, int K, int N, int stride_src, uint8_t* dst) {
        for (int n0 = 0; n0 < N; n0 += 32) {
            for (int k = 0; k < K; k += 4) {
                const int8_t* src0 = src;
                const int8_t* src1 = src + stride_src;
                const int8_t* src2 = src + stride_src * 2;
                const int8_t* src3 = src + stride_src * 3;
                for (int n = 0; n < 32; n++) {
                    // (-1,0,1) => (0,1,2), the value of two-digits base-3 number as the LUT index
                    int lut_idx_01 = (src0[n] + 1) + (src1[n] + 1) * 3;
                    int lut_idx_23 = (src2[n] + 1) + (src3[n] + 1) * 3;
                    if (n < 16) {
                        // low-4bits stores first (4 x 16)
                        dst[n] = lut_idx_01;
                        dst[n + 16] = lut_idx_23;
                    } else {
                        // high-4bits stores second (4 x 16)
                        dst[n - 16] |= (lut_idx_01 << 4);
                        dst[n - 16 + 16] |= (lut_idx_23 << 4);
                    }
                }
                src += 4 * stride_src;
                dst += 32;
            }
        }
    };

    static auto pack_4x32_to_32_avx2 =
        SIMDJit::create([](SIMDJit* jit, SReg src, SReg K, SReg N, SReg stride_src, SReg dst) {
            SReg psrc0, psrc2, n, k;
            VReg r0, r1, r2, r3, one;
            one.load(0x01010101);
            jit->for_loop(n, 0, N, 32, [&] {
                psrc0 = src + n;
                psrc2 = psrc0 + stride_src * 2;
                jit->for_loop(k, 0, K, 4, [&] {
                    r0.load(psrc0);
                    r1.load(psrc0 + stride_src);
                    r2.load(psrc2);
                    r3.load(psrc2 + stride_src);

                    jit->vpaddb(r0, r0, one);
                    jit->vpaddb(r1, r1, one);
                    jit->vpaddb(r2, r2, one);
                    jit->vpaddb(r3, r3, one);

                    // r0 = r1*3 + r0 lut_idx_01
                    jit->vpaddb(r0, r0, r1);
                    jit->vpaddb(r1, r1, r1);
                    jit->vpaddb(r0, r0, r1);

                    // r2 = r3*3 + r2 lut_idx_23
                    jit->vpaddb(r2, r2, r3);
                    jit->vpaddb(r3, r3, r3);
                    jit->vpaddb(r2, r2, r3);

                    jit->vperm2i128(r1.ymm(), r0.ymm(), r2.ymm(), get_imm8<4>(0, 2));
                    jit->vperm2i128(r3.ymm(), r0.ymm(), r2.ymm(), get_imm8<4>(1, 3));

                    jit->vpsllw(r3, r3, 4);
                    jit->vpor(r1, r1, r3);

                    r1.store(dst);
                    dst += 32;
                    psrc0 += stride_src * 4;
                    psrc2 += stride_src * 4;
                });
            });
        });

    py::array_t<uint8_t> PackedW({static_cast<int>((N / 32) * (K / 4)), 32});
    parallel_st(N, 32, [&](int ithr, int nthr, int g0, int g1) {
        // pack_4x32_to_32(ternary_w.data(0, g0*32), K, (g1 - g0)*32, N, PackedW.mutable_data(g0 * (K/4), 0));
        (*pack_4x32_to_32_avx2)(ternary_w.data(0, g0 * 32),
                                K,
                                (g1 - g0) * 32,
                                N,
                                PackedW.mutable_data(g0 * (K / 4), 0));
    });
    return PackedW;
}

py::array_t<int32_t> mm_tl1(py::array_t<int8_t> x, py::array_t<uint8_t> packedW, int K, int N) {
    ASSERT(x.ndim() == 2);
    auto M = x.shape(0);
    ASSERT(x.shape(1) == K);

    static auto scratch_cnt = M * K / 2 * 16;
    static auto scratch = alloc_cache_aligned<int8_t>(M * K / 2 * 16);
    if (scratch_cnt < M * K / 2 * 16) {
        scratch_cnt = M * K / 2 * 16;
        scratch = alloc_cache_aligned<int8_t>(scratch_cnt);
    }

    auto precomput_lut = SIMDJit::create([](SIMDJit* jit, SReg src, SReg dst) {
        auto lut_w_tl1 = jit->get_vreg();
        auto ones = jit->get_vreg();
        auto vx = jit->get_vreg();
        auto lut01 = jit->get_vreg();
        auto lut23 = jit->get_vreg();
        auto lut_bias = jit->get_vreg();
        auto unpack_map = jit->get_vreg();
        auto vtmp = jit->get_vreg();

        // [0,1][0,1]... repeat 16 times
        lut_w_tl1.load(TL1_plus1);
        ones.load(0x01010101);

        // vx.load(0x01030103);
        jit->vpbroadcastw(vx, jit->ptr[src.r64()]);
        jit->vpmaddubsw(lut01, lut_w_tl1, vx);  // [(w+1)u8 x (x[0])i8 + (w+1)u8 x (x[1])i8] => i16
        jit->vpmaddubsw(lut_bias, ones, vx);    // [(1)u8 x (x[0])i8 + (1)u8 x (x[1])i8] => i16
        jit->vpsubw(lut01, lut01, lut_bias);    // [(w) x (x[0]) + (w) x (x[1])] => i16

        jit->vpbroadcastw(vx, jit->ptr[src.r64() + 2]);
        // vx.load(0x03010301);
        jit->vpmaddubsw(lut23, lut_w_tl1, vx);
        jit->vpmaddubsw(lut_bias, ones, vx);
        jit->vpsubw(lut23, lut23, lut_bias);

        // pack lut01's low-8bits with lut23's
        static int8_t unpack_low_high[32] = {
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
        };

        // jit->jcout(jit->jcout.as_i16, "lut01=", lut01);
        // jit->jcout(jit->jcout.as_i16, "lut23=", lut23);
        // jit->jcout(jit->jcout.as_x8, "lut01=", lut01);
        // jit->jcout(jit->jcout.as_x8, "lut23=", lut23);

        unpack_map.load(unpack_low_high);
        jit->vpshufb(lut01, lut01, unpack_map);
        jit->vpshufb(lut23, lut23, unpack_map);
        jit->vpermq(lut01.ymm(), lut01.ymm(), get_imm8<2>(0, 2, 1, 3));
        jit->vpermq(lut23.ymm(), lut23.ymm(), get_imm8<2>(0, 2, 1, 3));
        jit->vperm2i128(vtmp.ymm(), lut01.ymm(), lut23.ymm(), get_imm8<4>(0, 2));
        // jit->jcout(jit->jcout.as_x8, "low =", vtmp);
        vtmp.store(dst);
        jit->vperm2i128(vtmp.ymm(), lut01.ymm(), lut23.ymm(), get_imm8<4>(1, 3));
        vtmp.store(dst + 32);
        // jit->jcout(jit->jcout.as_x8, "high=", vtmp);
    });

    // (*precomput_lut)(); abort();

    // pre-compute LUT from x
    auto* lut_base = scratch.get();
    {
        auto* psrc = x.data(0, 0);
        auto* pdst = lut_base;
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k += 4) {
                // precomput LUT for
                //    low_8bits_lut(x[0,1]), low_8bits_lut(x[2,3])
                //    high_8bits_lut(x[0,1]), high_8bits_lut(x[2,3])
                (*precomput_lut)(psrc + k, pdst);
                pdst += 64;
            }
            psrc += K;
        }
    }

    auto lut_gemm_tl1 = SIMDJit::create([](SIMDJit* jit, SReg pa, SReg pb, SReg pc) {
        auto vec_w0 = jit->get_vreg();
        auto vec_w1 = jit->get_vreg();
        auto vc0 = jit->get_vreg();
        auto vc1 = jit->get_vreg();
        auto maskF = jit->get_vreg();
        auto vec_lo0 = jit->get_vreg();
        auto vec_hi0 = jit->get_vreg();
        auto vec_lo1 = jit->get_vreg();
        auto vec_hi1 = jit->get_vreg();

        auto vec_lut0123_lo = jit->get_vreg();
        auto vec_lut0123_hi = jit->get_vreg();

        maskF.load(0x0F0F0F0F);

        vec_w0.load(pb);
        vec_lut0123_lo.load(pa);
        vec_lut0123_hi.load(pa + 32);

        jit->vpsrlw(vec_w1, vec_w0, 4);
        jit->vpand(vec_w0, vec_w0, maskF);
        jit->vpand(vec_w1, vec_w1, maskF);

        jit->vpshufb(vec_lo0, vec_lut0123_lo, vec_w0);
        jit->vpshufb(vec_hi0, vec_lut0123_hi, vec_w0);
        jit->vpshufb(vec_lo1, vec_lut0123_lo, vec_w1);
        jit->vpshufb(vec_hi1, vec_lut0123_hi, vec_w1);

        jit->vpunpcklbw(vec_w0, vec_lo0, vec_hi0);
        jit->vpunpckhbw(vec_w1, vec_lo0, vec_hi0);

        vc0.load(pc);
        vc1.load(pc + 32);

        jit->vpaddw(vc0, vc0, vec_w0);
        jit->vpaddw(vc0, vc0, vec_w1);

        jit->vpunpcklbw(vec_w0, vec_lo1, vec_hi1);
        jit->vpunpckhbw(vec_w1, vec_lo1, vec_hi1);
        jit->vpaddw(vc1, vc1, vec_w0);
        jit->vpaddw(vc1, vc1, vec_w1);

        vc0.store(pc);
        vc1.store(pc + 32);
    });

    py::array_t<int32_t> y({static_cast<int>(M), static_cast<int>(N)});
    parallel_st(N, 32, [&](int ithr, int nthr, int g0, int g1) {
        auto* pdst = y.mutable_data(0, 0);
        auto* pw0 = packedW.data(0, 0);
        auto* pa0 = scratch.get();
        for (int m = 0; m < M; m++) {
            for (int g = g0; g < g1; g++) {
                auto* pa = pa0;
                int n0 = g * 32;
                auto* pb = pw0 + n0 * (K / 4);
                int16_t acc[32] = {0};
                for (int k = 0; k < K; k += 4, pa += 64, pb += 32) {
                    // 1x4 (LUT: 64bytes) * 4x32(widx: 32bytes) => 1x32 int16
                    (*lut_gemm_tl1)(pa, pb, acc);
                }
                for (int n = 0; n < 32; n++)
                    pdst[n0 + n] = acc[n];
            }
            pa0 += K / 4 * 64;
            pdst += N;
        }
    });
    return y;
}

// (M, K, psrc, stride_src, pdst, stride_dst)
std::shared_ptr<SIMDJit> get_tl1_build_lut() {
    auto jit = std::make_shared<SIMDJit>("TL1_build_LUT");
    auto M = jit->get_arg();
    auto K = jit->get_arg();
    auto psrc = jit->get_arg();
    auto stride_src = jit->get_arg();
    auto pdst = jit->get_arg();
    auto stride_dst = jit->get_arg();

    auto src0123 = jit->get_vreg();
    auto src01 = jit->get_vreg();
    auto src23 = jit->get_vreg();
    // src01.load(0x01010101);
    // src23.load(0x01010101);

    auto tmp = jit->get_sreg();
    auto allw = jit->get_vreg();
    auto all1 = jit->get_vreg();
    auto lut01 = jit->get_vreg();
    auto lut23 = jit->get_vreg();
    auto lut_base = jit->get_vreg();
    auto shuffle_eo = jit->get_vreg();
    auto shuffle_01 = jit->get_vreg();
    auto shuffle_23 = jit->get_vreg();
    static uint8_t v_shuffle_even_odd[] = {
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
    };
    shuffle_eo.load(v_shuffle_even_odd);
    shuffle_01.load(0x01000100);
    shuffle_23.load(0x03020302);
    all1.load(0x01010101);
    jit->mov(tmp, reinterpret_cast<uintptr_t>(TL1_plus1));
    jit->vmovdqu(allw, jit->ptr[tmp.r64()]);

    jit->do_while_(M > 0, [&] {
        jit->for_loop(tmp, 0, K, 4, [&] {
            jit->vpbroadcastd(src0123, jit->ptr[psrc.r64() + tmp.r64()]);
            jit->vpshufb(src01, src0123, shuffle_01);
            jit->vpshufb(src23, src0123, shuffle_23);
            jit->vpmaddubsw(lut01, allw, src01);
            jit->vpmaddubsw(lut23, allw, src23);
            jit->vpmaddubsw(lut_base, all1, src01);
            jit->vpsubw(lut01, lut01, lut_base);

            jit->vpmaddubsw(lut_base, all1, src23);
            jit->vpsubw(lut23, lut23, lut_base);

            // jit->jcout(jit->jcout.as_i8, src01, "\n", allw);
            // jit->jcout(jit->jcout.as_i16, lut01, "\n", jit->jcout.as_x8, lut01);
            jit->vpshufb(lut01, lut01, shuffle_eo);
            jit->vpshufb(lut23, lut23, shuffle_eo);
            // jit->jcout(lut01);
            jit->vpermq(lut01.ymm(), lut01, 0xd8);
            jit->vpermq(lut23.ymm(), lut23, 0xd8);
            jit->jcout(jit->jcout.as_x8, lut01, "\n", lut23, "\n");

            jit->vperm2i128(src01.ymm(), lut01.ymm(), lut23, 0x20);
            jit->vperm2i128(src23.ymm(), lut01.ymm(), lut23, 0x31);
            jit->vmovdqu(jit->ptr[pdst.r64()], src01);
            jit->vmovdqu(jit->ptr[pdst.r64() + 32], src23);
            pdst += 64;
            jit->jcout(jit->jcout.as_x8, src01, "\n", src23);
        });
        pdst = pdst + stride_dst - K * (64 / 4);
        psrc = psrc + stride_src;
        M--;
    });

    jit->return_();
    jit->finalize();
    return jit;
}

std::shared_ptr<SIMDJit> get_tl1_mkernel() {
    auto jit = std::make_shared<SIMDJit>("TL1");

    auto cnt = jit->get_arg();
    auto pa = jit->get_arg();
    auto stride_a = jit->get_arg();
    auto pb = jit->get_arg();
    auto pc = jit->get_arg();
    auto stride_c = jit->get_arg();
    auto do_accumulate = jit->get_arg();

    auto maskF = jit->get_vreg();
    maskF.load(0x0F0F0F0F);

    auto vc0 = jit->get_vreg();
    auto vc1 = jit->get_vreg();
    auto vec_w0 = jit->get_vreg();
    auto vec_w1 = jit->get_vreg();

    auto vec_lo0 = jit->get_vreg();
    auto vec_hi0 = jit->get_vreg();
    auto vec_lo1 = jit->get_vreg();
    auto vec_hi1 = jit->get_vreg();

    auto vec_lut0123_lo = jit->get_vreg();
    auto vec_lut0123_hi = jit->get_vreg();

    // 256bit-ymm =>
    //   2 x [32 x 4bit] LUT-index
    //   LUT using 2x`vpshufb`+`vpunpckhbw` => [16 x 16bit]
    //
    jit->do_while_(cnt > 0, [&] {
        // load weight in 4bits LUT-index : 2 x [32 x 4bit]
        jit->vmovdqu(vec_w0, jit->ptr[pb.r64()]);
        jit->vmovdqu(vec_lut0123_lo, jit->ptr[pa.r64()]);
        jit->vmovdqu(vec_lut0123_hi, jit->ptr[pa.r64() + 32]);

        jit->vpsrlw(vec_w1, vec_w0, 4);
        jit->vpand(vec_w0, vec_w0, maskF);
        jit->vpand(vec_w1, vec_w1, maskF);

        jit->vpshufb(vec_lo0, vec_lut0123_lo, vec_w0);
        jit->vpshufb(vec_hi0, vec_lut0123_hi, vec_w0);
        jit->vpshufb(vec_lo1, vec_lut0123_lo, vec_w1);
        jit->vpshufb(vec_hi1, vec_lut0123_hi, vec_w1);

        jit->vpunpcklbw(vec_w0, vec_lo0, vec_hi0);
        jit->vpunpckhbw(vec_w1, vec_lo0, vec_hi0);
        jit->vpaddw(vc0, vc0, vec_w0);
        jit->vpaddw(vc0, vc0, vec_w1);

        jit->vpunpcklbw(vec_w0, vec_lo1, vec_hi1);
        jit->vpunpckhbw(vec_w1, vec_lo1, vec_hi1);
        jit->vpaddw(vc1, vc1, vec_w0);
        jit->vpaddw(vc1, vc1, vec_w1);

        pa += 32 * 4;
        pb += 32;
        cnt -= 4;
    });

    jit->return_();
    jit->finalize();
    return jit;
}

void mbench_tl1() {
    LinuxPerf perf;
    auto jit = get_tl1_mkernel();
    auto M = 1;
    auto K = 4096;
    auto N = 32;
    auto A = alloc_cache_aligned<int8_t>(M * K * 16, 0);
    auto B = alloc_cache_aligned<int8_t>(K * N / 4, 0);
    auto C = alloc_cache_aligned<int32_t>(M * N, 0);
    // start test
    // warm-up code cache
    (*jit)(4, A.get(), K, B.get(), C.get(), N * 4, 0);

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N, M * K * 16, K * N / 4);
    perf(
        [&] {
            (*jit)(K, A.get(), K, B.get(), C.get(), N * 4, 0);
        },
        std::string("TL1"),
        4,                              // repeat
        K / 4,                          // kernel's internal loop count
        M * K * N * 2,                  // OPs
        (M * K * 16 * 0) + (K * N / 4)  // bytes read
    );
}

void tl1_init(py::module_& m) {
    m.def("mbench_tl1", &mbench_tl1);
    m.def("mm_tl1", &mm_tl1);
    m.def("pack_tl1", &pack_tl1);
}
