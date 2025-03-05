#include "common.hpp"
#include "linux_perf.hpp"
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
                    int lut_idx_01 = (src0[n] + 1) * 3 + (src1[n] + 1);
                    int lut_idx_23 = (src2[n] + 1) * 3 + (src3[n] + 1);
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
            Ymm r0, r1, r2, r3, one;
            one.load(0x01010101);
            jit->for_(n, 0, N, 32, [&] {
                psrc0 = src + n;
                psrc2 = psrc0 + stride_src * 2;
                jit->for_(k, 0, K, 4, [&] {
                    r0.load(psrc0);
                    r1.load(psrc0 + stride_src);
                    r2.load(psrc2);
                    r3.load(psrc2 + stride_src);

                    jit->vpaddb(r0, r0, one);
                    jit->vpaddb(r1, r1, one);
                    jit->vpaddb(r2, r2, one);
                    jit->vpaddb(r3, r3, one);

                    // r0 = r0*3 + r1 lut_idx_01
                    jit->vpaddb(r1, r1, r0);
                    jit->vpaddb(r0, r0, r0);
                    jit->vpaddb(r0, r0, r1);

                    // r2 = r2*3 + r3 lut_idx_23
                    jit->vpaddb(r3, r3, r2);
                    jit->vpaddb(r2, r2, r2);
                    jit->vpaddb(r2, r2, r3);

                    //   0	1	2	3	4	5	6	7 |
                    //	16	17	18	19	20	21	22	23|
                    //	8	9	10	11	12	13	14	15|
                    //	24	25	26	27	28	29	30	31|

                    jit->vpermq(r1.ymm(), r0.ymm(), get_imm8<2>(0,2,1,3));
                    jit->vpermq(r3.ymm(), r2.ymm(), get_imm8<2>(0,2,1,3));
                    //jit->vperm2i128(r1.ymm(), r0.ymm(), r2.ymm(), get_imm8<4>(0, 2));
                    //jit->vperm2i128(r3.ymm(), r0.ymm(), r2.ymm(), get_imm8<4>(1, 3));

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
        //pack_4x32_to_32(ternary_w.data(0, g0 * 32),
                                K,
                                (g1 - g0) * 32,
                                N,
                                PackedW.mutable_data(g0 * (K / 4), 0));
    });
    return PackedW;
}

std::shared_ptr<SIMDJit> get_lut_gemm_mkernel(int K_UNROLL = 16) {
    return SIMDJit::create([&](SIMDJit* jit, SReg pa, SReg pb, SReg pc, SReg K) {
        Ymm maskF;
        Ymm acc[4];
        maskF.load(0x0F0F0F0F);

        acc[0].load(0);
        acc[1].load(0);
        acc[2].load(0);
        acc[3].load(0);
        jit->do_while_(K > 0, [&] {
            // we can accumulate results into int16 w/o overflow up to 256 times
            //
            Ymm vec_c0, vec_c1;
            vec_c0.load(0);
            vec_c1.load(0);
            SReg K_inner;
            const int INNER_LOOPS = 64;
            K_inner = INNER_LOOPS;
            jit->do_while_(K_inner > 0, [&] {
                constexpr int UNROLL = 2;
                for(int unroll = 0; unroll < UNROLL; unroll++) {
                    // inner-most kernel does: [1x4] @ [4x32]
                    Ymm vec_lut01_lo, vec_lut01_hi;
                    Ymm vec_lut23_lo, vec_lut23_hi;
                    {
                        Ymm vec_w01, vec_w23;
                        jit->vbroadcasti128(vec_lut01_lo.ymm(), jit->ptr[pa.r64() + 16*(unroll*4 + 0)]);
                        jit->vbroadcasti128(vec_lut23_lo.ymm(), jit->ptr[pa.r64() + 16*(unroll*4 + 1)]);
                        jit->vbroadcasti128(vec_lut01_hi.ymm(), jit->ptr[pa.r64() + 16*(unroll*4 + 2)]);
                        jit->vbroadcasti128(vec_lut23_hi.ymm(), jit->ptr[pa.r64() + 16*(unroll*4 + 3)]);

                        vec_w01.load(pb + 32*unroll);
                        jit->prefetcht1(jit->ptr[pb.r64() + 4096 + 32*unroll]);

                        jit->vpsrlw(vec_w23, vec_w01, 4);
                        jit->vpand(vec_w01, vec_w01, maskF);
                        jit->vpand(vec_w23, vec_w23, maskF);

                        // jit->jcout(jit->jcout.as_i8, "\nvec_lut0123_lo=", vec_lut0123_lo, "\nvec_lut0123_hi=",
                        // vec_lut0123_hi); jit->jcout(jit->jcout.as_i8, "\nvec_w0=", vec_w0, "\nvec_w1=", vec_w1);
                        jit->vpshufb(vec_lut01_lo, vec_lut01_lo, vec_w01);  // k=0,1
                        jit->vpshufb(vec_lut01_hi, vec_lut01_hi, vec_w01);  // k=0,1
                        jit->vpshufb(vec_lut23_lo, vec_lut23_lo, vec_w23);  // k=2,3
                        jit->vpshufb(vec_lut23_hi, vec_lut23_hi, vec_w23);  // k=2,3
                    }
                    Ymm v01_n0, v01_n1;
                    Ymm v23_n0, v23_n1;
                    jit->vpunpcklbw(v01_n0, vec_lut01_lo, vec_lut01_hi);
                    jit->vpunpckhbw(v01_n1, vec_lut01_lo, vec_lut01_hi);
                    jit->vpunpcklbw(v23_n0, vec_lut23_lo, vec_lut23_hi);
                    jit->vpunpckhbw(v23_n1, vec_lut23_lo, vec_lut23_hi);

                    // jit->jcout(jit->jcout.as_i16, "\nvec_lo0=", vec_lo0, "\nvec_hi0=", vec_hi0);
                    jit->vpaddw(v01_n0, v01_n0, v23_n0);
                    jit->vpaddw(v01_n1, v01_n1, v23_n1);
                    jit->vpaddw(vec_c0, vec_c0, v01_n0);
                    jit->vpaddw(vec_c1, vec_c1, v01_n1);
                }
                pa += 64*UNROLL;
                pb += 32*UNROLL;
                K_inner -= 4*UNROLL;
            });

            Ymm vec_i32[4];
            jit->vpmovsxwd(vec_i32[0], vec_c0.xmm());
            jit->vextracti128(vec_i32[1].xmm(), vec_c0.ymm(), 1);
            jit->vpmovsxwd(vec_i32[1], vec_i32[1].xmm());

            jit->vpmovsxwd(vec_i32[2], vec_c1.xmm());
            jit->vextracti128(vec_i32[3].xmm(), vec_c1.ymm(), 1);
            jit->vpmovsxwd(vec_i32[3], vec_i32[3].xmm());

            jit->vpaddd(acc[0], acc[0], vec_i32[0]);
            jit->vpaddd(acc[1], acc[1], vec_i32[1]);
            jit->vpaddd(acc[2], acc[2], vec_i32[2]);
            jit->vpaddd(acc[3], acc[3], vec_i32[3]);

            K -= INNER_LOOPS;
        });
        for (int i = 0; i < 4; i++)
            acc[i].store(pc + i * 32);
    });
}

void mbench_tl1() {
    LinuxSimplePerf perf;
    auto jit = get_lut_gemm_mkernel();
    auto M = 1;
    auto K = 4096;
    auto N = 32;
    auto A = alloc_cache_aligned<int8_t>(M * K / 4 * 64, 0);
    auto B = alloc_cache_aligned<int8_t>(K * N / 4, 0);
    auto C = alloc_cache_aligned<int32_t>(M * N, 0);
    // start test
    // warm-up code cache
    (*jit)(A.get(), B.get(), C.get(), K);

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N, M * K * 16, K * N / 4);
    perf(
        [&] {
            (*jit)(A.get(), B.get(), C.get(), K);
        },
        std::string("TL1"),
        4,                              // repeat
        K / 4,                          // kernel's internal loop count
        M * K * N * 2,                  // OPs
        (M * K * 16 * 0) + (K * N / 4)  // bytes read
    );
}

py::array_t<int32_t> mm_tl1(py::array_t<int8_t> x, py::array_t<uint8_t> packedW, int K, int N) {
    ASSERT(x.ndim() == 2);
    auto M = x.shape(0);
    ASSERT(x.shape(1) == K);

    auto prof = LinuxPerf::Profile("mm_tl1", 0, M, K, N);

    static auto scratch_cnt = M * K / 4 * 64;
    static auto scratch = alloc_cache_aligned<int8_t>(M * K / 4 * 64);
    if (scratch_cnt < M * K / 4 * 64) {
        scratch_cnt = M * K / 4 * 64;
        scratch = alloc_cache_aligned<int8_t>(scratch_cnt);
    }

    static auto precomput_lut = SIMDJit::create([](SIMDJit* jit, SReg src, SReg dst, SReg K) {
        Ymm lut_w_tl1, ones, vx;
        Ymm lut01, lut23, lut_bias;
        Ymm unpack_map, vtmp;

        // [0,1][0,1]... repeat 16 times
        lut_w_tl1.load(TL1_plus1);
        ones.load(0x01010101);

        jit->do_while_(K > 0, [&] {
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
                0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
            };

            unpack_map.load(unpack_low_high);
            jit->vpshufb(lut01, lut01, unpack_map);
            jit->vpshufb(lut23, lut23, unpack_map);
            jit->vpermq(lut01.ymm(), lut01.ymm(), get_imm8<2>(0, 2, 1, 3));
            jit->vpermq(lut23.ymm(), lut23.ymm(), get_imm8<2>(0, 2, 1, 3));
            jit->vperm2i128(vtmp.ymm(), lut01.ymm(), lut23.ymm(), get_imm8<4>(0, 2));

            vtmp.store(dst);

            jit->vperm2i128(vtmp.ymm(), lut01.ymm(), lut23.ymm(), get_imm8<4>(1, 3));
            vtmp.store(dst + 32);

            src += 4;
            dst += 64;
            K -= 4;
        });
    });

    // (*precomput_lut)(); abort();

    // pre-compute LUT from x
    auto* lut_base = scratch.get();
    {
        auto prof = LinuxPerf::Profile("precompute", 0);
        auto* psrc = x.data(0, 0);
        auto* pdst = lut_base;
        for (int m = 0; m < M; m++) {
            (*precomput_lut)(psrc, pdst, K);
            pdst += K / 4 * 64;
            psrc += K;
        }
    }

    static auto lut_gemm_tl1 = get_lut_gemm_mkernel();

    py::array_t<int32_t> y({static_cast<int>(M), static_cast<int>(N)});
    parallel_st(N, 32, [&](int ithr, int nthr, int g0, int g1) {
        auto prof = LinuxPerf::Profile("lut_gemm_tl1", 0);
        auto* pdst = y.mutable_data(0, 0);
        auto* pw0 = packedW.data(0, 0);
        auto* pa0 = scratch.get();
        for (int m = 0; m < M; m++) {
            for (int g = g0; g < g1; g++) {
                auto* pa = pa0;
                int n0 = g * 32;
                auto* pb = pw0 + n0 * (K / 4);
                (*lut_gemm_tl1)(pa, pb, pdst + n0, K);
            }
            pa0 += K / 4 * 64;
            pdst += N;
        }
    });
    return y;
}

void tl1_init(py::module_& m) {
    m.def("mbench_tl1", &mbench_tl1);
    m.def("mm_tl1", &mm_tl1);
    m.def("pack_tl1", &pack_tl1);
}
