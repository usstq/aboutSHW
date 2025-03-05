#include "common.hpp"

std::shared_ptr<SIMDJit> get_i2s8_mkernel() {
    return SIMDJit::create(
        [](SIMDJit* jit, SReg cnt, SReg pa, SReg stride_a, SReg pb, SReg pc, SReg stride_c, SReg do_accumulate) {
            Ymm mask3;
            mask3.load(0x03030303);

            Ymm vax, vay, vb0, vb1, vbx, vby;
            Ymm vc[8];

            for (int i = 0; i < 8; i++) {
                jit->vpxor(vc[i], vc[i], vc[i]);
            }
            jit->do_while_(cnt > 0, [&] {
                jit->vpbroadcastd(vax, jit->ptr[pa.r64()]);
                jit->vpbroadcastd(vay, jit->ptr[pa.r64() + 4]);
                vb0.load(pb);
                vb1.load(pb + 32);

                // (4 x 2bit) => (4 x 8bit) -> VNNI
                jit->vpand(vbx, vb0, mask3);
                jit->vpand(vby, vb1, mask3);
                jit->vpdpbusd(vc[0], vbx, vax, Xbyak::VexEncoding);
                jit->vpdpbusd(vc[4], vby, vay, Xbyak::VexEncoding);

                jit->vpsrlw(vbx, vb0, 2);
                jit->vpsrlw(vby, vb1, 2);
                jit->vpand(vbx, vbx, mask3);
                jit->vpand(vby, vby, mask3);
                jit->vpdpbusd(vc[1], vbx, vax, Xbyak::VexEncoding);
                jit->vpdpbusd(vc[5], vby, vay, Xbyak::VexEncoding);

                jit->vpsrlw(vbx, vb0, 4);
                jit->vpsrlw(vby, vb1, 4);
                jit->vpand(vbx, vbx, mask3);
                jit->vpand(vby, vby, mask3);
                jit->vpdpbusd(vc[2], vbx, vax, Xbyak::VexEncoding);
                jit->vpdpbusd(vc[6], vby, vay, Xbyak::VexEncoding);

                jit->vpsrlw(vbx, vb0, 6);
                jit->vpsrlw(vby, vb1, 6);
                jit->vpand(vbx, vbx, mask3);
                jit->vpand(vby, vby, mask3);
                jit->vpdpbusd(vc[3], vbx, vax, Xbyak::VexEncoding);
                jit->vpdpbusd(vc[7], vby, vay, Xbyak::VexEncoding);

                jit->prefetcht1(jit->ptr[pb.r64() + 4096]);
                pa += 8;
                pb += 64;
                cnt -= 8;
            });
            jit->vpaddd(vc[0], vc[0], vc[4]);
            jit->vpaddd(vc[1], vc[1], vc[5]);
            jit->vpaddd(vc[2], vc[2], vc[6]);
            jit->vpaddd(vc[3], vc[3], vc[7]);
            jit->if_(do_accumulate != 0, [&] {
                jit->vpaddd(vc[0], vc[0], jit->ptr[pc.r64() + 32 * 0]);
                jit->vpaddd(vc[1], vc[1], jit->ptr[pc.r64() + 32 * 1]);
                jit->vpaddd(vc[2], vc[2], jit->ptr[pc.r64() + 32 * 2]);
                jit->vpaddd(vc[3], vc[3], jit->ptr[pc.r64() + 32 * 3]);
            });
            vc[0].store(pc + 32 * 0);
            vc[1].store(pc + 32 * 1);
            vc[2].store(pc + 32 * 2);
            vc[3].store(pc + 32 * 3);
        });
}

void mbench_i2s8() {
    LinuxSimplePerf perf;
    auto jit = get_i2s8_mkernel();
    auto M = 1;
    auto K = 4096;
    auto N = 32;
    auto A = alloc_cache_aligned<int8_t>(M * K, 0);
    auto B = alloc_cache_aligned<int8_t>(K * N / 4, 0);
    auto C = alloc_cache_aligned<int32_t>(M * N, 0);
    // start test
    // warm-up code cache
    (*jit)(4, A.get(), K, B.get(), C.get(), N * 4, 0);

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N, M * K, K * N / 4);
    perf(
        [&] {
            (*jit)(K, A.get(), K, B.get(), C.get(), N * 4, 0);
        },
        std::string("i2s8"),
        4,                   // repeat
        K / 4,               // kernel's internal loop count
        M * K * N * 2,       // OPs
        M * K + (K * N / 4)  // bytes read
    );
}

void pack_vnni_4x32(SIMDJit* jit, VReg v0, VReg v1, VReg v2, VReg v3) {
    auto va = jit->get_vreg();
    auto vb = jit->get_vreg();
    auto vc = jit->get_vreg();
    auto vd = jit->get_vreg();

    jit->vpunpcklbw(va, v0, v1);
    jit->vpunpcklbw(vb, v2, v3);
    jit->vpunpckhbw(vc, v0, v1);
    jit->vpunpckhbw(vd, v2, v3);

    // jit->jcout(va, "\n", vb, "\n", vc, "\n", vd, "\n");

    jit->vpunpcklwd(v0, va, vb);
    jit->vpunpcklwd(v1, vc, vd);
    jit->vpunpckhwd(v2, va, vb);
    jit->vpunpckhwd(v3, vc, vd);

    // jit->jcout(v0, "\n", v1, "\n", v2, "\n", v3, "\n");
    jit->vperm2i128(va.ymm(), v0.ymm(), v2.ymm(), 0x20);
    jit->vperm2i128(vb.ymm(), v1.ymm(), v3.ymm(), 0x20);
    jit->vperm2i128(vc.ymm(), v0.ymm(), v2.ymm(), 0x31);
    jit->vperm2i128(vd.ymm(), v1.ymm(), v3.ymm(), 0x31);

    // jit->jcout(va, "\n", vb, "\n", vc, "\n", vd, "\n");
    v0.load(va);
    v1.load(vb);
    v2.load(vc);
    v3.load(vd);
}

void test_parallel_st() {
    for (int n_works = 1; n_works < 32; n_works++) {
        std::cout << " =========== n_works = " << n_works << std::endl;
        parallel_st(n_works, 1, [&](int ithr, int nthr, int n0, int n1) {
#pragma omp critical
            {
                std::cout << "ithr=" << ithr << "/" << nthr << "  " << n0 << " ~ " << n1 << " = " << n1 - n0
                          << std::endl;
            }
        });
    }
}

py::array_t<int32_t> mm_i2s(py::array_t<int8_t> X, py::array_t<uint8_t> PackedW) {
    static auto jit = get_i2s8_mkernel();
    ASSERT(X.ndim() == 2);
    ASSERT(PackedW.ndim() == 2);
    auto M = X.shape(0);
    auto K = PackedW.shape(0);
    ASSERT(X.shape(1) == K);
    auto N = PackedW.shape(1) * 4;

    py::array_t<int32_t> Y({M, N});

    auto BN = 32;

    // test_parallel_st();
    parallel_st(N, BN, [&](int ithr, int nthr, int n0, int n1) {
        for (int n = n0; n < n1; n++) {
            (*jit)(K, X.data(), K, PackedW.data() + n * K * 8, Y.mutable_data(0, n * BN), N * sizeof(int32_t), 0);
        }
    });
    return Y;
}

py::array_t<uint8_t> pack_i2s(py::array_t<int8_t> W) {
    static auto jit = SIMDJit::create([](SIMDJit* jit, SReg src, SReg src_stride, SReg dst, SReg K) {
        Ymm v0, v1, v2, v3, one, mask3;
        SReg src2;
        src2 = src + src_stride * 2;
        one.load(0x01010101);
        mask3.load(0x03030303);
        jit->do_while_(K > 0, [&] {
            v0.load(src);
            v1.load(src + src_stride);
            v2.load(src2);
            v3.load(src2 + src_stride);

            // jit->jcout(jit->jcout.as_hex8, v0, "\n", v1, "\n", v2, "\n", v3, "\n");
            pack_vnni_4x32(jit, v0, v1, v2, v3);
            // jit->jcout(jit->jcout.as_hex8, v0, "\n", v1, "\n", v2, "\n", v3, "\n");

            // -1, 0, 1 ===(I2_S encoding)===> 0, 1, 2
            jit->vpaddb(v0, v0, one);
            jit->vpaddb(v1, v1, one);
            jit->vpaddb(v2, v2, one);
            jit->vpaddb(v3, v3, one);

            jit->vpand(v0, v0, mask3);
            jit->vpand(v1, v1, mask3);
            jit->vpand(v2, v2, mask3);
            jit->vpand(v3, v3, mask3);

            jit->vpsllw(v1, v1, 2);
            jit->vpsllw(v2, v2, 4);
            jit->vpsllw(v3, v3, 6);

            // jit->jcout(jit->jcout.as_hex8, v0, "\n", v1, "\n", v2, "\n", v3, "\n");

            jit->vpor(v0, v0, v1);
            jit->vpor(v2, v2, v3);
            jit->vpor(v0, v0, v2);

            // jit->jcout(v0, "\n");
            v0.store(dst);

            src += src_stride * 4;
            src2 += src_stride * 4;
            K -= 4;
            dst += 4 * 8;
        });
    });

    ASSERT(W.ndim() == 2);
    auto K = W.shape(0);
    auto N = W.shape(1);
    auto* psrc = W.mutable_data();
    ASSERT((K % 4) == 0);
    ASSERT((N % 32) == 0);

    py::array_t<int8_t> PackedW({K, N / 4});
    auto* pdst = PackedW.mutable_data();

    parallel_st(N / 32, 1, [&](int ithr, int nthr, int n0, int n1) {
        for (int n = n0; n < n1; n++) {
            // Kx32 => Kx8
            (*jit)(psrc + n * 32, N, pdst + n * (K * 8), K);
        }
    });
    return PackedW;
}

void i2s_init(py::module_& m) {
    m.def("mbench_i2s8", &mbench_i2s8);
    m.def("pack_i2s", &pack_i2s);
    m.def("mm_i2s", &mm_i2s);
}
