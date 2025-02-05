#include "common.hpp"
#include "simd_jit_tests.hpp"
// micro kernel for lut 1.58 bits
// how kernel can be built on more organized way?
//   3 ternary-valued (-1, 0, 1) elements along reduction axis (K) were encoded into
//      1(sign)bit-map + 4(index)bits-map

// 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs
/* I2_S:
 unpacked = packed - 1
    -1 : 00 (0)
     0 : 01 (1)
     1 : 10 (2)
I2_S Kernel adopts the vanilla multiply-then-addition manner to perform the matrix multiplication.
As shown in Table 1, it transforms each full-precision weight into a 2-bit representation offline.
During computation, it transforms the weights back to their original values and performs the vanilla
GEMV operations. We recommend using it with sufficient threads, since it allows the compiler to
generate efficient pipelined instruction sequences.
*/
struct JIT : public SIMDJit {
    using Ptr = std::shared_ptr<JIT>;
    JIT(const char* name) : SIMDJit(name) {}
    void mmi2s(VReg va, VReg vb0, VReg vc0, VReg vc1, VReg vc2, VReg vc3, VReg mask3) {
        auto vb = get_vreg();

        // vb is packed, each 2 bits correspoindg to 1 vnni
        vpand(vb, vb0, mask3);
        // VNNI instruction `vpdpbusd` requires u8 * i8
        // to save this `vpsubb(vb, vb, one)` step, we store (weight + 1) as u8
        // and keep activation as i8
        vpdpbusd(vc0, vb, va, Xbyak::VexEncoding);

        vpsrldq(vb, vb0, 2);
        vpand(vb, vb, mask3);
        vpdpbusd(vc1, vb, va, Xbyak::VexEncoding);

        vpsrldq(vb, vb0, 2);
        vpand(vb, vb, mask3);
        vpdpbusd(vc2, vb, va, Xbyak::VexEncoding);

        vpsrldq(vb, vb0, 2);
        vpand(vb, vb, mask3);
        vpdpbusd(vc3, vb, va, Xbyak::VexEncoding);
    }

    void mmw4(VReg va, VReg vb0, VReg vc0, VReg vc1, VReg maskF) {
        auto vb = get_vreg();
        vpand(vb, vb0, maskF);
        vpdpbusd(vc0, vb, va, Xbyak::VexEncoding);
        vpsrldq(vb, vb0, 4);

        vpand(vb, vb0, maskF);
        vpdpbusd(vc1, vb, va, Xbyak::VexEncoding);
    }
};

JIT::Ptr get_i2s_mkernel(int M) {
    auto jit = std::make_shared<JIT>("mk_i2s");
    ASSERT(M == 1 || M == 2 || M == 3);

    auto cnt = jit->get_arg(0);
    auto pa = jit->get_arg(1);
    auto stride_a = jit->get_arg(2);
    auto pb = jit->get_arg(3);

    auto mask3 = jit->get_vreg();
    {
        static int mask3_value = 0x03030303;
        auto addr = jit->get_sreg();
        jit->mov(addr, reinterpret_cast<uintptr_t>(&mask3_value));
        jit->vpbroadcastd(mask3, jit->ptr[addr.r64()]);
    }

    auto vc = jit->get_vregs(M * 4);

    if (M == 1) {
        auto va = jit->get_vreg();
        auto vb0 = jit->get_vreg();
        auto vbx = jit->get_vreg();

        jit->do_while_(cnt > 0, [&] {
            jit->vpbroadcastd(va, jit->ptr[pa.r64()]);
            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);
            // jit->mmi2s(va, vb0, vc0, vc1, vc2, vc3, mask3);
            jit->vpand(vbx, vb0, mask3);
            jit->vpdpbusd(vc[0], vbx, va, Xbyak::VexEncoding);

            jit->vpsrldq(vbx, vb0, 2);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[1], vbx, va, Xbyak::VexEncoding);

            jit->vpsrldq(vbx, vb0, 4);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[2], vbx, va, Xbyak::VexEncoding);

            jit->vpsrldq(vbx, vb0, 6);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[3], vbx, va, Xbyak::VexEncoding);

            pa += 4;
            pb += 32;
            cnt -= 4;
        });
    }
    if (M == 2) {
        auto va = jit->get_vregs(M);
        auto vb0 = jit->get_vreg();
        auto vbx = jit->get_vreg();

        jit->do_while_(cnt > 0, [&] {
            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);

            jit->vpbroadcastd(va[0], jit->ptr[pa.r64()]);
            jit->vpbroadcastd(va[1], jit->ptr[pa.r64() + stride_a.r64()]);

            // jit->mmi2s(va, vb0, vc0, vc1, vc2, vc3, mask3);
            jit->vpand(vbx, vb0, mask3);
            jit->vpdpbusd(vc[0], vbx, va[0], Xbyak::VexEncoding);
            jit->vpdpbusd(vc[4], vbx, va[1], Xbyak::VexEncoding);

            jit->vpsrldq(vbx, vb0, 2);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[1], vbx, va[0], Xbyak::VexEncoding);
            jit->vpdpbusd(vc[5], vbx, va[1], Xbyak::VexEncoding);

            jit->vpsrldq(vbx, vb0, 4);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[2], vbx, va[0], Xbyak::VexEncoding);
            jit->vpdpbusd(vc[6], vbx, va[1], Xbyak::VexEncoding);

            jit->vpsrldq(vbx, vb0, 6);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[3], vbx, va[0], Xbyak::VexEncoding);
            jit->vpdpbusd(vc[7], vbx, va[1], Xbyak::VexEncoding);

            pa += 4;
            pb += 32;
            cnt -= 4;
        });
    }
    if (M == 3) {
        auto va = jit->get_vreg();
        auto vb0 = jit->get_vreg();
        auto vbx = jit->get_vreg();

        jit->do_while_(cnt > 0, [&] {
            // jit->mmi2s(va, vb0, vc0, vc1, vc2, vc3, mask3);
            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);
            jit->vpand(vbx, vb0, mask3);
            jit->vpbroadcastd(va, jit->ptr[pa.r64()]);
            jit->vpdpbusd(vc[0], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64()]);
            jit->vpdpbusd(vc[4], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64() * 2]);
            jit->vpdpbusd(vc[8], vbx, va, Xbyak::VexEncoding);

            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);
            jit->vpbroadcastd(va, jit->ptr[pa.r64()]);
            jit->vpsrldq(vbx, vb0, 2);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[1], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64()]);
            jit->vpdpbusd(vc[5], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64() * 2]);
            jit->vpdpbusd(vc[9], vbx, va, Xbyak::VexEncoding);

            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);
            jit->vpbroadcastd(va, jit->ptr[pa.r64()]);
            jit->vpsrldq(vbx, vb0, 4);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[2], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64()]);
            jit->vpdpbusd(vc[6], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64() * 2]);
            jit->vpdpbusd(vc[10], vbx, va, Xbyak::VexEncoding);

            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);
            jit->vpbroadcastd(va, jit->ptr[pa.r64()]);
            jit->vpsrldq(vbx, vb0, 6);
            jit->vpand(vbx, vbx, mask3);
            jit->vpdpbusd(vc[3], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64()]);
            jit->vpdpbusd(vc[7], vbx, va, Xbyak::VexEncoding);
            jit->vpbroadcastd(va, jit->ptr[pa.r64() + stride_a.r64() * 2]);
            jit->vpdpbusd(vc[11], vbx, va, Xbyak::VexEncoding);

            pa += 4;
            pb += 32;
            cnt -= 4;
        });
    }
    jit->return_();
    jit->finalize();
    return jit;
}


void mbench_i2s(int M) {
    LinuxSimplePerf perf;
    auto jit = get_i2s_mkernel(M);

    auto K = 4096;
    auto N = 32;
    auto A = alloc_cache_aligned<int8_t>(M * K, 0);
    auto B = alloc_cache_aligned<int8_t>(K * N / 4, 0);
    // start test
    // warm-up code cache
    (*jit)(4, A.get(), K, B.get());

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N, M * K, K * N / 4);
    perf(
        [&] {
            (*jit)(K, A.get(), K, B.get());
        },
        std::string("i2s_M") + std::to_string(M),
        4,                   // repeat
        K / 4,               // kernel's internal loop count
        M * K * N * 2,       // OPs
        M * K + (K * N / 4)  // bytes read
    );
}


void mbench_w4(int M) {
    ASSERT(M == 1 || M == 2);
    LinuxSimplePerf perf;
    auto jit = std::make_shared<JIT>("mk_i2s");
    {
        auto va = jit->get_vreg();
        auto vb0 = jit->get_vreg();
        auto vb1 = jit->get_vreg();

        auto vc00 = jit->get_vreg();
        auto vc01 = jit->get_vreg();
        auto vc02 = jit->get_vreg();
        auto vc03 = jit->get_vreg();

        auto vc10 = jit->get_vreg();
        auto vc11 = jit->get_vreg();
        auto vc12 = jit->get_vreg();
        auto vc13 = jit->get_vreg();

        auto maskF = jit->get_vreg();

        auto cnt = jit->get_arg(0);
        auto pa0 = jit->get_arg(1);
        auto strideA = jit->get_arg(2);
        auto pb = jit->get_arg(3);

        {
            auto addr = jit->get_sreg();
            static int maskF_value = 0x0F0F0F0F;
            jit->mov(addr, reinterpret_cast<uintptr_t>(&maskF_value));
            jit->vpbroadcastd(maskF, jit->ptr[addr.r64()]);
        }
        auto pa1 = jit->get_sreg();
        pa1 = pa0 + strideA;
        jit->do_while_(cnt > 0, [&]() {
            jit->vpbroadcastd(va, jit->ptr[pa0.r64()]);
            jit->vmovdqu(vb0, jit->ptr[pb.r64()]);
            jit->vmovdqu(vb1, jit->ptr[pb.r64() + 32]);
            jit->mmw4(va, vb0, vc00, vc01, maskF);
            jit->mmw4(va, vb1, vc02, vc03, maskF);

            if (M > 1) {
                jit->vpbroadcastd(va, jit->ptr[pa1.r64()]);
                jit->mmw4(va, vb0, vc10, vc11, maskF);
                jit->mmw4(va, vb1, vc12, vc13, maskF);
            }
            pa0 += 4;
            pa1 += 4;
            pb += 64;
            cnt -= 4;
        });
        jit->return_();
        jit->finalize();
    }

    auto K = 4096;
    auto N = 32;
    auto A = alloc_cache_aligned<int8_t>(K * M, 0);
    auto B = alloc_cache_aligned<int8_t>(K * N / 2, 0);
    // start test
    // warm-up code cache
    (*jit)(4, A.get(), K, B.get());

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N,  K * M, K * N / 2);
    perf(
        [&] {
            (*jit)(K, A.get(), K, B.get());
        },
        std::string("w4_M") + std::to_string(M),
        4,                 // repeat
        K / 4,             // kernel's internal loop count
        M * K * N * 2,     // OPs
        M * K + K * N / 2  // bytes read
    );
}

void mbench_w8(int M) {
    ASSERT(M > 0 && M < 5);
    LinuxSimplePerf perf;
    auto jit = std::make_shared<JIT>("mk_i2s");
    {
        auto va0 = jit->get_vreg();
        auto va1 = jit->get_vreg();
        auto va2 = jit->get_vreg();
        auto vbx = jit->get_vreg();

        auto vc00 = jit->get_vreg();
        auto vc01 = jit->get_vreg();
        auto vc02 = jit->get_vreg();
        auto vc03 = jit->get_vreg();

        auto vc10 = jit->get_vreg();
        auto vc11 = jit->get_vreg();
        auto vc12 = jit->get_vreg();
        auto vc13 = jit->get_vreg();

        auto vc20 = jit->get_vreg();
        auto vc21 = jit->get_vreg();
        auto vc22 = jit->get_vreg();
        auto vc23 = jit->get_vreg();

        auto cnt = jit->get_arg(0);
        auto pa = jit->get_arg(1);
        auto strideA = jit->get_arg(2);
        auto pb = jit->get_arg(3);

        jit->do_while_(cnt > 0, [&]() {
            if (M > 0)
                jit->vpbroadcastd(va0, jit->ptr[pa.r64()]);
            if (M > 1)
                jit->vpbroadcastd(va1, jit->ptr[pa.r64() + strideA.r64()]);
            if (M > 2)
                jit->vpbroadcastd(va2, jit->ptr[pa.r64() + strideA.r64() * 2]);

            jit->vmovdqu(vbx, jit->ptr[pb.r64()]);
            if (M > 0)
                jit->vpdpbusd(vc00, vbx, va0, Xbyak::VexEncoding);
            if (M > 1)
                jit->vpdpbusd(vc10, vbx, va1, Xbyak::VexEncoding);
            if (M > 2)
                jit->vpdpbusd(vc20, vbx, va2, Xbyak::VexEncoding);

            jit->vmovdqu(vbx, jit->ptr[pb.r64() + 32]);
            if (M > 0)
                jit->vpdpbusd(vc01, vbx, va0, Xbyak::VexEncoding);
            if (M > 1)
                jit->vpdpbusd(vc11, vbx, va1, Xbyak::VexEncoding);
            if (M > 2)
                jit->vpdpbusd(vc21, vbx, va2, Xbyak::VexEncoding);

            jit->vmovdqu(vbx, jit->ptr[pb.r64() + 32 * 2]);
            if (M > 0)
                jit->vpdpbusd(vc02, vbx, va0, Xbyak::VexEncoding);
            if (M > 1)
                jit->vpdpbusd(vc12, vbx, va1, Xbyak::VexEncoding);
            if (M > 2)
                jit->vpdpbusd(vc22, vbx, va2, Xbyak::VexEncoding);

            jit->vmovdqu(vbx, jit->ptr[pb.r64() + 32 * 3]);
            if (M > 0)
                jit->vpdpbusd(vc03, vbx, va0, Xbyak::VexEncoding);
            if (M > 1)
                jit->vpdpbusd(vc13, vbx, va1, Xbyak::VexEncoding);
            if (M > 2)
                jit->vpdpbusd(vc23, vbx, va2, Xbyak::VexEncoding);

            pa += 4;
            pb += 32 * 4;
            cnt -= 4;
        });
        jit->return_();
        jit->finalize();
    }

    auto K = 4096;
    auto N = 32;
    auto A = alloc_cache_aligned<int8_t>(K * M, 0);
    auto B = alloc_cache_aligned<int8_t>(K * N, 0);
    // start test
    // warm-up code cache
    (*jit)(4, A.get(), K, B.get());

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N,  K * M, K * N);
    perf(
        [&] {
            (*jit)(K, A.get(), K, B.get());
        },
        std::string("w8_M") + std::to_string(M),
        4,              // repeat
        K / 4,          // kernel's internal loop count
        M * K * N * 2,  // OPs
        M * K + K * N   // bytes read
    );
}

void i2s_init(py::module_& m);
void perf_init(py::module_& m);
void tl1_init(py::module_& m);

PYBIND11_MODULE(lut_gemm, m) {
    m.doc() = R"pbdoc(
    )pbdoc";

    i2s_init(m);
    tl1_init(m);
    perf_init(m);

    m.def("simd_test_basic", simd_test_basic);
    m.def("simd_test_tput", simd_test_tput);
    m.def("simd_test_printreg", simd_test_printreg);

    m.def("mbench_i2s", mbench_i2s);
    m.def("mbench_w4", mbench_w4);
    m.def("mbench_w8", mbench_w8);
}
