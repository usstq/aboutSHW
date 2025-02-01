#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "misc.hpp"
#include "simd_jit.hpp"
#include "simple_perf.hpp"

using ov::intel_cpu::SIMDJit;
using ov::intel_cpu::SReg;
using ov::intel_cpu::VReg;

namespace py = pybind11;

/*
TL1 Kernel preprocesses every two full-precision weights by packing them into 4-bit index (see
Table 2), and pre-computes their corresponding activations into 32 = 9 values. The index-value pairs
are stored in a lookup table to perform LUT computation [PPK+22, WCC+24]. GEMV processing
is performed using an int16 LUT and accumulation through addition. We recommend using it with a
limited number of threads when serving large models.

16-bit LUT:

*/

std::shared_ptr<SIMDJit> get_tl1_mkernel() {
    auto jit = std::make_shared<SIMDJit>("mk_i2s8");

    auto cnt = jit->get_arg(0);
    auto pa = jit->get_arg(1);
    auto stride_a = jit->get_arg(2);
    auto pb = jit->get_arg(3);
    auto pc = jit->get_arg(4);
    auto stride_c = jit->get_arg(5);
    auto do_accumulate = jit->get_arg(6);

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

        pa += 32*4;
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
    (*jit)(4, A.get(), K, B.get(), C.get(), N*4, 0);

    printf("(%d,%d,%d) size of A,B = %d, %d bytes\n", M, K, N, M * K * 16, K * N / 4);
    perf(
        [&] {
            (*jit)(K, A.get(), K, B.get(), C.get(), N*4, 0);
        },
        std::string("TL1"),
        4,                   // repeat
        K / 4,               // kernel's internal loop count
        M * K * N * 2,       // OPs
        (M * K * 16 * 0) + (K * N / 4)  // bytes read
    );
}

void tl1_init(py::module_& m) {
    m.def("mbench_tl1", &mbench_tl1);
}
