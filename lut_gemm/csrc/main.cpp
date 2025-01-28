#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "misc.hpp"
#include "simd_jit.hpp"
#include "simd_jit_tests.hpp"

using ov::intel_cpu::SIMDJit;
using ov::intel_cpu::VReg;

#define STRINGIFY(x)       #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

// micro kernel for lut 1.58 bits
// how kernel can be built on more organized way?
//   3 ternary-valued (-1, 0, 1) elements along reduction axis (K) were encoded into
//      1(sign)bit-map + 4(index)bits-map
std::shared_ptr<SIMDJit> get_lut_gemm_mkernel_1_58(int K) {
    auto jit = std::make_shared<SIMDJit>(__func__);
    jit->finalize();
    return jit;
}

py::array_t<int8_t> test(py::array_t<int8_t> Bmat) {
    ASSERT(Bmat.ndim() == 2);
    auto K = Bmat.shape(0);
    auto N = Bmat.shape(1);
    std::cout << " K,N = " << K << "," << N << std::endl;
    auto* ptr = Bmat.mutable_data();
    for (size_t i = 0; i < K * N; i++) {
        std::cout << static_cast<int>(ptr[i]) << std::endl;
    }
    // constexpr size_t elsize = sizeof(int8_t);
    // size_t strides[2]{1000 * 1000 * elsize, 1000 * elsize, elsize};
    auto a = py::array_t<int8_t>({2, 3});
    ptr = a.mutable_data();
    // std::cout << " shape = " << buf2.shape << ", strides = " << buf2.strides << std::endl;
    for (size_t i = 0; i < a.shape(0) * a.shape(1); i++) {
        ptr[i] = i;
    }
    return a;
}

PYBIND11_MODULE(lut_gemm, m) {
    m.doc() = R"pbdoc(
    )pbdoc";

    m.def("test", &test);

    m.def("simd_test_basic", simd_test_basic);
    m.def("simd_test_tput", simd_test_tput);
    m.def("simd_test_printreg", simd_test_printreg);
}
