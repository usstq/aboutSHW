/*
https://github.com/triSYCL/sycl/blob/sycl/unified/master/sycl/doc/extensions/experimental/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md
*/ 

#include <pybind11/numpy.h>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include "common.hpp"
#include "sycl/rms_kernel.hpp"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;
using namespace sycl;

extern sycl::queue sycl_queue;
extern std::vector<std::variant<cl_event, sycl::event>> all_events;

void test_esimd(tensor& a, tensor& b) {
    half* Buf1 = a;
    half* Buf2 = b;
    int Size = a.numel;
    //auto Dev = sycl_queue.get_device();
    //std::cout << "Running on " << Dev.get_info<sycl::info::device::name>() << "  Size=" << Size << "\n";
    sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Size / 8, [=](id<1> i) [[intel::sycl_explicit_simd]] {
            auto off = i * 8;
            simd<half, 8> Val1 = esimd::block_load<half, 8>(Buf1 + off);
            simd<half, 8> Val2 = esimd::block_load<half, 8>(Buf2 + off);
            esimd::block_store<half, 8>(Buf1 + off, Val1 * Val2);
        });
    });
}

tensor test_dpas(tensor& a, tensor& b) {
    half* pA = a;
    half* pB = b;
    tensor c({8, 8}, py::dtype("float32"));
    float *pC = c;
    int Size = a.numel;
    sycl_queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(1, [=](id<1> i) [[intel::sycl_explicit_simd]] {
            constexpr int M = 8;
            constexpr int N = 8;
            simd<half, M*16> A = esimd::block_load<half, M*16>(pA);

            //simd<half, N*16> B = esimd::gather<half, N*16>(pB);
            simd<float, M*N> C;
            //C = xmx::dpas<8, 8, float>(C, B, A);
            esimd::block_store<float, M*N>(pC, C);
        });
    });

    return c;
}

void rms(tensor& a, tensor& b, tensor& c, float eps) {
    half* pA = a;
    half* pB = b;
    auto& shape = a.get_shape();
    half *pC = c;
    int size = a.numel;
    int channel = shape.back();
    auto e = cldnn::sycl::details::rms_kernel(sycl_queue, pA, pB, pC, size / channel, channel, eps);
    all_events.emplace_back(e);
}

void init_ops(py::module_& m) {
    m.def("test_esimd", &test_esimd);
    m.def("test_dpas", &test_dpas);
    m.def("rms", &rms);
}
