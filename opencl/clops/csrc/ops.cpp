// CC=icpx pip install -e .
#include <sycl/CL/opencl.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;
using namespace sycl;

extern sycl::queue sycl_queue;

void test_esimd(float* Buf1, float* Buf2, int Size) {
  auto Dev = sycl_queue.get_device();
  std::cout << "Running on " << Dev.get_info<sycl::info::device::name>() << "  Size=" << Size << "\n";
  sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(Size/8, [=](id<1> i) [[intel::sycl_explicit_simd]] {
      auto off = i*8;
      simd<float, 8> Val1 = esimd::block_load<float, 8>(Buf1 + off);
      simd<float, 8> Val2 = esimd::block_load<float, 8>(Buf2 + off);
      esimd::block_store<float, 8>(Buf1 + off, Val1 * Val2);
    });
  });
}