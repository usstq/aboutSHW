#include <iostream>
#include <sstream>
#include "mxformat.hpp"

#include <vector>

#include "../perf/linux_perf.hpp"

#define JIT_DEBUG

#include "jit.h"

class Instruction : public jit_generator {
 public:
  Instruction() { create_kernel("Instruction"); }
  void generate() override {
    static std::vector<float> initv(16, 0.0f);
    mov(r8, reinterpret_cast<uintptr_t>(&initv[0]));
    vmovups(zmm0, ptr[r8]);
    vmovups(zmm1, ptr[r8]);
    vmovups(zmm2, ptr[r8]);
    vmovups(zmm3, ptr[r8]);
    vmovups(zmm4, ptr[r8]);
    vmovups(zmm5, ptr[r8]);
    vmovups(zmm6, ptr[r8]);
    vmovups(zmm7, ptr[r8]);
    vmovups(zmm31, ptr[r8]);
    auto regCnt = abi_param1;

    L(".lp");
#if 0
    // 0.5
    vaddps(zmm0, zmm31, zmm0);
    vaddps(zmm1, zmm31, zmm1);
    vaddps(zmm2, zmm31, zmm2);
    vaddps(zmm3, zmm31, zmm3);

    vaddps(zmm4, zmm31, zmm4);
    vaddps(zmm5, zmm31, zmm5);
    vaddps(zmm6, zmm31, zmm6);
    vaddps(zmm7, zmm31, zmm7);
#endif

#if 1
    // 0.5 FMA is powerful
    vfmadd132ps(zmm0, zmm31, zmm0);
    vfmadd132ps(zmm1, zmm31, zmm1);
    vfmadd132ps(zmm2, zmm31, zmm2);
    vfmadd132ps(zmm3, zmm31, zmm3);

    vfmadd132ps(zmm4, zmm31, zmm4);
    vfmadd132ps(zmm5, zmm31, zmm5);
    vfmadd132ps(zmm6, zmm31, zmm6);
    vfmadd132ps(zmm7, zmm31, zmm7);
#endif

#if 0
    // 1.66 : bfloat16 ALU has low throughput (1/2 of FP32 FMA)
    vdpbf16ps(zmm0, zmm31, zmm0);
    vdpbf16ps(zmm1, zmm31, zmm1);
    vdpbf16ps(zmm2, zmm31, zmm2);
    vdpbf16ps(zmm3, zmm31, zmm3);

    vdpbf16ps(zmm4, zmm31, zmm4);
    vdpbf16ps(zmm5, zmm31, zmm5);
    vdpbf16ps(zmm6, zmm31, zmm6);
    vdpbf16ps(zmm7, zmm31, zmm7);
#endif

    dec(regCnt);
    jne(".lp");
    ret();
  }
};

void test_CPI() {
  LinuxPerf::PerfEventGroup pevg({
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
  });
  Instruction inst;
  for(int i = 0; i < 10; i++) {
    auto pmc = pevg.rdpmc([&]() { inst(10000); }, true);
    std::cout << "CPI:" << (double)(pmc[0])/pmc[1] << std::endl;
  }
}

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
  MXFP4Brgemm() { create_kernel("MXFP4Brgemm"); }
  void generate() override {
    // read 16x(FP4)=64bit
    //VMOVDQU8 
    //  abi_param1 ~ abi_param6, RAX, R10, R11
    auto regSrcAddr = abi_param1;

    // load 
    vmovq(xmm0, ptr[regSrcAddr]);

  }
};



/*
test performance of MXFP4
*/

int main() {
  test_CPI();
  return 0;
}