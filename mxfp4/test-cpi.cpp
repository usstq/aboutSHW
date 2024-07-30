
#include "../perf/linux_perf.hpp"
#include "jit.h"

class InstructionLoop : public jit_generator {
 public:
  InstructionLoop() { create_kernel("InstructionLoop"); }

  void generate() override {
    static std::vector<float> initv(16, 0.0f);
    mov(rax, reinterpret_cast<uintptr_t>(&initv[0]));
    vmovups(zmm0, ptr[rax]);
    vmovups(zmm1, ptr[rax]);
    vmovups(zmm2, ptr[rax]);
    vmovups(zmm3, ptr[rax]);
    vmovups(zmm4, ptr[rax]);
    vmovups(zmm5, ptr[rax]);
    vmovups(zmm6, ptr[rax]);
    vmovups(zmm7, ptr[rax]);
    vmovups(zmm31, ptr[rax]);
    auto regCnt = abi_param1;

    Xbyak::Label loop_begin;

    align(64, false);
    L(loop_begin);
    // unroll enough times to lower the impact of loop-overhead instructions
    // these loop-overhead instructions usually can increase CPI by a lot.
    for(int unroll = 0; unroll < 100; unroll++) {
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

#if 0
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

#if 1
    // 2.05 : 
    //    considering vdpbf16ps performs 32x2 FOPS while vfmadd132ps only performs 16x2 FOPS
    //    the overall throuput of avx512-BF16 is half of avx512-FMA
    vdpbf16ps(zmm0, zmm31, zmm0);
    vdpbf16ps(zmm1, zmm31, zmm1);
    vdpbf16ps(zmm2, zmm31, zmm2);
    vdpbf16ps(zmm3, zmm31, zmm3);

    vdpbf16ps(zmm4, zmm31, zmm4);
    vdpbf16ps(zmm5, zmm31, zmm5);
    vdpbf16ps(zmm6, zmm31, zmm6);
    vdpbf16ps(zmm7, zmm31, zmm7);
#endif
    }
    dec(regCnt);
    jne(loop_begin, T_NEAR);

    ret();
  }
};

void test_CPI() {
  LinuxPerf::PerfEventGroup pevg({
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
      {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
  });
  InstructionLoop inst;
  for(int i = 0; i < 10; i++) {
    auto pmc = pevg.rdpmc([&]() { inst(10000); }, true);
    printf("CPI: %.2f\n", (double)(pmc[0])/pmc[1]);
  }
}

int main() {
    test_CPI();
    return 0;
}