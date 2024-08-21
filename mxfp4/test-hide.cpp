/*
    Can cpu-computation & memory-access be running in parallel perfectly?
    here we build a extreamly simple example to explore it.

    This test shows memory access can perfectly hide instruction CPU cycles
    and we can calculate the number of CPU cycles can be hiden perfectly by:
       
        CPU_cycles_hidden =  Memory_read_per_iteration[bytes] / Memory_BindWidth_per_Core[GB/s] * CPU_Frequency[GHz]

    so the higher the CPU frequency is (or the lower the memory bandwidth is), the more CPU cycles in computation can be hidden.
*/
#include <iostream>
#include <sstream>

#include <vector>

#include "../include/linux_perf.hpp"

#define JIT_DEBUG
#include "../include/jit.h"


static int NCOMP = getenv("NCOMP", 0);

class CPUMemParallel : public jit_generator {
public:
    int comp;
    CPUMemParallel(int comp) : comp(comp) {
        create_kernel("CPUMemParallel");
    }

    void generate() override {
        //  abi_param1 ~ abi_param6, RAX, R10, R11
        auto src_ptr = abi_param1;
        auto count = abi_param2;
        auto step = abi_param3;
        Xbyak::Label loop_begin;

        align(64, false);
        L(loop_begin);

        vmovdqu8(ymm0, ptr[src_ptr]);

        for(int i = 0; i < comp; i++) {
            vpandnd(zmm1, zmm0, zmm1);
        }

        prefetcht2(ptr[src_ptr + 4096]);
        lea(src_ptr, ptr[src_ptr + step]);

        dec(count);
        jnz(loop_begin, T_NEAR);
        ret();
    }
};


static int TSIZE = getenv("TSIZE", 16*1024*1024);
static int CLR_CACHE = getenv("CLR_CACHE", 1);

int test(LinuxPerf::PerfEventGroup& pevg, int comp, int step, int REPEAT) {
    CPUMemParallel jit_kernel(comp);

    tensor2D<uint8_t> tsrc(1, TSIZE);
    auto loop_count = TSIZE/32;
    auto* psrc = &tsrc[0];
    memset(psrc, 1, TSIZE);
    CLflush::run(psrc, TSIZE);

    for(int i = 0; i < REPEAT; i++) {
        if (CLR_CACHE) {
            CLflush::run(psrc, TSIZE);
        }
        pevg.rdpmc(
            [&]() {
                // repeat
                jit_kernel(psrc, loop_count, step);
            },
            jit_kernel.name(),
            loop_count,
            [&](uint64_t duration_ns, uint64_t* pmc){
                printf(" MemBW:%.2f(GB/s) @comp=%d step=%d", tsrc.size()*1.0/duration_ns, comp, step);
            });
    }
    return 0;
}

int main() {
    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
        //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},

        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x04, 0x04),"STALLS_TOTAL"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x06, 0x06),"STALLS_L3_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa3, 0x05, 0x05),"STALLS_L2_MISS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x40, 0x02),"BOUND_ON_STORES"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05), "BOUND_ON_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00), "ALL_LOADS"},

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00), "SPLIT_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00), "SPLIT_STORES"},
    });
    pevg.show_header();    
    for(int i = 0; i < 25; i++) {
        test(pevg, i, 0, 1);
        test(pevg, i, 32, 1);
    }
}