/*
    Can cpu-computation & memory-access be running in parallel perfectly?
    here we build a extreamly simple example to explore it.

    This test shows memory access can perfectly hide instruction CPU cycles
    and we can calculate the number of CPU cycles can be hiden perfectly by:
       
        CPU_cycles_hidden =  Memory_read_per_iteration[bytes] / Memory_BindWidth_per_Core[GB/s] * CPU_Frequency[GHz]

    so the higher the CPU frequency is (or the lower the memory bandwidth is), the more CPU cycles in computation can be hidden.

    https://en.wikipedia.org/wiki/Spinlock#Example_implementation
    https://github.com/cfsamson/articles-atomics-in-rust
    https://nullprogram.com/blog/2022/03/13/
    
*/
#include <iostream>
#include <sstream>

#include <vector>

#include "../include/linux_perf.hpp"

#define JIT_DEBUG
#include "../include/jit.h"

#include "omp.h"

static int NCOMP = getenv("NCOMP", 0);
static int NSTEP = getenv("NSTEP", 32);
static int PREFETCH_ADV = getenv("PREFETCH_ADV", 4096);

class CPUMemParallel : public jit_generator {
public:
    CPUMemParallel() {
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

        for(int i = 0; i < NCOMP; i++) {
            vpandnd(zmm1, zmm0, zmm1);
        }

        prefetcht2(ptr[src_ptr + PREFETCH_ADV]);
        lea(src_ptr, ptr[src_ptr + NSTEP]);

        dec(count);
        jnz(loop_begin, T_NEAR);
        ret();
    }
};


static int TSIZE = getenv("TSIZE", 16*1024*1024);

// start thread lib

int test_openmp(LinuxPerf::PerfEventGroup& pevg, int REPEAT) {
    CPUMemParallel jit_kernel;
    auto nthr = omp_get_max_threads();

    std::vector<tensor2D<uint8_t>> tsrc(nthr);

    for(int i = 0; i < nthr; i++) {
        tsrc[i] = tensor2D<uint8_t>(1, TSIZE);
        auto* psrc = tsrc[i].ptr();
        memset(psrc, 1, TSIZE);
        CLflush::run(psrc, TSIZE);
    }

    auto loop_count = TSIZE/32;

    for(int i = 0; i < REPEAT; i++) {
        bool clr_cache = ((i & 1) == 0) || true;
        if (clr_cache) {
            #pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                CLflush::run(tsrc[ithr].ptr(), TSIZE);
            }
        }

        pevg.rdpmc(
            [&]() {
                #pragma omp parallel
                {
                    int ithr = omp_get_thread_num();
                    jit_kernel(tsrc[ithr].ptr(), loop_count);
                }
            },
            jit_kernel.name(),
            loop_count,
            [&](uint64_t duration_ns, uint64_t* pmc){
                auto bw = TSIZE*1.0/duration_ns;
                printf(" MemBW:%.2f(GB/s) x %d = %.2f(GB/s) %s", bw, nthr, bw*nthr, clr_cache ? "======== with clflush":"");
            });
    }
    return 0;
}

// totally atomic & Lock free synchronization
int test_lock_free(LinuxPerf::PerfEventGroup& pevg, int REPEAT) {
    CPUMemParallel jit_kernel;
    auto nthr = omp_get_max_threads();
    auto loop_count = TSIZE/32;

    volatile uint64_t sync[64*64] = {0}; // leave extra space to avoid false-sharing

    auto sync_threads = [&](int ithr){
        auto id = ithr * 64;
        sync[id] ++;
        auto expected = sync[id];
        for(int i = 0; i < nthr; i++) {
            // some threads finished sync_threads() faster and may have
            // started & finished next task & added its sync index.
            // but it cannot goes further until current thread also reach that point.
            for(;;) {
                auto s = sync[i*64];
                if (s == expected || s == (expected + 1))
                    break;
            }
        }
    };

    #pragma omp parallel
    {
        // cooperate threads knows what to do next, so 
        int ithr = omp_get_thread_num();

        tensor2D<uint8_t> tsrc(1, TSIZE);
        memset(tsrc.ptr(), 1, TSIZE);

        for(int i = 0; i < REPEAT; i++) {
            CLflush::run(tsrc.ptr(), TSIZE); // clear cache

            sync_threads(ithr);

            if (ithr == 0) {
                pevg.rdpmc([&](){
                    jit_kernel(tsrc.ptr(), loop_count);
                    sync_threads(ithr);
                },
                jit_kernel.name(),
                loop_count,
                [&](uint64_t duration_ns, uint64_t* pmc){
                    auto bw = TSIZE*1.0/duration_ns;
                    printf(" MemBW:%.2f(GB/s) x %d = %.2f(GB/s) [%d]", bw, nthr, bw*nthr, i);
                });
            }
            else
            {
                jit_kernel(tsrc.ptr(), loop_count);
                sync_threads(ithr);
            }

            sync_threads(ithr);
        }
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
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xa6, 0x21, 0x05), "BOUND_ON_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x81, 0x00), "ALL_LOADS"},
        
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x01, 0x00), "XSNP_MISS"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x02, 0x00), "XSNP_NO_FWD"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x04, 0x00), "XSNP_FWD"},
        {PERF_TYPE_RAW, X86_RAW_EVENT(0xd2, 0x08, 0x00), "XSNP_NONE"},
        

        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x41, 0x00), "SPLIT_LOADS"},
        //{PERF_TYPE_RAW, X86_RAW_EVENT(0xd0, 0x42, 0x00), "SPLIT_STORES"},
    });
    printf("omp_get_max_threads()=%d\n", omp_get_max_threads());
    pevg.show_header();
    printf("============= openmp =============\n");
    test_openmp(pevg, 10);
    printf("============= lock_free =============\n");
    test_lock_free(pevg, 10);
}
