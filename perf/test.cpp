
// https://stackoverflow.com/questions/42088515/perf-event-open-how-to-monitoring-multiple-events

#include "../include/linux_perf.hpp"

#include "../thirdparty/xbyak/xbyak/xbyak.h"
#include <chrono>
#include <thread>
#include <sched.h>

class Jit : public Xbyak::CodeGenerator {
public:

/*
    RDI, RSI, RDX, RCX, R8, R9

*/
	Jit() : Xbyak::CodeGenerator(Xbyak::DEFAULT_MAX_CODE_SIZE, 0) {
    }

    template<class GEN_KERNEL>
    void setup_loop(GEN_KERNEL genk) {
        static std::vector<float> initv(16, 0.1f);
        mov(r8, reinterpret_cast<uintptr_t>(&initv[0]));
        vmovups(zmm0, ptr[r8]);
        vmovups(zmm1, ptr[r8]);

        L(".lp");
        genk();
        dec(edi);
		jne(".lp");
        ret();
    }

    template <typename... kernel_args_t>
    int operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = int (*)(const kernel_args_t... args);
        auto* fptr = (jit_kernel_func_t)getCode();
        return (*fptr)(std::forward<kernel_args_t>(args)...);
    }
};

void do_something() {
  int i;
  char* ptr;

  ptr = reinterpret_cast<char*>(malloc(100*1024*1024));
  for (i = 0; i < 100*1024*1024; i++) {
    ptr[i] = (char) (i & 0xff); // pagefault
  }
  free(ptr);
}

inline int omp_thread_count() {
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n;
}

void busy_sleep_ms(int duration_ms) {
    const auto now0 = std::chrono::system_clock::now();
    do{
        auto now1 = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now1 - now0);
        if (ms.count() > duration_ms) break;
    } while(1);
}
/*
uint32_t get_cpu_id() {
    uint32_t cpuid;
    if (getcpu(reinterpret_cast<uint32_t*>(&cpuid), nullptr) != 0) {
        perror("getcpu failed : ");
        abort();
    }
    return cpuid;
}
*/
#define get_cpu_id sched_getcpu

void test2() {
    Jit inst;
    inst.setup_loop([&](){
        for(int i = 0; i< 50; i++) {
            inst.vaddps(inst.zmm0, inst.zmm31, inst.zmm0);
            /*
            inst.vaddps(inst.zmm1, inst.zmm31, inst.zmm1);
            inst.vaddps(inst.zmm2, inst.zmm31, inst.zmm2);
            inst.vaddps(inst.zmm3, inst.zmm31, inst.zmm3);
            inst.vaddps(inst.zmm4, inst.zmm31, inst.zmm4);
            inst.vaddps(inst.zmm5, inst.zmm31, inst.zmm5);
            inst.vaddps(inst.zmm6, inst.zmm31, inst.zmm6);
            inst.vaddps(inst.zmm7, inst.zmm31, inst.zmm7);
            */
        }
    });

#if 0
    {
        auto prof0 = LinuxPerf::Profile("outer", 0);
        //std::this_thread::sleep_for(std::chrono::microseconds(500));
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    {
        auto prof0 = LinuxPerf::Profile("clock_gettime", 0);
        struct timespec tp0;
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp0) != 0) {
            perror("clock_gettime(CLOCK_MONOTONIC_RAW,...) failed!");
            abort();
        }
        uint64_t tpns = (tp0.tv_sec * 1000000000) + tp0.tv_nsec;
        if (tpns == 0) {
            printf("impossible!\n");
        }
    }

    {
        auto prof0 = LinuxPerf::Profile("_rdtsc", 0);
        if (_rdtsc() == 0) {
            printf("impossible!\n");
        }
    }

#endif
    LinuxPerf::ProfileScope pscope[256];

    auto nthr = omp_thread_count();
    printf("omp_thread_count = %d\n", nthr);

    uint32_t cpus[128];

    #pragma omp parallel for
    for(int ithr=0; ithr<nthr; ithr++) {
        cpus[ithr] = get_cpu_id();
    }

    for(int i=0;i<100;i++) {
        #pragma omp parallel for
        for(int ithr=0; ithr<nthr; ithr++) {
            pscope[ithr] = std::move(LinuxPerf::Profile("inner", 0));
        }

        #pragma omp parallel for
        for(int ithr=0; ithr<nthr; ithr++) {
            auto cpu = get_cpu_id();
            if (cpus[ithr] != cpu) {
                printf("%u->%u\n", cpus[ithr], cpu);
                cpus[ithr] = cpu;
            }
            inst(1000);
        }

        #pragma omp parallel for
        for(int ithr=0; ithr<nthr; ithr++) {
            pscope[ithr].finish();
        }

        busy_sleep_ms(10);
        /*
        auto pmc = pevg.rdpmc([&](){
            inst(1000);
        }, "test2", __LINE__, false);
        auto duration = pevg.refcycle2nano(pmc[2]);
        printf("CPU_freq:%.2f GHz, CPI: %.2f\n", static_cast<double>(pmc[0])/duration, static_cast<double>(pmc[0])/pmc[1]);
        */
    }
}

int main(int argc, char* argv[]) {
  LinuxPerf::Init();
  //test1(); return 0;
  test2();
  return 0;
}
