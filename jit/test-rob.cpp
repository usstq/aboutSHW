
#include "../perf/linux_perf.hpp"
#define JIT_DEBUG
#include "jit.h"
#include <cstdlib>


class InstructionLoop : public jit_generator {
public:
    int unroll_count;
    InstructionLoop(std::string ktype, int unroll_count) : unroll_count(unroll_count) {
        create_kernel("InstructionLoop");
    }
    void generate() override {
        auto pointer_chasing_start1 = abi_param1;
        auto pointer_chasing_start2 = abi_param2;
        auto pointer_chasing_count = abi_param3;

        Xbyak::Label loop_begin;
        align(64, false);
        L(loop_begin);
        mov(pointer_chasing_start1, ptr[pointer_chasing_start1]);
        for(int i = 0; i < unroll_count; i++) {
            nop();
        }
        mov(pointer_chasing_start2, ptr[pointer_chasing_start2]);
        for(int i = 0; i < unroll_count; i++) {
            nop();
        }
        dec(pointer_chasing_count);
        jne(loop_begin, T_NEAR);
        ret();        
    }
};

// https://stackoverflow.com/questions/9218724/get-random-element-and-remove-it
template <typename T>
void fast_remove_at(std::vector<T>& v, typename std::vector<T>::size_type n)
{
    std::swap(v[n], v.back());
    v.pop_back();
}

std::vector<void *> PointerChansing(int count, int valid_cnt) {
    std::vector<void *> ret(count, nullptr);
    std::vector<int> freelist(count - 1, 0);
    for(int i = 0; i < freelist.size(); i++) {
        freelist[i] = i+1;
    }

    int cur = 0;
    for(int i = 0; i < valid_cnt - 1; i++) {
        // random draw without replacement
        auto next_i = rand() % freelist.size();
        auto next_slot = freelist[next_i];
        fast_remove_at(freelist, next_i);
        ret[cur] = &ret[next_slot];
        cur = next_slot;
    }
    ret[cur] = &ret[0];
    return ret;
}

void clflush(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    for (int i = 0; i < bytes; i += 64) {
        _mm_clflushopt(p + i);
    }
    _mm_mfence();
};

void test_ROB(std::string ktype = {}) {
    static int index = 0;
    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    });
    if (index == 0)
        pevg.show_header();
    index ++;

    // initialize a 800MB pointer chasing table (>L3)
    printf("Initializing pointer-chansing table...\n");
    int chase_cnt = 800*1024*1024/sizeof(void*);
    int valid_cnt = chase_cnt/2000;
    auto chase1 = PointerChansing(chase_cnt, valid_cnt);
    auto chase2 = PointerChansing(chase_cnt, valid_cnt);

    printf("start.\n");
    double base_cycles = 0;
    for (int i = 0; i < 1024; i+=1) {
        InstructionLoop inst(ktype, i);
        clflush(&chase1[0], chase1.size()*sizeof(chase1[0]));
        clflush(&chase2[0], chase2.size()*sizeof(chase2[0]));
        printf("[%d] ", i);
        auto pmc = pevg.rdpmc(
            [&]() {
                inst(&chase1[0], &chase2[0], valid_cnt);
            },
            ktype,
            valid_cnt);
        if (i == 0) {
            base_cycles = pmc[0];
        } else {
            // unicode progress bar:  ░ ▄ █
            int inc_percentage = pmc[0]*50.0/base_cycles;
            for(int k = 0; k<inc_percentage; k++) printf("░");
            printf("\n");
        }
    }
}

int main() {
    test_ROB("rob");
}