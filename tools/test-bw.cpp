// g++ -O2 -fopenmp ./test-bw.cpp
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <thread>
#include <x86intrin.h>
#include "../include/jit.h"

class TestBWjit : public jit_generator {
public:
    TileConfig m_tile_cfg;
    TestBWjit() {
        create_kernel("TestBWjit");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // A0:4
                             {16, 64}, // A1:5
                             {16, 64}, // B0:6
                             {16, 64}, // B1:7
                         });
    }

    const TileConfig& tile_config() { return m_tile_cfg; }

    // to save push/pop: do not use `abi_save_gpr_regs`
    // uint8_t* base, int64_t size, uint64_t tsc_limit
    Xbyak::Reg64 reg_base = abi_param1;
    Xbyak::Reg64 reg_size = abi_param2;
    Xbyak::Reg64 reg_tsc_limit = abi_param3; // RDX
    Xbyak::Reg64 reg_tsc_0 = r8;
    Xbyak::Reg64 reg_repeats = r9;
    Xbyak::Reg64 reg_cnt = r10;
    Xbyak::Reg64 reg_tscL = r11;

    void generate() {
        Xbyak::Label loop_begin;
        Xbyak::Label loop_data;

        mov(reg_tscL, abi_param3); // RDX

        rdtsc(); // EDX:EAX
        sal(rdx, 32);
        or_(rax, rdx); // 64bit
        mov(reg_tsc_0, rax);

        xor_(reg_repeats, reg_repeats);

        align(64, false);
        L(loop_begin);

        xor_(reg_cnt, reg_cnt);
        L(loop_data);
        // for (int64_t i = 0; i < size; i += 64*4)
#if 0
        prefetcht0(ptr[reg_base + reg_cnt]);
        prefetcht0(ptr[reg_base + reg_cnt + 64*1]);
        prefetcht0(ptr[reg_base + reg_cnt + 64*2]);
        prefetcht0(ptr[reg_base + reg_cnt + 64*3]);
#else
        vmovaps(zmm0, ptr[reg_base + reg_cnt]);
        vmovaps(zmm1, ptr[reg_base + reg_cnt + 64]);
        vmovaps(zmm2, ptr[reg_base + reg_cnt + 64 * 2]);
        vmovaps(zmm3, ptr[reg_base + reg_cnt + 64 * 3]);
#endif
        add(reg_cnt, 64 * 4);
        cmp(reg_cnt, reg_size);
        jl(loop_data, T_NEAR);

        inc(reg_repeats);
        rdtsc(); // EDX:EAX
        sal(rdx, 32);
        or_(rax, rdx);       // 64bit
        sub(rax, reg_tsc_0); // tsc1 - tsc0
        cmp(rax, reg_tscL);  //
        jl(loop_begin, T_NEAR);

        mov(rax, reg_repeats);
        ret();
    }
};

inline int omp_thread_count() {
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n;
}
uint64_t rdtsc_calibrate(int seconds = 1) {

    uint64_t start_ticks;
    std::cout << "rdtsc is calibrating ... " << std::flush;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    auto tsc_diff = (__rdtsc() - start_ticks);
    std::cout << "done." << std::endl;
    return tsc_diff / seconds;
}

uint64_t get_tsc_ticks_per_second() {
    static auto tsc_ticks_per_second = rdtsc_calibrate();
    return tsc_ticks_per_second;
}

struct pretty_size {
    double sz;
    std::string txt;
    pretty_size(double sz, const char* unit = "") : sz(sz) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << std::setw(8);
        if (sz < 1024)
            ss << sz;
        else if (sz < 1024 * 1024)
            ss << (sz / 1024) << " K";
        else if (sz < 1024 * 1024 * 1024)
            ss << (sz / 1024 / 1024) << " M";
        else
            ss << (sz / 1024 / 1024 / 1024) << " G";
        ss << unit;
        txt = ss.str();
    }
    friend std::ostream& operator<<(std::ostream& os, const pretty_size& ps) {
        os << ps.txt;
        return os;
    }
};


double test_bw(double dur, int64_t size) {
    static TestBWjit jit;
    // allocate per-thread buffer
    auto tsc_second = get_tsc_ticks_per_second();

    uint8_t* data[128] = {0};
    int failed = 0;
#pragma omp parallel reduction(+ : failed)
    {
        int tid = omp_get_thread_num();
        data[tid] = reinterpret_cast<uint8_t*>(aligned_alloc(64, size));
        if (data[tid] == nullptr) {
            std::cout << "Error, aligned_alloc failed!" << std::endl;
            failed++;
        }
        // memset to 1 ensures physical pages are really allocated
        memset(data[tid], 1, size);
    }
    if (failed) {
        return 0;
    }

    // warm-up cache
    int64_t actual_reps[128];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        actual_reps[tid] = jit(data[tid], size, static_cast<uint64_t>(tsc_second / 10));
    }

    // start profile
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        actual_reps[tid] = jit(data[tid], size, static_cast<uint64_t>(dur * tsc_second));
    }
    auto t2 = std::chrono::steady_clock::now();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(data[tid]);
    }

    int64_t total_reps = 0;
#pragma omp parallel reduction(+ : total_reps)
    {
        int tid = omp_get_thread_num();
        total_reps += actual_reps[tid];
    }

    std::chrono::duration<double> dt = t2 - t1;
    auto bytes_per_sec = total_reps / dt.count() * size;

    return bytes_per_sec;
}

void test_all_bw(double duration) {
    auto test = [&](int64_t size) {
        auto OMP_NT = omp_thread_count();
        auto bw = test_bw(duration, size);
        std::cout << "(" << pretty_size(size) << "B buff " << pretty_size(bw/OMP_NT) << "B/s x " << OMP_NT << " threads) = " << pretty_size(bw) << "B/s" << std::endl;
    };

    test(15 * 1024);
    test(30 * 1024);

    for (int64_t KB = 1024; KB < 3072; KB += 256)
        test(KB * 1024); // L2

    test(13 * 1024 * 1024);        // 13MB L3
    test(56 * 1024 * 1024);        // 56MB L3
    test(128 * 1024 * 1024);       // 128MB L3 + DDR
    test(512 * 1024 * 1024);       // 512MB
    while(1)
        test(1024 * 1024 * 1024l);     // 1GB DDR
}

int main() {
    test_all_bw(1);
    return 0;
}