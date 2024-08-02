
#include "../perf/linux_perf.hpp"
#include "bf16.hpp"
#include "jit.h"


std::vector<std::string> all_support_ktypes;

bool is_ktype(std::string cur_ktype, std::string target_ktype) {
    // collect all_support_ktypes 
    if(std::find(all_support_ktypes.begin(), all_support_ktypes.end(), target_ktype) == all_support_ktypes.end()) {
        all_support_ktypes.push_back(target_ktype);
    }
    return cur_ktype == target_ktype;
}


class InstructionLoop : public jit_generator {
public:
    std::shared_ptr<ov::bfloat16> tilebf16;
    std::shared_ptr<ov::bfloat16> tileAbf16;
    std::string ktype;
    int unroll_count;

    std::shared_ptr<ov::bfloat16> alloc(int count, float default_value) {
        auto ret = std::shared_ptr<ov::bfloat16>(
                reinterpret_cast<ov::bfloat16*>(aligned_alloc(64, count * sizeof(ov::bfloat16))),
                [](void * p) { ::free(p); });
        
        for(int i = 0; i < count; i++) {
            ret.get()[i] = default_value;
        }
        return ret;
    }

    InstructionLoop(std::string ktype, int unroll_count) : ktype(ktype), unroll_count(unroll_count) {
        tilebf16 = alloc(1024*8/sizeof(ov::bfloat16), 0.1f);
        tileAbf16 = alloc(32*(128*32)/sizeof(ov::bfloat16), 0.1f);
        create_kernel("InstructionLoop");
    }

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
        auto tile_stride = abi_param2;

        auto tile_A_stride = abi_param3;
        auto tile_A0_addr = abi_param4;
        auto tile_A1_addr = abi_param5;

        mov(rax, reinterpret_cast<uintptr_t>(tilebf16.get()));
        mov(tile_stride, 64);

        mov(tile_A0_addr, reinterpret_cast<uintptr_t>(tileAbf16.get()));
        mov(tile_A1_addr, reinterpret_cast<uintptr_t>(tileAbf16.get()) + 16*(128*32));
        mov(tile_A_stride, 128*32);
        if (ktype == "tileloadd.stride1") {
            mov(tile_A_stride, 63*64);
        }

        tilezero(tmm0);
        tilezero(tmm1);
        tilezero(tmm2);
        tilezero(tmm3);
        tilezero(tmm4);
        tilezero(tmm5);
        tilezero(tmm6);
        tilezero(tmm7);

        Xbyak::Label loop_begin;

        align(64, false);
        L(loop_begin);
        // unroll enough times to lower the impact of loop-overhead instructions
        // these loop-overhead instructions usually can increase CPI by a lot.
        for (int unroll = 0; unroll < unroll_count; unroll++) {
            if (is_ktype(ktype,"vaddps")) {
                // 0.5
                vaddps(zmm0, zmm31, zmm0);
                vaddps(zmm1, zmm31, zmm1);
                vaddps(zmm2, zmm31, zmm2);
                vaddps(zmm3, zmm31, zmm3);

                vaddps(zmm4, zmm31, zmm4);
                vaddps(zmm5, zmm31, zmm5);
                vaddps(zmm6, zmm31, zmm6);
                vaddps(zmm7, zmm31, zmm7);
            }
            if (is_ktype(ktype, "fma")) {
                // 0.5 FMA is powerful
                vfmadd132ps(zmm0, zmm31, zmm0);
                vfmadd132ps(zmm1, zmm31, zmm1);
                vfmadd132ps(zmm2, zmm31, zmm2);
                vfmadd132ps(zmm3, zmm31, zmm3);

                vfmadd132ps(zmm4, zmm31, zmm4);
                vfmadd132ps(zmm5, zmm31, zmm5);
                vfmadd132ps(zmm6, zmm31, zmm6);
                vfmadd132ps(zmm7, zmm31, zmm7);
            }
            if (is_ktype(ktype, "vdpbf16ps")) {
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
            }
            if (is_ktype(ktype, "tileloadd")) {
                // 8.3 : Throughput from `software-optimization-manual`: 8
                tileloadd(tmm0, ptr[rax + tile_stride + 0]);
                tileloadd(tmm1, ptr[rax + tile_stride + 1024]);
                tileloadd(tmm2, ptr[rax + tile_stride + 1024*2]);
                tileloadd(tmm3, ptr[rax + tile_stride + 1024*3]);
                tileloadd(tmm4, ptr[rax + tile_stride + 1024*4]);
                tileloadd(tmm5, ptr[rax + tile_stride + 1024*5]);
                tileloadd(tmm6, ptr[rax + tile_stride + 1024*6]);
                tileloadd(tmm7, ptr[rax + tile_stride + 1024*7]);
            }
            if (is_ktype(ktype, "tileloadd.compact")) {
                // 8.3 : Throughput from `software-optimization-manual`: 8
                tileloadd(tmm0, ptr[rax + tile_stride]);
                tileloadd(tmm1, ptr[rax + tile_stride + 1024]);
            }
            if (is_ktype(ktype, "tileloadd.stride0")) {
                // 8.3 : Throughput from `software-optimization-manual`: 8
                tileloadd(tmm0, ptr[tile_A0_addr + tile_A_stride]);
                tileloadd(tmm1, ptr[tile_A1_addr + tile_A_stride]);
            }
            if (is_ktype(ktype, "tileloadd.stride1")) {
                // 8.3 : Throughput from `software-optimization-manual`: 8
                tileloadd(tmm0, ptr[tile_A0_addr + tile_A_stride]);
                tileloadd(tmm1, ptr[tile_A1_addr + tile_A_stride]);
            }
            if (is_ktype(ktype, "tdpbf16ps")) {
                // CPK:64.2
                //    CPI:15.9 : Throughput from `software-optimization-manual`: 16
                tdpbf16ps(tmm0, tmm4, tmm6);
                tdpbf16ps(tmm1, tmm4, tmm7);
                tdpbf16ps(tmm2, tmm5, tmm6);
                tdpbf16ps(tmm3, tmm5, tmm7);
            }
            if (is_ktype(ktype, "amx2x2")) {
                // CPK:64.2
                //    CPI:load & TMUL can run perfectly in parallel (indicating multiple t)
                tileloadd(tmm4, ptr[tile_A0_addr + tile_A_stride]);    // A0
                tileloadd(tmm6, ptr[rax + tile_stride + 1024*2]); // B0
                tdpbf16ps(tmm0, tmm4, tmm6);

                tileloadd(tmm5, ptr[tile_A1_addr + tile_A_stride]); // A1
                tdpbf16ps(tmm2, tmm5, tmm6);
                
                tileloadd(tmm7, ptr[rax + tile_stride + 1024*3]); // B1
                tdpbf16ps(tmm1, tmm4, tmm7);
                tdpbf16ps(tmm3, tmm5, tmm7);
            }
        }
        dec(regCnt);
        jne(loop_begin, T_NEAR);
        ret();
    }
};

void test_CPI(std::string ktype) {
    int LOOP_COUNT=10000;
    static int index = 0;
    LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    });
    if (index == 0)
        pevg.show_header();
    index ++;

    TileConfig tile_cfg;
    auto rows0 = 16;
    auto rows1 = 16;
    tile_cfg.reset(1,
                   0,
                   {
                       {rows0, 64},  // C00:0
                       {rows0, 64},  // C01:1
                       {rows1, 64},  // C10:2
                       {rows1, 64},  // C11:3
                       {rows0, 64},  // A0:4
                       {rows1, 64},  // A1:5
                       {16, 64},     // B0:6
                       {16, 64},     // B1:7
                   });
    TileConfigScope tcfg(tile_cfg);

    #define UNROLL_COUNT 100

    InstructionLoop inst(ktype, UNROLL_COUNT);
    inst(LOOP_COUNT); // warm-up
    for (int i = 0; i < 1; i++) {
        auto pmc = pevg.rdpmc(
            [&]() {
                inst(LOOP_COUNT);
            },
            ktype,
            UNROLL_COUNT*LOOP_COUNT);
    }
}

int main() {
    bool initAMX = initXTILE();

    // probing all_support_ktypes
    test_CPI("");
    for(auto ktype : all_support_ktypes) {
        test_CPI(ktype);
    }
    return 0;
}