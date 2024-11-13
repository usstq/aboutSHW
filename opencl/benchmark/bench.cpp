#include <cstdint>
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_TARGET_OPENCL_VERSION 300
#define NGEN_NO_OP_NAMES
#define NGEN_WINDOWS_COMPAT
#define NGEN_SAFE

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <array>
#include <fstream>

#include "ngen/ngen_opencl.hpp"
#include "ngen/ngen_register_allocator.hpp"
#include "opencl.hpp"
#include "../../include/misc.hpp"

#include "init.hpp"

using namespace ngen;

// g++ -o bench -std=c++17 bench.cpp -lOpenCL && ./bench

int loop_count = 10000, body_count = 100;
const int slm_size = 1 * 1024;
const int local = getenv("WG_SIZE", 8 * 16);    // one xecore
const int global = local * getenv("WG_NUM", 3200);

enum INST_TYPE {
    FLOAT_MOV,
    FLOAT_ADD,
    FLOAT_MUL,
    FLOAT_MAD,
    HALF_DPAS8x1,
    HALF_DPAS8x8,

    MEM_L1,
    MEM_SLM,
};
static const char* INST_TYPE_NAMES[] = {
    "FLOAT MOV",
    "FLOAT ADD",
    "FLOAT MUL",
    "FLOAT MAD",
    "HALF DPAS8x1",
    "HALF DPAS8x8",
    "MEM_L1",
    "MEM_SLM"
};

template <HW hw>
class JitKernelGenerator : public OpenCLCodeGenerator<hw>
{
protected:
    NGEN_FORWARD_OPENCL(hw);

public:
    RegisterAllocator _reg_alloc;

    // get cycle in low, high pair
    void get_cycle(const Subregister& low, const Subregister& high) {
        mov(1, low, tm0.ud(0));
        mov(1, high, tm0.ud(1));
    }

    // compute 64bit sub: (c_low, c_high) = (a_low, a_high) - (b_low, b_high)
    void subq(Subregister c_low, Subregister c_high, const Subregister& a_low, const Subregister& a_high, const Subregister& b_low, const Subregister& b_high) {
        auto low = _reg_alloc.alloc_sub<uint32_t>();
        auto low_acc =_reg_alloc.alloc_sub<uint32_t>();
        subb(1 | AccWrEn, low, a_low, b_low);
        mov(1, low_acc, acc0.ud(0));
        mov(1, c_low, low);
        add3(1, c_high, -low_acc, a_high, -b_high);
        _reg_alloc.release(low);
        _reg_alloc.release(low_acc);
    }

    JitKernelGenerator(int loop_count, int payload_counts, int slm_size, INST_TYPE i_type) : OpenCLCodeGenerator<hw>(), _reg_alloc(hw) {
        // Define kernel interface for OpenCL.
        newArgument("buffer", ExternalArgumentType::GlobalPtr);
        newArgument("alpha", DataType::f);
        requireLocalID(1);
        requireLocalSize();
        requireSLM(slm_size);
        if (i_type == HALF_DPAS8x8 || i_type == HALF_DPAS8x1)
            requireDPAS();
        requireSIMD((GRF::bytes(hw) == 64) ? 16 : 8);
        externalName(std::string("bench_") + INST_TYPE_NAMES[int(i_type)]);

        finalizeInterface();

        auto bufferSurface = Surface(getArgumentSurfaceIfExists("buffer"));     // Surface # for buffer.
        auto bufferPtr = getArgument("buffer");                                 // A64 pointer for buffer.
        auto alpha = getArgument("alpha");

        auto localSize = getLocalSize(0).uw();
        auto localID = getLocalID(0);               // Vector of local IDs.
        auto groupID = r0.ud(1);                    // Thread group (a.k.a. workgroup) IDs are in r0.ud(1) (X) r0.ud(6) (Y) r0.ud(7) (Z)

        // Local variables.
        auto globalID = r11.ud(0);
        auto header = r12;
        auto data = r13;
        auto temp = r14;

        _reg_alloc.claim(r0-r14);
        // Decide on load/store messages.
        bool useLSC = (hw >= HW::XeHPC);

        // All instructions use W (NoMask) by default.
        setDefaultNoMask();

        // Enable automatic SWSB for Gen12.
        setDefaultAutoSWSB();

        // Prologue for ATS+.
        prologue();

        // Enable IEEE denormals.
        or_(1 | Switch, cr0[0], cr0[0], 0x4C0);

        // Calculate global ID = (group ID) * (local size) + (local ID for lane 0).
        Label loop;
        auto loop_idx = _reg_alloc.alloc_sub<int32_t>();
        mov<uint32_t>(1, loop_idx, uint32_t(loop_count));
        auto beg_low = _reg_alloc.allocSub<uint32_t>();
        auto beg_high = _reg_alloc.allocSub<uint32_t>();
        get_cycle(beg_low, beg_high);

        switch (i_type) {
            case MEM_SLM:
                shl<uint32_t>(1, header, localID[0], 3);
                break;
            case MEM_L1:
                shl<uint32_t>(1, header, localID[0], 3);
                addc<uint32_t>(1, header, header, bufferPtr.ud(0));
                mov(1, temp.ud(0), acc0.ud(0));
                add(1, header.ud(1), bufferPtr.ud(1), temp.ud(0));
                break;
            default:
                break;
        }
        mark(loop);
        mul(1, globalID, groupID, localSize);
        add(1, globalID, globalID, localID[0]);
        if ((int)i_type >= MEM_L1) {
            // auto header = state.ra.alloc().ud();
            // mov<uint32_t>(1, header, slmOffset);
            // load(1 | ~leaderFlag, value, D32,                           SLM, header)
            // store(1 | leaderFlag, D32,                           SLM, header, value)
            auto d_regs = _reg_alloc.allocRange(payload_counts);
            for (int i = 0; i < payload_counts; i += 2) {
                switch (i_type) {
                    case MEM_SLM:
                        load(1, d_regs[i], D32 | V16T, SLM, header);
                        break;
                    case MEM_L1:
                        load(1, d_regs[i], D32 | V16T, A64, header);
                        break;
                    default: break;
                }
            }
        } else if ((int)i_type < (int)INST_TYPE::MEM_SLM) {
            if (i_type == INST_TYPE::HALF_DPAS8x1 || i_type == INST_TYPE::HALF_DPAS8x8) {
                // dpasw.8x8 (8|M0)         r46:f         null:f            r46:hf            r46.0:hf         {Atomic}
                // dpasw.8x8 (8|M0)         r46:f         r46:f             r29:hf            r8.0:hf          {Atomic}
                // dpasw.8x8 (8|M0)         r54:f         r54:f             r29:hf            r13.0:hf         {Atomic}
                // dpasw.8x8 (8|M0)         r62:f         r62:f             r29:hf            r18.0:hf         {Atomic}
                // dpasw.8x8 (8|M0)         r70:f         r70:f             r29:hf            r24.0:hf         {Atomic}
                // dpasw.8x8 (8|M0)         r78:f         r78:f             r38:hf            r8.0:hf          {Atomic}
                // dpasw.8x8 (8|M0)         r86:f         r86:f             r38:hf            r13.0:hf         {Atomic}
                // dpasw.8x8 (8|M0)         r94:f         r94:f             r38:hf            r18.0:hf         {Atomic}
                // dpasw.8x8 (8|M0)         r102:f        r102:f            r38:hf            r24.0:hf         {$5}
                // dpas.8x8 (8|M0)          r56:f         null:f            r56:hf            r56.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r56:f         r56:f             r16:hf            r48.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r64:f         r64:f             r16:hf            r40.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r72:f         r72:f             r16:hf            r32.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r80:f         r80:f             r16:hf            r24.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r88:f         r88:f             r8:hf             r48.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r96:f         r96:f             r8:hf             r40.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r104:f        r104:f            r8:hf             r32.0:hf         {Atomic}
                // dpas.8x8 (8|M0)          r112:f        r112:f            r8:hf             r24.0:hf         {$8}
                // dpasw(8 | Atomic, 8, 8, r46.f(), null, r46.hf(), r46.hf());
                // dpasw(8 | Atomic, 8, 8, r46.f(), r46.f(), r29.hf(), r8.hf());
                // dpasw(8 | Atomic, 8, 8, r54.f(), r54.f(), r29.hf(), r13.hf());
                // dpasw(8 | Atomic, 8, 8, r62.f(), r62.f(), r29.hf(), r18.hf());
                // dpasw(8 | Atomic, 8, 8, r70.f(), r70.f(), r29.hf(), r24.hf());
                // dpasw(8 | Atomic, 8, 8, r78.f(), r78.f(), r38.hf(), r8.hf());
                // dpasw(8 | Atomic, 8, 8, r86.f(), r86.f(), r38.hf(), r13.hf());
                // dpasw(8 | Atomic, 8, 8, r94.f(), r94.f(), r38.hf(), r18.hf());
                // dpasw(8, 8, 8, r102.f(), r102.f(), r38.hf(), r24.hf());

                auto c_regs = _reg_alloc.allocRange(8 * 8);
                auto a_regs = _reg_alloc.allocRange(8 * 4);
                auto b_regs = _reg_alloc.allocRange(8 * 2);
                // dpasw(8 | Atomic, 8, 8, c_regs[0].f(), c_regs[0].f(), b_regs[0].hf(), a_regs[0].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[8].f(), c_regs[8].f(), b_regs[0].hf(), a_regs[8].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[16].f(), c_regs[16].f(), b_regs[0].hf(), a_regs[16].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[24].f(), c_regs[24].f(), b_regs[0].hf(), a_regs[24].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[32].f(), c_regs[32].f(), b_regs[8].hf(), a_regs[0].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[40].f(), c_regs[40].f(), b_regs[8].hf(), a_regs[8].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[48].f(), c_regs[48].f(), b_regs[8].hf(), a_regs[16].hf());
                // dpasw(8 | Atomic, 8, 8, c_regs[56].f(), c_regs[56].f(), b_regs[8].hf(), a_regs[24].hf());
                auto r = i_type == INST_TYPE::HALF_DPAS8x1 ? 1 : 8;
                dpas(8, 8, r, c_regs[0].f(), c_regs[0].f(), a_regs[0].hf(), b_regs[0].hf());
                dpas(8, 8, r, c_regs[8].f(), c_regs[8].f(), a_regs[8].hf(), b_regs[0].hf());
                dpas(8, 8, r, c_regs[16].f(), c_regs[16].f(), a_regs[16].hf(), b_regs[0].hf());
                dpas(8, 8, r, c_regs[24].f(), c_regs[24].f(), a_regs[24].hf(), b_regs[0].hf());
                dpas(8, 8, r, c_regs[32].f(), c_regs[32].f(), a_regs[0].hf(), b_regs[8].hf());
                dpas(8, 8, r, c_regs[40].f(), c_regs[40].f(), a_regs[8].hf(), b_regs[8].hf());
                dpas(8, 8, r, c_regs[48].f(), c_regs[48].f(), a_regs[16].hf(), b_regs[8].hf());
                dpas(8, 8, r, c_regs[56].f(), c_regs[56].f(), a_regs[24].hf(), b_regs[8].hf());
                _reg_alloc.release(c_regs);
                _reg_alloc.release(a_regs);
                _reg_alloc.release(b_regs);
            } else {
                auto tmp_regs = _reg_alloc.allocRange(payload_counts);
                for (int i = 0; i < payload_counts; i++) {
                    if (i_type == INST_TYPE::FLOAT_MOV)
                        mov<float>(8, tmp_regs[i], float(-1));
                    else if (i_type == INST_TYPE::FLOAT_MUL)
                        mul<float>(8, tmp_regs[i], tmp_regs[i], float(-1));
                    else if (i_type == INST_TYPE::FLOAT_ADD)
                        add<float>(8, tmp_regs[i], tmp_regs[i], float(-1));
                    else if (i_type == INST_TYPE::FLOAT_MAD)
                        mad<float>(8, tmp_regs[i], tmp_regs[i], r13, r14);
                }
                _reg_alloc.release(tmp_regs);
            }
        }
        add<int32_t>(1 | gt | f0[0], loop_idx, loop_idx, int32_t(-1));
        jmpi(1 | f0[0], loop);
        auto end_low = _reg_alloc.allocSub<uint32_t>();
        auto end_high = _reg_alloc.allocSub<uint32_t>();
        get_cycle(end_low, end_high);
        auto result = _reg_alloc.alloc();
        subq(result.ud(0), result.ud(1), end_low, end_high, beg_low, beg_high);

        Label skip_write;
        // each workitem 0 of subgroup will write its timestamp(scalar) out.
        // it seems there should be bugs if only one item of a group wrote the result.(the first eu result is expected but got the second.) 
        mov<uint32_t>(1, temp.ud(0), localID[0]);
        // wi 0 of each subgroup
        and_<uint32_t>(1 | ze | f0[0], null, temp.ud(0), 7);
        jmpi(1 | ~f0[0], skip_write);
        // add offset: p = bufferPtr + globalID / 8 * 8
        //             global / 8 ==> got the sg index; globalID is the wi 0 of each sg
        //             global / 8 * 8 ==> each sg will write 8 bytes
        mov<uint32_t>(1, header, globalID);
        addc<uint32_t>(1, header, header, bufferPtr.ud(0));
        mov(1, temp.ud(0), acc0.ud(0));
        add(1, header.ud(1), bufferPtr.ud(1), temp.ud(0));
        store(1, D64, A64, header, result);
        mark(skip_write);

        // Scale data.
        // mul<float>(8, data, data, alpha);
        // Store updated data.
        // if (!useLSC)
        //     store(8, block_oword(2), bufferSurface, header, data);
        // else
        //     store(1, D32 | V8T, A64, header, data);

        // End thread. Must move r0 to one of r112-r127, then call threadend.
        mov<uint32_t>(8, r127, r0);
        threadend(r127);
    }
};

void runKernel(cl::Context& context, cl::CommandQueue queue, cl::Kernel kernel, INST_TYPE type) {
    cl_int status;

    int N = 128;
    std::vector<float> host_buffer(N);
    std::iota(host_buffer.begin(), host_buffer.end(), 1.0f);

    std::cout << std::fixed << std::setprecision(1);

    auto buffer_bytes = N * sizeof(float);
    cl::Buffer device_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, buffer_bytes, host_buffer.data());

    kernel.setArg(0, device_buffer);
    kernel.setArg(1, 0);

    const cl::NDRange offset_;
    cl::NDRange global_(global);
    cl::NDRange local_(local);
    std::vector<cl::Event> events_;
    std::vector<cl::Event> all_events(1);
    for (int i = 0; i < 10; i++) {
        queue.enqueueNDRangeKernel(kernel, offset_, global_, local_, &events_, &all_events[0]);
        queue.finish();

        for (auto& evt : all_events) {
            auto start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            auto end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            std::cout << "idx:" << i << ", cost: " << (end - start) * 1.0e-6 << "ms, ";
            cl::copy(queue, device_buffer, host_buffer.begin(), host_buffer.end());
            uint64_t* p = (uint64_t*)&host_buffer[0];
            if (type == MEM_SLM || type == MEM_L1) {
                double size = (double)loop_count * body_count * 4 * 8 * global / 8;
                std::cout << *p << " cycles, bytes/clk: " << ((double)loop_count * body_count * 4 * 8 / p[0]) << ", speed: " << size / (end - start) << " GB/s\n";
            } else {
                double computation = (double)loop_count * body_count * global / 8;
                static int flops_per_instr [] = {
                    1*8,        // FLOAT_MOV,
                    1*8,        // FLOAT_ADD,
                    1*8,        // FLOAT_MUL,
                    2*8,        // FLOAT_MAD,
                    1*16*8*2,   // HALF_DPAS8x1,
                    8*16*8*2,   // HALF_DPAS8x8,
                };
                computation *= flops_per_instr[(int)type];
                std::cout << *p << " cycles, CPI: " << ((p[0] * 1.0) / loop_count / body_count) << ", speed: " << computation / (end - start) << " GFLOPs/s\n";
            }
        }
    }
}

cl_kernel getKernel(cl::Context& ctx, cl::Device& dev, INST_TYPE i_type)
{
    auto context = ctx.get();
    auto device = dev.get();
    // Detect GPU hardware architecture.
    HW hw = JitKernelGenerator<HW::Unknown>::detectHW(context, device);
    const char *gpuString = "unknown";

    switch (hw) {
        case HW::Gen9:    gpuString = "Gen9"; break;
        case HW::Gen11:   gpuString = "Gen11"; break;
        case HW::Gen12LP: gpuString = "Gen12LP"; break;
        case HW::XeHP:    gpuString = "XeHP"; break;
        case HW::XeHPG:   gpuString = "XeHPG"; break;
        case HW::XeHPC:   gpuString = "XeHPC"; break;
        case HW::Xe2:     gpuString = "Xe2"; break;
        case HW::Xe3:     gpuString = "Xe3"; break;
#if XE3P
        case HW::Xe3p:    gpuString = "Xe3p"; break;
#endif
        default:          std::cerr << "Unknown GPU -- exiting.\n"; exit(2);
    }

    //std::cerr << "Found " << gpuString << " GPU.\n";

    // Create appropriate kernel generator object for the detected HW, and get a cl_kernel.
    switch (hw) {
        case HW::Gen9:    return JitKernelGenerator<HW::Gen9>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::Gen11:   return JitKernelGenerator<HW::Gen11>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::Gen12LP: return JitKernelGenerator<HW::Gen12LP>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::XeHP:    return JitKernelGenerator<HW::XeHP>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::XeHPG:   return JitKernelGenerator<HW::XeHPG>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::XeHPC:   return JitKernelGenerator<HW::XeHPC>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::Xe2:     return JitKernelGenerator<HW::Xe2>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
        case HW::Xe3:     return JitKernelGenerator<HW::Xe3>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
#if XE3P
        case HW::Xe3p:    return JitKernelGenerator<HW::Xe3p>(loop_count, body_count, slm_size, i_type).getKernel(context, device);
#endif
        default:          return nullptr;
    }
}

int main() {
    auto selected_platform = select_default_platform({"cl_intel_subgroups", "cl_intel_required_subgroup_size"});

    cl_context_properties properties[] = {
        CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL,
        (cl_context_properties)CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL | CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL | CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL,
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)selected_platform(),
        0};
    cl::Context context = cl::Context::getDefault(); // (CL_DEVICE_TYPE_GPU, &properties[0], NotifyFunction);
    cl::Device device = cl::Device::getDefault();
    auto cmd_queue = cl::CommandQueue(cl::QueueProperties::Profiling); // cl::QueueProperties::None

    std::cout << "\nloop_count = " << loop_count << "\n";
    std::cout << "body_count = " << body_count << "\n";
    std::cout << "slm_size   = " << slm_size / 1024 << "KB\n";
    std::cout << "local      = " << local << ", threads = " << local / 8 << ", threads/eu = " << std::max(1, local / 8 / 16) << "\n";
    std::cout << "global     = " << global << ", workgroup = " << global / local << "\n\n";
    for (int i = 0; i <= (int)MEM_SLM; i++) {
        int old_body_cout = -1;
        if (i == HALF_DPAS8x8 || i == HALF_DPAS8x1) {
            // dpas only unroll 8 times
            old_body_cout = body_count;
            body_count = 8;
        }
        cl::Kernel kernel(getKernel(context, device, (INST_TYPE)i));

        {
    #ifdef __linux__
            {
                auto open_file = [](std::string file_name) {
                    std::ofstream fw;
                    fw.open(file_name, std::ios::out);
                    if (!fw.is_open()) {
                        std::cout << "open [" << file_name << "] failed";
                        abort();
                    }
                    return fw;
                };
                auto exec = [](std::string cmd) {
                    std::array<char, 128> buffer;
                    std::string result;
                    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
                    if (!pipe) {
                        throw std::runtime_error("popen() failed!");
                    }
                    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
                        result += buffer.data();
                    }
                    return result;
                };
                auto program = kernel.getInfo<CL_KERNEL_PROGRAM>();  
                auto bins = program.getInfo<CL_PROGRAM_BINARIES>();
                for (int i = 0; i < bins.size(); i++) {
                    auto dump_bin_fpath = "./dev" + std::to_string(i) + ".bin";
                    auto fw = open_file(dump_bin_fpath);
                    fw.write(reinterpret_cast<const char*>(&bins[i][0]), bins[i].size());
                    fw.close();
                    exec(std::string("ocloc disasm -file ") + dump_bin_fpath + " -dump " + ".");
                }
            }
    #endif
        }
        std::cout << "Testing " << INST_TYPE_NAMES[i] << ":\n";
        runKernel(context, cmd_queue, kernel, (INST_TYPE)i);
        if (i == HALF_DPAS8x8 || i == HALF_DPAS8x1) {
            body_count = old_body_cout;
        }
    }
}