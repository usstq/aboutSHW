#pragma once
//#define CL_HPP_ENABLE_EXCEPTIONS
//#undef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <map>
#include <sstream>
#include <iomanip>
#include <cassert>
#include "../include/misc.hpp"
#include <filesystem>

static void _flush_cache() {
    static char _flush_cache1[32 * 1024 * 1024];
    static char _flush_cache2[32 * 1024 * 1024] = { 0 };
    memcpy(_flush_cache1, _flush_cache2, sizeof(_flush_cache2));
    if (_flush_cache1[std::rand() % sizeof(_flush_cache1)] == 13)
        std::cout << "impossible" << std::endl;
}


#define ANSI_COLOR_INFO "\033[32m"
#define ANSI_COLOR_ERROR "\033[31m"
#define ANSI_COLOR_RESET "\033[0m"

std::ostream& operator<<(std::ostream& os, const cl::detail::size_t_array& sz3) {
    os << "size_t_array[" << sz3[0] << "," << sz3[1] << "," << sz3[2] << "]";
    return os;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const cl::vector<T>& vt) {
    os << "vector[";
    const char * sep = "";
    for(int i = 0; i < vt.size(); i++) {
        os << sep << vt[i];
        sep = ",";
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const cl::NDRange& nd) {
    os << "NDRange(";
    const char * sep = "";
    for (int i= 0; i < nd.dimensions(); i++) {
        os << sep << nd.get()[i];
        sep = ",";
    }
    os << ")";
    return os;
}

struct CLkernels {
    std::map<std::string, cl::Kernel> kernel_map;
    cl::Program Program;

    CLkernels(const char* kernel_source) : Program(kernel_source) {

        auto show_build_info = [&](const char * ansi_color) {
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto& pair : buildInfo)
                std::cerr << ansi_color << "[BUILD_LOG]:" << pair.second << ANSI_COLOR_RESET << std::endl;
        };
        try {
            Program.build("-cl-std=CL3.0");
        }
        catch (...) {
            show_build_info(ANSI_COLOR_ERROR);
            abort();
        }
        show_build_info(ANSI_COLOR_INFO);

        cl::vector<cl::Kernel> kernels;
        if (Program.createKernels(&kernels) != CL_SUCCESS) {
            std::cerr << ANSI_COLOR_ERROR << "createKernels failed" << ANSI_COLOR_RESET << std::endl;
            abort();
        }

        {
            std::string directoryPath = ".build";
            if (!std::filesystem::exists(directoryPath)) {
                if (std::filesystem::create_directory(directoryPath)) {
                    std::cout << "Directory [" << directoryPath << "] created successfully!\n";
                } else {
                    std::cout << "Failed to create directory [" << directoryPath << "] .\n";
                }
            } else {
                std::cout << "Directory [" << directoryPath << "] already exists.\n";
            }
            auto open_file = [](std::string file_name) {
                std::ofstream fw;
                fw.open(file_name, std::ios::out);
                if (!fw.is_open()) {
                    std::cout << "open [" << file_name << "] failed";
                    abort();
                }
                return fw;
            };

            {
                auto fw = open_file(directoryPath + "/" + "CL_PROGRAM_SOURCE.cl");
                fw << Program.getInfo<CL_PROGRAM_SOURCE>();
            }
            {
                auto bins = Program.getInfo<CL_PROGRAM_BINARIES>();
                for(int i = 0; i < bins.size(); i++) {
                    auto fw = open_file(directoryPath + "/bin_" + std::to_string(i));
                    fw.write(reinterpret_cast<const char*>(&bins[i][0]), bins[i].size());
                }
            }
            // Program.getInfo<CL_PROGRAM_KERNEL_NAMES>() 
            std::cout << ANSI_COLOR_INFO << "Program source & binaries dumped to folder [" << directoryPath << "]" << ANSI_COLOR_RESET << std::endl;
        }

        for (auto& k : kernels) {
            auto kname = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
            kernel_map[kname] = k;

        }
    }

    void show_info(const char* kernel_name, cl::NDRange local_work_size, size_t sub_groups) {
        auto device = cl::Device::getDefault();
        auto& k = kernel_map[kernel_name];

        std::cout << kernel_name << " [getWorkGroupInfo] :" << "\n";
        std::cout << "    CL_KERNEL_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";
        std::cout << "    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << k.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << "\n";
        //std::cout << "    CL_KERNEL_GLOBAL_WORK_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(device) << "\n";
        std::cout << "    CL_KERNEL_COMPILE_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device) << "\n";
        std::cout << "    CL_KERNEL_LOCAL_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << "\n";
        std::cout << "    CL_KERNEL_PRIVATE_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device) << "\n";
        
        
        std::cout << kernel_name << " [getSubGroupInfo] :" << "\n";
        std::cout << "    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(device, local_work_size)  << "\n";
        std::cout << "    CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, local_work_size)  << "\n";
        std::cout << "    CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: " << sub_groups << " is " << k.getSubGroupInfo<CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT>(device, sub_groups)  << "\n";
        std::cout << "    CL_KERNEL_MAX_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";
        std::cout << "    CL_KERNEL_COMPILE_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_COMPILE_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";

        auto nargs = k.getInfo<CL_KERNEL_NUM_ARGS>();
        std::cout << " args " << nargs << " :" << std::endl;
        for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
            std::cout << "\t" << arg_idx << " " << k.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_idx) << " " << k.getArgInfo<CL_KERNEL_ARG_NAME>(arg_idx) << std::endl;
        }
    }

    template<typename ...Args>
    cl::Event submit(const char* kernel_name, const cl::EnqueueArgs& enqargs, Args&&... args) {
        //std::cout << kernel_name << " -> " << kernel_map[kernel_name].get() << ":" << clRetainKernel(kernel_map[kernel_name].get()) << std::endl;
        cl::KernelFunctor < Args...> kfunc(kernel_map[kernel_name]);
        return kfunc(enqargs, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    cl::Event call(const char* kernel_name, const cl::EnqueueArgs& enqargs, Args&&... args) {
        auto ev = submit(kernel_name, enqargs, std::forward<Args>(args)...);
        ev.wait();
        return ev;
    }
};

//https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_device_attribute_query.html
#define CL_DEVICE_IP_VERSION_INTEL                0x4250
#define CL_DEVICE_ID_INTEL                        0x4251
#define CL_DEVICE_NUM_SLICES_INTEL                0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL  0x4253
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL     0x4254
#define CL_DEVICE_NUM_THREADS_PER_EU_INTEL        0x4255
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL      0x4256

struct dev_info {
    std::string name;
    size_t num_EUs;
    size_t freq_MHz;
    double Tcycles_ps;

    dev_info() {
        auto context = cl::Context::getDefault();
        auto dev = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        name = dev.getInfo<CL_DEVICE_NAME>();
        num_EUs = dev.getInfo <CL_DEVICE_MAX_COMPUTE_UNITS>();
        freq_MHz = dev.getInfo <CL_DEVICE_MAX_CLOCK_FREQUENCY>();
        Tcycles_ps = 1e-6 * num_EUs * freq_MHz;
    }
};

static cl::Platform select_default_platform(std::vector<std::string> exts = {}) {
    // Filter for a 2.0 or newer platform and set it as the default
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform plat;
    for (int i = 0; i < platforms.size(); i++) {
        auto& p = platforms[i];
        std::string platver = p.getInfo<CL_PLATFORM_VERSION>();

        std::vector<cl::Device> devs;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devs);

        std::cout << "platform[" << i << "] : " << p.getInfo<CL_PLATFORM_VERSION>() << "; " << p.getInfo<CL_PLATFORM_NAME>() << std::endl;
        int usable_devs = 0;
        for (int k = 0; k < devs.size(); k++) {
            auto& dev = devs[k];
            std::cout << "  device[" << k << "] : " << dev.getInfo<CL_DEVICE_NAME>() << " " << dev.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_COMPUTE_UNITS: " << dev.getInfo <CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_CLOCK_FREQUENCY: " << dev.getInfo <CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "(MHz)" << std::endl;

            std::cout << "     max total EU-cycles/second: " << 1e-6 * dev.getInfo <CL_DEVICE_MAX_COMPUTE_UNITS>() * dev.getInfo <CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "(T-cycles/sec)" <<  std::endl;

            cl_uint v;
#define COUT_CL_INFO(qname) dev.getInfo(qname, &v); std::cout << "    " << #qname <<  ": " << v << std::endl;
            COUT_CL_INFO(CL_DEVICE_ID_INTEL);
            COUT_CL_INFO(CL_DEVICE_NUM_SLICES_INTEL);
            COUT_CL_INFO(CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL);
            COUT_CL_INFO(CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL);
            COUT_CL_INFO(CL_DEVICE_NUM_THREADS_PER_EU_INTEL);
            //COUT_CL_INFO(CL_DEVICE_FEATURE_CAPABILITIES_INTEL);

            std::cout << "    CL_DEVICE_MAX_WORK_GROUP_SIZE: " << dev.getInfo <CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << dev.getInfo <CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_WORK_ITEM_SIZES: ";
            auto maxWorkItems = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            for (auto& sz : maxWorkItems) {
                std::cout << sz << " ";
            }
            std::cout << std::endl;
            try {
                std::array<size_t, 8> value={0};
                dev.getInfo(CL_DEVICE_SUB_GROUP_SIZES_INTEL, &value);
                std::cout << "    CL_DEVICE_SUB_GROUP_SIZES_INTEL: " << value << std::endl;
            } catch (...) {
                std::cout << "    CL_DEVICE_SUB_GROUP_SIZES_INTEL: " << "???" << std::endl;
            }

            std::cout << "    CL_DEVICE_SVM_CAPABILITIES: ";
            auto svm_caps = dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>();
            if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) std::cout << " COARSE_GRAIN_BUFFER";
            if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) std::cout << " FINE_GRAIN_BUFFER";
            if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) std::cout << " FINE_GRAIN_SYSTEM";
            if (svm_caps & CL_DEVICE_SVM_ATOMICS) std::cout << " ATOMICS";
            std::cout << std::endl;

            auto dev_exts = dev.getInfo < CL_DEVICE_EXTENSIONS>();
            std::cout << "    CL_DEVICE_EXTENSIONS: " << dev_exts << std::endl;

            bool has_extension = true;
            for (auto& ext : exts) {
                if (dev_exts.find(ext) == std::string::npos) {
                    has_extension = false;
                    std::cout << "     lacks extension : " << ext << std::endl;
                    break;
                }
            }

            if (has_extension)
                usable_devs++;
        }

        if ((platver.find("OpenCL 2.") != std::string::npos ||
            platver.find("OpenCL 3.") != std::string::npos) && (usable_devs > 0)) {
            // Note: an OpenCL 3.x platform may not support all required features!
            plat = p;
        }
    }

    if (plat() == 0) {
        std::cout << "No OpenCL 2.0 or newer platform found.\n";
    }
    cl::Platform newP = cl::Platform::setDefault(plat);
    if (newP != plat) {
        std::cout << "Error setting default platform.\n";
    }
    std::cout << "platform selected: " << plat.getInfo<CL_PLATFORM_VERSION>() << "; " << plat.getInfo<CL_PLATFORM_NAME>() << std::endl;
    return newP;
}

struct workitem_info {
    uint32_t group_id0;
    uint32_t group_id1;
    uint32_t local_id0;
    uint32_t local_id1;
    uint32_t sub_group_id;
    uint32_t sub_group_local_id;
    uint32_t slice_id;
    uint32_t sub_slice_id;
    uint32_t eu_id;
    uint32_t eu_slot_id;
    uint64_t cycle_start;
    uint64_t cycle_dur;
    static void Dump(tensorND<workitem_info>& winfo, size_t latency_ns, size_t num_ops_per_workitem) {
        ChromeTraceDumpper dumpper("ocl.json");

        struct EUWork : workitem_info {
            int thr_cnt = 0;
            double num_OPs = 0;
            void dump(ChromeTraceDumpper & dumpper) {
                if (thr_cnt <= 0) return;
                if (slice_id != 0 || sub_slice_id > 1) return;
                std::stringstream ss;
                std::stringstream ss_cat;
                std::stringstream ss_pid;
                std::stringstream ss_tid;
                ss << "kernel(" << group_id0 << "," << group_id1 << ")";
                ss_pid << "slice.subslice:" << slice_id << "." << sub_slice_id;
                //ss_tid << "(" << local_id0 << "+" << thr_cnt << "," << local_id1 << ") EU" << eu_id;
                //ss_tid << "(" << group_id0 << "," << group_id1 << ")." << sub_group_id;
                ss_tid << "EU_" << eu_id << "." << eu_slot_id;
                //ss << "(" << local_id0 << "+" << thr_cnt << "," << local_id1 << ") sub-group:" << sub_group_id << "." << sub_group_local_id;
                dumpper.phb(ss.str(), ss_cat.str(), ss_pid.str(), ss_tid.str(), cycle_start, cycle_dur,
                {
                    {"local_id0",std::to_string(local_id0) + "+" + std::to_string(thr_cnt)},
                    {"local_id1",std::to_string(local_id1)},
                    {"OPS/cycle", std::to_string(num_OPs/cycle_dur)}
                });
            }
        };
        EUWork euwork;
        euwork.group_id0 = std::numeric_limits<uint32_t>::max();
        size_t total_thread_cnt = 0;

        {
            // collect & show some statistics
            struct SubSliceStat {
                uint64_t slice_id;
                uint64_t sub_slice_id;
                uint64_t thread_cnt = 0;
                uint64_t cycles_min = std::numeric_limits<uint64_t>::max();
                uint64_t cycles_max = std::numeric_limits<uint64_t>::min();
                double num_ops = 0;
            };
            std::vector<std::vector<SubSliceStat>> subslice_state;
            for(auto& w : winfo) {
                if (subslice_state.size() < w.slice_id + 1) {
                    subslice_state.resize(w.slice_id + 1);
                }
                auto& vs = subslice_state[w.slice_id];
                if (vs.size() < w.sub_slice_id + 1) {
                    vs.resize(w.sub_slice_id + 1);
                }
                auto& st = vs[w.sub_slice_id];

                st.slice_id = w.slice_id;
                st.sub_slice_id = w.sub_slice_id;
                st.thread_cnt ++;
                total_thread_cnt++;
                st.cycles_min = std::min(st.cycles_min, w.cycle_start);
                st.cycles_max = std::max(st.cycles_max, w.cycle_start + w.cycle_dur);
                st.num_ops += num_ops_per_workitem;
            }
            for(auto& vs : subslice_state) {
                for (auto& st : vs) {
                    if (st.thread_cnt == 0) continue;
                    std::cout << "subslice [" << st.slice_id << "." << st.sub_slice_id << "]:   "
                            << " cycles_min: " << st.cycles_min
                            << " cycles_dur: " << st.cycles_max - st.cycles_min
                            << " thread_cnt: " << st.thread_cnt
                            << " avg_OPS/cycle: " << st.num_ops/(st.cycles_max - st.cycles_min)
                            << std::endl;

                    std::stringstream ss;
                    std::stringstream ss_pid;
                    ss << "subslice[" << st.slice_id << "." << st.sub_slice_id << "]";
                    ss_pid << "slice.subslice:" << st.slice_id << "." << st.sub_slice_id;
                    dumpper.phX(ss.str(), "", ss_pid.str(), ss_pid.str(), st.cycles_min, st.cycles_max - st.cycles_min,
                    {
                        {"thread_cnt",std::to_string(st.thread_cnt)},
                        {"OPS",std::to_string(st.num_ops)},
                        {"OPS/per_thread",std::to_string(double(st.num_ops)/st.thread_cnt)},
                        {"avg_OPS/cycle", std::to_string(st.num_ops/(st.cycles_max - st.cycles_min))}
                    });
                }
            }
        }

        uint64_t min_cycle_start = std::numeric_limits<uint64_t>::max();
        uint64_t max_cycle_end = std::numeric_limits<uint64_t>::min();
        for(auto& w : winfo) {
            min_cycle_start = std::min(min_cycle_start, w.cycle_start);
            max_cycle_end = std::max(max_cycle_end, w.cycle_start + w.cycle_dur);
        }
        
        dev_info di;

        ECOUT(" total_thread_cnt : ", total_thread_cnt);
        ECOUT(" thread_per_EU : ", total_thread_cnt/8/di.num_EUs, " (assuming SIMD-8)");
        auto avg_freq_GHz = double(max_cycle_end - min_cycle_start)/latency_ns;
        auto normal_freq_GHz = di.freq_MHz*1e-3;
        ECOUT(" GPU_avg_freq : ", avg_freq_GHz, " (GHz)");
        ECOUT("              : ", avg_freq_GHz*100/(normal_freq_GHz), "% of ", normal_freq_GHz, "(GHz)");
        auto avg_gops = double(num_ops_per_workitem) * winfo.numel() /latency_ns;
        ECOUT(" GFLOPS/s     : ", avg_gops);
        ECOUT("              : ", avg_gops*100/(di.Tcycles_ps*1e3*8), "% of ", di.Tcycles_ps*8, "(Tcycles/s)");

        for(auto& w : winfo) {
            //std::cout << "(" << m << "," << n << ")  group(" << pw->group_id0 << "," << pw->group_id1 << ")(" << pw->local_id0 << "," << pw->local_id1 << ")(" << pw->sub_group_id << "," << pw->sub_group_local_id
            //          << ")  slice:" << pw->slice_id << "." << pw->sub_slice_id << " EU:" << pw->eu_id << "." << pw->eu_slot_id
            //          << " clock:" << pw->cycle_start << "+" << pw->cycle_dur
            //          << std::endl;
            //w.cycle_start -= min_cycle_start;
            if (euwork.group_id0 == w.group_id0 &&
                euwork.group_id1 == w.group_id1 &&
                euwork.sub_group_id == w.sub_group_id) {
                ASSERT(euwork.slice_id == w.slice_id);
                ASSERT(euwork.sub_slice_id == w.sub_slice_id);
                ASSERT(euwork.eu_id == w.eu_id);
                ASSERT(euwork.cycle_start == w.cycle_start);
                ASSERT(euwork.cycle_dur == w.cycle_dur);
                euwork.thr_cnt ++;
                euwork.num_OPs += num_ops_per_workitem;
            } else {
                euwork.dump(dumpper);
                reinterpret_cast<workitem_info&>(euwork) = w;
                euwork.thr_cnt = 1;
                euwork.num_OPs = num_ops_per_workitem;
            }
        }
        euwork.dump(dumpper);        
    }
};
