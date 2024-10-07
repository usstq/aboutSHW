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