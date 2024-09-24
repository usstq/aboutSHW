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

template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& nd) {
    os << "std::array<T=" << typeid(nd[0]).name() << ", N=" << N << ">{";
    const char * sep = "";
    for (int i= 0; i < N; i++) {
        os << sep << nd[i];
        sep = ",";
    }
    os << "}";
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
        std::cout << "      CL_KERNEL_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";
        std::cout << "      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << k.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << "\n";
        //std::cout << "      CL_KERNEL_GLOBAL_WORK_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(device) << "\n";
        std::cout << "      CL_KERNEL_COMPILE_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device) << "\n";
        std::cout << "      CL_KERNEL_LOCAL_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << "\n";
        std::cout << "      CL_KERNEL_PRIVATE_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device) << "\n";
        
        
        std::cout << kernel_name << " [getSubGroupInfo] :" << "\n";
        std::cout << "      CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(device, local_work_size)  << "\n";
        std::cout << "      CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, local_work_size)  << "\n";
        std::cout << "      CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: " << sub_groups << " is " << k.getSubGroupInfo<CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT>(device, sub_groups)  << "\n";
        std::cout << "      CL_KERNEL_MAX_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";
        std::cout << "      CL_KERNEL_COMPILE_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_COMPILE_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";
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
            std::cout << "\tdevice[" << k << "] : " << dev.getInfo<CL_DEVICE_NAME>() << " " << dev.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "\t\tCL_DEVICE_MAX_COMPUTE_UNITS: " << dev.getInfo <CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "\t\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " << dev.getInfo <CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << dev.getInfo <CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
            std::cout << "\t\tCL_DEVICE_MAX_WORK_ITEM_SIZES: ";
            auto maxWorkItems = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            for (auto& sz : maxWorkItems) {
                std::cout << sz << " ";
            }
            std::cout << std::endl;
            try {
                std::array<size_t, 8> value={0};
                dev.getInfo(CL_DEVICE_SUB_GROUP_SIZES_INTEL, &value);
                std::cout << "\t\tCL_DEVICE_SUB_GROUP_SIZES_INTEL: " << value << std::endl;
            } catch (...) {
                std::cout << "\t\tCL_DEVICE_SUB_GROUP_SIZES_INTEL: " << "???" << std::endl;
            }

            std::cout << "\t\tCL_DEVICE_SVM_CAPABILITIES: ";
            auto svm_caps = dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>();
            if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER) std::cout << " COARSE_GRAIN_BUFFER";
            if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) std::cout << " FINE_GRAIN_BUFFER";
            if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) std::cout << " FINE_GRAIN_SYSTEM";
            if (svm_caps & CL_DEVICE_SVM_ATOMICS) std::cout << " ATOMICS";
            std::cout << std::endl;

            auto dev_exts = dev.getInfo < CL_DEVICE_EXTENSIONS>();
            std::cout << "\t\tCL_DEVICE_EXTENSIONS: " << dev_exts << std::endl;

            bool has_extension = true;
            for (auto& ext : exts) {
                if (dev_exts.find(ext) == std::string::npos) {
                    has_extension = false;
                    std::cout << "\t\t lacks extension : " << ext << std::endl;
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




//===============================================================
template<typename T>
std::shared_ptr<T> alloc_cache_aligned(int count, T default_value) {
    auto ret = std::shared_ptr<T>(
            reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))),
            [](void * p) { ::free(p); });
    
    for(int i = 0; i < count; i++) {
        ret.get()[i] = default_value;
    }
    return ret;
}

template<typename T>
std::shared_ptr<T> alloc_cache_aligned(int count) {
    auto ret = std::shared_ptr<T>(
            reinterpret_cast<T*>(aligned_alloc(64, count * sizeof(T))),
            [](void * p) { ::free(p); });
    return ret;
}

template <typename T>
struct tensor2D {
    int shape[2] = {0};
    int stride_bytes = 0;
    std::shared_ptr<T> data;
    cl::Buffer ocl_buffer;

    uint64_t size() {
        return shape[0] * stride_bytes;
    }

    tensor2D() = default;

    tensor2D(int rows, int cols, bool initialize = true) {
        shape[0] = rows;
        shape[1] = cols;
        stride_bytes = cols * sizeof(T);
        data = alloc_cache_aligned<T>(rows * cols);
        if (initialize) {
            auto* pdata = data.get();
            for(int i = 0; i < rows * cols; i++)
                pdata[i] = 0;
        }
    }

    cl::Buffer& to_gpu() {
        if (ocl_buffer() == NULL) {
            ocl_buffer = cl::Buffer(data.get(), data.get() + shape[0] * shape[1], false);
        }
        return ocl_buffer;
    }

    void to_cpu() {
        cl::copy(ocl_buffer, data.get(), data.get() + shape[0] * shape[1]);
    }

    tensor2D<T> clone() {
        // deep copy
        tensor2D<T> ret;
        ret.shape[0] = shape[0];
        ret.shape[1] = shape[1];
        ret.stride_bytes = shape[1] * sizeof(T);
        ret.data = alloc_cache_aligned<T>(shape[0] * shape[1]);
        memcpy(ret.data.get(), data.get(), shape[0] * shape[1] * sizeof(T));
        return ret;
    }

    T* ptr(int i0 = 0, int i1 = 0) const {
        return data.get() + i0 * shape[1] + i1;
    }

    T& operator()(int i0, int i1) const {
        return *ptr(i0, i1);
    }

    // flatten 1D indexing
    T& operator[](int i) const {
        auto i0 = i / shape[1];
        auto i1 = i % shape[1];
        return *ptr(i0, i1);
    }

    std::string repr(int precision = 0, int width = 5) {
        std::stringstream ss;
        auto* pdata = data.get();
        ss << " (" << shape[0] << "," << shape[1] << ")" << typeid(*pdata).name() << " [\n";
        if (precision >= 0)
            ss << std::setprecision(precision);
        for(int m = 0; m < shape[0]; m++, pdata+=shape[1]) {
            ss << "  ";
            for(int n = 0; n < shape[1]; n++) {
                if (width >= 0)
                    ss << std::fixed << std::setw(width);

                ss << pdata[n] << ",";
            }
            ss << "    \\\\ row " << m <<  "\n";
        }
        ss << "]";
        return ss.str();
    }
};
