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

template <typename T, int TENSORND_MAXDIMS=8>
struct tensorND {
    size_t m_shape[TENSORND_MAXDIMS] = {0};
    size_t m_strides[TENSORND_MAXDIMS] = {0};
    size_t m_ndims = 0;
    size_t m_offset = 0;
    size_t m_numel = 0;
    std::shared_ptr<T> m_data;
    cl::Buffer ocl_buffer;

    size_t ndims() const { return m_ndims; }
    size_t numel() const  { return m_numel; }
    size_t size(int i) const {
        while (i < 0) i += m_ndims;
        while (i >= m_ndims) i -= m_ndims;
        return m_shape[i];
    }
    const size_t* shape() const { return m_shape; }
    const size_t* strides() const { return m_strides; }

    // coordinate
    struct coordinate {
        const tensorND<T, TENSORND_MAXDIMS>* m_tensor;
        int64_t m_value[TENSORND_MAXDIMS] = {0};
        size_t m_offset = 0;

        size_t offset() { return m_offset; }
        int64_t operator[](int i) {
            auto ndims = m_tensor->ndims();
            if (i < 0) i += ndims;
            return m_value[i];
        }

        coordinate(tensorND<T, TENSORND_MAXDIMS>* p_tensor, const std::vector<int>& idx = {}) : m_tensor(p_tensor) {
            for(int i = 0; i < TENSORND_MAXDIMS; i++) {
                if (i < idx.size())
                    m_value[i] = idx[i];
                else
                    m_value[i] = 0;
            }
        }
        coordinate& operator+=(int64_t carry) {
            auto* shape = m_tensor->shape();
            auto ndims = m_tensor->ndims();
            for (int r = ndims - 1; r >= 0; r--) {
                m_value[r] += carry;
                carry = 0;

                assert(m_value[r] >= 0);
                // get carry for next-higher digit
                while(m_value[r] >= shape[r]) {
                    m_value[r] -= shape[r];
                    carry ++;
                }

                if (carry == 0) break;
            }
            // update offset
            auto* strides = m_tensor->strides();
            m_offset = 0;
            for (int r = 0; r < ndims; r++) {
                m_offset += m_value[r] * strides[r];
            }
            return *this;
        }
    };

    coordinate get_coordinate() {
        return coordinate(this);
    }

    void resize(const std::vector<size_t>& dims) {
        assert(dims.size() <= TENSORND_MAXDIMS);
        m_ndims = dims.size();
        size_t s = 1;
        m_numel = 1;
        for(int64_t i = static_cast<int64_t>(m_ndims - 1); i >= 0; i--) {
            m_shape[i] = dims[i];
            m_strides[i] = s;
            s *= m_shape[i];
            m_numel *= m_shape[i];
        }
        m_data = alloc_cache_aligned<T>(m_numel);
    }

    void resize(const std::vector<size_t>& dims, T init) {
        resize(dims);
        auto* ptr = m_data.get();
        for(int i = 0; i < m_numel; i++) ptr[i] = init;
    }

    tensorND() = default;

    tensorND(const std::initializer_list<size_t>& dims) {
        resize(dims);
    }
    tensorND(const std::initializer_list<size_t>& dims, T init) {
        resize(dims, init);
    }

    cl::Buffer& to_gpu() {
        if (ocl_buffer() == NULL) {
            ocl_buffer = cl::Buffer(m_data.get(), m_data.get() + m_numel, false);
        }
        return ocl_buffer;
    }

    void to_cpu() {
        cl::copy(ocl_buffer, m_data.get(), m_data.get() + m_numel);
    }

    tensorND<T> clone() {
        // deep copy
        tensorND<T> ret;
        memcpy(ret.m_shape, m_shape, sizeof(m_shape));
        memcpy(ret.m_strides, m_strides, sizeof(m_strides));
        ret.m_ndims = m_ndims;
        ret.m_offset = m_offset;
        ret.m_numel = m_numel;
        ret.m_data = alloc_cache_aligned<T>(m_numel);
        memcpy(ret.m_data.get(), m_data.get(), m_numel * sizeof(T));
        return ret;
    }

    // reference element using index directly
    template <int dim>
    int64_t offset() const {
        return m_offset;
    }
    template <int dim, typename I>
    int64_t offset(I i) const {
        return m_offset + i * m_strides[dim];
    }
    template <int dim, typename I, typename... Is>
    int64_t offset(I i, Is... indices) const {
        return i * m_strides[dim] + offset<dim + 1>(indices...);
    }
    template <typename... Is>
    T* ptr(Is... indices) const {
        return reinterpret_cast<T*>(m_data.get()) + offset<0>(indices...);
    }
    /*
    template <typename... Is>
    void* ptr_v(Is... indices) const {
        return reinterpret_cast<void*>(m_data.get() + offset<0>(indices...) * m_element_size);
    }*/
    template <typename... Is>
    T& at(Is... indices) const {
#if 0
        std::cout << "at" << " ("; 
        int dummy[sizeof...(Is)] = {(std::cout << indices << ",", 0)...};
        std::cout << ") offset=" << offset<0>(indices...) << "" << std::endl;
#endif
        return *(m_data.get() + offset<0>(indices...));
    }
    //T& operator[](const coordinate<T, TENSORND_MAXDIMS>& coord) {
    //    return m_data.get()[coord.offset()];
    //}


    std::string repr(int precision = 0, int width = 5) {
        std::stringstream ss;
        auto* pdata = m_data.get();
        ss << " shape=(";
        const char * sep = "";
        for(int i = 0; i < m_ndims; i++) {
            ss << sep << m_shape[i];
            sep = ",";
        }
        ss << ") strides=(";
        sep = "";
        for(int i = 0; i < m_ndims; i++) {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << ") dtype=" << typeid(*pdata).name() << " [\n";
        if (precision >= 0) ss << std::setprecision(precision);

        auto coord = get_coordinate();

        for(int i = 0; i < m_numel; ) {
            if (m_ndims > 1) {
                ss << " [";
                sep = "";
                for(int n = 0; n < m_ndims - 1; n++) {
                    ss << sep << coord[n];
                    sep = ",";
                }
                ss << ", ...] : ";
            }
            for(int n = 0; n < m_shape[m_ndims - 1]; n++) {
                if (width >= 0)
                    ss << std::fixed << std::setw(width);
                ss << m_data.get()[coord.offset()];
                // std::cout << coord.offset() << "\n";
                coord += 1;
                i += 1;
            }
            ss << "\n";
        }
        ss << "]";
        return ss.str();
    }
};



inline int64_t getenv(const char * var, int64_t default_value) {
    const char * p = std::getenv(var);
    if (p) {
        char str_value[256];
        int len = 0;
        while(p[len] >= '0' && p[len] <= '9') {
            str_value[len] = p[len];
            len++;
        }
        str_value[len] = 0;

        char unit = p[len];
        int64_t unit_value = 1;
        // has unit?
        if (unit == 'K' || unit == 'k') unit_value = 1024;
        if (unit == 'M' || unit == 'm') unit_value = 1024*1024;
        if (unit == 'G' || unit == 'g') unit_value = 1024*1024*1024;

        default_value = std::atoi(str_value) * unit_value;
    }
    printf("\033[32mENV:\t %s = %lld %s\033[0m\n", var, default_value, p?"":"(default)");

    return default_value;
}
