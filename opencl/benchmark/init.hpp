#pragma once

#ifndef ASSERT
#    define ASSERT(cond)                                                     \
        if (!(cond)) {                                                       \
            std::stringstream ss;                                            \
            ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
            throw std::runtime_error(ss.str());                              \
        }
#endif

//=======================================================================================================
// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_device_attribute_query.html
#define CL_DEVICE_IP_VERSION_INTEL               0x4250
#define CL_DEVICE_ID_INTEL                       0x4251
#define CL_DEVICE_NUM_SLICES_INTEL               0x4252
#define CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL 0x4253
#define CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL    0x4254
#define CL_DEVICE_NUM_THREADS_PER_EU_INTEL       0x4255
#define CL_DEVICE_FEATURE_CAPABILITIES_INTEL     0x4256

#define ANSI_COLOR_INFO  "\033[32m"
#define ANSI_COLOR_ERROR "\033[31m"
#define ANSI_COLOR_RESET "\033[0m"

inline std::ostream& operator<<(std::ostream& os, const cl::detail::size_t_array& sz3) {
    os << "size_t_array[" << sz3[0] << "," << sz3[1] << "," << sz3[2] << "]";
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const cl::vector<T>& vt) {
    os << "vector[";
    const char* sep = "";
    for (int i = 0; i < vt.size(); i++) {
        os << sep << vt[i];
        sep = ",";
    }
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const cl::NDRange& nd) {
    os << "NDRange(";
    const char* sep = "";
    for (int i = 0; i < nd.dimensions(); i++) {
        os << sep << nd.get()[i];
        sep = ",";
    }
    os << ")";
    return os;
}

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
        int selected = -1;
        for (int k = 0; k < devs.size(); k++) {
            auto& dev = devs[k];
            std::cout << "  device[" << k << "] : " << dev.getInfo<CL_DEVICE_NAME>() << " " << dev.getInfo<CL_DEVICE_VENDOR>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_COMPUTE_UNITS: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_CLOCK_FREQUENCY: " << dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "(MHz)" << std::endl;

            std::cout << "     max total EU-cycles/second: " << 1e-6 * dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "(T-cycles/sec)"
                      << std::endl;
            std::cout << "    CL_DEVICE_LOCAL_MEM_SIZE: " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;

            cl_uint v;
#define COUT_CL_INFO(qname) \
    dev.getInfo(qname, &v); \
    std::cout << "    " << #qname << ": " << v << std::endl;
            // COUT_CL_INFO(CL_DEVICE_ID_INTEL);
            // COUT_CL_INFO(CL_DEVICE_NUM_SLICES_INTEL);
            // COUT_CL_INFO(CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL);
            // COUT_CL_INFO(CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL);
            // COUT_CL_INFO(CL_DEVICE_NUM_THREADS_PER_EU_INTEL);
            // COUT_CL_INFO(CL_DEVICE_FEATURE_CAPABILITIES_INTEL);

            std::cout << "    CL_DEVICE_MAX_WORK_GROUP_SIZE: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
            std::cout << "    CL_DEVICE_MAX_WORK_ITEM_SIZES: ";
            auto maxWorkItems = dev.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            for (auto& sz : maxWorkItems) {
                std::cout << sz << " ";
            }
            std::cout << std::endl;
            try {
                std::array<size_t, 8> value = {0};
                dev.getInfo(CL_DEVICE_SUB_GROUP_SIZES_INTEL, &value);
                std::cout << "    CL_DEVICE_SUB_GROUP_SIZES_INTEL: " << value << std::endl;
            } catch (...) {
                std::cout << "    CL_DEVICE_SUB_GROUP_SIZES_INTEL: " << "???" << std::endl;
            }

            std::cout << "    CL_DEVICE_SVM_CAPABILITIES: ";
            auto svm_caps = dev.getInfo<CL_DEVICE_SVM_CAPABILITIES>();
            if (svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
                std::cout << " COARSE_GRAIN_BUFFER";
            if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
                std::cout << " FINE_GRAIN_BUFFER";
            if (svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
                std::cout << " FINE_GRAIN_SYSTEM";
            if (svm_caps & CL_DEVICE_SVM_ATOMICS)
                std::cout << " ATOMICS";
            std::cout << std::endl;

            auto dev_exts = dev.getInfo<CL_DEVICE_EXTENSIONS>();
            std::cout << "    CL_DEVICE_EXTENSIONS: " << dev_exts << std::endl;

            bool has_extension = true;
            for (auto& ext : exts) {
                if (dev_exts.find(ext) == std::string::npos) {
                    has_extension = false;
                    std::cout << "     lacks extension : " << ext << std::endl;
                    break;
                }
            }

            if (has_extension) {
                selected = k;
                usable_devs++;
            }
        }

        if ((platver.find("OpenCL 2.") != std::string::npos || platver.find("OpenCL 3.") != std::string::npos) && (usable_devs > 0)) {
            // Note: an OpenCL 3.x platform may not support all required features!
            plat = p;
            cl::Device::setDefault(devs[selected]);
            break;
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
    std::cout << "  device selected: " << cl::Device::getDefault().getInfo<CL_DEVICE_NAME>() << std::endl;
    return newP;
}

// https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_driver_diagnostics.txt
inline void CL_CALLBACK NotifyFunction(const char* pErrInfo, const void* pPrivateInfo, size_t size, void* pUserData) {
    if (pErrInfo != NULL) {
        std::cerr << ANSI_COLOR_ERROR << "[cl_intel_driver_diagnostics]:" << pErrInfo << ANSI_COLOR_RESET << std::endl;
    }
}
