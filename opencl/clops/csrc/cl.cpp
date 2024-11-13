#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include "opencl.hpp"


#include <sycl/CL/opencl.h>
//#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

//using namespace sycl::ext::intel::esimd;
//using namespace sycl::ext::intel;
using namespace sycl;


#ifdef __linux__
#    include <dlfcn.h>
#endif
#include <omp.h>

#ifndef ASSERT
#    define ASSERT(cond)                                                     \
        if (!(cond)) {                                                       \
            std::stringstream ss;                                            \
            ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
            throw std::runtime_error(ss.str());                              \
        }
#endif

//=======================================================================================================
#define ANSI_COLOR_INFO  "\033[32m"
#define ANSI_COLOR_ERROR "\033[31m"
#define ANSI_COLOR_RESET "\033[0m"

std::ostream& operator<<(std::ostream& os, const cl::detail::size_t_array& sz3) {
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

std::ostream& operator<<(std::ostream& os, const cl::NDRange& nd) {
    os << "NDRange(";
    const char* sep = "";
    for (int i = 0; i < nd.dimensions(); i++) {
        os << sep << nd.get()[i];
        sep = ",";
    }
    os << ")";
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& nd) {
    os << "std::array<T=" << typeid(nd[0]).name() << ", N=" << N << ">{";
    const char* sep = "";
    for (int i = 0; i < N; i++) {
        os << sep << nd[i];
        sep = ",";
    }
    os << "}";
    return os;
}

namespace py = pybind11;

// CPU side generates GPU tasks:
//  - shape-infer
//  - allocate buffer from pool
//  - choose suitable (pre-built) kernel and enqueue it with parameters
//  - free unused buffer back to pool（by using smart-pointer/life-scope)
//
// and CPU can do this w/o waiting for GPU to finish, so CPU is like a JIT compiler spills GPU tasks on the fly.
// so CPU logic is simple and we can use python for that purpose w/o worry about performance
//
// CPU can also break anytime on any GPU tensor by waiting queue to finish, and copy data back to CPU side numpy array to check results
//
// because CPU does shape-infer, we need combine shape-information with cl::Buffer to form a clTensor python object
// and this object:
//    - is created from pool
//    - with shape/strides information
//    - can fill with numpy array data with compatible shape
//    - can copy from GPU back to CPU side
//    - can be passed to a OpenCL kernel
//
// to keep things simpler, we don't use out-of-order queue so no event concept is used too.
// (just relying on out-side profiling tools to optimize)
//
static auto CLDEBUG = std::getenv("CLDEBUG") ? atoi(std::getenv("CLDEBUG")) : 0;

#define DEBUG_MEMPOOL_SUMMARY (CLDEBUG & 1)
#define DEBUG_MEMPOOL_VERBOSE (CLDEBUG & 2)

static cl::CommandQueue cmd_queue;
static cl::Context ocl_context;
sycl::queue sycl_queue;
static std::vector<cl::Event> all_events;

// create temp tensor (from pool since we want it to be fast)
struct buffer_pool {
    std::multimap<size_t, void*> pool;
    size_t total_size = 0;
    size_t total_count = 0;
    size_t total_alloc_size = 0;
    size_t total_alloc_count = 0;

    void* alloc(size_t sz) {
        total_size += sz;
        total_count++;

        auto it = pool.find(sz);
        if (it != pool.end()) {
            pool.erase(it);
            if (DEBUG_MEMPOOL_VERBOSE)
                std::cout << "[buffer_pool] alloc from pool " << sz << " bytes @ " << it->second << std::endl;
            return it->second;
        }
        // allocate a new one
        total_alloc_size += sz;
        total_alloc_count++;

        void * p = sycl::malloc_device(sz, sycl_queue);
        if (DEBUG_MEMPOOL_VERBOSE)
            std::cout << "[buffer_pool] alloc new " << sz << " bytes @ " << p << std::endl;
        return p;
    }

    void free(void* buff, size_t sz) {
        pool.emplace(sz, buff);
        if (DEBUG_MEMPOOL_VERBOSE)
            std::cout << "[buffer_pool] free " << sz << " bytes @ " << buff << std::endl;
    }

    void show() {
        std::cout << "=== buffer_pool ===" << std::endl;
        size_t pool_size = 0;
        std::map<size_t, int> summary;
        for (auto const& p : pool) {
            auto sz = p.first;
            if (summary.count(sz) == 0)
                summary[sz] = 0;
            summary[sz]++;
            pool_size += p.first;
        }
        for (auto const& p : summary) {
            std::cout << "\t " << p.first << " bytes x " << p.second << std::endl;
        }
        std::cout << "=== totally pool/total/actual:  " << pool.size() << "/" << total_count << "/" << total_alloc_count << " buffers"
                  << " pool/total/actual = " << pool_size / 1e6 << "/" << total_size / 1e6 << "/" << total_alloc_size / 1e6 << " MB  ===" << std::endl;
    }

    ~buffer_pool() {
        if (DEBUG_MEMPOOL_SUMMARY)
            show();
    }
};
static buffer_pool g_buff_pool;

// composite of cl::Buffer & layout information
// like numpy array
struct tensor {
    std::vector<cl_uint> shape;
    std::vector<cl_uint> strides;
    cl_uint numel = 0;
    std::shared_ptr<void> p_buff;
    py::dtype dt;

    tensor() {}

    ~tensor() = default;

    const std::vector<cl_uint>& get_shape() const {
        return shape;
    }
    cl_uint get_numel() const {
        return numel;
    }
    py::dtype get_dtype() const {
        return dt;
    }

    template<class T>
    operator T* () {
        return reinterpret_cast<T*>(p_buff.get());
    }

    template <class SizeContainer>
    void resize(const SizeContainer& dims, py::dtype dtype) {
        dt = dtype;
        auto it_dims = dims.begin();
        auto it_dims_end = dims.end();
        numel = 1;
        for (; it_dims != it_dims_end; ++it_dims) {
            auto dim = *it_dims;
            shape.push_back(dim);
            numel *= dim;
        }

        cl_uint stride = 1;
        strides.resize(shape.size());
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        update_buff();
    }

    template <class SizeContainer>
    tensor(const SizeContainer& dims, py::dtype dtype) {
        resize(dims, dtype);
    }

    tensor(const py::array& arr) {
        resize(arr);
    }
    tensor(const std::vector<size_t>& dims, py::dtype dtype) {
        resize(dims, dtype);
    }
    void update_buff() {
        size_t sz = numel * dt.itemsize();
        void* p = g_buff_pool.alloc(sz);
        p_buff = std::shared_ptr<void>(p, [sz](void* pbuff) {
            g_buff_pool.free(pbuff, sz);
        });
    }

    void resize(const py::array& b) {
        py::buffer_info info = b.request();
        dt = b.dtype();
        cl_uint expect_stride = 1;
        numel = 1;
        shape.resize(info.ndim);
        strides.resize(info.ndim);
        for (int i = info.ndim - 1; i >= 0; --i) {
            numel *= info.shape[i];
            shape[i] = info.shape[i];
            strides[i] = info.strides[i] / info.itemsize;
            ASSERT(strides[i] == expect_stride);
            expect_stride *= shape[i];
        }
        update_buff();
        auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);

        sycl_queue.submit([&](handler &h) {
            h.memcpy(p_buff.get(), p_host, numel * dt.itemsize());
        });
        sycl_queue.wait();
    }

    py::array to_numpy() {
        // this shouldn't be a very frequent operation which requires optimizations
        // so we just allocate
        py::array ret(dt, shape);
        py::buffer_info info = ret.request();
        auto* p_host = reinterpret_cast<uint8_t*>(info.ptr);

        // make sure data is ready
        sycl_queue.submit([&](handler &h) {
            h.memcpy(p_host, p_buff.get(), numel * dt.itemsize());
        });
        sycl_queue.wait();
        return ret;
    }
};

//======================================================================================================

struct cl_kernels {
    std::map<std::string, cl::Kernel> kernel_map;
    cl::Program Program;

    std::string m_source;
    std::string m_options;

    cl_kernels(std::string source, std::string options, std::string dump_dir) : m_source(source), m_options(options), Program(ocl_context, source.c_str()) {
        cl_int build_error = CL_SUCCESS;
        try {
            build_error = Program.build((options + " -cl-std=CL3.0").c_str());
        } catch (cl::BuildError err) {
            std::stringstream ss;
            for (auto& pair : err.getBuildLog()) {
                ss << "build failed on device: " << pair.first.getInfo<CL_DEVICE_NAME>() << std::endl;
                ss << "build options: " << options << std::endl;
                ss << ANSI_COLOR_ERROR << err.message << std::endl;
                ss << pair.second << ANSI_COLOR_RESET << std::endl;
            }
            throw std::runtime_error(ss.str());
        }

        cl::vector<cl::Kernel> kernels;
        if (Program.createKernels(&kernels) != CL_SUCCESS) {
            std::cerr << ANSI_COLOR_ERROR << "createKernels failed" << ANSI_COLOR_RESET << std::endl;
            abort();
        }

        if (dump_dir.size() > 0) {
            std::string kernel_names = "";
            for (auto& k : kernels) {
                kernel_names += k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                kernel_names += ".";
            }
            std::string directoryPath = dump_dir + "/" + kernel_names;
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
                auto fw = open_file(directoryPath + "/src.cl");
                fw << Program.getInfo<CL_PROGRAM_SOURCE>();
            }
#ifdef __linux__
            {
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
                auto bins = Program.getInfo<CL_PROGRAM_BINARIES>();
                for (int i = 0; i < bins.size(); i++) {
                    auto dump_bin_fpath = directoryPath + "/dev" + std::to_string(i) + ".bin";
                    auto fw = open_file(dump_bin_fpath);
                    fw.write(reinterpret_cast<const char*>(&bins[i][0]), bins[i].size());
                    fw.close();
                    exec(std::string("ocloc disasm -file ") + dump_bin_fpath + " -dump " + directoryPath);
                }
            }
#endif
            // Program.getInfo<CL_PROGRAM_KERNEL_NAMES>()
            std::cout << ANSI_COLOR_INFO << "Program source & binaries dumped to folder [" << directoryPath << "]" << ANSI_COLOR_RESET << std::endl;
        }

        for (auto& k : kernels) {
            auto kname = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
            kernel_map[kname] = k;
            auto nargs = k.getInfo<CL_KERNEL_NUM_ARGS>();
            std::cout << "[kernel] " << kname << "(";
            const char* sep = "";
            for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
                std::cout << sep << k.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_idx) << " " << k.getArgInfo<CL_KERNEL_ARG_NAME>(arg_idx);
                sep = ", ";
            }
            std::cout << ")" << std::endl;
        }
    }

    cl::NDRange to_ndrange(const std::vector<int>& global_size) {
        cl::NDRange global_;
        if (global_size.size() == 1)
            global_ = cl::NDRange(global_size[0]);
        if (global_size.size() == 2)
            global_ = cl::NDRange(global_size[0], global_size[1]);
        if (global_size.size() == 3)
            global_ = cl::NDRange(global_size[0], global_size[1], global_size[2]);
        return global_;
    }

    void enqueue(std::string kernel_name, const std::vector<int>& global_size, const std::vector<int>& local_size, py::args args) {
        auto& kernel = kernel_map[kernel_name];

        auto nargs = kernel.getInfo<CL_KERNEL_NUM_ARGS>();
        int arg_idx = 0;
        for (auto& arg : args) {
            if (arg_idx >= nargs)
                throw std::runtime_error(std::string("arg index ") + std::to_string(arg_idx) + " exceeds nargs=" + std::to_string(nargs));
            if (py::isinstance<py::int_>(arg)) {
                auto i = arg.cast<int>();
                kernel.setArg(arg_idx, i);
            } else if (py::isinstance<py::float_>(arg)) {
                auto f = arg.cast<float>();
                kernel.setArg(arg_idx, f);
            } else if (py::isinstance<tensor>(arg)) {
                const auto& t = arg.cast<tensor>();
                kernel.setArg(arg_idx, t.p_buff.get());
            } else if (arg.is_none()) {
                kernel.setArg(arg_idx, sizeof(void*), nullptr);
            } else {
                throw std::runtime_error(std::string("Unknown kernel arg at index ") + std::to_string(arg_idx));
            }
            arg_idx++;
        }
        if (arg_idx < nargs)
            throw std::runtime_error(std::string("arg count ") + std::to_string(arg_idx) + " smaller than expected nargs=" + std::to_string(nargs));

        const cl::NDRange offset_;
        cl::NDRange global_ = to_ndrange(global_size);
        cl::NDRange local_ = to_ndrange(local_size);
        std::vector<cl::Event> events_;
        all_events.emplace_back();
        cmd_queue.enqueueNDRangeKernel(kernel, offset_, global_, local_, &events_, &all_events.back());
        return;
    }

    std::string info(const char* kernel_name, const std::vector<int>& local_size, size_t sub_groups) {
        std::stringstream ss;
        cl::NDRange local_work_size = to_ndrange(local_size);
        auto device = cl::Device::getDefault();
        auto& k = kernel_map[kernel_name];

        ss << kernel_name << " [getWorkGroupInfo] :" << "\n";
        ss << "    CL_KERNEL_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";
        ss << "    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << k.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) << "\n";
        // std::cout << "    CL_KERNEL_GLOBAL_WORK_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_GLOBAL_WORK_SIZE>(device) << "\n";
        ss << "    CL_KERNEL_COMPILE_WORK_GROUP_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device) << "\n";
        ss << "    CL_KERNEL_LOCAL_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device) << "\n";
        ss << "    CL_KERNEL_PRIVATE_MEM_SIZE: " << k.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device) << "\n";
        ss << "    CL_KERNEL_SPILL_MEM_SIZE_INTEL: " << k.getWorkGroupInfo<CL_KERNEL_SPILL_MEM_SIZE_INTEL>(device) << "\n";

        ss << kernel_name << " [getSubGroupInfo] :" << "\n";
        ss << "    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(device, local_work_size)
           << "\n";
        ss << "    CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, local_work_size)
           << "\n";
        ss << "    CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: " << sub_groups << " is " << k.getSubGroupInfo<CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT>(device, sub_groups) << "\n";
        ss << "    CL_KERNEL_MAX_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, local_work_size) << "\n";
        ss << "    CL_KERNEL_COMPILE_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_COMPILE_NUM_SUB_GROUPS>(device, local_work_size) << "\n";

        auto nargs = k.getInfo<CL_KERNEL_NUM_ARGS>();
        ss << " args " << nargs << " :" << std::endl;
        for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
            ss << "\t" << arg_idx << " " << k.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_idx) << " " << k.getArgInfo<CL_KERNEL_ARG_NAME>(arg_idx) << std::endl;
        }
        return ss.str();
    }
};

#ifdef __linux__
//======================================================================================================================
// [gcc + omp] based CPP kernels
union KArg {
    int64_t i;
    float f;
    void* p;
};
typedef void (*KERNEL_FUNC)(const int, const int, const std::vector<KArg>&);

static int global_so_id = 0;

struct cpp_kernels {
    std::string so_fname;
    void* dl_handle = nullptr;
    ~cpp_kernels() {
        if (dl_handle) {
            dlclose(dl_handle);
        }
    }

    cpp_kernels(std::string src, std::string options, std::string name) {
        std::stringstream ss;
        std::string so_fname;

        so_fname = "./lib-ckjit-gen-";
        so_fname += name;
        so_fname += std::to_string(global_so_id);
        so_fname += ".so";

        ss << "gcc -shared -o " << so_fname << " -march=native -Wall -fpic -x c++ - -lstdc++ ";
        ss << options;
        FILE* pipe = popen(ss.str().c_str(), "w");
        if (pipe == NULL) {
            perror("popen Error");
            abort();
        }

        fwrite(src.c_str(), src.size(), 1, pipe);
        if (pclose(pipe)) {
            perror("pclose Error");
            abort();
        }

        dl_handle = dlopen(so_fname.c_str(), RTLD_LAZY);
        if (!dl_handle) {
            fprintf(stderr, "dlopen Error: %s\n", dlerror());
            abort();
        }
    }

    std::map<std::string, KERNEL_FUNC> kernels;

    void call(std::string name, py::args args) {
        KERNEL_FUNC func;
        auto it = kernels.find(name);
        if (it == kernels.end()) {
            func = reinterpret_cast<KERNEL_FUNC>(dlsym(dl_handle, name.c_str()));
            if (!func) {
                fprintf(stderr, "Error: %s\n", dlerror());
                abort();
            }
            kernels[name] = func;
        } else {
            func = it->second;
        }

        std::vector<KArg> kargs;

        int arg_id = 0;
        for (auto& arg : args) {
            kargs.emplace_back();
            auto& karg = kargs.back();
            if (py::isinstance<py::int_>(arg)) {
                karg.i = arg.cast<int64_t>();
            } else if (py::isinstance<py::float_>(arg)) {
                karg.f = arg.cast<float>();
            } else if (py::isinstance<py::array>(arg)) {
                const auto& b = arg.cast<py::array>();
                py::buffer_info info = b.request();
                karg.p = info.ptr;
            } else {
                throw std::runtime_error(std::string("Unknown kernel arg at index ") + std::to_string(kargs.size()));
            }
        }

        int nthr = omp_get_max_threads();
#    pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            func(ithr, nthr, kargs);
        }
    }
};
#else

struct cpp_kernels {
    ~cpp_kernels() {}
    cpp_kernels(std::string src, std::string options, std::string name) {}
    void call(std::string name, py::args args) {
        throw std::runtime_error(std::string("cpp_kernels only works on Linux"));
    }
};
#endif
static bool enable_profile = false;

static void update_queue() {
    //cmd_queue = cl::CommandQueue(enable ? cl::QueueProperties::Profiling : cl::QueueProperties::None);
    auto opencl_gpu_selector = [](const sycl::device& d) {
        if (d.is_gpu() && d.get_backend() == sycl::backend::opencl) {
            return 1;
        }
        return -1;
    };

    if (enable_profile) {
        sycl::property_list propList{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
        sycl_queue = sycl::queue(opencl_gpu_selector, propList);
    } else {
        sycl::property_list propList{sycl::property::queue::in_order()};
        sycl_queue = sycl::queue(opencl_gpu_selector, propList);
    }
    cmd_queue = sycl::get_native<sycl::backend::opencl>(sycl_queue);
    ocl_context = sycl::get_native<sycl::backend::opencl>(sycl_queue.get_context());
}

void test_esimd(float* Buf1, float* Buf2, int Size);

PYBIND11_MODULE(cl, m) {
    update_queue();

    m.def("profiling", [&](bool enable) {
        enable_profile = enable;
        update_queue();
    });

    m.def("test_esimd", [&](tensor& a, tensor& b){
        test_esimd(a, b, a.numel);
    });

    py::class_<tensor>(m, "tensor")
        .def(py::init<>())
        .def(py::init<const py::array&>())
        .def(py::init<const std::vector<size_t>&, py::dtype>())
        .def("numpy", &tensor::to_numpy)
        .def_property_readonly("shape", &tensor::get_shape)
        .def_property_readonly("numel", &tensor::get_numel)
        .def_property_readonly("dtype", &tensor::get_dtype)
        .def(py::pickle(
            // https://docs.python.org/3/library/pickle.html#pickling-class-instances
            [](tensor& p) {  // __getstate__
                return py::make_tuple(p.to_numpy());
            },
            [](py::tuple t) {  // __setstate__
                return tensor(t[0].cast<py::array>());
            }));

    py::class_<cl_kernels>(m, "kernels")
        .def(py::init<std::string, std::string, std::string>(), py::arg("source") = "", py::arg("options") = "", py::arg("dump_dir") = "")
        .def("enqueue", &cl_kernels::enqueue)
        .def("info", &cl_kernels::info)
        .def(py::pickle(
            [](cl_kernels& p) {  // __getstate__
                return py::make_tuple(p.m_source, p.m_options);
            }, 
            [](py::tuple t) {  // __setstate__
                return cl_kernels(t[0].cast<std::string>(), t[1].cast<std::string>(), {});
            }));

    py::class_<cpp_kernels>(m, "cpp_kernels")
        .def(py::init<std::string, std::string, std::string>(), py::arg("source") = "", py::arg("options") = "", py::arg("name") = "")
        .def("call", &cpp_kernels::call);

    m.def("flush", []() {
        cmd_queue.flush();
    });

    m.def("dev_info", []() {
        py::dict result;
        const auto& devices = ocl_context.getInfo<CL_CONTEXT_DEVICES>();
        const auto& device = devices[0];
        result["CL_DEVICE_NAME"] = device.getInfo<CL_DEVICE_NAME>();
        result["CL_DEVICE_EXTENSIONS"] = device.getInfo<CL_DEVICE_EXTENSIONS>();
        result["CL_DEVICE_MAX_COMPUTE_UNITS"] = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        result["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
        result["CL_DEVICE_LOCAL_MEM_SIZE"] = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
        return result;
    });

    m.def("finish", []() {
        cmd_queue.finish();
        // return all event time-stamps
        std::vector<uint64_t> ret;
        if (enable_profile) {
            for (auto& evt : all_events) {
                ret.emplace_back();
                auto start = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
                auto end = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
                ret.back() = end - start;
            }
        }
        all_events.clear();
        return ret;
    });
}
