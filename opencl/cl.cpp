#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <map>
#include <vector>

#include "common.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#ifndef ASSERT
#    define ASSERT(cond)                                                     \
        if (!(cond)) {                                                       \
            std::stringstream ss;                                            \
            ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
            throw std::runtime_error(ss.str());                              \
        }
#endif

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
static auto CLDEBUG = getenv("CLDEBUG", 0);

#define DEBUG_MEMPOOL (CLDEBUG & 1)

// create temp cl_tensor (from pool since we want it to be fast)
struct cl_buffer_pool {
    std::multimap<size_t, cl::Buffer> pool;
    size_t total_size = 0;
    size_t total_count = 0;

    cl::Buffer alloc(size_t sz) {
        auto it = pool.find(sz);
        if (it != pool.end()) {
            pool.erase(it);
            if (DEBUG_MEMPOOL)
                std::cout << "[cl_buffer_pool] alloc from pool " << sz << " bytes, cl_mem: " << it->second.get() << std::endl;
            return it->second;
        }
        // allocate a new one
        total_size += sz;
        total_count++;
        cl::Buffer ret(CL_MEM_READ_WRITE, sz);
        if (DEBUG_MEMPOOL)
            std::cout << "[cl_buffer_pool] alloc new " << sz << " bytes, cl_mem: " << ret.get() << std::endl;
        return ret;
    }
    void free(cl::Buffer buff) {
        size_t sz = buff.getInfo<CL_MEM_SIZE>();
        pool.emplace(sz, buff);
        if (DEBUG_MEMPOOL)
            std::cout << "[cl_buffer_pool] free " << sz << " bytes, cl_mem: " << buff.get() << std::endl;
    }
    void show() {
        std::cout << "=== cl_buffer_pool ===" << std::endl;
        size_t pool_size = 0;
        for (auto const& p : pool) {
            std::cout << "\t" << p.first << " bytes, cl_mem: " << p.second.get() << std::endl;
            pool_size += p.first;
        }
        std::cout << "=== totally : " << pool.size() << "/" << total_count << " buffers, " << pool_size << "/" << total_size << " bytes ===" << std::endl;
    }

    ~cl_buffer_pool() {
        if (DEBUG_MEMPOOL)
            show();
    }
};

static cl_buffer_pool g_buff_pool;
static cl::CommandQueue cmd_queue;
static std::vector<cl::Event> all_events;

// composite of cl::Buffer & layout information
// like numpy array
struct cl_tensor {
    std::vector<cl_uint> shape;
    std::vector<cl_uint> strides;
    cl_uint numel;
    std::shared_ptr<cl::Buffer> p_buff;
    py::dtype dt;

    cl_tensor() = default;
    ~cl_tensor() = default;

    const std::vector<cl_uint>& get_shape() const {
        return shape;
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
    cl_tensor(const SizeContainer& dims, py::dtype dtype) {
        resize(dims, dtype);
    }

    cl_tensor(const py::array& arr) {
        resize(arr);
    }
    cl_tensor(const std::vector<size_t>& dims, py::dtype dtype) {
        resize(dims, dtype);
    }
    void update_buff() {
        auto* p = new cl::Buffer(g_buff_pool.alloc(numel * dt.itemsize()));
        p_buff = std::shared_ptr<cl::Buffer>(p, [](cl::Buffer* pbuff) {
            g_buff_pool.free(*pbuff);
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
        auto* p = reinterpret_cast<uint8_t*>(info.ptr);
        cl::copy(p, p + numel * dt.itemsize(), *p_buff);
    }

    py::array to_numpy() {
        // this shouldn't be a very frequent operation which requires optimizations
        // so we just allocate
        py::array ret(dt, shape);
        py::buffer_info info = ret.request();
        auto* p = reinterpret_cast<uint8_t*>(info.ptr);

        // make sure data is ready
        cmd_queue.finish();

        cl::copy(*p_buff, p, p + numel * dt.itemsize());
        return ret;
    }
};

//======================================================================================================

struct cl_kernels {
    std::map<std::string, cl::Kernel> kernel_map;
    cl::Program Program;

    cl_kernels(std::string source, std::string options, std::string dump_dir) : Program(source.c_str()) {
        auto throw_build_error = [&](const char* ansi_color) {
            std::stringstream ss;
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto& pair : buildInfo)
                ss << ansi_color << "[BUILD_LOG]:" << pair.second << ANSI_COLOR_RESET << std::endl;
            throw std::runtime_error(ss.str());
        };
        cl_int build_error = CL_SUCCESS;
        try {
            build_error = Program.build((options + " -cl-std=CL3.0").c_str());
        } catch (...) {
            throw_build_error(ANSI_COLOR_ERROR);
        }
        if (build_error != CL_SUCCESS)
            throw_build_error(ANSI_COLOR_ERROR);

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
            // Program.getInfo<CL_PROGRAM_KERNEL_NAMES>()
            std::cout << ANSI_COLOR_INFO << "Program source & binaries dumped to folder [" << directoryPath << "]" << ANSI_COLOR_RESET << std::endl;
        }

        for (auto& k : kernels) {
            auto kname = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
            kernel_map[kname] = k;
            auto nargs = k.getInfo<CL_KERNEL_NUM_ARGS>();
            std::cout << "[kernel] " << kname << " args " << nargs << " :" << std::endl;
            for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
                std::cout << "\t" << arg_idx << " " << k.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_idx) << " " << k.getArgInfo<CL_KERNEL_ARG_NAME>(arg_idx) << std::endl;
            }
        }
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
            } else if (py::isinstance<cl_tensor>(arg)) {
                const auto& t = arg.cast<cl_tensor>();
                kernel.setArg(arg_idx, *(t.p_buff));
            } else {
                throw std::runtime_error(std::string("Unknown kernel arg at index ") + std::to_string(arg_idx));
            }
            arg_idx++;
        }
        if (arg_idx < nargs)
            throw std::runtime_error(std::string("arg count ") + std::to_string(arg_idx) + " smaller than expected nargs=" + std::to_string(nargs));

        const cl::NDRange offset_;
        cl::NDRange global_;
        cl::NDRange local_;
        if (global_size.size() == 1)
            global_ = cl::NDRange(global_size[0]);
        if (global_size.size() == 2)
            global_ = cl::NDRange(global_size[0], global_size[1]);
        if (global_size.size() == 3)
            global_ = cl::NDRange(global_size[0], global_size[1], global_size[2]);

        if (local_size.size() == 1)
            local_ = cl::NDRange(local_size[0]);
        if (local_size.size() == 2)
            local_ = cl::NDRange(local_size[0], local_size[1]);
        if (local_size.size() == 3)
            local_ = cl::NDRange(local_size[0], local_size[1], local_size[2]);

        std::vector<cl::Event> events_;
        all_events.emplace_back();
        cmd_queue.enqueueNDRangeKernel(kernel, offset_, global_, local_, &events_, &all_events.back());
        return;
    }

    /*
        void show_info(std::string kernel_name, cl::NDRange local_work_size, size_t sub_groups) {
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
            std::cout << "    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(device,
       local_work_size)  << "\n"; std::cout << "    CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: " << local_work_size << " is " <<
       k.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, local_work_size)  << "\n"; std::cout << "    CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: " << sub_groups << "
       is " << k.getSubGroupInfo<CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT>(device, sub_groups)  << "\n"; std::cout << "    CL_KERNEL_MAX_NUM_SUB_GROUPS: " <<
       k.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, local_work_size)  << "\n"; std::cout << "    CL_KERNEL_COMPILE_NUM_SUB_GROUPS: " <<
       k.getSubGroupInfo<CL_KERNEL_COMPILE_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";
        }
    */
};

PYBIND11_MODULE(cl, m) {
    select_default_platform({"cl_intel_subgroups", "cl_intel_required_subgroup_size", "cl_intel_subgroup_matrix_multiply_accumulate"});

    // disable out-of-order execution
    // cl::CommandQueue::setDefault(cl::CommandQueue(cl::QueueProperties::None));
    cmd_queue = cl::CommandQueue(cl::QueueProperties::None);

    m.def("profiling", [](bool enable) {
        cmd_queue = cl::CommandQueue(enable ? cl::QueueProperties::Profiling : cl::QueueProperties::None);
    });

    py::class_<cl_tensor>(m, "tensor")
        .def(py::init<const py::array&>())
        .def(py::init<const std::vector<size_t>&, py::dtype>())
        .def("numpy", &cl_tensor::to_numpy)
        .def_property_readonly("shape", &cl_tensor::get_shape);

    py::class_<cl_kernels>(m, "kernels")
        .def(py::init<std::string, std::string, std::string>(), py::arg("source") = "", py::arg("options") = "", py::arg("dump_dir") = "")
        .def("enqueue", &cl_kernels::enqueue);

    m.def("flush", []() {
        cmd_queue.flush();
    });

    m.def("finish", []() {
        cmd_queue.finish();
        // return all event time-stamps
        std::vector<std::array<uint64_t, 5>> ret;
        for (auto& evt : all_events) {
            ret.emplace_back();
            ret.back()[0] = evt.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
            ret.back()[1] = evt.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
            ret.back()[2] = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            ret.back()[3] = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            ret.back()[4] = evt.getProfilingInfo<CL_PROFILING_COMMAND_COMPLETE>();
        }
        all_events.clear();
        return ret;
    });
}
