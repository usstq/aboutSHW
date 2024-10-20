#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"

#include <iostream>
#include <vector>
#include <map>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#ifndef ASSERT
#define ASSERT(cond) if (!(cond)) {\
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
    throw std::runtime_error(ss.str()); \
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

// create temp cl_tensor (from pool since we want it to be fast)
struct cl_buffer_pool {
    std::multimap<size_t, cl::Buffer> pool;
    size_t total_size = 0;
    size_t total_count = 0;

    cl::Buffer alloc(size_t sz) {
        auto it = pool.find(sz);
        if (it != pool.end()) {
            pool.erase(it);
            if (CLDEBUG)
                std::cout << "[cl_buffer_pool] alloc from pool " << sz << " bytes, cl_mem: " << it->second.get() << std::endl;
            return it->second;
        }
        // allocate a new one
        total_size += sz;
        total_count ++;
        cl::Buffer ret (CL_MEM_READ_WRITE, sz);
        if (CLDEBUG)
            std::cout << "[cl_buffer_pool] alloc new " << sz << " bytes, cl_mem: " << ret.get() << std::endl;
        return ret;
    }
    void free(cl::Buffer buff) {
        size_t sz = buff.getInfo<CL_MEM_SIZE>();
        pool.emplace(sz, buff);
        if (CLDEBUG)
            std::cout << "[cl_buffer_pool] free " << sz << " bytes, cl_mem: " << buff.get() << std::endl;
    }
    void show() {
        std::cout << "=== cl_buffer_pool ===" << std::endl;
        size_t pool_size = 0;
        for (auto const& p : pool) {
            std::cout << "\t" << p.first << " bytes, cl_mem: " << p.second.get() << std::endl;
            pool_size += p.first;
        }
        std::cout << "=== totally : " << pool.size() << "/" << total_count << " buffers, "
                  << pool_size << "/" << total_size << " bytes ===" << std::endl;
    }

    ~cl_buffer_pool() {
        if (CLDEBUG)
            show();
    }
};

static cl_buffer_pool g_buff_pool;


// composite of cl::Buffer & layout information
// like numpy array
template<class T>
struct cl_tensor {
    std::vector<cl_uint> shape;
    std::vector<cl_uint> strides;
    cl_uint numel;
    std::shared_ptr<cl::Buffer> p_buff;

    cl_tensor() = default;
    ~cl_tensor() = default;

    const std::vector<cl_uint> & get_shape() const {
        return shape;
    }
    template<class SizeContainer>
    void resize(const SizeContainer& dims) {
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

    template<class SizeContainer>
    cl_tensor(const SizeContainer& dims) {
        resize(dims);
    }
    void update_buff() {
        auto* p = new cl::Buffer(g_buff_pool.alloc(numel * sizeof(T)));
        p_buff = std::shared_ptr<cl::Buffer>(p,
            [](cl::Buffer * pbuff) {
                g_buff_pool.free(*pbuff);
            }
        );
    }

    void resize(py::array b) {
        py::buffer_info info = b.request();
        ASSERT(sizeof(T) == info.itemsize);
        cl_uint expect_stride = 1; 
        numel = 1;
        shape.resize(info.ndim);
        strides.resize(info.ndim);
        for(int i = info.ndim-1; i >= 0 ; --i) {
            numel *= info.shape[i];
            shape[i] = info.shape[i];
            strides[i] = info.strides[i]/sizeof(T);
            ASSERT(strides[i] == expect_stride);
            expect_stride *= shape[i];
        }
        update_buff();
        auto* p = reinterpret_cast<T*>(info.ptr);
        cl::copy(p, p + numel, *p_buff);
    }

    py::array to_numpy() {
        // this shouldn't be a very frequent operation which requires optimizations
        // so we just allocate
        std::vector<ssize_t> nshape(shape.begin(), shape.end());
        py::array_t<T> ret(nshape);
        py::buffer_info info = ret.request();
        auto* p = reinterpret_cast<T*>(info.ptr);
        cl::copy(*p_buff, p, p + numel);
        return ret;
    }
};



#if 0
// create from numpy array 
cl::Buffer& from_array() {
    if (ocl_buffer() == NULL) {
        ocl_buffer = cl::Buffer(m_data.get(), m_data.get() + m_numel, false);
        m_on_gpu = false;
    }
    if (!m_on_gpu) {
        cl::copy(m_data.get(), m_data.get() + m_numel, ocl_buffer);
        m_on_gpu = true;
    }
    return ocl_buffer;
}

py::array * to_numpy() {
    // this shouldn't be a very frequent operation which requires optimizations
    // so we just allocate 
    auto* p_shr_ptr = ;
    py::capsule free_when_done(p_shr_ptr, [](void* ptr) {
        delete reinterpret_cast<std::shared_ptr<void>*>(ptr);
    });

    auto ndims = t.ndims();
    auto* p_shape = t.shape();
    auto* p_strides = t.strides();
    std::vector<ssize_t> shape(p_shape, p_shape + ndims);
    std::vector<ssize_t> strides(p_strides, p_strides + ndims);

    cl::copy(ocl_buffer, m_data.get(), m_data.get() + m_numel);
    return py::array(shape, strides, t.ptr(), free_when_done);
}

template <class T>
const char* py_format() {
    return "?";
}
template <>
const char* py_format<float>() {
    return "f";
}
template <>
const char* py_format<int32_t>() {
    return "i";
}

template <class T>
py::buffer_info to_py_buffer(tensorND<T>& t) {
    auto ndims = t.ndims();
    std::vector<ssize_t> shape(ndims);
    std::vector<ssize_t> strides_in_bytes(ndims);
    for (size_t i = 0; i < ndims; i++) {
        shape[i] = t.shape()[i];
        strides_in_bytes[i] = t.strides()[i] * t.item_size();
    }

    return py::buffer_info(t.ptr(),         /* Pointer to buffer */
                           t.item_size(),   /* Size of one scalar */
                           py_format<T>(),  /* Python struct-style format descriptor */
                           ndims,           /* Number of dimensions */
                           shape,           /* Buffer dimensions */
                           strides_in_bytes /* Strides (in bytes) for each index */
    );
}

template <class T>
tensorND<T> from_array(py::array b, bool copy = false) {
    py::buffer_info info = b.request();
    tensorND<T> ret;
    auto strides = info.strides;
    for (auto& v : strides)
        v /= sizeof(T);
    if (copy) {
        tensorND<T> src;
        src.resize(info.shape, strides, reinterpret_cast<T*>(info.ptr));
        ret.resize(info.shape);
        ret = src;
    } else {
        ret.resize(info.shape, strides, reinterpret_cast<T*>(info.ptr));
    }
    return ret;
}

template <class T>
py::array to_numpy(tensorND<T>& t) {
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    // Create a Python object that will free the allocated memory when destroyed:
    auto* p_shr_ptr = new std::shared_ptr<void>(t.m_data);
    py::capsule free_when_done(p_shr_ptr, [](void* ptr) {
        delete reinterpret_cast<std::shared_ptr<void>*>(ptr);
    });

    auto ndims = t.ndims();
    auto* p_shape = t.shape();
    auto* p_strides = t.strides();
    std::vector<ssize_t> shape(p_shape, p_shape + ndims);
    std::vector<ssize_t> strides(p_strides, p_strides + ndims);

    return py::array(shape, strides, t.ptr(), free_when_done);
}
#endif
//======================================================================================================

static cl::CommandQueue cmd_queue;

struct cl_kernels {
    std::map<std::string, cl::Kernel> kernel_map;
    cl::Program Program;

    cl_kernels(std::string source, std::string options) : Program(source.c_str()) {
        auto show_build_info = [&](const char * ansi_color) {
            cl_int buildErr = CL_SUCCESS;
            auto buildInfo = Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
            for (auto& pair : buildInfo)
                std::cerr << ansi_color << "[BUILD_LOG]:" << pair.second << ANSI_COLOR_RESET << std::endl;
        };
        try {
            Program.build((options + " -cl-std=CL3.0").c_str());
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
            auto nargs = k.getInfo<CL_KERNEL_NUM_ARGS>();
            std::cout << "[kernel] " << kname << " args " << nargs << " :" << std::endl;
            for (int arg_idx = 0; arg_idx < nargs; arg_idx++) {
                std::cout << "\t" << arg_idx << " " << k.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_idx) << " " << k.getArgInfo<CL_KERNEL_ARG_NAME>(arg_idx) << std::endl;
            }
        }
    }

    void call(std::string kernel_name,
              const std::vector<int>& global_size,
              const std::vector<int>& local_size,
              py::args args) {
        auto& kernel = kernel_map[kernel_name];

        int arg_idx = 0;
        for(auto& arg : args) {
            if (py::isinstance<py::int_>(arg)) {
                auto i = arg.cast<int>();
                kernel.setArg(arg_idx, i);
            } else if (py::isinstance<py::float_>(arg)) {
                auto f = arg.cast<float>();
                kernel.setArg(arg_idx, f);
            } else if (py::isinstance<cl_tensor<float>>(arg)) {
                const auto& t = arg.cast<cl_tensor<float>>();
                kernel.setArg(arg_idx, *(t.p_buff));
            } else {
                throw std::runtime_error(std::string("Unknown kernel arg at index ") + std::to_string(arg_idx));
            }
            arg_idx ++;
        }

        const cl::NDRange offset_;
        cl::NDRange global_;
        cl::NDRange local_;
        if (global_size.size() == 1) global_ = cl::NDRange(global_size[0]);
        if (global_size.size() == 2) global_ = cl::NDRange(global_size[0], global_size[1]);
        if (global_size.size() == 3) global_ = cl::NDRange(global_size[0], global_size[1], global_size[2]);

        if (local_size.size() == 1) local_ = cl::NDRange(local_size[0]);
        if (local_size.size() == 2) local_ = cl::NDRange(local_size[0], local_size[1]);
        if (local_size.size() == 3) local_ = cl::NDRange(local_size[0], local_size[1], local_size[2]);

        std::vector<cl::Event> events_;
        cl::Event event;
        cmd_queue.enqueueNDRangeKernel(
            kernel,
            offset_,
            global_,
            local_,
            &events_,
            &event);
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
        std::cout << "    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(device, local_work_size)  << "\n";
        std::cout << "    CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: " << local_work_size << " is " << k.getSubGroupInfo<CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE>(device, local_work_size)  << "\n";
        std::cout << "    CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: " << sub_groups << " is " << k.getSubGroupInfo<CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT>(device, sub_groups)  << "\n";
        std::cout << "    CL_KERNEL_MAX_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_MAX_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";
        std::cout << "    CL_KERNEL_COMPILE_NUM_SUB_GROUPS: " << k.getSubGroupInfo<CL_KERNEL_COMPILE_NUM_SUB_GROUPS>(device, local_work_size)  << "\n";
    }
*/

};


PYBIND11_MODULE(cl, m) {
    select_default_platform({"cl_intel_subgroups","cl_intel_required_subgroup_size"});

    cl::CommandQueue::setDefault(cl::CommandQueue(cl::QueueProperties::None));
    cmd_queue = cl::CommandQueue::getDefault();

    py::class_<cl_tensor<float>>(m, "tensor_f32")
        .def(py::init([](py::array b) {
            cl_tensor<float> t;
            t.resize(b);
            return t;
        }))
        .def(py::init([](std::vector<size_t> shape) {
            return cl_tensor<float>(shape);
        }))
        .def("numpy", &cl_tensor<float>::to_numpy)
        .def_property_readonly("shape", &cl_tensor<float>::get_shape);


    py::class_<cl_kernels>(m, "kernels")
        .def(py::init<std::string, std::string>())
        .def("call",  &cl_kernels::call);

    m.def("flush", [](){
        cmd_queue.flush();
    });

    m.def("finish", [](){
        cmd_queue.finish();
    });
}
