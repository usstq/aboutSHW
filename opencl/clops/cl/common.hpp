
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>

#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <iostream>

#ifndef ASSERT

template <typename... TS>
void _write_all(std::ostream& os, TS&&... args) {
    int dummy[sizeof...(TS)] = {(os << std::forward<TS>(args), 0)...};
    (void)dummy;
}

#ifdef __x86_64__
#    define TRAP_INST() __asm__("int3");
#endif

#ifdef __aarch64__
#    define TRAP_INST() __asm__("brk #0x1");
#endif

#define ASSERT(cond, ...)                                                                   \
        if (!(cond)) {                                                                      \
            std::stringstream ss;                                                           \
            _write_all(ss, __FILE__, ":", __LINE__, " ", #cond, " failed:", ##__VA_ARGS__); \
            std::cout << "\033[31m" << ss.str() << "\033[0m" << std::endl;                  \
            /* TRAP_INST();*/                                                               \
            throw std::runtime_error(ss.str());                                             \
        }

#endif


#ifdef SYCL_LANGUAGE_VERSION
#include <sycl/sycl.hpp>
using namespace sycl;
struct ocl_queue {
    cl_platform_id platform;
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;

    sycl::queue sycl_queue;
    
    std::vector<std::variant<cl_event, sycl::event>> events;
    bool enable_profile;

    ocl_queue() {
        update_queue(false);
    }

    std::vector<uint64_t> finish() {
        sycl_queue.wait();
        // return all event time-stamps
        std::vector<uint64_t> ret;
        if (enable_profile) {
            for (auto& e : events) {
                ret.emplace_back(0);
                if (e.index() == 0) {
                    auto evt = std::get<cl_event>(e);
                    if (evt != 0) {
                        cl_ulong start, end;
                        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
                        clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
                        ret.back() = end - start;
                    }
                } else {
                    auto evt = std::get<sycl::event>(e);
                    auto start = evt.get_profiling_info<info::event_profiling::command_start>();
                    auto end = evt.get_profiling_info<info::event_profiling::command_end>();
                    ret.back() = end - start;
                }
            }
        }
        events.clear();
        return ret;        
    }

    void* malloc_host(size_t sz) {
        return sycl::malloc_host(sz, sycl_queue);
    }

    void* malloc_device(size_t sz) {
        return sycl::malloc_device(sz, sycl_queue);
    }

    void memcpy_HtoD(void* pdst, void* psrc, size_t bytes) {
        sycl_queue.submit([&](handler& h) {
            h.memcpy(pdst, psrc, bytes);
        });
        sycl_queue.wait();
    }

    void memcpy_DtoH(void* pdst, void* psrc, size_t bytes) {
        sycl_queue.submit([&](handler& h) {
            h.memcpy(pdst, psrc, bytes);
        });
        sycl_queue.wait();
    }

    template<class MAP>
    void get_device_info(MAP& result) {
        auto device = sycl_queue.get_device();
        result["CL_DEVICE_NAME"] = device.get_info<sycl::info::device::name>();
        result["CL_DEVICE_EXTENSIONS"] = device.get_info<sycl::info::device::extensions>();
        result["CL_DEVICE_MAX_COMPUTE_UNITS"] = device.get_info<sycl::info::device::max_compute_units>();
        result["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = device.get_info<sycl::info::device::max_clock_frequency>();
        result["CL_DEVICE_LOCAL_MEM_SIZE"] = device.get_info<sycl::info::device::local_mem_size>();
    }

    void update_queue(bool _enable_profile = false) {
        std::cout << "update_queue:" << std::endl;
        auto opencl_gpu_selector = [](const sycl::device& d) {
            std::cout << "opencl_gpu_selector: "
                    << (d.is_cpu() ? "[CPU]" : "")
                    << (d.is_gpu() ? "[GPU]" : "")
                    << d.get_info<sycl::info::device::name>();
            if (d.is_gpu() && d.get_backend() == sycl::backend::opencl) {
                // return higher score 1
                std::cout << "<====" << std::endl;
                return 1;
            }
            std::cout << std::endl;
            return 0;
        };

        enable_profile = _enable_profile;
        if (enable_profile) {
            sycl::property_list propList{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
            sycl_queue = sycl::queue(opencl_gpu_selector, propList);
        } else {
            sycl::property_list propList{sycl::property::queue::in_order()};
            sycl_queue = sycl::queue(opencl_gpu_selector, propList);
        }
        context = sycl::get_native<sycl::backend::opencl>(sycl_queue.get_context());
        device = sycl::get_native<sycl::backend::opencl>(sycl_queue.get_device());
        queue = sycl::get_native<sycl::backend::opencl>(sycl_queue);
        platform = sycl::get_native<sycl::backend::opencl>(sycl_queue.get_context().get_platform());
    }

    template<typename RET, typename... Args>
    void call_ext(const char * func_name, Args&&... args) {
        using ext_func_t = RET (*)(Args...);
        ext_func_t ext_func = clGetExtensionFunctionAddressForPlatform(platform, func_name);
        ASSERT(ext_func != nullptr, "clGetExtensionFunctionAddressForPlatform()", func_name, " returns nullptr");
        return (*ext_func)(std::forward<Args>(args)...);
    }
};

#else

template<typename T>
T getDeviceInfo(cl_device_id device, cl_device_info param_name) {
    size_t bytes = 0;
    clGetDeviceInfo(device, param_name, 0, nullptr, &bytes);
    ASSERT(bytes == sizeof(T), "clGetDeviceInfo needs more space");
    T ret;
    clGetDeviceInfo(device, param_name, sizeof(ret), &ret, nullptr);
    return ret;
}

template<>
inline std::string getDeviceInfo<std::string>(cl_device_id device, cl_device_info param_name) {
    size_t bytes = 0;
    clGetDeviceInfo(device, param_name, 0, nullptr, &bytes);
    std::string ret(bytes, 0);
    clGetDeviceInfo(device, param_name, bytes, &ret[0], nullptr);
    return ret;
}

struct ocl_queue {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    bool enable_profile;
    std::vector<cl_event> events;

    ocl_queue() {
        select_device();
        update_queue(false);
    }

    std::vector<uint64_t> finish() {
        ASSERT(clFinish(queue) == CL_SUCCESS);
        std::vector<uint64_t> ret;
        if (enable_profile) {
            for (auto& evt : events) {
                ret.emplace_back(0);
                if (evt != 0) {
                    cl_ulong start, end;
                    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
                    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
                    ret.back() = end - start;
                }
            }
        }
        events.clear();
        return ret;
    }

    void* malloc(size_t sz) {
        return clSVMAlloc(context, CL_MEM_READ_WRITE, sz, 64);
    }

    void memcpy_HtoD(void* pdst, void* psrc, size_t bytes) {
        ASSERT(clFinish(queue) == CL_SUCCESS);
        clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, pdst, bytes, 0, nullptr, nullptr);
        std::memcpy(pdst, psrc, bytes);
        clEnqueueSVMUnmap(queue, pdst, 0, nullptr, nullptr);
        ASSERT(clFinish(queue) == CL_SUCCESS);
    }

    void memcpy_DtoH(void* pdst, void* psrc, size_t bytes) {
        ASSERT(clFinish(queue) == CL_SUCCESS);
        clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, psrc, bytes, 0, nullptr, nullptr);
        std::memcpy(pdst, psrc, bytes);
        clEnqueueSVMUnmap(queue, psrc, 0, nullptr, nullptr);
        ASSERT(clFinish(queue) == CL_SUCCESS);
    }

    template<class MAP>
    void get_device_info(MAP& result) {
        result["CL_DEVICE_NAME"] = getDeviceInfo<std::string>(device, CL_DEVICE_NAME);
        result["CL_DEVICE_EXTENSIONS"] = getDeviceInfo<std::string>(device, CL_DEVICE_EXTENSIONS);
        result["CL_DEVICE_MAX_COMPUTE_UNITS"] = getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
        result["CL_DEVICE_MAX_CLOCK_FREQUENCY"] = getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
        result["CL_DEVICE_LOCAL_MEM_SIZE"] = getDeviceInfo<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
    }

    void select_device(std::vector<std::string> exts = {}) {
        cl_int error;
        cl_uint n = 0;
        clGetPlatformIDs(0, nullptr, &n);
        std::vector<cl_platform_id> ids(n);
        clGetPlatformIDs(n, ids.data(), nullptr);
        int gpu_dev_index = 0;
        platform = 0;
        std::vector<std::pair<cl_device_id, cl_platform_id>> device_ids;
        for(int i = 0; i < n; i++) {
            auto platform_id = ids[i];
            cl_uint n_devs = 0;
            clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devs);
            if (n_devs == 0) continue;

            std::vector<cl_device_id> dev_ids(n_devs);
            error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, n_devs, dev_ids.data(), nullptr); ASSERT(error == CL_SUCCESS);

            for(int k = 0; k < n_devs; k++) {
                auto devid = dev_ids[k];
                std::cout << " GPU device [" << gpu_dev_index << "] : "
                          << getDeviceInfo<std::string>(devid, CL_DEVICE_NAME)
                          << " by \"" << getDeviceInfo<std::string>(devid, CL_DEVICE_VENDOR) << "\" : "
                          << getDeviceInfo<cl_uint>(devid, CL_DEVICE_MAX_COMPUTE_UNITS) << " EUs "
                          << getDeviceInfo<cl_uint>(devid, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz "
                          << getDeviceInfo<cl_ulong>(devid, CL_DEVICE_LOCAL_MEM_SIZE) << " bytes SLM"
                          << std::endl;
                device_ids.push_back({devid, platform_id});
                gpu_dev_index++;
            }
        }
        device = device_ids[0].first;
        platform = device_ids[0].second;
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &error); ASSERT(error == 0);
        return;
    }

    void update_queue(bool _enable_profile = false) {
        enable_profile = _enable_profile;
        cl_queue_properties queue_properties[] = {
                    CL_QUEUE_PROPERTIES,
                    enable_profile ? CL_QUEUE_PROFILING_ENABLE : 0ul,
                    0ul};

        cl_int error;
        queue = clCreateCommandQueueWithProperties(context, device, queue_properties, &error); ASSERT(error == 0);
    }

    template<typename RET, typename... Args>
    void call_ext(const char * func_name, Args&&... args) {
        using ext_func_t = RET (*)(Args...);
        ext_func_t ext_func = clGetExtensionFunctionAddressForPlatform(platform, func_name);
        ASSERT(ext_func != nullptr, "clGetExtensionFunctionAddressForPlatform()", func_name, " returns nullptr");
        return (*ext_func)(std::forward<Args>(args)...);
    }    
};
#endif


// composite of cl::Buffer & layout information
// like numpy array
struct tensor {
    std::vector<cl_uint> shape;
    std::vector<cl_uint> strides;
    cl_uint numel = 0;
    std::shared_ptr<void> p_buff;
    py::dtype dt;
    int offset = 0;

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

    template <class T>
    operator T*() const {
        //if (py::dtype::of<T>() != dt)
        //    throw std::runtime_error(std::string("unable to cast from tensor of dtype ") + dt.kind() + " to " + py::dtype::of<T>().kind());
        return reinterpret_cast<T*>(data());
    }

    operator void*() const {
        return data();
    }

    void* data() const {
        return reinterpret_cast<int8_t*>(p_buff.get()) + offset;
    }

    uintptr_t addr() const {
        return reinterpret_cast<uintptr_t>(p_buff.get()) + offset;
    }

    void set_offset(int off) {
        offset = off;
    }
    int get_offset() {
        return offset;
    }

    void resize(const std::vector<cl_uint>& dims, py::dtype dtype);

    tensor(const py::array& arr) {
        resize(arr);
    }
    tensor(const std::vector<cl_uint>& dims, py::dtype dtype) {
        resize(dims, dtype);
    }
    void update_buff();
    void resize(const py::array& b);
    py::array to_numpy();
};

struct cl_kernels {
    std::map<std::string, cl_kernel> kernel_map;
    cl_program m_prog;
    std::string m_source;
    std::string m_options;
    cl_kernels() = default;

    void check_error(const char* name, cl_int error, bool show_build_log = false);
    cl_kernels(py::bytes bin_bytes, std::string options);
    cl_kernels(std::string source, std::string options, std::string dump_dir);

    void setup(std::string source, std::string options, std::string dump_dir);

    cl_kernel get_kernel(std::string kernel_name);
    void set_arg(cl_kernel kernel, int idx, int arg);
    void set_arg(cl_kernel kernel, int idx, float arg);
    void set_arg(cl_kernel kernel, int idx, void* arg);

    template<typename... Ts>
    void set_args(cl_kernel kernel, Ts... args) {
        int arg_id = 0;
        // since initializer lists guarantee sequencing, this can be used to
        // call a function on each element of a pack, in order:
        int dummy[sizeof...(Ts)] = {(set_arg(kernel, arg_id++, args), 0)...};
    }

    void enqueue(cl_kernel kernel, const std::vector<size_t>& global_size, const std::vector<size_t>& local_size);

    void enqueue_py(std::string kernel_name, const std::vector<size_t>& global_size, const std::vector<size_t>& local_size, py::args args);
    std::string info(const char* kernel_name, const std::vector<int>& local_size, size_t sub_groups);
};


// all jit-based/performance-aware function should be a functor/callable because:
//   - it needs to hold reference to kernel (to save build time & resources)
//   - it needs to do other compile time preparation work and hold the relevant
//     runtime-data-struct (to make runtime faster)
// to optimze compile-time-workload itself, the functor instance itself should be
// cached with compile-time parameter as the key.
//
// because it's a functor, which supposed to have no states, so cache-factory should
// always return shared_ptr to constant object, so it won't behave differently when being
// called by different caller, and this also ensure it's multi-threading safe since it
// won't modify it's content.
//
template <typename... TTypes>
class tuple_hasher {
private:
    typedef std::tuple<TTypes...> Tuple;
    template <int N>
    size_t hash(Tuple& value) const {
        return 0;
    }
    template <int N, typename THead, typename... TTail>
    size_t hash(Tuple& value) const {
        constexpr int Index = N - sizeof...(TTail) - 1;
        return std::hash<THead>()(std::get<Index>(value)) ^ hash<N, TTail...>(value);
    }

public:
    size_t operator()(Tuple value) const {
        auto hv = hash<sizeof...(TTypes), TTypes...>(value);
        return hv;
    }
};

// create const object with internal cache with constructor-args as the key
// this helps reduces construction time overhead, and perfectly suitable
// for caching functor/callable.
template <class T, typename... CArgs>
std::shared_ptr<const T> make_cacheable(CArgs... cargs) {
    std::shared_ptr<const T> sptr;
    auto key = std::make_tuple(cargs...);
    static std::unordered_map<decltype(key), std::weak_ptr<const T>, tuple_hasher<CArgs...>> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        auto& wptr = it->second;
        sptr = wptr.lock();
        if (!sptr) {
            sptr = std::make_shared<T>(cargs...);
            // ECOUT("make_cacheable re-constructed: ", typeid(T).name(), "(", cargs..., ")");
            wptr = sptr;
        }
    } else {
        sptr = std::make_shared<T>(cargs...);
        // ECOUT("make_cacheable constructed: ", typeid(T).name(), "(", cargs..., ")");
        cache.emplace(std::make_pair(key, std::weak_ptr<const T>(sptr)));
    }
    return sptr;
}