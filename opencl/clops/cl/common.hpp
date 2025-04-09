
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <sycl/CL/opencl.h>

#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

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