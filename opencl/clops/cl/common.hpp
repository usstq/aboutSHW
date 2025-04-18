
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>

#include <vector>
#include <memory>

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

    template <class T>
    operator T*() const {
        //if (py::dtype::of<T>() != dt)
        //    throw std::runtime_error(std::string("unable to cast from tensor of dtype ") + dt.kind() + " to " + py::dtype::of<T>().kind());
        return reinterpret_cast<T*>(p_buff.get());
    }

    operator void*() const {
        return p_buff.get();
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
