#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

#define ANSI_COLOR_INFO  "\033[32m"
#define ANSI_COLOR_ERROR "\033[31m"
#define ANSI_COLOR_RESET "\033[0m"

namespace py = pybind11;

#ifdef __linux__
//======================================================================================================================
// [gcc + omp] based CPP kernels
union KArg {
    int64_t i;
    float f;
    void* p;
};
typedef void (*KERNEL_FUNC)(const std::vector<KArg>&);

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

        ss << "gcc -fopenmp -shared -o " << so_fname << " -Wall -fpic -x c++ - -lstdc++ ";
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

        func(kargs);
    }
};
#    else

struct cpp_kernels {
    ~cpp_kernels() {}
    cpp_kernels(std::string src, std::string options, std::string name) {}
    void call(std::string name, py::args args) {
        throw std::runtime_error(std::string("cpp_kernels only works on Linux"));
    }
};

#endif

void init_gemm(py::module_& m);

PYBIND11_MODULE(pycpp, m) {

    init_gemm(m);

    py::class_<cpp_kernels>(m, "kernels")
         .def(py::init<std::string, std::string, std::string>(), py::arg("source") = "", py::arg("options") = "", py::arg("name") = "")
         .def("call", &cpp_kernels::call);
}
