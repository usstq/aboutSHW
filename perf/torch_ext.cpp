/*
<torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:
 - The ATen library, which is our primary API for tensor computation,
 - pybind11, which is how we create Python bindings for our C++ code,
 - Headers that manage the details of interaction between ATen and pybind11.
*/
#include <torch/extension.h>

#include "../include/linux_perf.hpp"

auto _xx = LinuxPerf::Init();

struct PerfData {
    LinuxPerf::ProfileScope pscope[256];
    int NT;
    const std::string title;
    bool is_finished;
    bool all_threads;

    PerfData(const std::string& title, bool all_threads) : NT(0), title(title), is_finished(false), all_threads(all_threads) {
        pscope[0] = std::move(LinuxPerf::Profile(title, 0));

        if (all_threads) {
            NT = at::get_num_threads();
            at::parallel_for(0, NT, 0, [&](int64_t i0, int64_t i1) {
                if (i0 > 0) pscope[i0] = std::move(LinuxPerf::Profile(title, 0));
            });
        }
    }
    void finish() {
        if (!is_finished) {
            if (all_threads) {
                at::parallel_for(0, NT, 0, [&](int64_t i0, int64_t i1) {
                    if (i0 > 0) pscope[i0].finish();
                });
            }
            pscope[0].finish();
            is_finished = true;
        }
    }
    ~PerfData() {
        finish();
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<PerfData>(m, "PerfData")
        .def(pybind11::init<const std::string&, bool>())
        .def("finish", &PerfData::finish)
        .def("__enter__", [&] (PerfData& r) { })
        .def("__exit__",
        [&] (PerfData& r,
            const pybind11::object& exc_type,
            const pybind11::object& exc_value,
            const pybind11::object& traceback)
        { 
                r.finish(); 
        });
}
