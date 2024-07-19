/*
<torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:
 - The ATen library, which is our primary API for tensor computation,
 - pybind11, which is how we create Python bindings for our C++ code,
 - Headers that manage the details of interaction between ATen and pybind11.
*/
#include <torch/extension.h>

#include "linux_perf.hpp"

thread_local PerfEventGroup pevg({
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
    //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "SW_CONTEXT_SWITCHES"},
    {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "SW_TASK_CLOCK"},
    {PERF_TYPE_RAW, 0x01b2, "PORT_0"},
});

struct PerfData {
    PerfEventGroup::ProfileScope pscope[256];
    int NT;
    const std::string title;
    bool is_finished;

    PerfData(const std::string& title) : NT(0), title(title), is_finished(false) {
        NT = at::get_num_threads();
        at::parallel_for(0, NT, 0, [&](int64_t i0, int64_t i1) {
            pscope[i0] = std::move(pevg.start_profile(title, 0));
        });
    }
    void finish() {
        if (!is_finished) {
            at::parallel_for(0, NT, 0, [&](int64_t i0, int64_t i1) {
                pscope[i0].finish();
            });
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
        .def(pybind11::init<const std::string&>())
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
