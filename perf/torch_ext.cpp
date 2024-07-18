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

thread_local PerfEventGroup::ProfileScope pscope;

void start(std::string title, int id) {
    auto NT = at::get_num_threads();
    // std::cout << "aten::get_num_threads() is " << NT << std::endl;
    at::parallel_for(0, NT, 0, [&](int64_t i0, int64_t i1) {
        pscope = std::move(pevg.start_profile(title, id));
    });
}

void finish() {
    auto NT = at::get_num_threads();
    at::parallel_for(0, NT, 0, [&](int64_t i0, int64_t i1) {
        pscope.finish();
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("start", &start, "start");
    m.def("finish", &finish, "finish");
}
