#include "../include/linux_perf.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

//auto _xx = LinuxPerf::Init();

static LinuxPerf::PerfEventGroup& get_pevtg() {
    thread_local LinuxPerf::PerfEventGroup pevg({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
        //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},
        //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},
        //{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "SW_CONTEXT_SWITCHES"},
        //{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "SW_TASK_CLOCK"},
        //{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "SW_PAGE_FAULTS"}
    });
    return pevg;
}

struct PerfData {
    LinuxPerf::ProfileScope pscope[1];
    const std::string title;
    bool is_finished;

    PerfData(const std::string& title) : title(title), is_finished(false) {
        auto& pevg = get_pevtg();
        auto* pd = pevg._profile(title, 0);
        pscope[0] = std::move(LinuxPerf::ProfileScope(&pevg, pd));
    }
    ~PerfData() {
        if (!is_finished) {
            pscope[0].finish();
            is_finished = true;
        }
    }
    PerfData* enter() {
        return this;
    }
    bool exit(py::object exc_type, py::object exc_value, py::object traceback) {
        if (!is_finished) {
            pscope[0].finish();
            is_finished = true;
        }
    }
    std::vector<uint64_t> finish() {
        if (!is_finished) {
            std::vector<uint64_t> ret(32, 0);
            pscope[0].finish(&ret[0]);
            is_finished = true;
            return ret;
        }
        return {};
    }
};

PYBIND11_MODULE(perf, m) {
    pybind11::class_<PerfData>(m, "perf")
        .def(pybind11::init<const std::string&>())
        .def("finish", &PerfData::finish)
        .def("__enter__", &PerfData::enter)
        .def("__exit__", &PerfData::exit);
}
