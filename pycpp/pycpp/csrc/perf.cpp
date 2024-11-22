#include "../include/linux_perf.hpp"

//auto _xx = LinuxPerf::Init();

struct PerfData {
    LinuxPerf::ProfileScope pscope[1];
    const std::string title;
    bool is_finished;

    PerfData(const std::string& title) : title(title), is_finished(false) {
        pscope[0] = std::move(LinuxPerf::Profile(title, 0));
    }
    void finish() {
        if (!is_finished) {
            pscope[0].finish();
            is_finished = true;
        }
    }
    ~PerfData() {
        finish();
    }
};

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(perf, m) {
    pybind11::class_<PerfData>(m, "perf")
        .def(pybind11::init<const std::string&>())
        .def("finish", &PerfData::finish)
        .def("__enter__", [&](PerfData& r) {})
        .def("__exit__", [&](PerfData& r, const pybind11::object& exc_type, const pybind11::object& exc_value, const pybind11::object& traceback) {
            r.finish();
        });
}
