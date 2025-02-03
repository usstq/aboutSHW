#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "linux_perf.hpp"
namespace py = pybind11;

// auto _xx = LinuxPerf::Init();
struct PerfData {
    LinuxPerf::ProfileScope pscope;
    std::vector<LinuxPerf::PerfEventGroup::Config> m_configs;
    std::shared_ptr<LinuxPerf::PerfEventGroup> m_pevg;
    std::string m_name;
    std::map<std::string, uint64_t> m_evts;
    uint64_t m_rounds = 0;
    uint64_t m_flops = 0;
    uint64_t m_kernel_loops = 0;
    uint64_t m_mem_rbytes = 0;

    PerfData()
        : m_configs{
              {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
              {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
          } {
        m_pevg = std::make_shared<LinuxPerf::PerfEventGroup>(m_configs);
        m_pevg->dump_limit = 999999;
    }

    PerfData(const std::vector<std::string>& pmu_config) {
        for (auto& str : pmu_config)
            m_configs.emplace_back(str);

        m_pevg = std::make_shared<LinuxPerf::PerfEventGroup>(m_configs);
        m_pevg->dump_limit = 999999;
    }

    ~PerfData() {}

    PerfData* name(const std::string& title) {
        m_name = title;
        return this;
    }

    PerfData* verbose(const std::string& title, uint64_t rounds, uint64_t kernel_loops, uint64_t kernel_flops, uint64_t kernel_mem_rbytes) {
        m_rounds = rounds;
        m_name = title;
        m_kernel_loops = kernel_loops;
        m_flops = kernel_flops;
        m_mem_rbytes = kernel_mem_rbytes;
        return this;
    }

    PerfData* enter() {
        auto* pd = m_pevg->_profile(m_name, 0);
        pscope = std::move(LinuxPerf::ProfileScope(m_pevg.get(), pd));
        return this;
    }

    bool exit(py::object exc_type, py::object exc_value, py::object traceback) {
        pscope.finish(&m_evts);
        if (m_rounds > 0) {
            // verbose mode, will automatically display performance numbers
            double avg_cycles = 0;
            printf("\033[36m");
            for (int i = 0; i < m_configs.size(); i++) {
                auto& evt_name = m_configs[i].name;
                if (m_configs[i].is_cpu_cycles())
                    avg_cycles = m_evts[evt_name] / (double)(m_rounds);
                printf("%s(%lu) ", evt_name.c_str(), m_evts[evt_name]/m_rounds);
            }
            auto dt_ns = m_evts["ns"] / (double)(m_rounds);

            printf("\033[33m");
            printf(" [%s] %.3f ms", m_name.c_str(), dt_ns * 1e-6);

            if (avg_cycles > 0) {
                printf("  %.1f GHz", avg_cycles / dt_ns);
                if (m_kernel_loops > 0)
                    printf("  CPI:%.1f", avg_cycles / m_kernel_loops);
            }
            if (m_flops > 0) {
                printf("  %.1f GFLOPS", m_flops / dt_ns);
            }
            if (m_mem_rbytes > 0) {
                printf("  %.1f GB/s", m_mem_rbytes / dt_ns);
            }
            printf("\033[0m");
            printf("\n");
            // reset verbose mode
            m_rounds = 0;
        }
        return false;  // false: we don't handle exceptions
    }

    std::map<std::string, uint64_t> events() {
        return m_evts;
    }
};

void perf_init(py::module_& m) {
    pybind11::class_<PerfData>(m, "perf")
        .def(pybind11::init<>())
        .def(pybind11::init<const std::vector<std::string>&>())
        .def("events", &PerfData::events)
        .def("name", &PerfData::name)
        .def("verbose", &PerfData::verbose, py::arg("title") = "?", py::arg("rounds") = 1, py::arg("kernel_loops") = 0, py::arg("kernel_flops") = 0, py::arg("kernel_mem_rbytes") = 0)
        .def("__enter__", &PerfData::enter)
        .def("__exit__", &PerfData::exit);
}
