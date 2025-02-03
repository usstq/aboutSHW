#pragma once
#include <inttypes.h>
#include <linux/perf_event.h> /* Definition of PERF_* constants */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h> /* Definition of SYS_* constants */
#include <unistd.h>

#include <vector>
#include <map>

#include "misc.hpp"
#define TOTAL_EVENTS 6

struct LinuxPerf {
    // Executes perf_event_open syscall and makes sure it is successful or exit
    static long perf_event_open(struct perf_event_attr* hw_event,
                                pid_t pid,
                                int cpu,
                                int group_fd,
                                unsigned long flags) {
        int fd;
        fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
        ASSERT(fd != -1, "perf_event_open failed");
        return fd;
    }

    // Helper function to setup a perf event structure (perf_event_attr; see man perf_open_event)
    void configure_event(struct perf_event_attr* pe, uint32_t type, uint64_t config) {
        memset(pe, 0, sizeof(struct perf_event_attr));
        pe->type = type;
        pe->size = sizeof(struct perf_event_attr);
        pe->config = config;
        pe->read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
        pe->disabled = 1;
        pe->exclude_kernel = 1;
        pe->exclude_hv = 1;
    }

    // Format of event data to read
    // Note: This format changes depending on perf_event_attr.read_format
    // See `man perf_event_open` to understand how this structure can be different depending on event config
    // This read_format structure corresponds to when PERF_FORMAT_GROUP & PERF_FORMAT_ID are set
    struct read_format {
        uint64_t nr;
        struct {
            uint64_t value;
            uint64_t id;
        } values[TOTAL_EVENTS];
    };

    uint64_t pe_val[TOTAL_EVENTS + 1];  // Counter value array corresponding to fd/id array.

    struct event {
        std::string name;
        uint32_t type;
        uint64_t config;
        int fd;                // fd[0] will be the group leader file descriptor
        int id;                // event ids for file descriptors
        uint64_t value;
        perf_event_attr attr;  // Configuration structure for perf events (see man perf_event_open)
        bool is_cycles() {
            return type == PERF_TYPE_HARDWARE && config == PERF_COUNT_HW_CPU_CYCLES;
        }
        bool is_instructions() {
            return type == PERF_TYPE_HARDWARE && config == PERF_COUNT_HW_INSTRUCTIONS;
        }
        event(const std::string _name, const uint64_t _config = 0) : name(_name), config(_config) {
            if (name == "CPU_CYCLES" || name == "C") {
                type = PERF_TYPE_HARDWARE;
                config = PERF_COUNT_HW_CPU_CYCLES;
            } else if (name == "INSTRUCTIONS" || name == "I") {
                type = PERF_TYPE_HARDWARE;
                config = PERF_COUNT_HW_INSTRUCTIONS;
            } else if (name == "STALLED_CYCLES_FRONTEND") {
                type = PERF_TYPE_HARDWARE;
                config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
            } else if (name == "STALLED_CYCLES_BACKEND") {
                type = PERF_TYPE_HARDWARE;
                config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
            } else {
                type = PERF_TYPE_RAW;
                config = _config;
            }
            auto* pe = &attr;
            memset(pe, 0, sizeof(struct perf_event_attr));
            pe->type = type;
            pe->size = sizeof(struct perf_event_attr);
            pe->config = config;
            pe->read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
            pe->disabled = 1;
            pe->exclude_kernel = 1;
            pe->exclude_hv = 1;
        }
    };

    std::vector<event> m_events;

    std::string m_ev_name_cycles;
    std::string m_ev_name_instructions;

    // default : only hw cycles & instructions
    LinuxPerf() : LinuxPerf({{"C", 0}, {"I", 0}}) {}

    LinuxPerf(const std::vector<event>& events) {
        m_events = events;
        for(int i = 0; i < m_events.size(); i++) {
            if (m_events[i].is_cycles()) m_ev_name_cycles = m_events[i].name;
            if (m_events[i].is_instructions()) m_ev_name_instructions = m_events[i].name;
        }
        // Create event group leader
        m_events[0].fd = perf_event_open(&m_events[0].attr, 0, -1, -1, 0);
        ASSERT(m_events[0].fd >= 0);
        ioctl(m_events[0].fd, PERF_EVENT_IOC_ID, &m_events[0].id);
        // Let's create the rest of the events while using fd[0] as the group leader
        for (int i = 1; i < m_events.size(); i++) {
            m_events[i].fd = perf_event_open(&m_events[i].attr, 0, -1, m_events[0].fd, 0);
            ASSERT(m_events[i].fd >= 0);
            ioctl(m_events[i].fd, PERF_EVENT_IOC_ID, &m_events[i].id);
        }
    }
    LinuxPerf(const std::initializer_list<event>& events) : LinuxPerf(std::vector<event>(events)) {}

    inline uint64_t get_time_ns() {
        struct timespec tp0;
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp0) != 0) {
            perror("clock_gettime(CLOCK_MONOTONIC_RAW,...) failed!");
            abort();
        }
        return (tp0.tv_sec * 1000000000) + tp0.tv_nsec;
    }

    uint64_t duration_ns;
    struct read_format counter_results;

    void start() {
        // Reset counters and start counting; Since fd[0] is leader, this resets and enables all counters
        // PERF_IOC_FLAG_GROUP required for the ioctl to act on the group of file descriptors
        ioctl(m_events[0].fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(m_events[0].fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        duration_ns = get_time_ns();
    }

    std::map<std::string, uint64_t> m_evs;

    const std::map<std::string, uint64_t>& stop() {
        duration_ns = get_time_ns() - duration_ns;
        m_evs.clear();
        m_evs["ns"] = duration_ns;

        // Stop all counters
        ioctl(m_events[0].fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        // Read the group of counters and print result
        read(m_events[0].fd, &counter_results, sizeof(read_format));
        ASSERT(m_events.size() == counter_results.nr);

        for (int i = 0; i < counter_results.nr; i++) {
            for (int j = 0; j < m_events.size(); j++) {
                if (counter_results.values[i].id == m_events[j].id) {
                    m_events[j].value = counter_results.values[i].value;
                    m_evs[m_events[j].name] = m_events[j].value;
                }
            }
        }
        return m_evs;
    }

    ~LinuxPerf() {
        // Close counter file descriptors
        for (int i = 0; i < TOTAL_EVENTS; i++) {
            close(m_events[i].fd);
        }
    }

    // profiling callable
    template<class F>
    void operator()(F func, std::string name, int n_repeats = 1, int loop_count = 1, double ops = 0, double bytes = 0) {
        for(int round = 0; round < n_repeats; round++) {
            start();
            func();
            auto evs = stop();

            printf("\e[0;36m[%8s] No.%d/%d %lu(ns) ", name.c_str(), round, n_repeats, evs["ns"]);

            for (auto const& x : evs) {
                if (x.first != "ns")
                    printf("%s/", x.first.c_str());
            }
            printf("(per-iter): ");
            for (auto const& x : evs) {
                if (x.first != "ns")
                    printf("%.1f ", (double)x.second/loop_count);
            }

            if (!m_ev_name_cycles.empty() && !m_ev_name_instructions.empty())
                printf("CPI: %.2f ", (double)evs[m_ev_name_cycles]  / (double)evs[m_ev_name_instructions]);
            if (!m_ev_name_cycles.empty())
                printf("%.2f(GHz) ", (double)evs[m_ev_name_cycles]/evs["ns"]);

            if (ops > 0)
                printf("%.1f(GOP/S) ", ops/evs["ns"]);
            if (bytes > 0)
                printf("%.1f(GB/S) ", bytes/evs["ns"]);
            if (ops > 0 && bytes > 0)
                printf("%.1f(OP/B) ", ops/bytes);

            printf("\e[0m\n");
        }
    }
};