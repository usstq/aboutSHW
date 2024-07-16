/* Until glibc provides a proper stub ... */
#include <linux/perf_event.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>

__attribute__((weak))
int perf_event_open(struct perf_event_attr *attr, pid_t pid,
		    int cpu, int group_fd, unsigned long flags)
{
	return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <x86intrin.h>
#include <sys/mman.h>
#include <thread>

struct PerfEventGroup {
    int group_fd = -1;
    uint64_t read_format;

    struct event {
        int fd = -1;
        uint64_t id = 0;
        uint64_t pmc_index = 0;
        perf_event_mmap_page* pmeta = nullptr;
        const char * name = "?";
        char format[16];
    };
    std::vector<event> events;

    uint64_t read_buf[512]; // 4KB
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t pmc_width;
    uint64_t pmc_mask;
    uint64_t values[32];
    uint32_t tsc_time_shift;
    uint32_t tsc_time_mult;
    uint32_t refcycle_time_mult;
    
    // ref_cpu_cycles even id
    // this event is fixed function counter provided by most x86 CPU
    // and it provides TSC clock which is:
    //    - very high-resolution (<1ns or >1GHz)
    //    - independent of CPU-frequency throttling
    int ref_cpu_cycles_evid = -1;

    uint64_t operator[](int i) {
        if (i < events.size()) {
            return values[i];
        } else {
            printf("PerfEventGroup: operator[] with index %d oveflow (>%lu)\n", i, events.size());
            abort();
        }
        return 0;
    }
    
    PerfEventGroup() = default;

    struct Config {
        uint32_t type;
        uint64_t config;
        const char * name;
    };
    PerfEventGroup(const std::vector<Config> type_configs) {
        for(auto& tc : type_configs) {
            if (tc.type == PERF_TYPE_SOFTWARE) {
                add_sw(tc.config);
            }
            if (tc.type == PERF_TYPE_HARDWARE) {
                add_hw(tc.config);
            }
            if (tc.type == PERF_TYPE_RAW) {
                add_raw(tc.config);
            }
            events.back().name = tc.name;
            sprintf(events.back().format, "%%%lulu, ", strlen(tc.name));
        }
        show_header();
    }

    void show_header() {
        printf(" === PerfEventGroup ===\n\e[33m");
        for(auto& ev : events) {
            printf("%s, ", ev.name);
        }
        printf("\e[0m\n");
    }

    void add_raw(uint64_t config, bool pinned=false) {
        perf_event_attr pea;
        memset(&pea, 0, sizeof(struct perf_event_attr));
        pea.type = PERF_TYPE_RAW;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.exclude_kernel = 1;
        pea.exclude_hv = 1;
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;        
        if (pinned && group_fd == -1) {
            // pinned: It applies only to hardware counters and only to group leaders
            pea.pinned = 1;
        }
        if (group_fd == -1) {
            pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        }
        add(&pea);
    }

    void add_hw(uint64_t config, bool pinned=false) {
        perf_event_attr pea;
        memset(&pea, 0, sizeof(struct perf_event_attr));
        pea.type = PERF_TYPE_HARDWARE;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.exclude_kernel = 1;
        pea.exclude_hv = 1;
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;        
        if (pinned && group_fd == -1) {
            // pinned: It applies only to hardware counters and only to group leaders
            pea.pinned = 1;
        }
        if (group_fd == -1) {
            pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        }
        add(&pea);
    }

    void add_sw(uint64_t config) {
        perf_event_attr pea;
        memset(&pea, 0, sizeof(struct perf_event_attr));
        pea.type = PERF_TYPE_SOFTWARE;
        pea.size = sizeof(struct perf_event_attr);
        pea.config = config;
        pea.disabled = 1;
        pea.exclude_kernel = 1;
        pea.exclude_hv = 1;
        //pea.pinned = 1;   //sw event cannot set pinned!!!
        pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID ;
        add(&pea);
    }

    void add(perf_event_attr* pev_attr, pid_t pid = 0, int cpu = -1) {
        event ev;
        ev.fd = perf_event_open(pev_attr, pid, cpu, group_fd, 0);
        if (ev.fd < 0) {
            perror("perf_event_open");
            abort();
        }
        ioctl(ev.fd, PERF_EVENT_IOC_ID, &ev.id);

        size_t mmap_length = sysconf(_SC_PAGESIZE)*2;
        ev.pmeta = reinterpret_cast<perf_event_mmap_page*>(mmap(NULL, mmap_length, PROT_READ, MAP_SHARED, ev.fd, 0));
        if (ev.pmeta == MAP_FAILED) {
            perror("mmap perf_event_mmap_page failed:");
            close(ev.fd);
            abort();
        }

        if (group_fd == -1) {
            group_fd = ev.fd;
            read_format = pev_attr->read_format;
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_REF_CPU_CYCLES) {
            ref_cpu_cycles_evid = events.size();
        }
        printf("perf_event_open : fd=%d, id=%lu\n", ev.fd, ev.id);

        events.push_back(ev);
    }

    void enable() {
        ioctl(group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        // PMC index is only valid when being enabled
        for(auto& ev : events) {
            if (ev.pmeta->cap_user_rdpmc) {
                uint32_t seqlock;
                do {
                    seqlock = ev.pmeta->lock;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    ev.pmc_index = ev.pmeta->index;
                    pmc_width = ev.pmeta->pmc_width;
                    pmc_mask = 1;
                    pmc_mask = (pmc_mask << pmc_width) - 1;
                    if (ev.pmeta->cap_user_time) {
                        tsc_time_shift = ev.pmeta->time_shift;
                        tsc_time_mult = ev.pmeta->time_mult;
                        //printf("time: %u,%u\n", tsc_time_shift, tsc_time_mult);
                    }
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                } while (ev.pmeta->lock != seqlock || (seqlock & 1));
            }
        }
        /*
        UnHalted Reference Cycles — Event select 3CH, Umask 01H
            This event counts reference clock cycles at a fixed frequency while the clock signal on the core is running. The
            event counts at a fixed frequency, irrespective of core frequency changes due to performance state transitions.
            Processors may implement this behavior differently. Current implementations use the core crystal clock, TSC or
            the bus clock. Because the rate may differ between implementations, software should calibrate it to a time
            source with known frequency.

        here we need to calibrate Reference Cycles (TSC)
        */
        if (ref_cpu_cycles_evid >= 0) {
            auto ref_cycles_t0 = rdpmc(ref_cpu_cycles_evid);
            auto tsc_t0 = _rdtsc();
            auto ref_cycles_t1 = ref_cycles_t0;
            auto tsc_t1 = tsc_t0;
            // (tsc >> time_shift) * time_mult = 0.5e9 nanoseconds
            // tsc = (0.5e9 << time_shift / time_mult)
            uint64_t tsc_span = 500*1000*1000; // 100ms
            tsc_span = (tsc_span << tsc_time_shift) / tsc_time_mult;
            do {
                ref_cycles_t1 = rdpmc(ref_cpu_cycles_evid);
                tsc_t1 = _rdtsc();
            } while (tsc_t1 - tsc_t0 < tsc_span);

            refcycle_time_mult = tsc_time_mult * (ref_cycles_t1 - ref_cycles_t0) / (tsc_t1 - tsc_t0);

            //printf("tsc_time_shift=%u, tsc_time_mult=%u, refcycle_time_mult=%u\n",tsc_time_shift,tsc_time_mult,refcycle_time_mult);
        }
    }

    uint64_t refcycle2nano(uint64_t cyc) {
        uint64_t quot, rem;
        quot  = cyc >> tsc_time_shift;
        rem   = cyc & (((uint64_t)1 << tsc_time_shift) - 1);
        return quot * refcycle_time_mult + ((rem * refcycle_time_mult) >> tsc_time_shift);
    }

    uint64_t tsc2nano(uint64_t cyc) {
        uint64_t quot, rem;
        quot  = cyc >> tsc_time_shift;
        rem   = cyc & (((uint64_t)1 << tsc_time_shift) - 1);
        return quot * tsc_time_mult + ((rem * tsc_time_mult) >> tsc_time_shift);
    }

    void disable() {
        ioctl(group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        for(auto& ev : events) {
            ev.pmc_index = 0;
        }
    }

    uint64_t rdpmc(int i, uint64_t base = 0) {
        return (_rdpmc(events[i].pmc_index - 1) - base) & pmc_mask;
    }

    template<class FN>
    std::vector<uint64_t> rdpmc(FN fn, bool verbose = false) {
        int cnt = events.size();
        std::vector<uint64_t> pmc(cnt, 0);
        for(int i = 0; i < cnt; i++) {
            pmc[i] = _rdpmc(events[i].pmc_index - 1);
        }
        fn();
        for(int i = 0; i < cnt; i++) {
            pmc[i] = (_rdpmc(events[i].pmc_index - 1) - pmc[i]) & pmc_mask;
        }
        if (verbose) {
            printf("\e[33m");
            for(int i = 0; i < cnt; i++) {
                printf(events[i].format, pmc[i]);
            }
            printf("\e[0m\n");
        }
        return pmc;
    }

    void read(bool verbose = false) {
        for(int i = 0; i < events.size(); i++) values[i] = 0;

        if (::read(group_fd, read_buf, sizeof(read_buf)) == -1) {
            perror("read perf event failed:");
            abort();
        }

        uint64_t * readv = read_buf;
        auto nr = *readv++;
        if (verbose) printf("number of counters:\t%lu\n", nr);
        time_enabled = 0;
        time_running = 0;
        if (read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) {
            time_enabled = *readv++;
            if (verbose) printf("time_enabled:\t%lu\n", time_enabled);
        }
        if (read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) {
            time_running = *readv++;
            if (verbose) printf("time_running:\t%lu\n", time_running);
        }

        for (int i = 0; i < nr; i++) {
            auto value = *readv++;
            auto id = *readv++;
            for (int k = 0; k < events.size(); k++) {
                if (id == events[k].id) {
                    values[k] = value;
                }
            }
        }

        if (verbose) {
            for (int k = 0; k < events.size(); k++) {
                printf("\t[%d]: %lu\n", k, values[k]);
            }
        }
    }

    ~PerfEventGroup() {
        for(auto & ev : events) {
            close(ev.fd);
        }
    }
};

