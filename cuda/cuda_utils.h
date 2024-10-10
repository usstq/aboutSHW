#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <deque>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <stdio.h>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <memory>
#include <array>
#include "../include/misc.hpp"

struct CUDADevice {
    CUDADevice(int id) {
        ECOUT("cudaSetDevice(0)");
        ASSERT(cudaSetDevice(id) == cudaSuccess);

        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, id);
        std::cout << " cudaGetDeviceProperties(..., " << id << " ) : \n";
        std::cout << "\t totalGlobalMem     : " << dev_prop.totalGlobalMem << " (" << dev_prop.totalGlobalMem / (1024*1024) << " MB)" << std::endl;
        std::cout << "\t sharedMemPerBlock  : " << dev_prop.sharedMemPerBlock << std::endl;
        std::cout << "\t regsPerBlock       : " << dev_prop.regsPerBlock << std::endl;
        std::cout << "\t warpSize           : " << dev_prop.warpSize << std::endl;
        std::cout << "\t memPitch           : " << dev_prop.memPitch << std::endl;
        std::cout << "\t maxThreadsPerBlock : " << dev_prop.maxThreadsPerBlock << std::endl;
        std::cout << "\t totalConstMem      : " << dev_prop.totalConstMem << std::endl;
        std::cout << "\t major          : " << dev_prop.major << std::endl;
        std::cout << "\t minor          : " << dev_prop.minor << std::endl;
        std::cout << "\t clockRate              : " << dev_prop.clockRate << "(KHz)" << std::endl;
        std::cout << "\t multiProcessorCount    : " << dev_prop.multiProcessorCount << " (each SM has 128 CUDA-cores)" << std::endl;
        std::cout << "\t kernelExecTimeoutEnabled: " << dev_prop.kernelExecTimeoutEnabled << std::endl;
        std::cout << "\t integrated         : " << dev_prop.integrated << std::endl;
        std::cout << "\t canMapHostMemory   : " << dev_prop.canMapHostMemory << std::endl;
        std::cout << "\t computeMode        : " << dev_prop.computeMode << std::endl;
        // each SM has 128 CUDA cores which has 2*frequency FLOPS/s
        std::cout << "\t ... peak performance        : " << 1e-9 * dev_prop.multiProcessorCount * dev_prop.clockRate * 128 * 2  << "(TFLOP/s)" << std::endl;
    }

    ~CUDADevice() {
        ECOUT("cudaDeviceReset()");
        ASSERT(cudaDeviceReset() == cudaSuccess);
    }
};

#define CEIL_DIV(x, a) (((x) + (a) - 1)/(a))
#define WRAP_SIZE 32


__global__ void _tensor_rand(int M, float *A) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < M) {
    A[x] = (x % 256)/255.0f - 0.5f;
  }
}

// https://stackoverflow.com/questions/7876624/timing-cuda-operations
struct CUDATimer {
    std::chrono::system_clock::time_point host_start;
    std::chrono::system_clock::time_point host_stop;
    cudaEvent_t start;
    cudaEvent_t stop;
    const char* func_name;
    int lineno;
    int id;
    const char* annotation;
    uint64_t bytes;
    uint64_t flops;
    std::stringstream postscript;

    CUDATimer(int id, const char* func_name = "", int lineno = 0, const char * annotation = nullptr,
              uint64_t bytes = 0, uint64_t flops = 0)
        : id(id), func_name(func_name), lineno(lineno), annotation(annotation), bytes(bytes), flops(flops) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CUDATimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start() {
        cudaEventRecord(start, 0);
        host_start = std::chrono::system_clock::now();
    }
    void Stop() {
        host_stop = std::chrono::system_clock::now();
        cudaEventRecord(stop, 0);
    }
    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

struct prettyTime {
    const char* unit;
    float t;
    prettyTime(float t0) : t(t0) {
        unit = "sec";
        if (t <= 1e-3) {
            t *= 1e6;
            unit = " us";
        } else if (t <= 1.0f) {
            t *= 1e3;
            unit = " ms";
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const prettyTime& dt);
};

inline std::ostream& operator<<(std::ostream& os, const prettyTime& dt) {
    os << std::fixed << std::setw(7) << std::setprecision(3) << dt.t << dt.unit;
    return os;
}

struct AutoCUDATimer {
    std::deque<CUDATimer> timers;

    // return avg device duration in seconds
    double finish() {
        float elapsed_since_last_stop;
        cudaEvent_t tlast = nullptr;
        double avg_dev_dur_nano_seconds = 0;
        for (auto& t : timers) {
            if (!tlast)
                tlast = t.start;
            cudaEventElapsedTime(&elapsed_since_last_stop, tlast, t.start);
            elapsed_since_last_stop *= 1e-3; // in seconds
            auto host_dur = std::chrono::duration<double>(t.host_stop - t.host_start).count();
            auto dev_dur = t.Elapsed() * 1e-3; // in seconds
            avg_dev_dur_nano_seconds += t.Elapsed() * 1e6;

            std::cout << "\033[1;96m [AutoCUDATimer # " << t.id << "] @host " << prettyTime(host_dur)
                << " | @device (+" << prettyTime(elapsed_since_last_stop) << ") " << prettyTime(dev_dur);
            if (t.bytes)
                std::cout << " " << std::fixed << std::setw(7) << std::setprecision(3)
                          << t.bytes * 1e-9 / dev_dur << " GB/s";
            if (t.flops) {
                double flops = t.flops * 1e-9 / dev_dur;
                auto unit = " GFLOP/s";
                if (flops > 1000.0) {
                    flops /= 1000.0;
                    unit = " TFLOP/s";
                }
                std::cout << " " << std::fixed << std::setw(7) << std::setprecision(3) << flops << unit;
            }
            std::cout << "\t  " << t.annotation << " (" << t.func_name << ":" << t.id << ") "
                      << t.bytes/1e6 << " MB " << t.flops/1e9 << " Gflops  \033[0m" << t.postscript.str() << std::endl;
            tlast = t.stop;
        }
        avg_dev_dur_nano_seconds /= timers.size();
        timers.clear();
        return avg_dev_dur_nano_seconds;
    }
    ~AutoCUDATimer() {
        finish();
    }
    template<typename Callable>
    void timeit(Callable c, const char* name = nullptr, int id = 0) {
        timers.emplace_back(name, id);
        timers.back().Start();
        c();
        timers.back().Stop();
    }
};

static AutoCUDATimer gpu_timers;

// TIMEIT_BEGIN(annotation, bytes, flops)
#define TIMEIT_BEGIN(...) gpu_timers.timers.emplace_back(__func__, __LINE__, __VA_ARGS__); gpu_timers.timers.back().Start();
#define TIMEIT_END() gpu_timers.timers.back().Stop();
#define TIMEIT_PS(ps) gpu_timers.timers.back().postscript = ps;

#define TIMEIT_FINISH() gpu_timers.finish();

#define TIMEIT(...) do { \
    gpu_timers.timers.emplace_back(__func__, __LINE__); \
    gpu_timers.timers.back().Start(); \
    __VA_ARGS__ \
    gpu_timers.timers.back().Stop(); \
} while(0)

static int get_sequence_id() {
    static int id = 0;
    return id++;
}

template<typename F>
double cuda_timeit(F func, const char * func_name, int lineno, const char * annotation, size_t bytes, size_t flops, int repeat = 1) {
    auto& list = getenvs("CUDATIMEIT");
    auto myid = get_sequence_id();
    auto skip = false;
    if (!list.empty()) {
        if (std::find(list.begin(), list.end(), myid) == list.end())
            skip = true;
    }

    std::cout << "cuda_timeit #" << myid << " " << func_name << ":" << lineno << " " << annotation << " x " << repeat
              << "  " << bytes << "(bytes) "  << flops << "(flops)" <<  (skip ? " ... SKIPPED":"") << std::endl;
    if (skip) return 0;

    AutoCUDATimer gpu_timers;

    for(int i = 0; i < repeat; i++) {
        gpu_timers.timers.emplace_back(myid, func_name, lineno, annotation, bytes, flops);
        auto& timer = gpu_timers.timers.back();
        timer.Start();
        func(i, timer.postscript);
        timer.Stop();
    }
    return gpu_timers.finish();
}

std::stringstream& cuda_timeit_last_ps() {
    auto& timer = gpu_timers.timers.back();
    return timer.postscript;
}

#define CUDA_CALL(...) if ( __VA_ARGS__ != cudaSuccess) {\
    auto err = cudaGetLastError(); \
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << "  Error  " << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << std::endl; \
    std::cout << ss.str(); \
    throw std::runtime_error(ss.str()); \
}

std::ostream& operator<<(std::ostream& os, const dim3& d) {
    os << "dim3{" << d.x << ", " << d.y << ", " << d.z << "}";
    return os;
}

struct thread_info {
    int64_t blk_x = -1;
    int64_t blk_y = -1;
    uint64_t thr_x0 = 0;
    uint64_t thr_y0 = 0;
    uint64_t smid = 0;
    uint64_t warpid = 0;
    uint64_t clk_start = 0;
    uint64_t clk_dur = 0;

    static void dump(thread_info * tinfo_base, int tnum, double avg_dur_ns, size_t thr_bytes = 0, size_t thr_ops = 0) {
        auto* ptinfo = tinfo_base;
        // calibrate all clocks from different SMs
        std::vector<uint64_t> clock_min(128, std::numeric_limits<uint64_t>::max());
        std::vector<uint64_t> clock_max(128, std::numeric_limits<uint64_t>::min());
        std::vector<uint64_t> thread_cnt(128, 0);
        uint64_t sm_cnt = 0;
        for(int i = 0; i < tnum; i++, ptinfo ++) {
            sm_cnt = std::max(sm_cnt, ptinfo->smid+1);
            if (ptinfo->clk_dur > 0) {
                clock_min[ptinfo->smid] = std::min(clock_min[ptinfo->smid], ptinfo->clk_start);
                clock_max[ptinfo->smid] = std::max(clock_max[ptinfo->smid], ptinfo->clk_start + ptinfo->clk_dur);
                thread_cnt[ptinfo->smid] ++;
            }
            //ECOUT("SM ", smid , blk_x, ",", blk_y, " clock_start= ", clk_start, ", ", clk_dur);
        }
        ECOUT2("==========SM statistics:==========");
        uint64_t clock_overall_dur = std::numeric_limits<uint64_t>::min();
        for(uint64_t smid = 0; smid < sm_cnt; smid++) {
            auto clock_dur = clock_max[smid] - clock_min[smid];
            clock_overall_dur = std::max(clock_overall_dur, clock_dur);
            ECOUT2("SM ", std::fixed, std::setw(3), smid , " clock: ", clock_min[smid], " + ", clock_dur,
                " ", thread_cnt[smid], "(threads)",
                " ", (thread_cnt[smid]*thr_bytes)/clock_dur, " (bytes/cycle)",
                " ", (thread_cnt[smid]*thr_ops)/clock_dur, " (ops/cycle)"
                );
        }
        ECOUT2(" clock_overall_dur = ", clock_overall_dur);
        ECOUT2(" avg_dur_ns = ", avg_dur_ns, "(ns)");
        ECOUT2(" GPU_avg_frequency = ", clock_overall_dur / avg_dur_ns, " (GHz)");
        if (thr_ops)
            ECOUT2(" average compute   = ", (tnum * thr_ops)/avg_dur_ns, " (GOP/second)");
        if (thr_bytes)
            ECOUT2(" average Bandwidth = ", (tnum * thr_bytes)/avg_dur_ns, " (GB/second)");

        ptinfo = tinfo_base;
        for(int i = 0; i < tnum; i++, ptinfo ++) {
            ptinfo->clk_start -= clock_min[ptinfo->smid];
        }

        // 32 threads from same warp (if block-size in X direction is larger than 32)
        struct warp_info : public thread_info {
            uint64_t thr_cnt = 0;
            size_t thr_ops;
            size_t thr_bytes;
            ChromeTraceDumpper& dumpper;
            warp_info(ChromeTraceDumpper& dumpper, size_t thr_ops, size_t thr_bytes) : dumpper(dumpper), thr_ops(thr_ops), thr_bytes(thr_bytes) {}
            void dump() {
                if (thr_cnt > 0) {
                    std::stringstream ss;
                    ss << "block(" << blk_x <<"," << blk_y << ")";
                    dumpper.phX(ss.str(), "", 
                        std::string("SM_") + std::to_string(thread_info::smid),
                        std::string("warp_") + std::to_string(thread_info::warpid),
                        clk_start, clk_dur,
                        {
                            {"thr_x0",std::to_string(thr_x0) + "+" + std::to_string(thr_cnt)},
                            {"thr_y0",std::to_string(thr_y0)},
                            {"ops/cycle", std::to_string(double(thr_ops * thr_cnt)/clk_dur)},
                            {"bytes/cycle", std::to_string(double(thr_bytes * thr_cnt)/clk_dur)}
                        });
                }
            }
        };

        ChromeTraceDumpper dumpper("ct.json");
        warp_info warp(dumpper, thr_ops, thr_bytes);
        ptinfo = tinfo_base;
        for (int n = 0; n < tnum; n++, ptinfo ++) {
            if (warp.blk_x == ptinfo->blk_x
                && warp.blk_y == ptinfo->blk_y
                && warp.smid == ptinfo->smid
                && warp.clk_start == ptinfo->clk_start && warp.clk_dur == ptinfo->clk_dur
                && warp.warpid == ptinfo->warpid
                && warp.thr_y0 == ptinfo->thr_y0) {
                warp.thr_cnt++;
                //if (warp.thr_cnt == WRAP_SIZE) {
                //    warp.dump(dumpper);
                //    warp.thr_cnt = 0;
                //}
            } else {
                warp.dump();
                memcpy(&warp, ptinfo, sizeof(*ptinfo));
                warp.thr_cnt = 1;
            }
        }
        warp.dump();
    }
};
