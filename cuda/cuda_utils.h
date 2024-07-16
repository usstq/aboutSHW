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

#define ASSERT(cond) if (!(cond)) {\
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << " " << #cond << " failed!"; \
    throw std::runtime_error(ss.str()); \
}

template<typename T>
struct tensor2D {
    int shape[2];
    int size;
    T* ptr_host;
    T* ptr_dev;
    bool on_device;
    tensor2D(int d0, int d1, bool _on_device = false) {
        size = d0 * d1;
        shape[0] = d0;
        shape[1] = d1;
        ptr_host = new T[size];
        ASSERT(cudaMalloc((void**)&ptr_dev, size * sizeof(T)) == cudaSuccess);
        on_device = _on_device;
    }
    ~tensor2D() {
        cudaFree(ptr_dev);
        delete[] ptr_host;
    }
    void to_host(bool do_copy = true) {
        if (on_device) {
            if (do_copy) {
                // maybe we need wait for device code to finish the work here?
                ASSERT(cudaDeviceSynchronize() == cudaSuccess);
                ASSERT(cudaMemcpy(ptr_host, ptr_dev, size * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
            }
            on_device = false;
        }
    }
    void to_dev(bool do_copy = true) {
        if (!on_device) {
            if (do_copy) {
                ASSERT(cudaMemcpy(ptr_dev, ptr_host, size * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
            }
            on_device = true;
        }
    }
    template<typename DT>
    friend std::ostream& operator<<(std::ostream& os, const tensor2D<DT>& dt);
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const tensor2D<T>& t) {
    const char* Tname = "?";
    if (std::is_same<T, int>::value) Tname = "int";
    if (std::is_same<T, float>::value) Tname = "float";
    os << "tensor2D<" << Tname << ">[" << t.shape[0] << "," << t.shape[1] << "]" << (t.on_device ? "@device" : "@cpu");
    if (!t.on_device) {
        os << " values={";
        int i = 0;
        const char* sep = "";
        for (; i < 8 && i < t.size; i++) {
            os << sep << t.ptr_host[i];
            sep = ",";
        }
        if (i < t.size)
            os << "...";
        os << "}";
    }
    return os;
}

// https://stackoverflow.com/questions/7876624/timing-cuda-operations
struct CUDATimer {
    std::chrono::system_clock::time_point host_start;
    std::chrono::system_clock::time_point host_stop;
    cudaEvent_t start;
    cudaEvent_t stop;
    const char* name;
    int id;
    const char* annotation;
    uint64_t bytes;
    uint64_t flops;

    CUDATimer(const char* name = "", int id = 0, const char * annotation="",
              uint64_t bytes = 0, uint64_t flops = 0.0f)
        : name(name), id(id), annotation(annotation), bytes(bytes), flops(flops) {
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

    void finish() {
        float elapsed_since_last_stop;
        cudaEvent_t tlast = nullptr;
        for (auto& t : timers) {
            if (!tlast)
                tlast = t.start;
            cudaEventElapsedTime(&elapsed_since_last_stop, tlast, t.start);
            elapsed_since_last_stop *= 1e-3; // in seconds
            auto host_dur = std::chrono::duration<double>(t.host_stop - t.host_start).count();
            auto dev_dur = t.Elapsed() * 1e-3; // in seconds

            std::cout << "\033[1;96m [AutoCUDATimer] @host " << prettyTime(host_dur)
                << " | @device (+" << prettyTime(elapsed_since_last_stop) << ") " << prettyTime(dev_dur);
            if (t.bytes)
                std::cout << " " << std::fixed << std::setw(7) << std::setprecision(3)
                          << t.bytes * 1e-9 / dev_dur << " GB/s";
            if (t.flops)
                std::cout << " " << std::fixed << std::setw(7) << std::setprecision(3)
                          << t.flops * 1e-9 / dev_dur << " GFLOPS/s";
            std::cout << "\t  " << t.annotation << " (" << t.name << ":" << t.id << ") \033[0m" << std::endl;
            tlast = t.stop;
        }
        timers.clear();
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

#define TIMEIT_BEGIN(...) gpu_timers.timers.emplace_back(__func__, __LINE__, __VA_ARGS__); gpu_timers.timers.back().Start();
#define TIMEIT_END() gpu_timers.timers.back().Stop();
#define TIMEIT_FINISH() gpu_timers.finish();

#define TIMEIT(...) do { \
    gpu_timers.timers.emplace_back(__func__, __LINE__); \
    gpu_timers.timers.back().Start(); \
    __VA_ARGS__ \
    gpu_timers.timers.back().Stop(); \
} while(0)

#define CUDA_CALL(...) if ( __VA_ARGS__ != cudaSuccess) {\
    auto err = cudaGetLastError(); \
    std::stringstream ss; \
    ss << __FILE__ << ":" << __LINE__ << "  Error  " << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << std::endl; \
    std::cout << ss.str(); \
    throw std::runtime_error(ss.str()); \
}