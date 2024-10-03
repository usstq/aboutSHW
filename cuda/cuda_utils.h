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
        std::cout << "\t clockRate              : " << dev_prop.clockRate << std::endl;
        std::cout << "\t multiProcessorCount    : " << dev_prop.multiProcessorCount << std::endl;
        std::cout << "\t kernelExecTimeoutEnabled: " << dev_prop.kernelExecTimeoutEnabled << std::endl;
        std::cout << "\t integrated         : " << dev_prop.integrated << std::endl;
        std::cout << "\t canMapHostMemory   : " << dev_prop.canMapHostMemory << std::endl;
        std::cout << "\t computeMode        : " << dev_prop.computeMode << std::endl;
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

template<typename T>
struct tensor2D {
    const int64_t shape[2];
    const int64_t size;
    const int64_t size_bytes;
    std::shared_ptr<T> ptr_host;
    std::shared_ptr<T> ptr_dev;
    bool on_device;
    tensor2D(int64_t d0, int64_t d1, bool _on_device = false) : shape{d0, d1}, size(d0 * d1), size_bytes(d0 * d1 * sizeof(T))  {
        ptr_host = std::shared_ptr<T>(new T[size], std::default_delete<T[]>());
        T * ptr;
        ASSERT(cudaMalloc((void**)&ptr, size * sizeof(T)) == cudaSuccess);
        ptr_dev = std::shared_ptr<T>(ptr, [](void * p){cudaFree(p);});
        on_device = _on_device;
    }

    void rand() {
        _tensor_rand<<< dim3(size), dim3(32) >>>(size, ptr_dev.get());
    }
    void zero() {
        if (on_device) {
            ASSERT(cudaMemset(ptr_dev.get(), 0, size * sizeof(T)) == cudaSuccess);
        } else {
            std::memset(ptr_host.get(), 0, size * sizeof(T));
        }
    }

    ~tensor2D() {
    }

    operator T*() const {
        if (on_device) return ptr_dev.get();
        return ptr_host.get();
    }

    void to_host(bool do_copy = true) {
        if (on_device) {
            if (do_copy) {
                // maybe we need wait for device code to finish the work here?
                ASSERT(cudaDeviceSynchronize() == cudaSuccess);
                ASSERT(cudaMemcpy(ptr_host.get(), ptr_dev.get(), size * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
            }
            on_device = false;
        }
    }
    void to_dev(bool do_copy = true) {
        if (!on_device) {
            if (do_copy) {
                ASSERT(cudaMemcpy(ptr_dev.get(), ptr_host.get(), size * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
            }
            on_device = true;
        }
    }

    bool operator==(const tensor2D<T>& rhs) const {
        if (shape[0] != rhs.shape[0]) return false;
        if (shape[1] != rhs.shape[1]) return false;

        assert(!on_device && !rhs.on_device);

        auto * p_lhr = ptr_host.get();
        auto * p_rhr = rhs.ptr_host.get();
        for(int64_t i = 0; i < shape[0]; i++) {
            for(int64_t j = 0; j < shape[1]; j++) {
                if (*p_lhr != *p_rhr) {
                    std::cout << "mismatch at (" << i << ", " << j << ")   " << *p_lhr << " != " << *p_rhr << std::endl;
                    return false;
                }
                p_lhr ++;
                p_rhr ++;
            }
        }
        return true;
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
            os << sep << t.ptr_host.get()[i];
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

template<class T>
struct CArray {
    const T* data;
    std::size_t N;
    CArray(const T* data, std::size_t N) : data(data), N(N) {}
};

template<class T>
CArray<T> carray(const T* data, std::size_t N) {
    return CArray<T>(data, N);
}

template<class T>
CArray<T> carray(const tensor2D<T>& t) {
    return CArray<T>(t.ptr_host.get(), t.size);
}

template<class T>
std::ostream& operator<<(std::ostream& os, const CArray<T>& arr) {
    T last_v = arr.data[0];
    int last_c = 1;
    const char * sep = "";
    //os << "CArray<" << typeid(arr.data[0]).name() << "," << arr.N << ">{";
    os << "{";
    for(int i = 1; i < arr.N; i++) {
        auto cur_v = arr.data[i];
        if (last_v != cur_v) {
            os << sep << last_v;
            if (last_c > 1) os << "...x" << last_c;
            last_v = cur_v;
            last_c = 1;
        } else {
            last_c ++;
        }
    }
    os << sep << last_v;
    if (last_c > 1) os << "...x" << last_c;
    os << "}";
    return os;
}


