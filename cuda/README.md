# CUDA programming

as shown [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#the-cuda-compilation-trajectory), nvcc needs local Host's C/C++ compiler to work.
On my windows, this requires adding following path containing `cl.exe` (VC's command-line) into environment variable Path:

 - C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64

after that we can compile cuda source code directly using nvcc (since we are focusing on experiments, not building a big project, no make system is used).
```bash
nvcc ./kernel.cu
```

## nvidia GPU code-name & series
https://wiki.gentoo.org/wiki/NVIDIA
 - INDEPENDENT THREAD SCHEDULING since [Volta](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

## CUDA programming Introductions
[CUDA and Application to Task-Based Programming | Eurographics'2021 Tutorial]
(https://cuda-tutorial.github.io/) videos: [part1](https://www.youtube.com/watch?v=6kT7vVHCZIc) [part2](https://www.youtube.com/watch?v=mrDWmnXC5Ck).

## Hardware
whitepapers from nvidia:
 - [pascal-architecture-whitepaper.pdf](https://images.nvidia.cn/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
 - [volta-architecture-whitepaper.pdf](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

[Pascal Architecture1](https://www.anandtech.com/show/10325/the-nvidia-geforce-gtx-1080-and-1070-founders-edition-review/4), 
[Pascal Architecture2](https://www.anandtech.com/show/11172/nvidia-unveils-geforce-gtx-1080-ti-next-week-699)

CUDA core is just scalar ALU; GPU executes multiple threads in time-division multiplexing fasion, but unlike CPU:
 - context-switching of HW threads is designed to be very fast (on cycle level);
 - HW-threads keep all states in register files (with stack in mem?), never swap to memory;
 - HW-scheduler is invented to switching context (unlike OS running on CPU using SW-scheduler);
 - thus number of concurrent threads supported are limited by register file size & HW-scheduler capability;
 - HW-threads are grouped in unit of 32 into Warp, to get higher ALU throughput.
 - number of wraps/HW-threads supported is far more than number of CUDA cores(to hide mem-latency).
 
 **for example on my GTX-1070, 2048 threads (or 64 warps) per SM are supported,
 but each SM has only 4 Warp schedulers, 16x oversubscride.**

## GPU disassembly code
use `CUDA C++` in https://godbolt.org/ to inspect the generated PTX & SASS code.

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

## GPU Memory hierarchy
[GPU cache is designed with significant different purpose from CPU's](https://www.rastergrid.com/blog/gpu-tech/2021/01/understanding-gpu-caches/)

**Cache Coherency**
> As a result, GPU caches are usually incoherent, and require explicit flushing
> and/or invalidation of the caches in order to recohere (i.e. to have a coherent
> view of the data between the GPU cores and/or other devices).

**Per Core Instruction Cache**
> One thing to keep in mind from performance point of view is that
> on GPUs an instruction cache miss can potentially stall thousands
> of threads instead of just one, as in the case of CPUs, so generally it¡¯s highly recommended
> for shader/kernel code to be small enough to completely fit into this cache.

**Per Core Data Cache**
> "Thus reuse of cached data on GPUs usually does not happen in the time domain
>  as in case of CPUs (i.e. subsequent instructions of the same thread accessing
>  the same data multiple times), but rather in the spatial domain (i.e. instructions
>  of different threads on the same core accessing the same data).

> Accordingly, using memory as temporary storage and relying on caching for fast
> repeated access is not a good idea on GPUs, unlike on CPUs. However, the larger
> register space and the availability of shared memory make up for that in practice."

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory

## Micro-architecture level optimization:

https://forums.developer.nvidia.com/t/instruction-latency/3579/13
 - instruction latency hidding

http://bebop.cs.berkeley.edu/pubs/volkov2008-benchmarking.pdf

 