# CUDA programming

as shown [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#the-cuda-compilation-trajectory), nvcc needs local Host's C/C++ compiler to work.
On my windows, this requires adding following path containing `cl.exe` (VC's command-line) into environment variable Path:

 - C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64

after that we can compile cuda source code directly using nvcc (since we are focusing on experiments, not building a big project, no make system is used).

```bash
mkdir build
# device code PTX & cubin are kept under build folder
nvcc --generate-line-info --keep --keep-dir build  kernel.cu
# disassemble device side code with line-info
nvdisasm --print-line-info-inline -c build\kernel.sm_52.cubin
# profile all metrics
nvprof.exe --metrics all .\a.exe
```
also we can use `CUDA C++` in https://godbolt.org/ to inspect the generated PTX & SASS code.

## nvidia GPU code-name & series
https://wiki.gentoo.org/wiki/NVIDIA
 - INDEPENDENT THREAD SCHEDULING since [Volta](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

## CUDA programming Introductions
[CUDA and Application to Task-Based Programming | Eurographics'2021 Tutorial]
(https://cuda-tutorial.github.io/) videos:
 - [part1](https://www.youtube.com/watch?v=6kT7vVHCZIc)
 - [part2](https://www.youtube.com/watch?v=mrDWmnXC5Ck).

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

Here are some [detailed HW-features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).

## GPU Memory hierarchy
[GPU cache is designed with significant different purpose from CPU's](https://www.rastergrid.com/blog/gpu-tech/2021/01/understanding-gpu-caches/)

**Cache Coherency**
> As a result, GPU caches are usually incoherent, and require explicit flushing
> and/or invalidation of the caches in order to recohere (i.e. to have a coherent
> view of the data between the GPU cores and/or other devices).

**Per Core Instruction Cache**
> One thing to keep in mind from performance point of view is that
> on GPUs an instruction cache miss can potentially stall thousands
> of threads instead of just one, as in the case of CPUs, so generally it��s highly recommended
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

## Occupancy

on my GTX-1070 card:
 - Maximum Threads per SM    : 2048 (64 Wraps)
 - Maximum Blocks per SM     : 32
 - Maximum Threads per Block : 1024

Thus, to get full occupancy, we need : `2048(threads) = num_blocks_per_SM * block_size`, with `num_blocks_per_SM <= 32`, suppose registers/share_mem are not bottleneck, we can archieve 100% occupancy only in following cases:
 - num_blocks_per_SM = 1  : `block_size = 2048`
 - num_blocks_per_SM = 2  : `block_size = 1024`
 - num_blocks_per_SM = 4  : `block_size = 512`
 - num_blocks_per_SM = 8  : `block_size = 256`
 - num_blocks_per_SM = 16 : `block_size = 128`
 - num_blocks_per_SM = 32 : `block_size = 64`

Other limitations are:
 - Shared memory per Block    : 48 KiB
 - Shared memory per SM       : 96 KiB
 - Registers per Block        : 65536
 - Registers per SM           : 65536

HW level:
 - totally 16 SMs
 - Warp schedulers per SM    : 4 
    - only 128 threads can run in-parallel in single cycle
    - if 100% Occupancy is archieved, each thread can only run on 1/16 cycles (over-subscribed)
    - over-subscription helps neighbouring threads to run in pipeline for hiding memory access latency:
      threads in wrap0 issue 32 loads, then it blocks on data to be ready to access, then
      next threads in wrap1 do the same thing, until the data is ready, wrap0 is waken to
      do computations. thus memory-access latency is hidden by this over-subscription.

 - each thread has one CUDA core, and each CUDA core provide 2-FLOPS per cycle (with FMADD)
 - CUDA core clock runs between `600MHz ~ 1.645MHz`
 - thus single precision FLOP/s at different GPU frequency:
  `FLOP/s = 16(SMs) * 128(CUDA-cores) * 2(FLOPS) * GPU_freq = 4096 * GPU_freq`
   @ `1.645 (GHz)` `FLOP/s = 6.7 (TeraFLOP/S)`
   @ `1.4   (GHz)` `FLOP/s = 5.7 (TeraFLOP/S)`
   @ `0.76  (GHz)` `FLOP/s = 3.1 (TeraFLOP/S)`
 - also all 2048 CUDA cores must be working togeteher to reach this peak FLOP/s,
   single CUDA core only has `2(FLOPS) * GPU_freq` 

## GPU frequency throttling

All Compute Instances on a GPU share the same clock frequencies. To get consistent metric values with multi-pass collection, it is recommended to lock the GPU clocks during the profiling session. CLI tool nvidia-smi can be used to configure a fixed frequency for the whole GPU by calling `nvidia-smi --lock-gpu-clocks=tdp,tdp`. This sets the GPU clocks to the base TDP frequency until you reset the clocks by calling `nvidia-smi --reset-gpu-clocks`.
