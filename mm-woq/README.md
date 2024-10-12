# Prototyping Weight-only quantization of FC (with perf-tunning)

## [test-cpu.cpp](./test-cpu.cpp)

Using `SM=0 CLC=0` config, no cache miss happens and we can measure the loop-kernel latency in clock cycles.
based on the memory bind-width CPU frequency we can see how much cycles is allowed for CPU to decompress weights.

On target machine, MLC shows 97GB/s peak DDR bandwidth, so we can estimate the throughput in terms of cycles per loop-kernel, using 8x P-cores we have 12.125GB/s bandwidth for each P-core

| kernel name | data-load-size(bytes)| data-throuput <br>(ns) | data-throuput<br> (cycles @5.0GHz~@5.7GHz) | compute-cycles | comment |
|-------------|----------------------|-------------|-------------|-------------|-------------|
|mm_avx2_f32_reg_x1s | 8xf32=32  | 0.32 | 13~15     | 4   | memory-bound  |
| mm_avx2_i8_reg_x1s | 8xi8=8    | 0.08 | 3.29~3.76 | 4   | compute-bound |
| mm_avx2_i8_reg_x4s | 4x8xi8=32 | 0.32 | 13~15     | 4.3 | compute-bound |
| mm_avx2_i4_reg_x4s | 4x8xi4=16 | 0.16 | 6.5~7.5   | 5   | memory-bound  |


Even in the case where analysis shows memory-bound, we still need to take extra care to make sure DDR-access is really in parallel with ALU computation by:
 - make each core access (physical) contiguous data, to best use HW-prefetcher and avoid cache-coherent protocol overhead
 - prefetching ahead at by 4KB to help HW-prefetch when crossing page-boundary

```bash
# on 8x P-Cores of RPL with 97GB/s DDR bandwidth
#  Intel(R) Core(TM) i9-14900K
$ g++ -fopenmp -O2 -march=native ./test.cpp

# check Cycles per loop-kernel to see compute-cycles
$ SM=0 CLC=0  numactl -C0,2,4,6,8,10,12,14 ./a.out
============== round 1 ===================
      8306345,        14687725,  [mm_avx2_f32_reg_x1] 1456.722 us CPU:5.70(GHz) CPI:0.57  CPK:4.0x2097152 MemBW: 368.55(GB/s)
      8268451,        16785311,  [mm_avx2_f32_reg_x1s] 1452.234 us CPU:5.69(GHz) CPI:0.49 CPK:3.9x2097152 MemBW: 369.69(GB/s)   + exploit HW-pefetcher
      8314721,        18881865,  [mm_avx2_i8_reg_x1s] 1458.295 us CPU:5.70(GHz) CPI:0.44  CPK:4.0x2097152 MemBW: 92.04(GB/s)    + Weight-Compressed INT8
      2263130,        11538131,  [mm_avx2_i8_x4s<0>] 396.880 us CPU:5.70(GHz) CPI:0.20    CPK:4.3x524288 MemBW: 338.18(GB/s)    + register-blocking
      2280403,        12062454,  [mm_avx2_i8_x4s<4096>] 399.937 us CPU:5.70(GHz) CPI:0.19 CPK:4.3x524288 MemBW: 335.60(GB/s)    + SW-prefetch across page boundary 
      2636302,        12062461,  [mm_avx2_i4_x4s<4096>] 463.847 us CPU:5.68(GHz) CPI:0.22 CPK:5.0x524288 MemBW: 144.68(GB/s)    + Weight-Compressed INT4

# 
$ SM=1 CLC=1  numactl -C0,2,4,6,8,10,12,14 ./a.out
ENV:     SM = 1 
ENV:     CLC = 1 
ENV:     M = 1 (default)
ENV:     N = 32768 (default)
ENV:     K = 4096 (default)
 +0.000 ms test.cpp:349 main()  nthr=8
 +143.313 ms test.cpp:362 main()  ====================================
 +0.014 ms test.cpp:363 main()     A matrix size = 0.016384 MB(1,000,000 bytes)
 +0.003 ms test.cpp:364 main()     C matrix size = 0.131072 MB(1,000,000 bytes)
 +0.002 ms test.cpp:365 main()  Bf32 matrix size = 536.871 MB(1,000,000 bytes)
 +0.001 ms test.cpp:366 main()   Bi8 matrix size = 134.218 MB(1,000,000 bytes)
 +0.002 ms test.cpp:367 main()   Bi4 matrix size = 67.1089 MB(1,000,000 bytes)
#0:HW_CPU_CYCLES, HW_INSTRUCTIONS, 
============== round 1 ===================
    118403706,        14849306,  [mm_avx2_f32_reg_x1] 20839.634 us CPU:5.68(GHz) CPI:7.97 CPK:56.5x2097152 MemBW: 25.76(GB/s)
     31509647,        16785196,  [mm_avx2_f32_reg_x1s] 5605.587 us CPU:5.62(GHz) CPI:1.88 CPK:15.0x2097152 MemBW: 95.77(GB/s)   + exploit HW-pefetcher
     10587958,        18881900,  [mm_avx2_i8_reg_x1s] 1926.707 us CPU:5.50(GHz) CPI:0.56 CPK:5.0x2097152 MemBW: 69.66(GB/s)     + Weight-Compressed INT8
      9105117,        11537966,  [mm_avx2_i8_x4s<0>] 1669.989 us CPU:5.45(GHz) CPI:0.79 CPK:17.4x524288 MemBW: 80.37(GB/s)      + register-blocking
      7515133,        12062255,  [mm_avx2_i8_x4s<4096>] 1390.277 us CPU:5.41(GHz) CPI:0.62 CPK:14.3x524288 MemBW: 96.54(GB/s)   + SW-prefetch across page boundary 
      3699274,        12062250,  [mm_avx2_i4_x4s<4096>] 726.595 us CPU:5.09(GHz) CPI:0.31 CPK:7.1x524288 MemBW: 92.36(GB/s)     + Weight-Compressed INT4
```

