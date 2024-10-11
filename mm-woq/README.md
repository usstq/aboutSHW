# Prototyping Weight-only quantization of FC (with perf-tunning)

## [test-cpu.cpp](./test-cpu.cpp)
```bash
# on 8x P-Cores of RPL with 97GB/s DDR bandwidth
#  Intel(R) Core(TM) i9-14900K
$ g++ -fopenmp -O2 -march=native ./test.cpp
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

