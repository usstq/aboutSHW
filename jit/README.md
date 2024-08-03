# JIT

[./jit.h](./jit.h) : a jit base class based on [xbayk](https://github.com/herumi/xbyak), providing some additional features:
 - `JIT_DEBUG=name` or `JIT_DEBUG=*` will show disassembly code of jit and triger `INT3` to cause GDB to stop for runtime debuging.
 - runtime logging infrastructure which can log`zmm` registers and show after kernel invokation, very helpful for debugging.

# Compute bound Analysis
[./test-cpi.cpp](./test-cpi.cpp) using jit to test instruction/instruction-sequence throughput using PMU.

This is measuring actual throughput rather than prediction done by static code analyzer like [uiCA](https://uica.uops.info/), this is helpful to understand & optimize computational bound.

## Instruction throughput 

As defined in [uops.info](https://www.uops.info/background.html), we can measure instruction throughput by profiling series of independent instructions of the same kind:

```bash
#0:HW_CPU_CYCLES, HW_INSTRUCTIONS, 
       400146,          820058,  [          vaddps]   210.544  us CPU:1.90(GHz) CPI:0.49 CPK:4.0
       400304,          820058,  [             fma]   210.632  us CPU:1.90(GHz) CPI:0.49 CPK:4.0
      1649953,          820059,  [       vdpbf16ps]   871.577  us CPU:1.89(GHz) CPI:2.01 CPK:16.5
      6672336,          820067,  [       tileloadd]   3522.035 us CPU:1.89(GHz) CPI:8.14 CPK:66.7
      1668442,          220065,  [ tileloadd.compact] 880.871  us CPU:1.89(GHz) CPI:7.58 CPK:16.7
      3492576,          220072,  [ tileloadd.stride0] 1845.048 us CPU:1.89(GHz) CPI:15.8 CPK:34.9
      1668329,          220131,  [ tileloadd.stride1] 928.314  us CPU:1.80(GHz) CPI:7.58 CPK:16.7
      6416448,          420236,  [       tdpbf16ps]   3389.623 us CPU:1.89(GHz) CPI:15.2 CPK:64.2
      7506139,          820074,  [  amx2x2.stride0]   3961.088 us CPU:1.89(GHz) CPI:9.15 CPK:75.1
      6411387,          820134,  [  amx2x2.stride1]   3384.627 us CPU:1.89(GHz) CPI:7.82 CPK:64.1
```


we can see the measurements are consistent with [official doc](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

 - `vaddps`/`vfmadd132ps` has 0.5 throughput
 - `vdpbf16ps` has 2.0 throughput
 - `tileloadd` has 8.0 throughput only if L1-cache bank-conflict can be avoided (by proper stride), check tileloadd.compact & tileloadd.stride1 for details.
 - `tdpbf16ps` has 16.0 throughput

Also we can measure instruction-sequence throughput like 2x2 tiles AMX-BF16 TMUL:

```c++
    tileloadd(tmm4, ptr[tile_A0_addr + tile_A_stride]); // Load A0
    tileloadd(tmm6, ptr[rax + tile_stride + 1024*2]);   // Load B0
    tdpbf16ps(tmm0, tmm4, tmm6);                        // TMUL A0 @ B0

    tileloadd(tmm5, ptr[tile_A1_addr + tile_A_stride]); // Load A1
    tdpbf16ps(tmm2, tmm5, tmm6);                        // TMUL A1 @ B0

    tileloadd(tmm7, ptr[rax + tile_stride + 1024*3]);   // Load B1
    tdpbf16ps(tmm1, tmm4, tmm7);                        // TMUL A0 @ B1
    tdpbf16ps(tmm3, tmm5, tmm7);                        // TMUL A1 @ B1
```

with proper strides, Load & TMUL can hide latency of each-other, so above sequence took `64 cycles` which is same as 4 TMUL sequence. this suggests that [register renaming](https://en.wikipedia.org/wiki/Register_renaming) works for Tile, so load & TMUL can run in parallel.

in `20.11.2 INTEL® AMX AND INTEL® AVX-512 INTERLEAVING (SW PIPELINING)` of `Intel architecture optimization reference manual`, it says data-independent AMX instructions & AVX instructions can be SW-pipelined to overcome the limited code window size of out-of-order CPU engine. `amx.avx.mix` is designed to test how well AVX512 & AMX instructions can be mixed.

we increase the number of different avx instructions inserted into the AMX-2x2-TMUL loop and see how well they can run in parallel:

we measure CPU cycles per iteration for following 3 loop:
 - 4 AMX-BF16 TMULs instrucions along. this measurement is always 64 cycles.
 - N avx instrucions running along. we use 4 independent avx instruction of same-kind, so this measurement is always `throughput * N`
 - 4 AMX-BF16 TMULs interleaved with N avx instructions, this is the most interesting measurement, call it `X`.

As `N` increasing, `X` would stable at 64 cycles for a while, and then increase along with N, the point of `N` where `X` starts to increase is the max number of avx instructions can be run in parallel or totally hidden by AMX TMUL.


| AVX512 instruction | Dispatch-Port | inflection point `N`(`N'`) | hidden latency `throughput * N` | Note |
| ------------------ | ------------- | -------------------  | ------------------------------- | ---- |
| `vpsllq`(avx512)   |   0           |    64 (64)           |     1 * 64                      | perfectly hidden|
| `vpxorq`(avx512)   |   0,5         |   128 (116)          |   0.5 * 128                     | perfectly hidden|
| `vdpbf16ps`(avx512) |  0,5         |    28 (28)           |     2 * 28                      | partially hidden|

> if we put all `N` avx instructions together, rather than evenly distributed between 4 TMULs, we may have less perfect hiding (depending on avx instruction type used), denoted as `N'`

