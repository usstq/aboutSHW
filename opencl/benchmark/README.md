# benchmark

Micro architectural benchmark based on ngen(https://github.com/oneapi-src/oneDNN/tree/main/src/gpu/intel/jit/ngen).

```bash
$ g++ -o bench -std=c++17 bench.cpp -lOpenCL && ./bench
Output from Arc 770:
...
Testing FLOAT MOV:
idx:0, cost: 43.2ms, 18441677524131985907 cycles, CPI: 18441677524132.0, speed: 9474.5 GFLOPs/s
idx:1, cost: 42.9ms, 3227568 cycles, CPI: 3.2, speed: 9554.8 GFLOPs/s
idx:2, cost: 42.9ms, 3226499 cycles, CPI: 3.2, speed: 9558.9 GFLOPs/s
idx:3, cost: 42.9ms, 3228910 cycles, CPI: 3.2, speed: 9558.9 GFLOPs/s
idx:4, cost: 42.8ms, 3225909 cycles, CPI: 3.2, speed: 9559.0 GFLOPs/s
idx:5, cost: 42.9ms, 3228326 cycles, CPI: 3.2, speed: 9558.7 GFLOPs/s
idx:6, cost: 42.9ms, 3228144 cycles, CPI: 3.2, speed: 9558.9 GFLOPs/s
idx:7, cost: 42.8ms, 3225362 cycles, CPI: 3.2, speed: 9558.9 GFLOPs/s
idx:8, cost: 42.8ms, 3226040 cycles, CPI: 3.2, speed: 9559.0 GFLOPs/s
idx:9, cost: 42.9ms, 3226350 cycles, CPI: 3.2, speed: 9558.9 GFLOPs/s
Testing FLOAT ADD:
idx:0, cost: 42.9ms, 3240709 cycles, CPI: 3.2, speed: 9547.8 GFLOPs/s
idx:1, cost: 42.9ms, 3242885 cycles, CPI: 3.2, speed: 9551.8 GFLOPs/s
idx:2, cost: 42.9ms, 3242721 cycles, CPI: 3.2, speed: 9551.7 GFLOPs/s
idx:3, cost: 42.9ms, 3241687 cycles, CPI: 3.2, speed: 9551.5 GFLOPs/s
idx:4, cost: 42.9ms, 3240314 cycles, CPI: 3.2, speed: 9551.7 GFLOPs/s
idx:5, cost: 42.9ms, 3240114 cycles, CPI: 3.2, speed: 9551.5 GFLOPs/s
idx:6, cost: 42.9ms, 3240440 cycles, CPI: 3.2, speed: 9551.7 GFLOPs/s
idx:7, cost: 42.9ms, 3242142 cycles, CPI: 3.2, speed: 9551.8 GFLOPs/s
idx:8, cost: 42.9ms, 3242151 cycles, CPI: 3.2, speed: 9551.9 GFLOPs/s
idx:9, cost: 42.9ms, 3243332 cycles, CPI: 3.2, speed: 9551.6 GFLOPs/s
Testing FLOAT MUL:
idx:0, cost: 42.9ms, 3240809 cycles, CPI: 3.2, speed: 9551.5 GFLOPs/s
idx:1, cost: 42.9ms, 3239683 cycles, CPI: 3.2, speed: 9551.8 GFLOPs/s
idx:2, cost: 42.9ms, 3240989 cycles, CPI: 3.2, speed: 9551.8 GFLOPs/s
idx:3, cost: 42.9ms, 3242748 cycles, CPI: 3.2, speed: 9551.6 GFLOPs/s
idx:4, cost: 42.9ms, 3241677 cycles, CPI: 3.2, speed: 9551.9 GFLOPs/s
idx:5, cost: 42.9ms, 3240841 cycles, CPI: 3.2, speed: 9551.6 GFLOPs/s
idx:6, cost: 42.9ms, 3241247 cycles, CPI: 3.2, speed: 9551.6 GFLOPs/s
idx:7, cost: 42.9ms, 3239542 cycles, CPI: 3.2, speed: 9551.7 GFLOPs/s
idx:8, cost: 42.9ms, 3242277 cycles, CPI: 3.2, speed: 9551.5 GFLOPs/s
idx:9, cost: 42.9ms, 3241589 cycles, CPI: 3.2, speed: 9551.7 GFLOPs/s
Testing FLOAT MAD:
idx:0, cost: 44.4ms, 4229114 cycles, CPI: 4.2, speed: 18468.3 GFLOPs/s
idx:1, cost: 44.4ms, 4232218 cycles, CPI: 4.2, speed: 18468.0 GFLOPs/s
idx:2, cost: 44.4ms, 4226827 cycles, CPI: 4.2, speed: 18468.0 GFLOPs/s
idx:3, cost: 44.4ms, 4215047 cycles, CPI: 4.2, speed: 18468.2 GFLOPs/s
idx:4, cost: 44.4ms, 4222687 cycles, CPI: 4.2, speed: 18468.2 GFLOPs/s
idx:5, cost: 44.4ms, 4222508 cycles, CPI: 4.2, speed: 18468.0 GFLOPs/s
idx:6, cost: 44.4ms, 4222471 cycles, CPI: 4.2, speed: 18468.4 GFLOPs/s
idx:7, cost: 44.4ms, 4222493 cycles, CPI: 4.2, speed: 18468.2 GFLOPs/s
idx:8, cost: 44.4ms, 4222781 cycles, CPI: 4.2, speed: 18468.3 GFLOPs/s
idx:9, cost: 44.4ms, 4222608 cycles, CPI: 4.2, speed: 18467.8 GFLOPs/s
Testing HALF MAD:
idx:0, cost: 44.4ms, 4231817 cycles, CPI: 4.2, speed: 36936.1 GFLOPs/s
idx:1, cost: 44.4ms, 4218582 cycles, CPI: 4.2, speed: 36935.4 GFLOPs/s
idx:2, cost: 44.4ms, 4232167 cycles, CPI: 4.2, speed: 36936.3 GFLOPs/s
idx:3, cost: 44.4ms, 4226200 cycles, CPI: 4.2, speed: 36935.6 GFLOPs/s
idx:4, cost: 44.4ms, 4219901 cycles, CPI: 4.2, speed: 36936.0 GFLOPs/s
idx:5, cost: 44.4ms, 4232411 cycles, CPI: 4.2, speed: 36936.6 GFLOPs/s
idx:6, cost: 44.4ms, 4221227 cycles, CPI: 4.2, speed: 36936.3 GFLOPs/s
idx:7, cost: 44.4ms, 4233625 cycles, CPI: 4.2, speed: 36936.3 GFLOPs/s
idx:8, cost: 44.4ms, 4231817 cycles, CPI: 4.2, speed: 36936.4 GFLOPs/s
idx:9, cost: 44.4ms, 4225690 cycles, CPI: 4.2, speed: 36935.9 GFLOPs/s
Testing HALF DPAS8x1:
idx:0, cost: 38.2ms, 1329640 cycles, CPI: 16.6, speed: 27466.9 GFLOPs/s
idx:1, cost: 38.2ms, 1328669 cycles, CPI: 16.6, speed: 27481.4 GFLOPs/s
idx:2, cost: 38.2ms, 1328974 cycles, CPI: 16.6, speed: 27481.3 GFLOPs/s
idx:3, cost: 38.2ms, 1362053 cycles, CPI: 17.0, speed: 27471.4 GFLOPs/s
idx:4, cost: 38.2ms, 1341621 cycles, CPI: 16.8, speed: 27466.1 GFLOPs/s
idx:5, cost: 38.2ms, 1350923 cycles, CPI: 16.9, speed: 27474.7 GFLOPs/s
idx:6, cost: 38.2ms, 1329241 cycles, CPI: 16.6, speed: 27481.2 GFLOPs/s
idx:7, cost: 38.2ms, 1329159 cycles, CPI: 16.6, speed: 27481.3 GFLOPs/s
idx:8, cost: 38.2ms, 1336289 cycles, CPI: 16.7, speed: 27479.1 GFLOPs/s
idx:9, cost: 38.2ms, 1329752 cycles, CPI: 16.6, speed: 27481.1 GFLOPs/s
Testing HALF DPAS8x8:
idx:0, cost: 64.2ms, 1803756 cycles, CPI: 22.5, speed: 130736.9 GFLOPs/s
idx:1, cost: 64.2ms, 1814306 cycles, CPI: 22.7, speed: 130728.2 GFLOPs/s
idx:2, cost: 64.2ms, 1807967 cycles, CPI: 22.6, speed: 130733.7 GFLOPs/s
idx:3, cost: 64.2ms, 1808509 cycles, CPI: 22.6, speed: 130733.3 GFLOPs/s
idx:4, cost: 64.2ms, 1804913 cycles, CPI: 22.6, speed: 130735.8 GFLOPs/s
idx:5, cost: 64.2ms, 1806685 cycles, CPI: 22.6, speed: 130734.1 GFLOPs/s
idx:6, cost: 64.2ms, 1825236 cycles, CPI: 22.8, speed: 130718.4 GFLOPs/s
idx:7, cost: 64.2ms, 1806993 cycles, CPI: 22.6, speed: 130733.9 GFLOPs/s
idx:8, cost: 64.2ms, 1804278 cycles, CPI: 22.6, speed: 130736.2 GFLOPs/s
idx:9, cost: 64.2ms, 1829811 cycles, CPI: 22.9, speed: 130714.4 GFLOPs/s
Testing MEM_L1:
idx:0, cost: 169.7ms, 8002120 cycles, bytes/clk: 4.0, speed: 9654.4 GB/s
idx:1, cost: 170.4ms, 8323071 cycles, bytes/clk: 3.8, speed: 9614.0 GB/s
idx:2, cost: 170.4ms, 8004643 cycles, bytes/clk: 4.0, speed: 9615.2 GB/s
idx:3, cost: 169.7ms, 8004210 cycles, bytes/clk: 4.0, speed: 9655.3 GB/s
idx:4, cost: 169.7ms, 8004188 cycles, bytes/clk: 4.0, speed: 9654.6 GB/s
idx:5, cost: 169.7ms, 8322744 cycles, bytes/clk: 3.8, speed: 9657.2 GB/s
idx:6, cost: 168.8ms, 8004374 cycles, bytes/clk: 4.0, speed: 9708.7 GB/s
idx:7, cost: 169.7ms, 8333965 cycles, bytes/clk: 3.8, speed: 9655.2 GB/s
idx:8, cost: 170.4ms, 8004963 cycles, bytes/clk: 4.0, speed: 9615.0 GB/s
idx:9, cost: 169.7ms, 8322400 cycles, bytes/clk: 3.8, speed: 9654.7 GB/s
Testing MEM_SLM:
idx:0, cost: 108.8ms, 4640758 cycles, bytes/clk: 6.9, speed: 15064.6 GB/s
idx:1, cost: 108.5ms, 4640725 cycles, bytes/clk: 6.9, speed: 15097.5 GB/s
idx:2, cost: 109.3ms, 4799067 cycles, bytes/clk: 6.7, speed: 14984.7 GB/s
idx:3, cost: 109.0ms, 4799124 cycles, bytes/clk: 6.7, speed: 15027.3 GB/s
idx:4, cost: 109.9ms, 4799140 cycles, bytes/clk: 6.7, speed: 14909.9 GB/s
idx:5, cost: 109.1ms, 4799186 cycles, bytes/clk: 6.7, speed: 15018.8 GB/s
idx:6, cost: 108.9ms, 4799122 cycles, bytes/clk: 6.7, speed: 15040.5 GB/s
idx:7, cost: 109.4ms, 4800709 cycles, bytes/clk: 6.7, speed: 14977.2 GB/s
idx:8, cost: 109.0ms, 4799157 cycles, bytes/clk: 6.7, speed: 15027.0 GB/s
idx:9, cost: 108.8ms, 4799140 cycles, bytes/clk: 6.7, speed: 15063.9 GB/s
```

### References

Hardware:
 - Intel-Graphics ISA https://www.intel.com/content/dam/develop/external/us/en/documents/micro2015-isa-igc-tutorial.pdf
 - Intel-Graphics-Compiler Virtual-ISA: https://github.com/intel/intel-graphics-compiler/tree/master/documentation/visa/instructions
 - ARC770 spec: https://www.techpowerup.com/gpu-specs/arc-a770.c3914
 - Detailed ARC GPU doc: https://www.x.org/docs/intel/ACM/intel-gfx-prm-osrc-acm-vol09-renderengine.pdf
