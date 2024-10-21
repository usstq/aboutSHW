#!/usr/bin/python3
import cl
import numpy as np
import os

print(dir(cl))

cl.profiling(True)

src = '''
    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void max_fma(__global float * x, int K) {
        //int i = get_global_linear_id();
        float c[FMACNT];
        for(int j = 0; j < FMACNT; j++) c[j] = j;
        float a = get_local_id(0);
        float b = get_local_id(1);
        for(int k = 0; k < K; k += FMACNT*UNROLL) {
            // following loop will be unrolled
            for(int unroll = 0; unroll < UNROLL; unroll ++)
                for(int j = 0; j < FMACNT; j++)
                    c[j] = fma(a, b, c[j]);
        }

        // prevent optimization
        float sum_c = 0;
        for(int j = 0; j < FMACNT; j++) sum_c += c[j];
        if (sum_c == 0) x[(int)c[0]] = 0;
    }
'''

k = cl.kernels(src, "-D FMACNT=4 -D UNROLL=4")
t = cl.tensor_f32([16])
K = 4096000
M = 44096
N = 32
k.enqueue("max_fma", [M, N],[32, N], t, K)
profiling = cl.finish()

dur_ns = profiling[0][-2] - profiling[0][-3]

print(f" {dur_ns}ns  {2*M*N*K*1e-3/dur_ns: .2f} TFLOPS")

