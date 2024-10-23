#!/usr/bin/python3
import cl
import numpy as np
import sys

print(dir(cl))

def test_basic():
    z = np.zeros([2, 3], dtype=np.float32)
    z[0,0] = 1.3
    z[0,1] = 2.3
    z[1,0] = 1.4
    z[1,1] = 2.4
    print(z)

    t = cl.tensor(z)
    print("+++++++++++")
    print(f"t.shape={t.shape}, t.numpy()={t.numpy()}")

    src = '''
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void set_ids(__global float * x) {
            int i = get_global_linear_id();
            x[i] = i;
        }
    '''
    k = cl.kernels(src, "")
    k.enqueue("set_ids", t.shape,[1,1], t)
    print("+++++++++++2222")
    print(t.numpy())

def test_buffer_pool():
    def test():
        t1 = cl.tensor([10,20], np.dtype(np.float32))
        t2 = cl.tensor([20,20], np.dtype(np.float32))
        return t1

    a = test()
    test()
    test()

def test_max_gflops():
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
    t = cl.tensor([16], np.dtype(np.float32))
    K = 4096000
    M = 44096
    N = 32
    k.enqueue("max_fma", [M, N],[1, N], t, K)
    profiling = cl.finish()

    dur_ns = profiling[0][-2] - profiling[0][-3]

    print(f" {dur_ns}ns  {2*M*N*K*1e-3/dur_ns: .2f} TFLOPS")

cl.profiling(True)
test_basic()
test_buffer_pool()
test_max_gflops()