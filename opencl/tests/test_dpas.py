#!/usr/bin/python3
import cl
import numpy as np
import os
import ctypes

print(dir(cl))

cl.profiling(True)

src = '''
    ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void mmXMX(__global half * A, __global half * B, __global float *C, __global ulong *perf,  int M, int K, int N, int REP) {
        int m = get_global_id(0);
        int n = get_global_id(1);

        int8 a0, a1;
        float8 acc0, acc1, acc2, acc3;
        int8 b0, b1;

        // a : (M8 x 16)
        *(uint8*)&a0 = intel_sub_group_block_read_ui8((__global uint*)(A));
        *(uint8*)&a1 = intel_sub_group_block_read_ui8((__global uint*)(A + 8*K));
        
        // b : (8 x SIMD-8)
        *(uint8*)&b0 = intel_sub_group_block_read_ui8((__global uint*)(B));
        *(uint8*)&b1 = intel_sub_group_block_read_ui8((__global uint*)(B + 16*8));

        acc0 = 0;
        acc1 = 0;
        acc2 = 0;
        acc3 = 0;
        
        int i = get_global_linear_id();
        perf[i] = intel_get_cycle_counter();
        for(int r = 0; r < REP; r++) {
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc3);
        }
        perf[i] = intel_get_cycle_counter() - perf[i];

        intel_sub_group_block_write_ui8((__global uint*)(C), *(uint8*)&acc0);
        intel_sub_group_block_write_ui8((__global uint*)(C + 8*N), *(uint8*)&acc1);
        intel_sub_group_block_write_ui8((__global uint*)(C + 8*N*2), *(uint8*)&acc2);
        intel_sub_group_block_write_ui8((__global uint*)(C + 8*N*3), *(uint8*)&acc3);
    }


    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void mmALU(__global half * A, __global half * B, __global float * C, int M, int K, int N) {
        int m = get_global_id(0);
        int n = get_global_id(1);
        float sum = 0;
        for(int k = 0; k < K; k++) {
            sum += A[m*K + k] * B[n*K + k];
        }
        if (m < M && n < N)
            C[m*N + n] = sum;
    }
'''

kernel = cl.kernels(src, "-D FMACNT=4 -D UNROLL=4", "./dump")

K = 16
M = 8*2
N = 8

def repack_dpas(Bi):
    rows, cols = Bi.shape
    Bo = np.zeros([rows//2, cols*2], dtype=Bi.dtype)
    for r in range(rows):
        for c in range(cols):
            r1 = r // 2
            c1 = c*2 + (r % 2)
            Bo[r1, c1] = Bi[r, c]
    return Bo

np.random.seed(0)
A0 = np.random.randint(-2, 2, [M, K]).astype(np.float16)
B0 = np.random.randint(-2, 2, [K, N]).astype(np.float16)
#for k in range(K):
#    for n in range(N):
#        B0[k, n] = k*100 + n

C0 = np.matmul(A0, B0).astype(np.float32)

A = cl.tensor(A0)
B = cl.tensor(repack_dpas(B0))
C = cl.tensor([M, N], np.dtype(np.float32))

'''
print(f"B0=\n{B0.astype(np.int32)}")

for k in range(K):
    probeA0 = np.zeros_like(A0)
    probeA0[:, k] = 1
    probeA = cl.tensor(probeA0)
    kernel.enqueue("mmXMX", [8, 1], [8, 1], probeA, B, C, M, K, N)
    print(f"k={k} {C.numpy().astype(np.int32)}")
'''

P = cl.tensor([8], np.dtype(np.ulonglong))

kernel.enqueue("mmXMX", [8, 1], [8, 1], A, B, C, P, M, K, N, 1)
C1 = C.numpy()
if not np.allclose(C0, C1):
    print("mmXMX misMatch!!!")
    print("C0=\n", C0)
    print("C1=\n",C1)
else:
    print("mmXMX Match!!!")




P = cl.tensor([8], np.dtype(np.ulonglong))
REP = 100000
kernel.enqueue("mmXMX", [8, 1], [8, 1], A, B, C, P, M, K, N, REP)
P0 = P.numpy()

REP = 100000 + 10
kernel.enqueue("mmXMX", [8, 1], [8, 1], A, B, C, P, M, K, N, REP)
P1 = P.numpy()

print((P1 - P0)/10)
print(P0/REP)
print(P1/REP)

profiling = cl.finish()
dur_ns = profiling[0][-2] - profiling[0][-3]
print(f" {dur_ns}ns  {2*M*N*K*1e-3/dur_ns: .2f} TFLOPS")

