#!/usr/bin/python3
import cl
import numpy as np
import os
import ctypes

src='''
    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void testLS(__global int * A, __global int * B, __global int * C) {
        int m = get_global_id(0);

        int8 b;
        // load matrix of shape:
        //      8(row: uint8) x 8(cols: SIMD-width)
        //
        *(uint8*)&b = intel_sub_group_block_read_ui8((__global uint*)(A));

        // so each SIMD-lane (work-item) is a column of the matrix
        B[m*8 + 0] = b.s0;  // row0, col-i 
        B[m*8 + 1] = b.s1;  // row1, col-i
        B[m*8 + 2] = b.s2;  // row2, ....
        B[m*8 + 3] = b.s3;  // ...
        B[m*8 + 4] = b.s4;  // ...
        B[m*8 + 5] = b.s5;  
        B[m*8 + 6] = b.s6;  //
        B[m*8 + 7] = b.s7;  // row7, col-i

        // store matrix in GRF
        intel_sub_group_block_write_ui8((__global uint*)(C), *(uint8*)&b);
    }

    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void testLS_us8(__global half * A, __global half * B, __global half * C) {
        int m = get_global_id(0);

        half8 b;
        // load matrix of shape:
        //      8(row: uint8) x 8(cols: SIMD-width)
        //
        *(ushort8*)&b = intel_sub_group_block_read_us8((__global ushort*)(A));

        // so each SIMD-lane (work-item) is a column of the matrix
        B[m*8 + 0] = b.s0;  // row0, col-i 
        B[m*8 + 1] = b.s1;  // row1, col-i
        B[m*8 + 2] = b.s2;  // row2, ....
        B[m*8 + 3] = b.s3;  // ...
        B[m*8 + 4] = b.s4;  // ...
        B[m*8 + 5] = b.s5;  
        B[m*8 + 6] = b.s6;  //
        B[m*8 + 7] = b.s7;  // row7, col-i

        // store matrix in GRF
        intel_sub_group_block_write_us8((__global ushort*)(C), *(ushort8*)&b);
    }

'''
kernel = cl.kernels(src, "-D FMACNT=4 -D UNROLL=4", "./dump")


A0 = np.zeros((8, 8), dtype=np.int32)
for i in range(8):
    for j in range(8):
        A0[i, j] = i*10 + j

print("============= intel_sub_group_block_read_ui8")
A = cl.tensor(A0)
B = cl.tensor(A0.shape, A0.dtype)
C = cl.tensor(A0.shape, A0.dtype)
kernel.enqueue("testLS", [8, 1], [8,1], A, B, C)

print("============= A")
print(A.numpy())
print("============= B")
print(B.numpy())
print("============= C")
print(C.numpy())


print("============= intel_sub_group_block_read_us8")
A0 = A0.astype(np.float16)
A = cl.tensor(A0)
B = cl.tensor(A0.shape, A0.dtype)
C = cl.tensor(A0.shape, A0.dtype)
kernel.enqueue("testLS_us8", [8, 1], [8,1], A, B, C)

print("============= A")
print(A.numpy())
print("============= B")
print(B.numpy())
print("============= C")
print(C.numpy())
