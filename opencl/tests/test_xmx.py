#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare

src = r'''
// Prepack A from mk to MK(8m16k) for fp16
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void prepackA(__global half * src, __global half * dest) {
    int m_id = get_global_id(0);
    int K_id = get_global_id(1);
    int m_size = get_global_size(0);
    int K_size = get_global_size(1);
    int s_offset = m_id * K_size + K_id;
    int d_offset = m_id/8 * K_size * 8 + K_id * 8 + m_id%8;
    uint8 data = *((__global uint8*)src + s_offset);
    *((__global uint8*)dest + d_offset) = data;
}

// Prepack B from kn to NK(8k8n2k) for fp16, Similiar with AMX prepack
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void prepackB(__global half * src, __global half * dest) {
    int k_id = get_global_id(0);
    int n_id = get_global_id(1);
    int k_size = get_global_size(0);
    int n_size = get_global_size(1);
    int s_offset = k_id * n_size + n_id;
    int d_offset = (n_id/8 * k_size/16 + k_id/16)*128 + ((k_id)/2)%8 * 16 + (n_id%8)*2 + (k_id)%2;
    half data = *(src + s_offset);
    //printf("[id](%d, %d),[offset](%d, %d)\n",m_id, K_id, s_offset, d_offset);
    *(dest + d_offset) = data;
}

// After prepacking, A and B can be blocked loaded.
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_8x8_PREPACKED(__global half * A, __global half * B, __global float * C,  int K) {
    int k_loop = (K)/ 16;
    float8 acc = 0.f;
    int channel = get_sub_group_local_id();
    int8 a, b;
    uint8 res;

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < k_loop; i++) {
        a = as_int8(intel_sub_group_block_read8((__global uint*)(A+128*i)));
        b = as_int8(intel_sub_group_block_read8((__global uint*)(B+128*i)));
        res = as_uint8(intel_sub_group_f16_f16_matrix_mad_k16(a, b,  acc));
        acc = as_float8(res);
    }
    intel_sub_group_block_write8((__global uint*)C, res);
}

//https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
//https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_8x8(__global half * A, __global half * B, __global float * C) {
    
    int8 a = as_int8(intel_sub_group_block_read8((__global uint*)A));
    int channel = get_sub_group_local_id();
    float8 acc = 0.f;
    int8 b = *((int8*)(B) + channel);
    uint8 res = as_uint8(intel_sub_group_f16_f16_matrix_mad_k16(a, b,  acc));
    intel_sub_group_block_write8((__global uint*)C, res);
}

//https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_split_matrix_multiply_accumulate.html
//https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPASW.md
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_split_8x16(__global half * A, __global half * B, __global half * C) {
    int sg = get_sub_group_id();
    int sglid = get_sub_group_local_id();
    //int n = get_global_id(0);
    float8 c =0.f;
    int4 a = as_int4(intel_sub_group_block_read4((__global uint*)(A+sg*64)));
    __global half* b_sg_base = B+sg*128;
    int8 b = *(__global int8*)(b_sg_base+16*sglid);
    c = intel_sub_group_f16_f16_split_matrix_mad_k16(a, b, c);
    half8 hc = convert_half8(c);
    __global half* C_sg_base = C+sg*8+sglid;
#if 1
    C_sg_base[0*16] = hc.s0;
    C_sg_base[1*16] = hc.s1;
    C_sg_base[2*16] = hc.s2;
    C_sg_base[3*16] = hc.s3;
    C_sg_base[4*16] = hc.s4;
    C_sg_base[5*16] = hc.s5;
    C_sg_base[6*16] = hc.s6;
    C_sg_base[7*16] = hc.s7;
#else
    //Can't use hc[i] to access??, printf can show value, but can't write into C buffer.
    for (int i = 0; i < 7; i++) {
        C_sg_base[16*i] = hc[i];
        printf("[%f?=%f], ", hc[i], C_sg_base[16*i]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    printf("\n");
#endif
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_split_8x32(__global half * A, __global half * B, __global half * C, int N) {
    int sg = get_sub_group_id();
    int channel = get_sub_group_local_id();
    float8 acc=0.f;
    float8 c0 =0.f;
    float8 c1 =0.f;
#if 1
    int4 a = as_int4(intel_sub_group_block_read4((__global uint*)(A+sg%2*64)));
#else
    int8 a = as_int8(intel_sub_group_block_read8((__global uint*)(A)));
#endif
    __global half* b_sg_base = B+sg*256;
    int8 b0 = *(__global int8*)(b_sg_base+16*channel);
    int8 b1 = *(__global int8*)(b_sg_base+128+16*channel);
#if 1
    c0 = intel_sub_group_f16_f16_split_matrix_mad_k16(a, b0, acc);
    c1 = intel_sub_group_f16_f16_split_matrix_mad_k16(a, b1, acc);
#else
    c0 = intel_sub_group_f16_f16_matrix_mad_k16(a, b0, acc);
    c1 = intel_sub_group_f16_f16_matrix_mad_k16(a, b1, acc);
#endif
    half8 hc0 = convert_half8(c0);
    half8 hc1 = convert_half8(c1);
    __global half* C00 = C+sg*16+channel;
    __global half* C01= C+sg*16+8+channel;
    C00[0*N] = hc0.s0;
    C00[1*N] = hc0.s1;
    C00[2*N] = hc0.s2;
    C00[3*N] = hc0.s3;
    C00[4*N] = hc0.s4;
    C00[5*N] = hc0.s5;
    C00[6*N] = hc0.s6;
    C00[7*N] = hc0.s7;

    C01[0*N] = hc1.s0;
    C01[1*N] = hc1.s1;
    C01[2*N] = hc1.s2;
    C01[3*N] = hc1.s3;
    C01[4*N] = hc1.s4;
    C01[5*N] = hc1.s5;
    C01[6*N] = hc1.s6;
    C01[7*N] = hc1.s7;   
}
'''
kernels = cl.kernels(src, f"", "./dump/")
# cl.profiling(True)
## case 1: XMX output [8,8]
M = 8
N = 8
K = 16 
vRANGE = 3
A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)

C = np.matmul(A, B.transpose(1,0))

tA = cl.tensor(A)
tB = cl.tensor(B)
tC = cl.tensor([M, N], np.dtype(np.float32))
kernels.enqueue("XMX_8x8", [8],[8], tA, tB, tC)
compare(C, tC.numpy())

## case 2: XMX output [8,8], K=128, accumluated K with prepack A,B inputs.
K=128
M=8
N=8
A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
B_PACK = np.zeros([K, N]).astype(np.float16)
A_PACK = np.zeros([M, K]).astype(np.float16)
C =  np.matmul(A, B)
# K_tensor =  np.full((1), K, dtype=int)
# tK = cl.tensor(K_tensor)
tA = cl.tensor(A)
tB = cl.tensor(B)
tB_PACK = cl.tensor(B_PACK)
tA_PACK =cl.tensor(A_PACK)
tC = cl.tensor([M, N], np.dtype(np.float32))
kernels.enqueue("prepackA", [int(M), int(K/16)], [1, 1], tA, tA_PACK)
kernels.enqueue("prepackB", tB.shape,tB.shape, tB, tB_PACK)
kernels.enqueue("XMX_8x8_PREPACKED", [8],[8], tA_PACK, tB_PACK, tC, K)
compare(C, tC.numpy())

## case 3: XMX output [8,16] , split on 2 subgroups
M = 8
N = 16
K = 16 
A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
C = np.matmul(A, B.transpose(1,0))
tC = cl.tensor([M, N], np.dtype(np.float16))
tA = cl.tensor(A)
tB = cl.tensor(B)
kernels.enqueue("XMX_split_8x16", [16],[16], tA, tB, tC)
# cl.finish()
compare(C, tC.numpy())

## case 4: XMX output [8,32] , split on 2 subgroups
M = 8
N = 2048
K = 16 
A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
C = np.matmul(A, B.transpose(1,0))
tC = cl.tensor([M, N], np.dtype(np.float16))
tA = cl.tensor(A)
tB = cl.tensor(B)
# there would be 1024/8=128 subgroups.
kernels.enqueue("XMX_split_8x32", [int(N/2)],[int(N/2)], tA, tB, tC,N) 
# cl.finish()
compare(C, tC.numpy())
print("-----------------")

