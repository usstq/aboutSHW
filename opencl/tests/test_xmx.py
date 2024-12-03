#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *


def test_gemm_multi_dss():
    src = r'''
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_GEMM(__global half * A, __global half * B, __global half * C, int M, int N, int K) {
        //M, N divided by work groups evenly
        //assert(M%get_num_groups() == 0 && N%get_num_groups() == 0);
        int m = M/get_num_groups(0);
        int n = N/get_num_groups(1);
        int lgid_m = get_group_id(0);
        int lgid_n = get_group_id(1);

        int sg_nb = get_num_sub_groups();
        //used HW threads  per VE engine
        int sg_per_col = get_local_size(0);
        //used VE engines per DSS
        int sg_per_row = sg_nb/sg_per_col;

        //The subgroups in one row can be 2-paired
        //assert(sg_per_row % 2 == 0);
        //assert(m%sg_per_col == 0 && n%VEs == 0);
        int m_per_sg = m / sg_per_col;
        int n_per_sg = n / sg_per_row;
        int m_block = 8;

        int sgid = get_sub_group_id();
        int sgidx_m = sgid / sg_per_row;
        int sgidx_n = sgid % sg_per_row;
        //A,B,C base of WG
        __global half * local_A = A + m * lgid_m * K;
        __global half * local_B = B + n * lgid_n  * K;
        __global half * local_C = C + m * lgid_m * N + n * lgid_n;
        //A,B,C base of SG
        __global half *sg_A = local_A + sgidx_m * m_per_sg * K;
        __global half *sg_B = local_B + sgidx_n * n_per_sg * K;
        __global half *sg_C = local_C + (sgidx_m * m_per_sg) * N +  sgidx_n * n_per_sg;
        __global half * AA = sg_A;
        __global half * BB = sg_B;
        __global half * CC = sg_C;
        int chan_id = get_sub_group_local_id();

        float8 c0 =0.f;
        float8 c1 =0.f;
        //accumulate K to output C[m_sz_per_sg,n_sz_per_sg]
        int loop_1 = 1;
        __attribute__((opencl_unroll_hint))
        // @todo: To use intel_sub_group_f16_f16_split_matrix_mad_k16() in a loop. The loop number should be a const or MACRO. 
        //        Using "; i < m_per_sg/m_block;" would have accuracy issue with split implementation.
        // @bug: (int i = 0; i <  m_per_sg/m_block; i++) {
        for (int i = 0; i < M_BLK; i++) {
            c0 = 0.f;
            c1 = 0.f;
        #if DPASW
            int4 a = as_int4(intel_sub_group_block_read4((__global uint*)(AA+sgid%2*64)));
        #else
            int8 a = as_int8(intel_sub_group_block_read8((__global uint*)(AA)));
        #endif
            __global half * BB0 =  BB+16*chan_id;
            global half * BB1 =  BB+128+16*chan_id;
            int8 b0 = *(__global int8*)(BB0);
            int8 b1 = *(__global int8*)(BB1);
        #if DPASW
            c0 = intel_sub_group_f16_f16_split_matrix_mad_k16(a, b0, c0);
            c1 = intel_sub_group_f16_f16_split_matrix_mad_k16(a, b1, c1);
        #else
            c0 = intel_sub_group_f16_f16_matrix_mad_k16(a, b0, c0);
            c1 = intel_sub_group_f16_f16_matrix_mad_k16(a, b1, c1);
        #endif
            //printf("sgid:%d, chan_id:%d, a[0]:%d, c0:%f, c1:%f, b0:%f, b1:%f\n", sgid, chan_id, a.s0, c0.s0, c1.s0, b0.s0, b1.s1);

            half8 hc0 = convert_half8(c0);
            half8 hc1 = convert_half8(c1);
            __global half* C00 = CC+chan_id;
            __global half* C01= CC+8+chan_id;
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
            AA += 8*K;
            CC += 8*N;
        }
    }
'''
    # Each subgroup would output MBLOCK * 8 = 32 rows of data.
    kernels = cl.kernels(src, "-D M_BLK=4 -D DPASW=1")
    RESERVE_DSS = 8
    M = 256 * RESERVE_DSS
    N = 256 * RESERVE_DSS
    # ONE XMX would accumulate 16
    K = 16
    vRANGE = 3
    #np.random.seed(0);
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    # B layout is [N, K]
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    C = np.matmul(A, B.transpose(1,0))
    tC = cl.tensor([M, N], np.dtype(np.float16))
    tA = cl.tensor(A)
    tB = cl.tensor(B)

    # One DSS would calculate to output C [256, 256]. One DSS(WG) would has [8, 128] work items. 8 is hw threads, 128 = 16 *8 = EUs * subgroup_size.
    # subgroup number on on DSS is 8*128/subgroup_size = 128;  Matrix C [256, 256] is divided by subgroups [8, 16].
    # Each subgroup would output C [256/8, 256/16] = [32, 16]. XMX HW would output [8,8] once. so there would be 8 XMX instructions invocation for one subgroup.
    # To use the intel_sub_group_f16_f16_split_matrix_mad_k16(), subgroups in one row should be even.
    # Consider one subgroup is one VE core using 1 hw threads.
    # Subgroup size is 8, doesn't mean can only output 8 columns on the C.
    # work items doesn't mean one element. one work item means  a collection of parallel executions of a kernel.
    kernels.enqueue("XMX_GEMM", [8 * RESERVE_DSS, 128 * RESERVE_DSS],[8, 128], tA, tB, tC, M, N, K)

    cl.finish()
    compare(C, tC.numpy())

def test_xmx_8x8():
    src = r'''
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
'''
    kernels = cl.kernels(src, f"")
    cl.profiling(True)
    # ## case 1: XMX output [8,8]
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
    cl.finish()
    compare(C, tC.numpy())

def test_xmx_prepack_8x8():
    src = r'''
    // Prepack A from mk to MK(8m16k) for fp16
    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void prepackA(__global half * src, __global half * dest) {
        int m_id = get_global_id(0);
        int K_id = get_global_id(1);
        int m_size = get_global_size(0);
        int K_size = get_global_size(1);
        int s_offset = m_id * K_size + K_id;
        int d_offset = m_id/8 *K_size*8  + K_id * 8 + m_id%8;
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
    '''
    kernels = cl.kernels(src, f"")
    vRANGE = 3
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

def test_split_xmx():
    src = r'''
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
#endif
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_split_8x32(__global half * A, __global half * B, __global half * C, int N) {
    int sg = get_sub_group_id();
    int channel = get_sub_group_local_id();
    float8 acc=0.f;
    float8 c0 =0.f;
    float8 c1 =0.f;
#if 0
    int4 a = as_int4(intel_sub_group_block_read4((__global uint*)(A+sg%2*64)));
#else
    int8 a = as_int8(intel_sub_group_block_read8((__global uint*)(A)));
#endif
    __global half* b_sg_base = B+sg*256;
    int8 b0 = *(__global int8*)(b_sg_base+16*channel);
    int8 b1 = *(__global int8*)(b_sg_base+128+16*channel);
#if 0
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
    vRANGE = 3
    kernels = cl.kernels(src, f"")
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
    cl.finish()
    compare(C, tC.numpy())
    M = 8
    N = 2048
    K = 16
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    C = np.matmul(A, B.transpose(1,0))
    tC = cl.tensor([M, N], np.dtype(np.float16))
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    kernels.enqueue("XMX_split_8x32", [int(N/2)],[int(N/2)], tA, tB, tC,N)
    cl.finish()
    compare(C, tC.numpy())

def test_xmx_gflops():
    
    src = '''

        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void xmx_tput(__global half * A, __global half * B, __global float * C, int K) {
            int id = get_global_linear_id();
            int sgid = id /get_sub_group_size();
            global half *sgA = A + sgid *128;
            global half *sgB = B + sgid *128;
            global float *CC = C + id *10;
            global float *sgC = C + sgid *64;
//DPASW = 1
#if DPASW

            int4 a0=0;
    #if BLOCKA

            a0 = as_int4(intel_sub_group_block_read4((__global uint*)(sgA)));
    //BLOCKA = 0
    #else
            a0.s0 = *(__global uint*)(sgA);
            a0.s1 = *(__global uint*)(sgA+2);
            a0.s2 = *(__global uint*)(sgA+4);
            a0.s3 = *(__global uint*)(sgA+6);
    //End BLOCKA
    #endif


//DPASW = 0
#else
            int8 a0;
    #if BLOCKA
            a0 = as_int8(intel_sub_group_block_read8((__global uint*)(sgA)));
    //BLOCKA = 0
    #else
            a0.s0 = *(__global uint*)(sgA);
            a0.s1 = *(__global uint*)(sgA+2);
            a0.s2 = *(__global uint*)(sgA+4);
            a0.s3 = *(__global uint*)(sgA+6);
            a0.s4 = *(__global uint*)(sgA+8);
            a0.s5 = *(__global uint*)(sgA+10);
            a0.s6 = *(__global uint*)(sgA+12);
            a0.s7 = *(__global uint*)(sgA+14);
            /*  Initilized  with IMM would cause perf drop to from 116 TFLOPS to 90 TFLOPS. 
                a0(int8) is shared by all the daps ISA. Extra copy a0 in GRF regristers level cause the perf drop to 90 TFLOPS. 
                a0.s1 = 1;
                a0.s2 = 2;
                a0.s3 = 3;
                a0.s4 = 4;
                a0.s5 = 5;
                a0.s6 = 6;
                a0.s7 = 7;
                expected dpas8x8:
                        dpas.8x8 (8|M0)          r115:f        null:f            r115:hf           r115.0:hf        {Atomic}
                        dpas.8x8 (8|M0)          r115:f        r115:f            r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r107:f        r107:f            r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r99:f         r99:f             r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r91:f         r91:f             r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r83:f         r83:f             r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r75:f         r75:f             r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r67:f         r67:f             r35:hf            r51.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r59:f         r59:f             r35:hf            r51.0:hf         {$7}
                But real case:
                        mov (8|M0)               r24.0<1>:d    0:w
                        mov (8|M0)               r25.0<1>:d    1:w
                        mov (8|M0)               r26.0<1>:d    2:w
                        mov (8|M0)               r27.0<1>:d    3:w
                        mov (8|M0)               r28.0<1>:d    4:w
                        mov (8|M0)               r29.0<1>:d    5:w
                        mov (8|M0)               r30.0<1>:d    6:w
                        mov (8|M0)               r31.0<1>:d    7:w
                        mov (8|M0)               r38.0<1>:ud   r38.0<8;8,1>:ud                  {$5.dst}
                        sync.nop                             null                             {Compacted,A@1}
                        sync.nop                             null                             {Compacted,$6.dst}
                        dpas.8x8 (8|M0)          r110:f        null:f            r110:hf           r110.0:hf        {Atomic}
                        dpas.8x8 (8|M0)          r110:f        r110:f            r38:hf            r8.0:hf          {Atomic}
                        dpas.8x8 (8|M0)          r102:f        r102:f            r38:hf            r16.0:hf         {$6}
                (W)     cmp (8|M0)    (lt)f1.1   null<1>:d     r33.7<0;1,0>:d    r33.6<0;1,0>:d
                        mov (8|M0)               r8.0<1>:d     0:w                               {$6.src}
                        mov (8|M0)               r9.0<1>:d     1:w
                        mov (8|M0)               r10.0<1>:d    2:w
                        mov (8|M0)               r11.0<1>:d    3:w
                        mov (8|M0)               r12.0<1>:d    4:w
                        mov (8|M0)               r13.0<1>:d    5:w
                        mov (8|M0)               r14.0<1>:d    6:w
                        mov (8|M0)               r15.0<1>:d    7:w
                        mov (8|M0)               r16.0<1>:d    0:w
                        mov (8|M0)               r17.0<1>:d    1:w
                        mov (8|M0)               r18.0<1>:d    2:w
                        mov (8|M0)               r19.0<1>:d    3:w
                        mov (8|M0)               r20.0<1>:d    4:w
                        mov (8|M0)               r21.0<1>:d    5:w
                        mov (8|M0)               r22.0<1>:d    6:w
                        mov (8|M0)               r23.0<1>:d    7:w
                        sync.nop                             null                             {Compacted,A@1}
                        sync.nop                             null                             {Compacted,$7.dst}
                        dpas.8x8 (8|M0)          r94:f         null:f            r94:hf            r94.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r94:f         r94:f             r38:hf            r24.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r86:f         r86:f             r38:hf            r8.0:hf          {Atomic}
                        dpas.8x8 (8|M0)          r78:f         r78:f             r38:hf            r16.0:hf         {$7}
                        mov (8|M0)               r24.0<1>:d    0:w                               {$7.src}
                        mov (8|M0)               r25.0<1>:d    1:w
                        mov (8|M0)               r26.0<1>:d    2:w
                        mov (8|M0)               r27.0<1>:d    3:w
                        mov (8|M0)               r28.0<1>:d    4:w
                        mov (8|M0)               r29.0<1>:d    5:w
                        mov (8|M0)               r30.0<1>:d    6:w
                        mov (8|M0)               r31.0<1>:d    7:w
                        mov (8|M0)               r8.0<1>:d     0:w
                        mov (8|M0)               r9.0<1>:d     1:w
                        mov (8|M0)               r10.0<1>:d    2:w
                        mov (8|M0)               r11.0<1>:d    3:w
                        mov (8|M0)               r12.0<1>:d    4:w
                        mov (8|M0)               r13.0<1>:d    5:w
                        mov (8|M0)               r14.0<1>:d    6:w
                        mov (8|M0)               r15.0<1>:d    7:w
                        mov (8|M0)               r16.0<1>:d    0:w
                        mov (8|M0)               r17.0<1>:d    1:w
                        mov (8|M0)               r18.0<1>:d    2:w
                        mov (8|M0)               r19.0<1>:d    3:w
                        mov (8|M0)               r20.0<1>:d    4:w
                        mov (8|M0)               r21.0<1>:d    5:w
                        mov (8|M0)               r22.0<1>:d    6:w
                        mov (8|M0)               r23.0<1>:d    7:w
                        sync.nop                             null                             {Compacted,A@1}
                        sync.nop                             null                             {Compacted,$8.dst}
                (W)     sync.nop                             null
                        dpas.8x8 (8|M0)          r70:f         null:f            r70:hf            r70.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r70:f         r70:f             r38:hf            r24.0:hf         {Atomic}
                        dpas.8x8 (8|M0)          r62:f         r62:f             r38:hf            r8.0:hf          {Atomic}
                        dpas.8x8 (8|M0)          r54:f         r54:f             r38:hf            r16.0:hf         {$8}
                (f1.1)  goto.b (8|M0)                        L3856                  L2528
            */
    // End BLOCKA
    #endif


//End of DPASW
#endif
            int8 b0 = as_int8(intel_sub_group_block_read8((__global uint*)(sgB)));
            float8 c0= 0.f;
            float8 c1= 0.f;
            float8 c2= 0.f;
            float8 c3= 0.f;
            float8 c4= 0.f;
            float8 c5= 0.f;
            float8 c6= 0.f;
            float8 c7= 0.f;
            float8 c8= 0.f;
            float8 c9= 0.f;

            __attribute__((opencl_unroll_hint(1)))
            for(int k = 0; k < K/20; k ++) {
#if DPASW
                c0 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c0);
                c1 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c1);
                c2 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c2);
                c3 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c3);
                c4 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c4);
                c5 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c5);
                c6 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c6);
                c7 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c7);
                c8 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c8);
                c9 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c9);

                c0 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c0);
                c1 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c1);
                c2 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c2);
                c3 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c3);
                c4 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c4);
                c5 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c5);
                c6 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c6);
                c7 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c7);
                c8 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c8);
                c9 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c9);
                
#else
                c0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c0);
                c1 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c1);
                c2 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c2);
                c3 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c3);
                c4 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c4);
                c5 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c5);
                c6 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c6);
                c7 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c7);
                c8 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c8);
                c9 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c9);

                c0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c0);
                c1 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c1);
                c2 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c2);
                c3 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c3);
                c4 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c4);
                c5 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c5);
                c6 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c6);
                c7 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c7);
                c8 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c8);
                c9 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c9);
#endif

            }
            //enusre the accumulate OPs not optimized by compiler.
            CC[0] = c0.s0;
            CC[1] = c1.s0;
            CC[2] = c2.s0;
            CC[3] = c3.s0;
            CC[4] = c4.s0;
            CC[5] = c5.s0;
            CC[6] = c6.s0;
            CC[7] = c7.s0;
            CC[8] = c8.s0;
            CC[9] = c9.s0;
        }
    '''
    RESERVE_DSS = 32
    SUB_GROUPS = RESERVE_DSS * 16 * 8;
    A = np.random.randint(-3, 3, [SUB_GROUPS*128]).astype(np.float16)
    B = np.random.randint(-3, 3, [SUB_GROUPS*128]).astype(np.float16)
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    tC = cl.tensor([SUB_GROUPS*80], np.dtype(np.float32))
    K = 10240000
    DPASW_LIST = [0, 1]
    BLOCKED_LIST = [0, 1]
    for DPASW in DPASW_LIST:
        for BLOCKA in BLOCKED_LIST:
            k = kernel_cache(src, options=f"-DDPASW={DPASW} -DBLOCKA={BLOCKA}")
            k.enqueue("xmx_tput", [1024* RESERVE_DSS], [1024], tA, tB, tC, K)
            profiling = cl.finish()
            dur_ns = profiling[0]
            print(f" DPASW={DPASW} BLOCKA={BLOCKA}: {dur_ns}ns  {128* RESERVE_DSS* K*(64*16*2)*1e-3/dur_ns: .2f} TFLOPS")



def test_gemm_one_dss():
    
    
    src = r'''
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void XMX_GEMM(__global half * A, __global half * B, __global half * C, int M, int N, int K) {
        int m = M/get_num_groups(0);
        int n = N/get_num_groups(1);
        int lgid_m = get_group_id(0);
        int lgid_n = get_group_id(1);

        int sg_nb = get_num_sub_groups();
        //used HW threads  per VE engine
        int sg_per_col = get_local_size(0);
        //used VE engines per DSS
        int sg_per_row = sg_nb/sg_per_col;

        //The subgroups in one row can be 2-paired
        //assert(sg_per_row % 2 == 0);
        //assert(m%sg_per_col == 0 && n%VEs == 0);
        int m_per_sg = m / sg_per_col;
        int n_per_sg = n / sg_per_row;

        int sgid = get_sub_group_id();
        int sgidx_m = sgid / sg_per_row;
        int sgidx_n = sgid % sg_per_row;
        int chan_id = get_sub_group_local_id();

        //A,B,C base of WG
        __global half *wg_A = A + m * lgid_m * K;
        //prepacked B laytout is N{16nK[4k2n(8k8n2k)]}, perf group base is same with planar NK layout.
        __global half *wg_B = B + n * lgid_n  * K;
        __global half *wg_C = C + m * lgid_m * N + n * lgid_n;
        
        __local half buffB[256 * 64];
        __local half buffA[64 * 256];
        __local half* local_B_ptr = buffB;
        __local half* local_A_ptr = buffA;

        //C base of SG
        __global half *sg_C = wg_C + (sgidx_m * m_per_sg) * N +  sgidx_n * n_per_sg;

        // B is prepacked to N{16nK[4k2n(8k8n2k)]} , () means in DPAS,   [] means in local memory, {} means one xe core.
        // when doing copying B, `16n` dimension is divided by 16 VEs(divided result:sgidx_n), `4k2n` combined dimension is divided by 6 HW threads(divided result: sgidx_m ). Each HW would copy  (8k8n2k) elements.
        // The stide on `16n` diemsnion is `K/64*1024` , the stride on `4k2n` is 128.
        __global half *copy_B_src = wg_B + sgidx_n * 1024 * K/64 +  sgidx_m * 128;
        int copy_B_dest_offset = sgidx_n * 1024 + sgidx_m * 128;
        int sg_B_offset_local = sgidx_n * 1024;

        //Prepack A from mk to M{8mK[4k4m(8m16k)]}, a is `8m` dim, b is `4k` dim, c is `4m` dim.
        int a =  sgidx_m;
        int b =  sgidx_n / 4;
        int c =  sgidx_n % 4;
        int copy_A_src_m = (a *4 + c) * 8 + chan_id;
        int copy_A_src_k = 0;
        int copy_A_dest_offset = a * 2048 + (b*4+c)*128 + chan_id*16;
        int sg_A_offset_local = sgidx_m * 2048;

        float8 c00, c01, c10, c11, c20, c21,c30, c31;
        int8 b0, b1;
        int8 a0, a1, a2, a3;
        __global half *CC00, *CC01, *CC10, *CC11, *CC20, *CC21, *CC30, *CC31;
        __global half *BB0, *BB1;
        int K_BLK, loop_k_slm;
        uint8 data_B;
        uint data_A;
        c00 = 0.f;
        c01 = 0.f;
        c10 = 0.f;
        c11 = 0.f;
        c20 = 0.f;
        c21 = 0.f;
        c30 = 0.f;
        c31 = 0.f;

        int i;
        half16 dataA;
        for (K_BLK = 0; K_BLK < 32; K_BLK++) {
            copy_A_src_k = (K_BLK*4 + b) * 16;
            
            dataA = *(half16*)(wg_A + (copy_A_src_m)*K+copy_A_src_k);
            *(__local half16*)(buffA+copy_A_dest_offset) = dataA;

            data_B = intel_sub_group_block_read8((__global uint*)(copy_B_src));
            intel_sub_group_block_write8((__local uint*)(buffB+copy_B_dest_offset), data_B);
            barrier(CLK_LOCAL_MEM_FENCE);

            local_B_ptr = buffB + sg_B_offset_local;
            local_A_ptr = buffA + sg_A_offset_local;
            for (loop_k_slm = 0; loop_k_slm < 4; loop_k_slm++) {
                b0 = as_int8(intel_sub_group_block_read8((__local uint*)(local_B_ptr)));
                b1 = as_int8(intel_sub_group_block_read8((__local uint*)(local_B_ptr+128)));
                a0 = as_int8(intel_sub_group_block_read8((__local uint*)(local_A_ptr)));
                a1 = as_int8(intel_sub_group_block_read8((__local uint*)(local_A_ptr+128)));
                a2 = as_int8(intel_sub_group_block_read8((__local uint*)(local_A_ptr+256)));
                a3 = as_int8(intel_sub_group_block_read8((__local uint*)(local_A_ptr+384)));
                c00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c00);
                c01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, c01);
                c10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, c10);
                c11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, c11);
                c20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, c20);
                c21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, c21);
                c30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, c30);
                c31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, c31);
                //accumulate next 16 k, A_step = 128*4, B_step = 128*2
                local_A_ptr +=512;
                local_B_ptr +=256;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            //B layout,  N{16nK[4k2n(8k8n2k)]}, stide of `K` dim  = 4*2*8*8*2 =1024
            copy_B_src += 1024;
        }

        half8 hc00 = convert_half8(c00);
        half8 hc01 = convert_half8(c01);
        half8 hc10 = convert_half8(c10);
        half8 hc11 = convert_half8(c11);
        half8 hc20 = convert_half8(c20);
        half8 hc21 = convert_half8(c21);
        half8 hc30 = convert_half8(c30);
        half8 hc31 = convert_half8(c31);

        CC00 = sg_C+chan_id;
        CC01= sg_C+8+chan_id;
        CC10 = sg_C+8*N+chan_id;
        CC11= sg_C+8*N+8+chan_id;
        CC20 = sg_C+16*N+chan_id;
        CC21= sg_C+16*N+8+chan_id;
        CC30 = sg_C+24*N+chan_id;
        CC31= sg_C+24*N+8+chan_id;

        CC00[0*N] = hc00.s0;
        CC00[1*N] = hc00.s1;
        CC00[2*N] = hc00.s2;
        CC00[3*N] = hc00.s3;
        CC00[4*N] = hc00.s4;
        CC00[5*N] = hc00.s5;
        CC00[6*N] = hc00.s6;
        CC00[7*N] = hc00.s7;

        CC01[0*N] = hc01.s0;
        CC01[1*N] = hc01.s1;
        CC01[2*N] = hc01.s2;
        CC01[3*N] = hc01.s3;
        CC01[4*N] = hc01.s4;
        CC01[5*N] = hc01.s5;
        CC01[6*N] = hc01.s6;
        CC01[7*N] = hc01.s7;
        
        CC10[0*N] = hc10.s0;
        CC10[1*N] = hc10.s1;
        CC10[2*N] = hc10.s2;
        CC10[3*N] = hc10.s3;
        CC10[4*N] = hc10.s4;
        CC10[5*N] = hc10.s5;
        CC10[6*N] = hc10.s6;
        CC10[7*N] = hc10.s7;

        CC11[0*N] = hc11.s0;
        CC11[1*N] = hc11.s1;
        CC11[2*N] = hc11.s2;
        CC11[3*N] = hc11.s3;
        CC11[4*N] = hc11.s4;
        CC11[5*N] = hc11.s5;
        CC11[6*N] = hc11.s6;
        CC11[7*N] = hc11.s7;

        CC20[0*N] = hc20.s0;
        CC20[1*N] = hc20.s1;
        CC20[2*N] = hc20.s2;
        CC20[3*N] = hc20.s3;
        CC20[4*N] = hc20.s4;
        CC20[5*N] = hc20.s5;
        CC20[6*N] = hc20.s6;
        CC20[7*N] = hc20.s7;

        CC21[0*N] = hc21.s0;
        CC21[1*N] = hc21.s1;
        CC21[2*N] = hc21.s2;
        CC21[3*N] = hc21.s3;
        CC21[4*N] = hc21.s4;
        CC21[5*N] = hc21.s5;
        CC21[6*N] = hc21.s6;
        CC21[7*N] = hc21.s7;
        
        CC30[0*N] = hc30.s0;
        CC30[1*N] = hc30.s1;
        CC30[2*N] = hc30.s2;
        CC30[3*N] = hc30.s3;
        CC30[4*N] = hc30.s4;
        CC30[5*N] = hc30.s5;
        CC30[6*N] = hc30.s6;
        CC30[7*N] = hc30.s7;

        CC31[0*N] = hc31.s0;
        CC31[1*N] = hc31.s1;
        CC31[2*N] = hc31.s2;
        CC31[3*N] = hc31.s3;
        CC31[4*N] = hc31.s4;
        CC31[5*N] = hc31.s5;
        CC31[6*N] = hc31.s6;
        CC31[7*N] = hc31.s7;
    }
'''

    prepack =  r'''

    // Prepack A from mk to M{8mK[4k4m(8m16k)]} for fp16,  () means in DPAS,   [] means in local memory, {} means one xe core.
    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void prepackA(__global half * src, __global half * dest) {
        int m_id = get_global_id(0);
        int K_id = get_global_id(1);
        int m_size = get_global_size(0);
        int K_size = get_global_size(1);
        int s_offset = m_id * K_size + K_id;
        int d_offset = m_id/32 * K_size *32 + K_id * 32 + m_id%32;
        uint8 data = *((__global uint8*)src + s_offset);
        *((__global uint8*)dest + d_offset) = data;
    }

    // Prepack B from kn to N{16nK[4k2n(8k8n2k)]} for fp16,  () means in DPAS,   [] means in local memory, {} means one xe core.
    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void prepackB(__global half * src, __global half * dest) {
        int n_id = get_global_id(0);
        int k_id = get_global_id(1);
        int n_size = get_global_size(0);
        int k_size = get_global_size(1);
        int s_offset = n_id * k_size + k_id;
        int d_offset = (n_id/16)*(k_size/16*256) +(k_id/16*256)+((n_id/8)%2)*128+ ((k_id/2)%8) * 16 + (n_id%8)*2 + (k_id)%2;
        half data = *(src + s_offset);
        *(dest + d_offset) = data;
    }
    '''
    # Each subgroup would output MBLOCK * 8 = 32 rows of data.
    prepack_ker = cl.kernels(prepack, "")
    kernels = cl.kernels(src, "")
    M = 512
    N = 512
    # ONE XMX would accumulate 16
    K = 2048
    vRANGE = 10
    #np.random.seed(0);
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    # B layout is [N, K]
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    B_PACK = B
    A_PACK = A

    C = np.matmul(A, B.transpose(1,0))
    tC = cl.tensor([M, N], np.dtype(np.float16))
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    tB_pack = cl.tensor(B_PACK)
    tA_pack = cl.tensor(A_PACK)
    prepack_ker.enqueue("prepackA", [N, int(K/16)],[8, 64], tA, tA_pack)
    prepack_ker.enqueue("prepackB", [N, K],[8, 128], tB, tB_pack)
    cl.finish()
    # One DSS would calculate to output C [256, 256]. One DSS(WG) would has [8, 128] work items. 8 is hw threads, 128 = 16 *8 = EUs * subgroup_size.
    # subgroup number on on DSS is 8*128/subgroup_size = 128;  Matrix C [256, 256] is divided by subgroups [8, 16].
    # Each subgroup would output C [256/8, 256/16] = [32, 16]. XMX HW would output [8,8] once. so there would be 8 XMX instructions invocation for one subgroup.
    # To use the intel_sub_group_f16_f16_split_matrix_mad_k16(), subgroups in one row should be even.
    # Consider one subgroup is one VE core using 1 hw threads.
    # Subgroup size is 8, doesn't mean can only output 8 columns on the C.
    # work items doesn't mean one element. one work item means  a collection of parallel executions of a kernel.
    kernels.enqueue("XMX_GEMM", [16, 256],[8, 128], tA, tB_pack, tC, M, N, K)
    cl.finish()

    compare(C, tC.numpy())

## step by step to understand how to use XMX and how to distribute workloads on GPU A770
cl.profiling(True)
# test_xmx_8x8()
# test_xmx_prepack_8x8()
# test_split_xmx()
# test_gemm_multi_dss()
# test_xmx_gflops()
# test_xmx_gflops()
test_gemm_one_dss()

print("-------------------------")
