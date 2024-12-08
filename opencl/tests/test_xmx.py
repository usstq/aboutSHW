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

#  Current unroll=40 achieve about 124 TFLOPs, enlarging unroll to 100 would achieve about 128 TFLOPS
#  DPASW=0 BLOCKA=0 UNROLL=2: 689804791ns   124.53 TFLOPS
#  DPASW=0 BLOCKA=1 UNROLL=2: 690417291ns   124.42 TFLOPS
#  DPASW=1 BLOCKA=0 UNROLL=2: 689496666ns   124.58 TFLOPS
#  DPASW=1 BLOCKA=1 UNROLL=2: 689496875ns   124.58 TFLOPS
def test_xmx_gflops():

    src = '''

        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void xmx_tput(__global half * A, __global half * B, __global float * C, int K) {
            int id = get_global_linear_id();
            int sgid = id /get_sub_group_size();
            global half *sgA = A + sgid *128;
            global half *sgB = B + sgid *128;
            // every workitem update 10 C elements.
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
                a0(int8) is shared by all the DPAS ISAs. Extra copy a0 in GRF regristers level cause the perf drop to 90 TFLOPS.
                The disassembly shows there DPAS is split by extra loading A0 and sync ISAs.
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
                        ........
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
                        ..........

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
            for(int k = 0; k < K/40; k ++) {
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
    # DPASW_LIST = [0]
    # BLOCKED_LIST = [0]
    for DPASW in DPASW_LIST:
        for BLOCKA in BLOCKED_LIST:
            k = kernel_cache(src, options=f"-DDPASW={DPASW} -DBLOCKA={BLOCKA}")
            k.enqueue("xmx_tput", [1024* RESERVE_DSS], [1024], tA, tB, tC, K)
            profiling = cl.finish()
            dur_ns = profiling[0]
            print(f" DPASW={DPASW} BLOCKA={BLOCKA} : {dur_ns}ns  {128* RESERVE_DSS* K*(64*16*2)*1e-3/dur_ns: .2f} TFLOPS")

def test_mm_tput():
    
    src = r'''
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void XMX_GEMM(__global half * A, __global half * B, __global half * C, int M, int N, int K) {
    
        int lgid_m = get_group_id(0);
        int lgid_n = get_group_id(1);
        int sg_nb = get_num_sub_groups();
        int chan_id = get_sub_group_local_id();
        int sgid = get_sub_group_id();

        int sgidx_m = sgid / N_WG_SGS;
        int sgidx_n = sgid % N_WG_SGS;

        //A,B,C base of WG
        __global half *wg_A = A + M_BLOCK * lgid_m * K;
        
        __local half buffA[K_BLOCK * M_BLOCK];
        __local half buffB[N_BLOCK * K_BLOCK];

        //C base of SG
        __global half *sg_C = (C + M_BLOCK * lgid_m * N + N_BLOCK * lgid_n) + (sgidx_m * SG_M) * N +  sgidx_n * SG_N;
#if DEBUG == 0
        // B is prepacked to N{16nK[4k2n(8k8n2k)]} , () means in DPAS,   [] means in local memory, {} means one xe core.
        // when doing copying B, `16n` dimension is divided by 16 VEs(divided result:sgidx_n), `4k2n` combined dimension is divided by 6 HW threads(divided result: sgidx_m ). Each HW would copy  (8k8n2k) elements.
        // The stide on `16n` diemsnion is `K/64*1024` , the stride on `4k2n` is 128.

        __global half *copy_B_src = ( B + N_BLOCK * lgid_n  * K) + sgidx_n * 1024 +  sgidx_m * 128;
        int copy_B_dest_offset = sgidx_n * 1024 + sgidx_m * 128;

        //Prepack A from mk to M{8mK[4k4m(8m16k)]}, a is `8m` dim, b is `4k` dim, c is `4m` dim.
        __global half* sg_copy_A_src = ( A + M_BLOCK * lgid_m  * K)+(sgidx_m *16 + sgidx_n ) * 2 * K;
        __local half* sg_copy_A_dest = buffA + sgidx_m*2048 + sgidx_n*32;

#endif
        int sg_B_offset_local = sgidx_n * 1024;
#if USE_DPAS
        int sg_A_offset_local = sgidx_m * 2048;
#elif USE_DPASW
        int sg_A_offset_local = sgidx_m * 2048 + (sgid % 2) * 64;
#endif
        float8 c00, c01, c10, c11, c20, c21,c30, c31;

        int kblk2, kblk;
        c00 = 0.f;
        c01 = 0.f;
        c10 = 0.f;
        c11 = 0.f;
        c20 = 0.f;
        c21 = 0.f;
        c30 = 0.f;
        c31 = 0.f;
        __local uint*  local_B_ptr = (__local uint*)(buffB + sg_B_offset_local);
        __local uint*  local_A_ptr = (__local uint*)(buffA + sg_A_offset_local);
        //XMX compute
        {
            __attribute__((opencl_unroll_hint(1)))
            for (kblk2 = 0; kblk2 < K/K_BLOCK; kblk2++) {
    #if DEBUG == 0
            {
                uint4 r0 = intel_sub_group_block_read4((__global uint *)(sg_copy_A_src));
                uint4 r1 = intel_sub_group_block_read4((__global uint *)(sg_copy_A_src + K));
                uint2 a0 = (uint2)(r0.s0, r1.s0);
                uint2 a1 = (uint2)(r0.s1, r1.s1);
                uint2 a2 = (uint2)(r0.s2, r1.s2);
                uint2 a3 = (uint2)(r0.s3, r1.s3);
                intel_sub_group_block_write2((__local uint*)(sg_copy_A_dest), a0);
                intel_sub_group_block_write2((__local uint*)(sg_copy_A_dest + 32*16*1), a1);
                intel_sub_group_block_write2((__local uint*)(sg_copy_A_dest + 32*16*2), a2);
                intel_sub_group_block_write2((__local uint*)(sg_copy_A_dest + 32*16*3), a3);

                uint8 data_B = intel_sub_group_block_read8((__global uint*)(copy_B_src));
                intel_sub_group_block_write8((__local uint*)(buffB+copy_B_dest_offset), data_B);
                barrier(CLK_LOCAL_MEM_FENCE);
            }
    #endif
                __attribute__((opencl_unroll_hint(1)))
                for (kblk = 0; kblk < K_BLOCK/16; kblk++) {
    #if USE_DPAS
                    int8 b0 = as_int8(intel_sub_group_block_read8(local_B_ptr));
                    int8 b1 = as_int8(intel_sub_group_block_read8(local_B_ptr+64));
                    int8 a0 = as_int8(intel_sub_group_block_read8(local_A_ptr));
                    int8 a1 = as_int8(intel_sub_group_block_read8(local_A_ptr+64));
                    int8 a2 = as_int8(intel_sub_group_block_read8(local_A_ptr+128));
                    int8 a3 = as_int8(intel_sub_group_block_read8(local_A_ptr+192));
                    c00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, c00);
                    c01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, c01);
                    c10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, c10);
                    c11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, c11);
                    c20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, c20);
                    c21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, c21);
                    c30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, c30);
                    c31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, c31);
    #elif USE_DPASW
                    int8 b0 = as_int8(intel_sub_group_block_read8(local_B_ptr));
                    int8 b1 = as_int8(intel_sub_group_block_read8(local_B_ptr+64));
                    int4 a0 = as_int4(intel_sub_group_block_read4(local_A_ptr));
                    int4 a1 = as_int4(intel_sub_group_block_read4(local_A_ptr+64));
                    int4 a2 = as_int4(intel_sub_group_block_read4(local_A_ptr+128));
                    int4 a3 = as_int4(intel_sub_group_block_read4(local_A_ptr+192));
                    c00 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b0, c00);
                    c01 = intel_sub_group_f16_f16_split_matrix_mad_k16(a0, b1, c01);
                    c10 = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b0, c10);
                    c11 = intel_sub_group_f16_f16_split_matrix_mad_k16(a1, b1, c11);
                    c20 = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b0, c20);
                    c21 = intel_sub_group_f16_f16_split_matrix_mad_k16(a2, b1, c21);
                    c30 = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b0, c30);
                    c31 = intel_sub_group_f16_f16_split_matrix_mad_k16(a3, b1, c31);
    #endif
                    //accumulate next 16 k, A_step = 64*4, B_step = 64*2
                    local_A_ptr +=256;
                    local_B_ptr +=128;
                }
    #if DEBUG == 0
                barrier(CLK_LOCAL_MEM_FENCE);
                //B layout,  N{16nK[4k2n(8k8n2k)]}, stide of `K` dim  = 4*2*8*8*2 =1024
                copy_B_src += 16*1024;
                sg_copy_A_src += K_BLOCK;
                local_A_ptr -= 1024;
                local_B_ptr -= 512;

    #endif
            }
        }//end of XMX compute

#if  DEBUG == 0
            __local ushort * scratch_buffA = (__local ushort *)buffA + sgid * 128;
            __local ushort * scratch_buffB = (__local ushort *)buffB + sgid * 128;

            int scratch_offset = chan_id * 8;
            int sgc_offset = chan_id * N;

            intel_sub_group_block_write_us8((scratch_buffA), as_ushort8(convert_half8(c00)));
            intel_sub_group_block_write_us8((scratch_buffA+64), as_ushort8(convert_half8(c01)));
            *(__global half8*)(sg_C+sgc_offset) = *(__local half8*)(scratch_buffA +scratch_offset);
            *(__global half8*)(sg_C+8+sgc_offset) = *(__local half8*)(scratch_buffA+64 +scratch_offset);

            intel_sub_group_block_write_us8((scratch_buffB), as_ushort8(convert_half8(c10)));
            intel_sub_group_block_write_us8((scratch_buffB+64), as_ushort8(convert_half8(c11)));
            *(__global half8*)(sg_C+sgc_offset+8*N) = *(__local half8*)(scratch_buffB +scratch_offset);
             *(__global half8*)(sg_C+8+sgc_offset+8*N) = *(__local half8*)(scratch_buffB+64 +scratch_offset);

            intel_sub_group_block_write_us8((scratch_buffA), as_ushort8(convert_half8(c20)));
            intel_sub_group_block_write_us8((scratch_buffA+64), as_ushort8(convert_half8(c21)));
            *(__global half8*)(sg_C+sgc_offset+16*N) = *(__local half8*)(scratch_buffA +scratch_offset);
             *(__global half8*)(sg_C+8+sgc_offset+16*N) = *(__local half8*)(scratch_buffA+64 +scratch_offset);

            intel_sub_group_block_write_us8((scratch_buffB), as_ushort8(convert_half8(c30)));
            intel_sub_group_block_write_us8((scratch_buffB+64), as_ushort8(convert_half8(c31)));
            *(__global half8*)(sg_C+sgc_offset+24*N) = *(__local half8*)(scratch_buffB +scratch_offset);
             *(__global half8*)(sg_C+8+sgc_offset+24*N) = *(__local half8*)(scratch_buffB+64 +scratch_offset);
#endif
    }
'''

    preprocess =  r'''

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
        //int d_offset = (n_id/16)*(k_size/16*256) +(k_id/16*256)+((n_id/8)%2)*128+ ((k_id/2)%8) * 16 + (n_id%8)*2 + (k_id)%2;
        int d_offset = (n_id/256)*k_size*256 + (k_id/64)*64*256 + (n_id/16%16)*1024+ (k_id/16%4)*256+(n_id/8%2)*128 + (k_id/2%8)*16 + (n_id%8)*2 + (k_id%2);
        half data = *(src + s_offset);
        *(dest + d_offset) = data;
    }

    __kernel void res_C(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[m_idx*K+i]*B[n_idx*K+i];
        CC[m_idx*N+n_idx] = sum;
    }
    '''
    # Each subgroup would output MBLOCK * 8 = 32 rows of data.

    SG_M = (4*8)
    SG_N = (2*8)
    M_WG_SGS = 8
    N_WG_SGS = 16
    M_BLOCK = (SG_M * M_WG_SGS)
    N_BLOCK = (SG_N * N_WG_SGS)
    K_BLOCK = 64
    SG_SIZE=8

    M = 2048
    N = 1024
    # ONE XMX would accumulate 16
    K = 2048
    vRANGE = 3
    #np.random.seed(0);
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    # B layout is [N, K]
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    B_PACK = B
    A_PACK = A
    DEBUG = 0

    prepocess_ker = kernel_cache(preprocess, "")
    kernels = kernel_cache(src, options=f"-DSG_SIZE={SG_SIZE} -DK_BLOCK={K_BLOCK} -DM_BLOCK={M_BLOCK} -DN_BLOCK={N_BLOCK} -DSG_M={SG_M} -DSG_N={SG_N} -DN_WG_SGS={N_WG_SGS} -DDEBUG={DEBUG} -DUSE_DPASW=1")
    tRefC = cl.tensor([M, N], np.dtype(np.float16))
    tC = cl.tensor([M, N], np.dtype(np.float16))
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    tB_pack = cl.tensor(B_PACK)
    #tA_pack = cl.tensor(A_PACK)
    # prepocess_ker.enqueue("prepackA", [N, int(K/16)],[8, 64], tA, tA_pack)
    prepocess_ker.enqueue("prepackB", [N, K],[8, 128], tB, tB_pack)
    prepocess_ker.enqueue("res_C", [M, N],[8, 128], tA, tB, tRefC, N, K)
    cl.finish()
    # One DSS would calculate to output C [256, 256]. One DSS(WG) would has [8, 128] work items. 8 is hw threads, 128 = 16 *8 = EUs * subgroup_size.
    # subgroup number on on DSS is 8*128/subgroup_size = 128;  Matrix C [256, 256] is divided by subgroups [8, 16].
    # Each subgroup would output C [256/8, 256/16] = [32, 16]. XMX HW would output [8,8] once. so there would be 8 XMX instructions invocation for one subgroup.
    # To use the intel_sub_group_f16_f16_split_matrix_mad_k16(), subgroups in one row should be even.
    # Consider one subgroup is one VE core using 1 hw threads.
    # Subgroup size is 8, doesn't mean can only output 8 columns on the C.
    # work items doesn't mean one element. one work item means  a collection of parallel executions of a kernel.
    for i in range(0, 10):
        kernels.enqueue("XMX_GEMM", [int(M/SG_M), int(N/SG_N*SG_SIZE)],[8, 128], tA, tB_pack, tC, M, N, K)
    res  = cl.finish()
    for ns in res:
        flops = (M * K * N * 2)
        rd_bytes = (M*K + K*N)*2
        total_work_groups = M//M_BLOCK * N//N_BLOCK
        actual_rd = total_work_groups * (M_BLOCK*K + N_BLOCK*K)*2
        print(f"XMX_tput: {flops*1e-9:.1f} Gflop / {ns*1e-3:.1f} us = {flops/ns*1e-3:.3f} TFlops/s    {rd_bytes/1e6:.1f}=>{actual_rd/1e6:.1f} MB  / {ns*1e-3:.1f} us = {actual_rd/ns:.1f} GB/s")
    if DEBUG != 1:
        compare(tRefC.numpy(), tC.numpy())

## step by step to understand how to use XMX and how to distribute workloads on GPU A770
cl.profiling(True)
# test_xmx_8x8()
# test_xmx_prepack_8x8()
# test_split_xmx()
# test_gemm_multi_dss()
# test_xmx_gflops()
test_mm_tput()
print("-------------------------")
