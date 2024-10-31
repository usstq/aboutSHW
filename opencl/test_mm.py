#!/usr/bin/python3
import os
os.environ['cl_intel_driver_diagnostics'] = "1"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
import torch

print(dir(cl))

'''
https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/intel-xe-gpu-architecture.html

    we assume A(MxK) & B(KxN) sub matrix has been copied into SLM,
    and 16-EUs within current WG will compute C(MxN) to reach compute-bound.

    we will keep C in each thead's GRF, so each EU thread is responsible for
    producing non-overlapping sub-C matrix, given sub-A & sub-B in SLM

        CL_DEVICE_LOCAL_MEM_SIZE : 65536 x Bytes, 32768 x half
            
            A: 16384 half
            B: 16384 half

    each EU's GRF can store 128*8*32 bytes or 8192 floats, so shape of sub-C must < [90 x 90]
    and there are 16-EU in WG: [4*90, 4*90]
        
        Let's assume there is nothing shared between WG, so each WG would take (M+N)*K*2 bytes in
    and generate M*N*K*2 flops computations using time T:

        mem-bw:  (M + N)*K*2 /T  <  Bm
        compute: M*N*K*2 / T     <  Bc

    best possible time to compte C  : Tc = (M*N*K*2)/Bc
    best possible time to load A & B: Tm = (M + N)*K*2/Bm
        we can reach compute-bound if Tc > Tm
            or Tc/Tm > 1
            or [(M*N/Bc)]/[(M+N)/Bm] > 1
            or (M*N)/(M+N) > Bc/Bm
    
    if mem-bw is evenly distributed amoung all sub-slices, we have :
        per sub-splice Bm = 16(=512/32)GB/s, 
    given XMX performs 64 fp16-MAD per-cycle, we have:
        Bc = 64*2*16(EU)*2.4G = 4915.2 GFLOP/s
    
    so we need (M*N)/(M+N) > Bc/Bm = 4915.2/16 ~= 307
    if we further assume M==N, then M/2 > 307, M > 614

    but as we can see, GRF cannot hold such big M

    Given GRF's size, we can assume M = N = 256 (4*64), give SLM size, we have K = 16384/256 = 64.
    so:
        Tm = (256 + 256)*64*2/16e9 = 4.096e-06 = 4 us
        Tc = (256*256*64*2)/4915e9 = 1.706e-06 = 1.7us

    
    assume SLM is filled with data, can all XMX within a WG reach peak performance?
'''

# https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups.html
#  uint8   intel_sub_group_block_read8( const __global uint* p );
#  void    intel_sub_group_block_write8( __global uint* p, uint8 data );

# https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html
#   float8 intel_sub_group_f16_f16_matrix_mad_k16(int8 a, int8 b, float8 acc);

torch.manual_seed(0)
cl.profiling(True)

def compare(ref, opt, atol=0.01, rtol=0.01):
    if not np.allclose(ref, opt, atol=atol, rtol=rtol):
        pos = np.where(np.isnan(opt))
        if len(pos[0]) > 0:
            print(f'========================================================')
            print(f'pos nan = {len(pos)}x{len(pos[0])} {pos}')
            print(f'opt nan = {opt[pos]}')
            raise("failed.")

        pos = np.where(np.abs(ref - opt) > atol)
        print(f'========================================================')
        print(f'compare failed (ref={ref.dtype} {ref.shape}, opt={opt.dtype} {opt.shape}, atol={atol}, rtol={rtol})')
        print(f'pos = {len(pos)}x{len(pos[0])} {pos}')
        print(f'ref_val = {ref[pos]}')
        print(f'opt_val = {opt[pos]}')
        raise Exception("failed.")
    else:
        print(f'compare allclose success!')

def XMX_basic():
    src = r'''
        ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
        __kernel void XMX_packB_16x8(__global half * src, __global half * dst) {
            int n = get_global_id(0);
            int k = get_global_id(1);
            int N = get_global_size(0);
            int K = get_global_size(1);
            int off = n*2 + (k%2) + (k/2)*16;
            dst[off] = src[n*K + k];
        }

        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void XMX_basic_8x16x8(__global half * A, __global half * B, __global float * C) {
            //int i = get_global_linear_id();
            
            int8 a = as_int8(intel_sub_group_block_read8((__global uint*) A));
            int8 b = as_int8(intel_sub_group_block_read8((__global uint*) B));
            float8 acc = (float8)(0.0f);

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);

            intel_sub_group_block_write8((__global uint*)C, as_uint8(acc));
        }
        
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void XMX_tput_8x16x8(__global half * A, __global half * B, __global float * C, int loop_count, int p) {
            int8 a = as_int8(intel_sub_group_block_read8((__global uint*) A));
            int8 b = as_int8(intel_sub_group_block_read8((__global uint*) B));

            float8 acc[ACC_CNT];

            for(int i = 0; i < 8; i++) acc[i] = (float8)i;

            __attribute__((opencl_unroll_hint(1)))
            for(int i = 0; i < loop_count; i++) {
                
                __attribute__((opencl_unroll_hint(ACC_CNT)))
                for(int k = 0; k < ACC_CNT; k++)
                    acc[k] = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc[k]);
            }

            intel_sub_group_block_write8((__global uint*)C, as_uint8(acc[p]));
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    '''
    ACC_CNT = 10
    k = cl.kernels(src, f"-D ACC_CNT={ACC_CNT}", "./dump/")
    M = 8   #
    N = 8   #
    K = 16  # each GRF register is 32bytes/16half
    vRANGE = 3
    A = torch.randint(-vRANGE, vRANGE+1, [M, K]).half()
    B = torch.randint(-vRANGE, vRANGE+1, [N, K]).half()
    C = torch.matmul(A, B.transpose(1,0)).numpy()

    tA = cl.tensor(A.numpy())
    tB0 = cl.tensor(B.numpy())
    tB = cl.tensor(B.numpy())
    tC = cl.tensor([M, N], np.dtype(np.float32))
    k.enqueue("XMX_packB_16x8", [N, K],[1, 1], tB0, tB)
    k.enqueue("XMX_basic_8x16x8", [1, 8],[1, 8], tA, tB, tC)

    compare(C, tC.numpy())

    cl.finish()

    loop_cnt = 1280
    num_sub_groups = 16*8
    k.enqueue("XMX_tput_8x16x8", [num_sub_groups, 8],[num_sub_groups, 8], tA, tB, tC, loop_cnt, 0)
    k.enqueue("XMX_tput_8x16x8", [num_sub_groups, 8],[num_sub_groups, 8], tA, tB, tC, loop_cnt, 0)
    k.enqueue("XMX_tput_8x16x8", [num_sub_groups, 8],[num_sub_groups, 8], tA, tB, tC, loop_cnt, 0)
    k.enqueue("XMX_tput_8x16x8", [num_sub_groups, 8],[num_sub_groups, 8], tA, tB, tC, loop_cnt, 0)
    latency_ns = cl.finish()
    for ns in latency_ns:
        flops = loop_cnt * ACC_CNT * num_sub_groups * (M*K*N) * 2
        print(f" {flops} / {ns} = {flops/ns:.3f} GFlops/s")
'''
only 1 EU, 8 thread, 8*8 work-items

GRF 

XMX load from GRF
'''


XMX_basic()