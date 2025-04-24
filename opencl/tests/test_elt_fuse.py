#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *
import torch as torch


def ALIGN_UP(a, b):
    return ((a + (b -1)) // b *b)

def DIV_UP(a, b):
    return ((a + (b -1)) // b)

def test_fused_eltwise(M, N):
    SG_SZ=16

    kernel_src =  r'''__attribute__((intel_reqd_sub_group_size(SG_SZ)))\
    __kernel void fused(__global half *up_data, __global half *gate_data,  __global half *out_data, int M, int N, int N_sg, int M_sg) {
        int sgid = get_sub_group_id();
        int gid0 = get_group_id(0);
        int gid1 = get_group_id(1);
        int n_idx = (gid1 * N_sg  +  sgid % N_sg) * SG_SZ * N_FACTOR;
        if (n_idx >= N)
            return;
        int m_idx = (gid0 * M_sg  +  sgid / N_sg) * M_FACTOR;
        if (m_idx >= M)
            return;
        int offset = m_idx * N + n_idx;
        up_data += offset;
        gate_data += offset;
        out_data += offset;
        __attribute__((opencl_unroll_hint))
        for (int mm = 0; mm < M_FACTOR; mm++) {
            __global half *out_ptr = out_data + mm * N;
            __global half *up_ptr = up_data + mm * N;
            __global half *gate_ptr = gate_data + mm * N;
             __attribute__((opencl_unroll_hint))
            for (int jj = 0; jj < N_FACTOR; jj++) {
                half up_val = as_half(intel_sub_group_block_read_us((const __global ushort*)(up_ptr)));
                half gate_val = as_half(intel_sub_group_block_read_us((const __global ushort*)(gate_ptr)));
                up_val = up_val /(1 + exp(0-up_val));
                gate_val *= up_val;
                intel_sub_group_block_write_us((const __global ushort*)(out_ptr), as_short(gate_val));
                out_ptr += SG_SZ;
                gate_ptr += SG_SZ;
                up_ptr += SG_SZ;
            }
        }
    }
    '''
    
    check_acc = True
    nfactor = 4
    mfactor = 8

    rep = 20
    cl.profiling(True)
    vRANGE=1
    up_tensor = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)
    gate_tensor = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)
    ref_tensor = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)
    N_WGZ=1024
    assert N % (SG_SZ*nfactor)==0 and N_WGZ %SG_SZ == 0 and N_WGZ <=1024
    assert M % (mfactor)==0
    opt_kernel = kernel_cache(kernel_src, options=f"-DSG_SZ={SG_SZ} -DN_FACTOR={nfactor} -DM_FACTOR={mfactor}")


    # np.random.seed(0)
    up_list = [cl.tensor(up_tensor) for _ in range(rep)]
    gate_list = [cl.tensor(gate_tensor) for _ in range(rep)]
    out_list = [cl.tensor(up_tensor) for _ in range(rep)]

    lws = [1, 1, N_WGZ]
    gws = [M//mfactor, DIV_UP(N//nfactor, N_WGZ), N_WGZ]
    for i in range(0, rep):
        opt_kernel.enqueue("fused", gws, lws, up_list[i], gate_list[i], out_list[i], M, N,  N_WGZ//SG_SZ, 1)
    ns=cl.finish()
    print(ns)


    if check_acc:
        print(f'----------------------------------------------------------------------------------------------------------------------------------')
        print(f'| M:{M}, N:{N}:')
        print(f'| [TEST]: GWS:{gws}, LWS:{lws} MFACTOR:{mfactor} NFACTOR:{nfactor}')
        print(f'----------------------------------------------------------------------------------------------------------------------------------')

        up = torch.from_numpy(up_tensor)
        gate =  torch.from_numpy(gate_tensor)
        silu=torch.nn.SiLU()
        out = torch.mul(silu(up), gate).numpy()
        compare(out, out_list[0].numpy(), rtol=0.1)


cl.profiling(True)
# def test_transposeB_acc():
#     for rank in range(16//16, 256//16):
#         for out_state in range(512//16, 3840//16):
#             #unroll is 4, SG_AZ*4  = 64, input_state%64 == 0
#             for input_state in (1024, 1536, 2048, 2560, 3072):
#                 test_single_lora_transposeB(rank*16, input_state, out_state*16, 1, True)

# def test_transposeB_perf():
#     #Check perf of cusmtomer minicpm
#     for rank in [64]:
#         for out_state in [512, 1536, 3840]:
#             for input_state in [1536]:
#                 test_single_lora_transposeB(rank, input_state, out_state, 20)
#     #Increase workload to ensure memory bound. 
#     for rank in [1536*2]:
#         for out_state in [2048*4]:
#             for input_state in [1536*2]:
#                 test_single_lora_transposeB(rank, input_state, out_state, 20)

# tranposeB = False test
# test_single_lora()

# tranposeB = True test
# Check accuray, will take a while.
# test_transposeB_acc()
# test_transposeB_perf()
test_fused_eltwise(3192, 8192)