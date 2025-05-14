#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *


def ALIGN_UP(a, b):
    return ((a + (b -1)) // b *b)

def ALIGN_DOWN(a, b):
    return (a // b * b)


cl_kernel_sources_ref = r'''
    __kernel void gemm(__global half * A, __global half *B,  __global half *CC, int M,  int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i], B[n_idx*K + i], sum);
        CC[m_idx * N + n_idx] = sum;
    }
'''

# BM = regM*sgM
# BN = regN*sgN*SG_SZ
# GWS = [M//regM, N//(regN)]
# LWS = [sgM, sgN * SG_SZ]
# GWS:[256, 16], LWS:[16, 16], M:1024/64, N:64/64, K:1536/128
def gen_store_C(regM, regN):
    src = ""
    for m in range(regM):
        for n in range(regN):
            src +=  f"\n\tptrC[{n}] = sum{m}_{n};"
        src += f"\n\tptrC += N;\n\n"
    return src

def test_FMA(M, N, K,regM, regN, sgM, sgN, bk):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    gen_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * regN;
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx * K;
        __global half *ptrC = C + m_idx * N + n_idx;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int i = 0; i < K; i += BK) {
            __attribute__((opencl_unroll_hint))
            for(int j = 0; j < BK; j += SG_SZ) {

            // copy B & A into shared local mem

                '''  + "\n\t\t ".join([f"half aa{m} = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * K)));" for m in range(regM)]) + r'''
                '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * K)));" for n in range(regN)]) + r'''
                ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                ptrB += SG_SZ;
                ptrA += SG_SZ;
            }
        }
        ''' + "\n\t\t\t".join([f"sum{m}_{n} = sub_group_reduce_add(sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''

        if (get_sub_group_local_id() == 0) {

        ''' +  gen_store_C(regM, regN) + r'''
        }
    }
    '''
    print(gen_src)
    cl.profiling(True)

    SG_SZ = 16
    BM = regM*sgM
    BN = regN*sgN

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 20
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    B_tr = np.transpose(B)
    # print(B_tr)
    C = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor(C) for _ in range(REPEAT)]

    kernel_opt = kernel_cache(gen_src, options=f"-DSG_SZ={SG_SZ}  -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN} -DBK={bk}")

    GWS = [M//regM , N//(regN)*SG_SZ]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        kernel_opt.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time in ns:
        print(f'TPUT: [NO_SLM]:{flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}')
    print("----------------------------------------------------")
    if 1:
        tC_ref = cl.tensor(C)
        kernel_ref = kernel_cache(cl_kernel_sources_ref, options=f"-DSG_SZ={SG_SZ}")
        kernel_ref.enqueue("gemm", [M, N], [1 , 1], tA_list[0], tB_list[0],tC_ref, M, N, K)
        cl.finish()
        compare(tC_ref.numpy(), tC_slm_list[0].numpy())

SG_SZ =16
regM=8
sgM=8

regN=8
sgN=8

WGS_M = 16
WGS_N = 32

bk=64
K = bk*64
M = regM * sgM * WGS_M
N = regN * sgN * WGS_N

assert K%bk == 0
# A770 FMA tput: 512 XVE *16 lanes *2 FMA_OPS *2.4G = 39.3 TFLOPS

test_FMA(M, N, K,regM, regN, sgM, sgN, bk)
