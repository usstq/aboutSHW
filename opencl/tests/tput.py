
#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

def test_FMA_basic(regM, regN):
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *C, int M, int N, int K) {
        int sg_id = get_sub_group_id();
        __local half lA[BM*BK];
        __local half lB[BK*BN];
        __local half* ptrA = lA;
        __local half* ptrB = lB;
        ''' + "half " + ",".join([f"sum{m}_{n}" for m in range(regM) for n in range(regN)]) + r''';

        for(int i = 0; i < K; i += BK) {
            // copy B & A into shared local mem
            for(int k = 0; k < BK; k+=SG_SZ) {
                ''' +  "\n\t\t".join([f"ushort input{m} = intel_sub_group_block_read_us((const __local ushort*)(ptrA + {16*m}));" for m in range(regM)]) + r'''


                //__attribute__((opencl_unroll_hint))
                for (int j = 0; j < SG_SZ; j++) {
                    ''' + "\n\t\t\t".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __local ushort*)(ptrB + j*32 + {16*n})));" for n in range(regN)]) + r'''
                    ''' + "\n\t\t\t".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, j));" for m in range(regM)]) + r'''
                    ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                }
                ptrA += SG_SZ*8;
                ptrB += SG_SZ*32;
            }
        }
        sum0_0 = ''' + "+".join([f"sum{m}_{n}" for m in range(regM) for n in range(regN)]) + r''';
        intel_sub_group_block_write_us((__global ushort*)(C), as_short(sum0_0));
    }
    '''
    print(src)
    cl.profiling(True)
    M = 3072
    N = 128
    K = 1536
    # np.random.seed(0)
    vRANGE = 2
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C_ref = np.matmul(A, B)
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 16
    BK=128
    BM = regM*16
    BN = N
    
    assert BN % (regN * SG_SZ) == 0 , " BN:{BN} is not multiple of {regN * SG_SZ}"
    assert M % BM == 0 , " M:{M} is not multiple of BM:{BM}"
    assert BK % SG_SZ == 0 , " BK:{BK} is not multiple of SG_SZ:{SG_SZ}"


    
    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN}")
    tC = cl.tensor([M, N], np.dtype(np.float16))
    GWS = [M//regM, N//(regN)]
    LWS = [BM//regM, BN//(regN)]
    for i in range(0, 20):
        kernel.enqueue("gemm", GWS, LWS, tA, tB,tC, M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2 
    for time in ns:
        print(f'TPUT: {flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')


test_FMA_basic(4, 4)
