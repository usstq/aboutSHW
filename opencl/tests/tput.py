
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
                    ptrB += 32;

                }
                ptrA += SG_SZ*8;
            }
        }
        sum0_0 = ''' + "+".join([f"sum{m}_{n}" for m in range(regM) for n in range(regN)]) + r''';
        intel_sub_group_block_write_us((__global ushort*)(C), as_short(sum0_0));
    }
    '''
    print(src)
    cl.profiling(True)
    M = 3072
    N = 64
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
    assert K % BK == 0 , " BK:{BK} is not multiple of SG_SZ:{SG_SZ}"



    
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

# BM = regM*sgM
# BN = regN*sgN*SG_SZ
# GWS = [M//regM, N//(regN)]
# LWS = [sgM, sgN * SG_SZ]
# GWS:[256, 16], LWS:[16, 16], M:1024/64, N:64/64, K:1536/128
def test_FMA_4x4(M, N, K,regM, regN, sgM, sgN):
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *C, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        int regM = 4;
        int regN = 4;
        __local half lA[BM*BK];
        __local half lB[BK*BN];
        //__local half* ptrA = lA;
        //__local half* ptrB = lB;
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;


        half sum0_0 = 0;
        half sum0_1 = 0;
        half sum0_2 = 0;
        half sum0_3 = 0;
        half sum1_0 = 0;
        half sum1_1 = 0;
        half sum1_2 = 0;
        half sum1_3 = 0;
        half sum2_0 = 0;
        half sum2_1 = 0;
        half sum2_2 = 0;
        half sum2_3 = 0;
        half sum3_0 = 0;
        half sum3_1 = 0;
        half sum3_2 = 0;
        half sum3_3 = 0;

        for(int i = 0; i < K; i += BK) {
            // copy B & A into shared local mem
            for(int j = 0; j < BK; j+=SG_SZ) {
                ushort input0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0));
                ushort input1 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + K));
                ushort input2 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 2*K));
                ushort input3 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 3*K));

                //__attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {
                    half bb0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0)));
                    half bb1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 16)));
                    half bb2 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 32)));
                    half bb3 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 48)));
                    half aa0 = as_half(intel_sub_group_broadcast(input0, kk));
                    half aa1 = as_half(intel_sub_group_broadcast(input1, kk));
                    half aa2 = as_half(intel_sub_group_broadcast(input2, kk));
                    half aa3 = as_half(intel_sub_group_broadcast(input3, kk));
                    sum0_0 = fma(aa0, bb0, sum0_0);
                    sum0_1 = fma(aa0, bb1, sum0_1);
                    sum0_2 = fma(aa0, bb2, sum0_2);
                    sum0_3 = fma(aa0, bb3, sum0_3);
                    sum1_0 = fma(aa1, bb0, sum1_0);
                    sum1_1 = fma(aa1, bb1, sum1_1);
                    sum1_2 = fma(aa1, bb2, sum1_2);
                    sum1_3 = fma(aa1, bb3, sum1_3);
                    sum2_0 = fma(aa2, bb0, sum2_0);
                    sum2_1 = fma(aa2, bb1, sum2_1);
                    sum2_2 = fma(aa2, bb2, sum2_2);
                    sum2_3 = fma(aa2, bb3, sum2_3);
                    sum3_0 = fma(aa3, bb0, sum3_0);
                    sum3_1 = fma(aa3, bb1, sum3_1);
                    sum3_2 = fma(aa3, bb2, sum3_2);
                    sum3_3 = fma(aa3, bb3, sum3_3);
                    ptrB += N;
                }
                ptrA +=SG_SZ;
            }
        }
    
        intel_sub_group_block_write_us((__global ushort*)(ptrC), as_short(sum0_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+16), as_short(sum0_1));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+32), as_short(sum0_2));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+48), as_short(sum0_3));
        ptrC += N;
        intel_sub_group_block_write_us((__global ushort*)(ptrC), as_short(sum1_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+16), as_short(sum1_1));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+32), as_short(sum1_2));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+48), as_short(sum1_3));
        ptrC += N;
        intel_sub_group_block_write_us((__global ushort*)(ptrC), as_short(sum2_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+16), as_short(sum2_1));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+32), as_short(sum2_2));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+48), as_short(sum2_3));
        ptrC += N;
        intel_sub_group_block_write_us((__global ushort*)(ptrC), as_short(sum3_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+16), as_short(sum3_1));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+32), as_short(sum3_2));
        intel_sub_group_block_write_us((__global ushort*)(ptrC+48), as_short(sum3_3));
    }
    '''
    cl.profiling(True)

    # np.random.seed(0)
    vRANGE = 1
    REPEAT = 100
    A = np.random.randint(-vRANGE, vRANGE, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C_ref = np.matmul(A, B)
    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]
    SG_SZ = 16
    BK= 128
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    assert M % BM == 0 , f'M:{M} is not multiple of BM:{BM}'
    assert BK % SG_SZ == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'
    assert K % BK == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'

    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    GWS = [M//regM, N//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"

    for i in range(0, REPEAT):
        kernel.enqueue("gemm", GWS, LWS, tA_list[i], tB_list[i],tC_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time in ns:
        print(f'TPUT: {flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    compare(C_ref, tC_list[0].numpy())


# test_FMA_basic(4, 4)
for n in (1536,):
    for m in (3072,):
        for k in (128,):
            test_FMA_4x4(m, n, k, 4, 4, sgM=8, sgN=2)