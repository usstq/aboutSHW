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

def DIV_UP(a, b):
    return (a + b -1) // b

def test_FMA_basic_fp16():
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *C, int N, int K) {
        int sg_id = get_sub_group_id();
        int n_idx = sg_id * SG_SZ;
        if ((n_idx + SG_SZ) > N)
            n_idx = N - SG_SZ;

        __global half *B_ptr = B+n_idx;
        __global half *C_ptr = C+n_idx;
        __global half *A_ptr = A;


        half sum = 0.f;
        for(int i = 0; i < K; i += SG_SZ) {
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                B_ptr += N;
            }
            A_ptr += SG_SZ;
        }
        intel_sub_group_block_write_us((const __global ushort*)(C_ptr), as_short(sum));
    }
    '''
    N = 32
    K = 16
    # np.random.seed(0)
    vRANGE = 2
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C_ref = np.matmul(A, B)
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 16
    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ}")
    C = cl.tensor([1, N], np.dtype(np.float16))
    kernel.enqueue("gemm", [ALIGN_UP(N, SG_SZ)],[ALIGN_UP(N, SG_SZ)], tA, tB, C, N, K)
    cl.finish()
    compare(C_ref, C.numpy())
    

def test_FMA_basic_fp32():
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global float * A, __global float *B,  __global float *C, int N, int K) {
        int sg_id = get_sub_group_id();
        int n_idx = sg_id * SG_SZ;
        if ((n_idx + SG_SZ) > N)
            n_idx = N - SG_SZ;

        __global float *B_ptr = B+n_idx;
        __global float *C_ptr = C+n_idx;
        __global float *A_ptr = A;


        float sum = 0.f;
        for(int i = 0; i < K; i += SG_SZ) {

            uint input = intel_sub_group_block_read_ui((const __global uint*)(A_ptr));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                float bb = as_float(intel_sub_group_block_read_ui((const __global uint*)(B_ptr)));
                float aa = as_float(sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                B_ptr += N;
            }
            A_ptr += SG_SZ;
        }
        intel_sub_group_block_write_ui((const __global uint*)(C_ptr), as_uint(sum));
    }
    '''
    N = 31
    K = 16
    # np.random.seed(0)
    vRANGE = 2
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float32)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float32)
    C_ref = np.matmul(A, B)
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 8
    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ}")
    C = cl.tensor([1, N], np.dtype(np.float32))
    kernel.enqueue("gemm", [ALIGN_UP(N, SG_SZ)],[ALIGN_UP(N, SG_SZ)], tA, tB, C, N, K)
    cl.finish()
    compare(C_ref, C.numpy())
    
def test_FMA_blocking_fp32(M, N, K,regM, regN, sgM, sgN, withscale = False, withSum=False):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    ''' + f'{func}' + r'''(__global float * A, __global float *B,  __global float *C, __global float *alpha,  __global float *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        int lid =  get_sub_group_local_id();

        int regM = 1;
        int regN = 1;

        __local uint TAIL_MEMORY[SG_SZ*sgN*sgM];
        __local uint* A_tail = &TAIL_MEMORY[(sgid_M * sgN + sgid_N)*SG_SZ];
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;

        __global float *ptrA = A + m_idx * K;
        __global float *ptrB = B + n_idx;
        __global float *ptrC = C + m_idx * N + n_idx;
        float sum0_0 = 0;;
        int i = 0;
        for(i = 0; (i+SG_SZ) <= K; i += SG_SZ) {
                uint input0 = intel_sub_group_block_read_ui((const __global uint*)(ptrA + 0 * K));
                //uint input0 = as_uint(*(ptrA + 0 * K + lid));
                for (int kk = 0; kk < SG_SZ; kk++) {
                     float bb0 = as_float(intel_sub_group_block_read_ui((const __global uint*)(ptrB + 0 * SG_SZ)));
                     float aa0 = as_float(sub_group_broadcast(input0, kk));
                     sum0_0 = fma(aa0, bb0, sum0_0);
                    ptrB += N;
                }
                ptrA +=SG_SZ;
        }
        if (i < K) {
                int tail_len = K - i;
                int offset = (lid < tail_len) ? lid : 0;
# if 1
                *(A_tail + lid) = as_uint(*(ptrA + 0 * K + offset));
                //barrier(CLK_LOCAL_MEM_FENCE);
                uint src0 = intel_sub_group_block_read_ui((const __local uint*)(A_tail));
                //uint src0 = as_uint(*(A_tail + lid));
                for (int kk = 0; kk < tail_len; kk++) {
                        float bbb0 = as_float(intel_sub_group_block_read_ui((const __global uint*)(ptrB + 0 * SG_SZ)));
                        float aaa0 = as_float(sub_group_broadcast(src0, kk));
                        sum0_0 = fma(aaa0, bbb0, sum0_0);
                        ptrB += N;
                }
#else
                uint src0 = as_uint(*(ptrA + 0 * K + offset));
                for (int kk = 0; kk < tail_len; kk++) {
                        float bbb0 = as_float(intel_sub_group_block_read_ui((const __global uint*)(ptrB + 0 * SG_SZ)));
                        float aaa0 = as_float(sub_group_broadcast(src0, kk));
                        sum0_0 = fma(aaa0, bbb0, sum0_0);
                    ptrB += N;
                }

#endif
        }    
        intel_sub_group_block_write_ui((__global uint*)(ptrC + SG_SZ * 0), as_uint(sum0_0));
        ptrC += N;
}
    '''
    

    # print(src)
    cl.profiling(True)

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 1
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float32)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float32)
    alpha = np.random.rand(N).astype(np.float32)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float32)
    if withscale:
        C_ref = np.multiply(np.matmul(A, B), alpha)
    elif withSum:
        C_ref = np.add(np.matmul(A, B), mainInput)
    else:
        C_ref = np.matmul(A, B)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tAlpha_list =  [cl.tensor(alpha) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float32)) for _ in range(REPEAT)]
    tMaininput_list = [cl.tensor(mainInput) for _ in range(REPEAT)]

    SG_SZ = 8
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    
    assert M >= regM , f'M:{M} is smaller than regM:{regM}'
    assert N >= regN*SG_SZ , f'N:{N} is smaller than :regN*SG_SZ {regN*SG_SZ}'
    assert N % SG_SZ == 0 , f'N:{N} is not multiple of SG_SZ:{SG_SZ}'

    # assert BK % SG_SZ == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'
    # assert K % BK == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'

    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    GWS = [ALIGN_UP(M, BM)//regM , ALIGN_UP(N, BN)//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()
    for i in range(0, REPEAT):
        kernel.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
    ns = cl.finish()
    total_wgs = (M // BM) * (N //BN)
    flops = M * N * K * 2
    rd_b = (M+N) * K *2
    total_rd_b = total_wgs * (BM +BN) * K * 2
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}, SGM:{sgM}, SGN:{sgN}')
    print("----------------------------------------------------")
    # print(C_ref)
    # print(tC_list[0].numpy())
    compare(C_ref, tC_list[0].numpy())

def test_FMA_blocking_fp16(M, N, K,regM, regN, sgM, sgN, withscale = False, withSum=False):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        int lid =  get_sub_group_local_id();

        int regM = 1;
        int regN = 1;

        __local ushort TAIL_MEMORY[SG_SZ*sgN*sgM];
        __local ushort* A_tail = &TAIL_MEMORY[(sgid_M * sgN + sgid_N)*SG_SZ];
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;

        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;
        half sum0_0 = 0;;
        int i = 0;
        for(i = 0; (i+SG_SZ) <= K; i += SG_SZ) {
                ushort input0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0 * K));
                //ushort input0 = as_ushort(*(ptrA + 0 * K + lid));
                for (int kk = 0; kk < SG_SZ; kk++) {
                     half bb0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SG_SZ)));
                     half aa0 = as_half(intel_sub_group_broadcast(input0, kk));
                     sum0_0 = fma(aa0, bb0, sum0_0);
                    ptrB += N;
                }
                ptrA +=SG_SZ;
        }
        if (i < K) {
                int tail_len = K - i;
                int offset = (lid < tail_len) ? lid : 0;
# if 1
                *(A_tail + lid) = as_ushort(*(ptrA + 0 * K + offset));
                barrier(CLK_LOCAL_MEM_FENCE);
                ushort src0 = intel_sub_group_block_read_us((const __local ushort*)(A_tail));
                //ushort src0 = as_ushort(*(A_tail + lid));
                for (int kk = 0; kk < tail_len; kk++) {
                        half bbb0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SG_SZ)));
                        half aaa0 = as_half(intel_sub_group_broadcast(src0, kk));
                        sum0_0 = fma(aaa0, bbb0, sum0_0);
                        ptrB += N;
                }
#else
                ushort src0 = as_ushort(*(ptrA + 0 * K + offset));
                for (int kk = 0; kk < tail_len; kk++) {
                        half bbb0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SG_SZ)));
                        half aaa0 = as_half(intel_sub_group_broadcast(src0, kk));
                        sum0_0 = fma(aaa0, bbb0, sum0_0);
                    ptrB += N;
                }

#endif
        }    
        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum0_0));
        ptrC += N;
}
    '''
    

    # print(src)
    cl.profiling(True)

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 1
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    alpha = np.random.rand(N).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)
    if withscale:
        C_ref = np.multiply(np.matmul(A, B), alpha)
    elif withSum:
        C_ref = np.add(np.matmul(A, B), mainInput)
    else:
        C_ref = np.matmul(A, B)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tAlpha_list =  [cl.tensor(alpha) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]
    tMaininput_list = [cl.tensor(mainInput) for _ in range(REPEAT)]

    SG_SZ = 16
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    
    assert M >= regM , f'M:{M} is smaller than regM:{regM}'
    assert N >= regN*SG_SZ , f'N:{N} is smaller than :regN*SG_SZ {regN*SG_SZ}'
    assert N % SG_SZ == 0 , f'N:{N} is not multiple of SG_SZ:{SG_SZ}'

    # assert BK % SG_SZ == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'
    # assert K % BK == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'

    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    GWS = [ALIGN_UP(M, BM)//regM , ALIGN_UP(N, BN)//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()
    for i in range(0, REPEAT):
        kernel.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
    ns = cl.finish()
    total_wgs = (M // BM) * (N //BN)
    flops = M * N * K * 2
    rd_b = (M+N) * K *2
    total_rd_b = total_wgs * (BM +BN) * K * 2
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}, SGM:{sgM}, SGN:{sgN}')
    print("----------------------------------------------------")
    # print(C_ref)
    # print(tC_list[0].numpy())
    compare(C_ref, tC_list[0].numpy())

'''FP32 N-tail summary: 
# No limitation for now.
''' 
# test_FMA_basic_fp32()

'''FP16 N-tail summary: 
# N must be odd when M=1 or M > 1;
''' 
# test_FMA_basic_fp16()

# print("-------------------------")
cl.profiling(True)
# test_FMA_blocking_fp16(4, N=16, K=24, regM=1, regN=1, sgM=2, sgN=1)

'''FP16 K-tail summary: 
# when m = 1, K can be odd or even  for FP16. so no limitation for 2nd token.
# when m > 1, K <16, K can be odd or even for FP16.
# when m >1 and K >=16, K must be odd for the tail case(K%16 !=0).
# above all, not aligned K must be even for FP16. doesn't know why.
''' 
for k_len in range (16, 32, 2):    
    test_FMA_blocking_fp16(2, N=16, K=k_len, regM=1, regN=1, sgM=1, sgN=1)


'''FP32 K-tail summary: 
# No limitation for now.
''' 
# for k_len in range (16, 48):    
#     test_FMA_blocking_fp32(4, N=16, K=k_len, regM=1, regN=1, sgM=1, sgN=1)


print("-------------------------")