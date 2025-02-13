#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *
def test_lora():
    reference =  r'''
    __kernel void Wa_gemm(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[m_idx*K+i]*B[i*N+n_idx];
        CC[m_idx*N+n_idx] = sum;
    }
    __kernel void split_Wa_gemm(__global half * A, __global half *BQ, __global half *BK,__global half *BV, __global half *CC, int RANK, int K, int N) {
        int n = get_global_id(1);
        int n_blk = get_group_id(1);
        __global half*B = NULL;
        if (n_blk == 0)
            B = BQ;
        else if (n_blk == 1)
            B = BK;
        else
            B = BV;
        int m = get_global_id(0);
        int sub_n = get_local_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[m*K+i]*B[i*RANK+sub_n];
        CC[m*N+n] = sum;
    }
    '''
    opt_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void Wa_gemm(__global half * A, __global half *B,  __global half *temp_res, int N, int K) {
        int gid =  get_group_id(0);
        int sgid = get_sub_group_id();
        __global half *C_ptr = temp_res + N * gid + sgid * SG_SZ;
        int k_start = gid * K_PER_WG;
        int k_num = (k_start + K_PER_WG) > K ?  (K-k_start): K_PER_WG;
        __global half* weight_ptr= B + k_start * N + SG_SZ * sgid;
        __global half* input_ptr= A + k_start;

        half sum = 0;
        for (int k_blk = 0; k_blk < k_num/SG_SZ; k_blk++) {
            ushort input = intel_sub_group_block_read_us((const __global ushort*)((input_ptr + k_blk * SG_SZ)));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)weight_ptr));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                weight_ptr += N;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''
    MINICPM_HIDDEN_SZ = 1536
    RANK = 64
    N = RANK * 3
    K = MINICPM_HIDDEN_SZ
    np.random.seed(0)
    vRANGE = 3
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    # C = np.matmul(A, B)
    ref_ker = kernel_cache(reference, "")
    # BQ = B[:, 0:RANK].flatten().reshape(K, RANK)
    # BK = B[:, RANK:2*RANK].flatten().reshape(K, RANK)
    # BV = B[:, 2*RANK:N].flatten().reshape(K, RANK)
    tRefC = cl.tensor([1, N], np.dtype(np.float16))
    tC = cl.tensor([1, N], np.dtype(np.float16))
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    # tBQ = cl.tensor(BQ)
    # tBK = cl.tensor(BK)
    # tBV = cl.tensor(BV)

    ref_ker.enqueue("Wa_gemm", [1, N],[1, RANK], tA, tB, tRefC, N, K)
    cl.finish()
    # ref_ker.enqueue("split_Wa_gemm", [1, N],[1, RANK], tA, tBQ, tBK, tBV, tRefC, RANK, K, N)
    SG_SZ = 16
    # There are 8 Xecores, Run in parallel in 8 Xecores. can also try multiple of xecore number.
    WG_NUM = 8
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"
    N_SG_NUM = N // SG_SZ
    K_SG_NUM = K // SG_SZ
    K_PER_WG = ((K_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    tTemp = cl.tensor([WG_NUM, N], np.dtype(np.float16))    
    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DK_PER_WG={K_PER_WG}")
    opt_kernel.enqueue("Wa_gemm", [N*WG_NUM],[N], tA, tB, tTemp, N, K)
    C = np.zeros([1, N], dtype=np.float16)
    tempRes = tTemp.numpy()
    for i in range(N):
        for j in range(WG_NUM):
            C[0][i] += tempRes[j][i]
    compare(tRefC.numpy(), C)
    cl.finish()


def test_16x16():
    reference =  r'''
    __kernel void Wa_gemm(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[m_idx*K+i]*B[i*N+n_idx];
        CC[m_idx*N+n_idx] = sum;
    }
    __kernel void split_Wa_gemm(__global half * A, __global half *BQ, __global half *BK,__global half *BV, __global half *CC, int RANK, int K, int N) {
        int n = get_global_id(1);
        int n_blk = get_group_id(1);
        __global half*B = NULL;
        if (n_blk == 0)
            B = BQ;
        else if (n_blk == 1)
            B = BK;
        else
            B = BV;
        int m = get_global_id(0);
        int sub_n = get_local_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[m*K+i]*B[i*RANK+sub_n];
        CC[m*N+n] = sum;
    }
    '''
    opt_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void Wa_gemm(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int gid =  get_group_id(0);
        int sgid = get_sub_group_id();
        __global half *C_ptr = CC + gid* N + 16*sgid;
        __global half* weight_ptr= B + gid * K_PER_WG * N + 16*sgid;
        __global half* input_ptr= A + gid * K_PER_WG;

        half sum = 0;
        for (int k_blk = 0; k_blk < K_PER_WG/16; k_blk++) {
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(input_ptr + k_blk * 16));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)weight_ptr));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                //if (j == 0)
                    //printf("input[%d]:%f, aa:%f, bb:%f\n",get_sub_group_local_id(), input, aa, bb);
                sum = fma(aa, bb, sum);
                weight_ptr += N;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''

    N = 192
    K = 64
    WG_NUM = 4
    K_PER_WG = K // WG_NUM
    np.random.seed(0)
    vRANGE = 3
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    # B layout is [K, N]
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.matmul(A, B)
    # print("A:", A)
    # print("B0:", B[0,:])
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 16
    tTemp = cl.tensor([WG_NUM,N], np.dtype(np.float16))     
    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DK_PER_WG={K_PER_WG}")
    opt_kernel.enqueue("Wa_gemm", [N*WG_NUM],[N], tA, tB, tTemp, N, K)
    # compare(C, CC.numpy())
    CC = np.zeros([1, N], dtype=np.float16)
    tempRes = tTemp.numpy()
    for i in range(N):
        for j in range(WG_NUM):
            CC[0][i] += tempRes[j][i]
    compare(C, CC)
    cl.finish()

cl.profiling(True)
test_lora()
#test_16x16()
print("-------------------------")