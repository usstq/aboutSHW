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
        int k_start = gid * K_BLOCK;
        int k_len = (k_start + K_BLOCK) > K ?  (K-k_start): K_BLOCK;
        __global half* weight_ptr= B + k_start * N + SG_SZ * sgid;
        __global half* input_ptr= A + k_start;

        half sum = 0;
        for (int i = 0; i < k_len; i+=SG_SZ) {
            input_ptr += SG_SZ;
            ushort input = intel_sub_group_block_read_us((const __global ushort*)input_ptr);
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < 16; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)weight_ptr));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                if (j == 0)
                    printf("k_start:[%d],sgid:[%d],input[%d]:%f, aa:%f, bb:%f\n",k_start, sgid, get_sub_group_local_id(), input, aa, bb);
                sum = fma(aa, bb, sum);
                weight_ptr += N;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''

    # RANK = 64
    # N = RANK * 3
    # K = 1536
    RANK = 16
    N = RANK
    K = 16
    np.random.seed(0)
    vRANGE = 3
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    # B layout is [K, N]
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.matmul(A, B)
    print("A:", A)
    print("B0:", B[0,:])
    # ref_ker = kernel_cache(reference, "")
    # BQ = B[:, 0:RANK].flatten().reshape(K, RANK)
    # BK = B[:, RANK:2*RANK].flatten().reshape(K, RANK)
    # BV = B[:, 2*RANK:N].flatten().reshape(K, RANK)
    # tRefC = cl.tensor([1, N], np.dtype(np.float16))
    # tC = cl.tensor([1, N], np.dtype(np.float16))
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    # tBQ = cl.tensor(BQ)
    # tBK = cl.tensor(BK)
    # tBV = cl.tensor(BV)

    # ref_ker.enqueue("Wa_gemm", [1, N],[1, RANK], tA, tB, tRefC, N, K)
    # cl.finish()
    # compare(C, tRefC.numpy())

    # ref_ker.enqueue("split_Wa_gemm", [1, N],[1, RANK], tA, tBQ, tBK, tBV, tRefC, RANK, K, N)
    # compare(C, tRefC.numpy())
    SG_SZ = 16
    # There are 8 Xecores, Run in parallel in 8 Xecores. can also try multiple of xecore number.
    K_IN_PARALLEL = 1
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"
    K_SG_NUM = K // SG_SZ
    K_BLOCK = ((K_SG_NUM + K_IN_PARALLEL - 1) // K_IN_PARALLEL) * SG_SZ
    N_SG_NUM = N // SG_SZ
    tempRes = cl.tensor([K_IN_PARALLEL, N], np.dtype(np.float16))    
    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DK_BLOCK={K_BLOCK}")
    opt_kernel.enqueue("Wa_gemm", [N*K_IN_PARALLEL],[N], tA, tB, tempRes, N, K)
    compare(C, tempRes.numpy())
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
        __global half *C_ptr = CC + 16*sgid;
        __global half* weight_ptr= B + 16*sgid;
        __global half* input_ptr= A;

        half sum = 0;
        for (int k_blk = 0; k_blk < K/16; k_blk++) {
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
    K = 64*3
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
    CC = cl.tensor([N], np.dtype(np.float16))    
    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ}")
    opt_kernel.enqueue("Wa_gemm", [N],[N], tA, tB, CC, N, K)
    compare(C, CC.numpy())
    # print("C:", C)
    # print("CC:", CC.numpy())
    cl.finish()

cl.profiling(True)
# test_lora()
test_16x16()
print("-------------------------")