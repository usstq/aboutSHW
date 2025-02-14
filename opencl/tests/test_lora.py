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
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        float sum = 0.f;
        for (int i = 0; i < K; i++)
            sum += A[m_idx*K+i]*B[i*N+n_idx];
        CC[m_idx*N+n_idx] = sum;
    }
    __kernel void gemmA_with_split_wei(__global half * A, __global half *BQ, __global half *BK,__global half *BV, __global half *CC, int RANK, int K, int N) {
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

    __kernel void gemmB(__global half * input, __global half *B_Q,  __global half *B_K, __global half *B_V, __global half *CC, int rank) {
        int wgs = get_num_groups(0);
        int wg_id = get_group_id(0);
        int lid = get_local_id(0);
        int n_idx = get_global_id(0);

        __global half *A = input;
        __global half *BN = B_Q + n_idx;
        int stride = TOKEN_HIDDEN_SZ;

        if (wg_id ==  wgs - 2) {
            A += rank;
            BN = B_K + lid;
            stride = KV_HIDDEN_SZ;
        } else if (wg_id ==  wgs - 1) {
            A += rank * 2;
            BN = B_V + lid;
            stride = KV_HIDDEN_SZ;
        }

        float sum = 0.f;
        for (int i = 0; i < rank; i++)
            sum += A[i]*BN[i*stride];
        CC[n_idx] = sum;
    }
    '''
    opt_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A,  __global half *BQ, __global half *BK,__global half *BV, __global half *temp_res, int N, int K) {
        int gid =  get_group_id(0);
        int sgid = get_sub_group_id();
        __global half *C_ptr = temp_res + N * gid + sgid * SG_SZ;
        int k_start = gid * K_PER_WG;
        int k_blk_num = ((k_start + K_PER_WG) > K ?  (K-k_start): K_PER_WG) / SG_SZ;
        int sg_in_rank = RANK / SG_SZ;
        int rank_idx = sgid / sg_in_rank;
        __global half* B = NULL;
        B = (rank_idx == 0) ? BQ : (rank_idx == 1 ?  BK : BV);
        __global half* weight_ptr= B + k_start * RANK + (sgid % sg_in_rank) * SG_SZ;
        __global half* input_ptr= A + k_start;

        half sum = 0;
        for (int k_blk = 0; k_blk < k_blk_num; k_blk++) {
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(input_ptr));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)weight_ptr));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                weight_ptr += RANK;
            }
            input_ptr += SG_SZ;
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''
    TOKEN_HIDDEN_SZ = 1536
    KV_HIDDEN_SZ = 512
    KV_GROUPS_SZ = TOKEN_HIDDEN_SZ // KV_HIDDEN_SZ

    RANK = 64
    N = RANK * 3
    K = TOKEN_HIDDEN_SZ
    # K = 16
    np.random.seed(0)
    vRANGE = 3
    inputA = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    weiA = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    # C = np.matmul(A, B)
    ref_ker = kernel_cache(reference, options=f"-DTOKEN_HIDDEN_SZ={TOKEN_HIDDEN_SZ} -DKV_HIDDEN_SZ={KV_HIDDEN_SZ}")
    # Split [K, N]  to  3 [K, RANK]. Slicing would keep the original stride.So flatten and reshape.
    weiA_Q = weiA[:, 0:RANK].flatten().reshape(K, RANK)
    weiA_K = weiA[:, RANK:2*RANK].flatten().reshape(K, RANK)
    weiA_V = weiA[:, 2*RANK:N].flatten().reshape(K, RANK)
    tRefC = cl.tensor([1, N], np.dtype(np.float16))
    tC = cl.tensor([1, N], np.dtype(np.float16))
    tInput = cl.tensor(inputA)
    tWeiA = cl.tensor(weiA)
    tWeiA_Q = cl.tensor(weiA_Q)
    tWeiA_K = cl.tensor(weiA_K)
    tWeiA_V = cl.tensor(weiA_V)

    ref_ker.enqueue("gemmA", [1, N],[1, RANK], tInput, tWeiA, tRefC, N, K)
    # Each WG would ouput [1, RANK]
    ref_ker.enqueue("gemmA_with_split_wei", [1, N],[1, RANK], tInput, tWeiA_Q, tWeiA_K, tWeiA_V, tC, RANK, K, N)
    cl.finish()
    compare(tRefC.numpy(), tC.numpy())
    SG_SZ = 16
    # Run in parallel in 8 Xecores.
    WG_NUM = 8
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0 and RANK//SG_SZ >= 1, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    assert N <=256, f"'N'  {N} exceeds max value 256"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"

    N_SG_NUM = N // SG_SZ
    K_SG_NUM = K // SG_SZ
    K_PER_WG = ((K_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    tTempC = cl.tensor([WG_NUM, N], np.dtype(np.float16))
    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DK_PER_WG={K_PER_WG} -DRANK={RANK}")
    # N columns in one WG. The WG_NUM would separate K.
    opt_kernel.enqueue("gemmA", [N*WG_NUM],[N], tInput, tWeiA_Q, tWeiA_K, tWeiA_V, tTempC, N, K)
    cl.finish()
    C = np.zeros([1, N], dtype=np.float16)
    tempC = tTempC.numpy()
    for i in range(N):
        for j in range(WG_NUM):
            C[0][i] += tempC[j][i]
    compare(tRefC.numpy(), C)
    # GEMM B
    inputB = tRefC.numpy()
    inputB_Q = inputB[:, 0:RANK].flatten().reshape(1, RANK)
    inputB_K = inputB[:, RANK:2*RANK].flatten().reshape(1, RANK)
    inputB_V = inputB[:, 2*RANK:N].flatten().reshape(1, RANK)

    weiB_Q = np.random.randint(-vRANGE, vRANGE+1, [RANK, TOKEN_HIDDEN_SZ]).astype(np.float16)
    weiB_K = np.random.randint(-vRANGE, vRANGE+1, [RANK, KV_HIDDEN_SZ]).astype(np.float16)
    weiB_V = np.random.randint(-vRANGE, vRANGE+1, [RANK, KV_HIDDEN_SZ]).astype(np.float16)

    ret_Q = np.matmul(inputB_Q, weiB_Q)
    ret_K = np.matmul(inputB_K, weiB_K)
    ret_V = np.matmul(inputB_V, weiB_V)
    ret = np.concatenate((ret_Q, ret_K, ret_V), axis=1, dtype=np.float16)

    tInputB = cl.tensor(inputB)
    tWeiB_Q = cl.tensor(weiB_Q)
    tWeiB_K = cl.tensor(weiB_K)
    tWeiB_V = cl.tensor(weiB_V)
    tRetRef = cl.tensor([1, TOKEN_HIDDEN_SZ+KV_HIDDEN_SZ*2], np.dtype(np.float16))
    ref_ker.enqueue("gemmB", [TOKEN_HIDDEN_SZ+KV_HIDDEN_SZ*2],[KV_HIDDEN_SZ], tInputB, tWeiB_Q, tWeiB_K, tWeiB_V, tRetRef, RANK)
    cl.finish()
    compare(ret, tRetRef.numpy())

# C = Matmul(A, B): A[1, 16], B [16, 16], C [1, 16]
def test_FMA_basic():
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *C, int N, int K) {
        half sum = 0;
        ushort input = intel_sub_group_block_read_us((const __global ushort*)(A));
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < SG_SZ; j++) {
            half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B));
            half aa = as_half(intel_sub_group_broadcast(input, j));
            sum = fma(aa, bb, sum);
            B += N;
        }
        intel_sub_group_block_write_us((const __global ushort*)C, as_short(sum));
    }
    '''
    N = 16
    K = 16
    np.random.seed(0)
    vRANGE = 3
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C_ref = np.matmul(A, B)
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 16
    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ}")
    C = cl.tensor([1, N], np.dtype(np.float16))
    kernel.enqueue("gemm", [N],[N], tA, tB, C, N, K)
    cl.finish()
    compare(C_ref, C.numpy())
cl.profiling(True)
test_lora()
test_FMA_basic()
print("-------------------------")