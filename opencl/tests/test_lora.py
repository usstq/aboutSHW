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
    __kernel void reduceA(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk) {
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int n_idx = get_global_id(0);
        size_t blk_num = (K + k_blk -1 )/ k_blk;
        half sum = 0.f;
        for (int i = 0; i < blk_num; i++) {
            half sub_sum = 0.f;
            int k_idx = i*k_blk;
            // incase K%k_blk != 0;
            if ((k_idx + k_blk) > K)
                k_blk = K - k_idx;
            for (size_t sub_k = 0; sub_k < k_blk; sub_k++) {
                sub_sum = fma(A[k_idx], B[k_idx*N+n_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        CC[n_idx] = sum;
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
        half sum = 0.f;
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
            stride = KV_PROJ_SZ;
        } else if (wg_id ==  wgs - 1) {
            A += rank * 2;
            BN = B_V + lid;
            stride = KV_PROJ_SZ;
        }

        half sum = 0.f;
        for (int i = 0; i < rank; i++)
            //sum += A[i]*BN[i*stride];
            sum = fma(A[i], BN[i*stride], sum);
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
        int k_len = (k_start + K_PER_WG) > K ?  (K-k_start): K_PER_WG;
        int sg_in_rank = RANK / SG_SZ;
        int rank_idx = sgid / sg_in_rank;
        __global half* B = NULL;
        B = (rank_idx == 0) ? BQ : (rank_idx == 1 ?  BK : BV);
        __global half* weight_ptr= B + k_start * RANK + (sgid % sg_in_rank) * SG_SZ;
        __global half* input_ptr= A + k_start;

        half sum = 0;
        for (int kk = 0;  kk < k_len; kk += SG_SZ) {
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

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * input_ptr,  __global half *BQ, __global half *BK,__global half *BV,  __global half *CC, int A_stride) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int sg_idx = wg_id * sg_num  +  sg_id;
        int id_sg_local = get_sub_group_local_id();
#if USE_SLM_GEMMB
        __local half reduce[RANK];
#else
        half reduce[RANK] = {0.f};
#endif

        __global half *C_ptr = CC + sg_idx * SG_SZ;
        // Default A, B for Q projection
        __global half *A_ptr = input_ptr;
        __global half *B_ptr = BQ + sg_idx * SG_SZ;
        int B_stride = KV_PROJ_SZ * KV_GROUP_SZ;

        if (sg_idx >= KV_PROJ_SZ * (KV_GROUP_SZ + 1) / SG_SZ) {
            // V projection
            A_ptr += RANK * 2;
            B_ptr = BV + sg_idx * SG_SZ - KV_PROJ_SZ * (KV_GROUP_SZ + 1);
            B_stride = KV_PROJ_SZ;
        } else if (sg_idx >=KV_PROJ_SZ * KV_GROUP_SZ / SG_SZ) {
            // K projection
            A_ptr += RANK;
            B_ptr = BK + sg_idx * SG_SZ - KV_PROJ_SZ * KV_GROUP_SZ;
            B_stride = KV_PROJ_SZ;
        }

        //1. Reduce
#if USE_SLM_GEMMB
        //EACH WG would reduce the data and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        //sg would diverge here. not all the sgs can satisfy 'offset < RANK'.
        for (int offset = sg_id * SG_SZ; offset < RANK; offset += local_sz) {
            __global half *subA_ptr = A_ptr + offset;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)subA_ptr));
                reduce[offset + id_sg_local] += partial_val;
                subA_ptr += A_stride;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#else

        for (int i = 0; i < PART_NUM; i++) {
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < RANK; j++) {
                reduce[j] += A_ptr[i * A_stride + j];
            }
        }
#endif
        //2.  GEMMB
        half sum = 0;
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            half input = reduce[id_sg_local + kk];
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += B_stride;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''
    TOKEN_HIDDEN_SZ = 1536
    KV_PROJ_SZ = 512
    KV_GROUP_SZ = TOKEN_HIDDEN_SZ // KV_PROJ_SZ
    RANK = 64
    N = RANK * 3
    K = TOKEN_HIDDEN_SZ
    WG_NUM = 7
    SG_SZ = 16
    # Run in parallel in 8 Xecores.
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0 and RANK//SG_SZ >= 1, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    assert N <=256, f"'N'  {N} exceeds max value 256"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"

    N_SG_NUM = N // SG_SZ
    K_SG_NUM = K // SG_SZ
    K_PER_WG = ((K_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    USE_RANDINT = 1
    # gemmb used 8 EUs per Xecore to balance the workload between Xecores(3, 4, 7, 8 Xecores can has)
    GEMMB_LOCAL_SIZE = 128
    # Each Xecore should share the same [1,RANK] input for gemmb.If  KV_PROJ_SZ % GEMMB_LOCAL_SIZE, means  Xecore  would use 2 different inputs.
    USE_SLM_GEMMB = 1 if KV_PROJ_SZ % GEMMB_LOCAL_SIZE == 0 else 0
    # USE_SLM_GEMMB = 1
    # np.random.seed(0)
    if USE_RANDINT:
        vRANGE = 2
        inputA = np.random.randint(-vRANGE, vRANGE, [1, K]).astype(np.float16)
        weiA = np.random.randint(-vRANGE, vRANGE, [K, N]).astype(np.float16)
        weiB_Q = np.random.randint(-vRANGE, vRANGE, [RANK, TOKEN_HIDDEN_SZ]).astype(np.float16)
        weiB_K = np.random.randint(-vRANGE, vRANGE, [RANK, KV_PROJ_SZ]).astype(np.float16)
        weiB_V = np.random.randint(-vRANGE, vRANGE, [RANK, KV_PROJ_SZ]).astype(np.float16)
    else :
        inputA = np.random.rand(1, K).astype(np.float16)
        weiA = np.random.rand(K, N).astype(np.float16)
        weiB_Q = np.random.rand(RANK, TOKEN_HIDDEN_SZ).astype(np.float16)
        weiB_K = np.random.rand(RANK, KV_PROJ_SZ).astype(np.float16)
        weiB_V = np.random.rand(RANK, KV_PROJ_SZ).astype(np.float16)
    # Split [K, N]  to  3 [K, RANK]. Slicing would keep the original stride.So flatten and reshape.
    weiA_Q = weiA[:, 0:RANK].flatten().reshape(K, RANK)
    weiA_K = weiA[:, RANK:2*RANK].flatten().reshape(K, RANK)
    weiA_V = weiA[:, 2*RANK:N].flatten().reshape(K, RANK)
    tOutputA_Ref = cl.tensor([1, N], np.dtype(np.float16))
    tInput = cl.tensor(inputA)
    tWeiA = cl.tensor(weiA)
    tWeiA_Q = cl.tensor(weiA_Q)
    tWeiA_K = cl.tensor(weiA_K)
    tWeiA_V = cl.tensor(weiA_V)

    tWeiB_Q = cl.tensor(weiB_Q)
    tWeiB_K = cl.tensor(weiB_K)
    tWeiB_V = cl.tensor(weiB_V)

    tOutputB_Ref = cl.tensor([1, TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2], np.dtype(np.float16))
    tOutputA = cl.tensor([WG_NUM, N], np.dtype(np.float16))
    toutputA_Reduce = cl.tensor([1, N], np.dtype(np.float16))
    tOutputB = cl.tensor([1, TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2], np.dtype(np.float16))

    ref_ker = kernel_cache(reference, options=f"-DTOKEN_HIDDEN_SZ={TOKEN_HIDDEN_SZ} -DKV_PROJ_SZ={KV_PROJ_SZ}")
    ref_ker.enqueue("gemmA", [N],[RANK], tInput, tWeiA, tOutputA_Ref, N, K, K_PER_WG)
    ref_ker.enqueue("gemmB", [TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2],[KV_PROJ_SZ], tOutputA_Ref, tWeiB_Q, tWeiB_K, tWeiB_V, tOutputB_Ref, RANK)
    cl.finish()
    # Each WG would ouput [1, RANK]
    #tC = cl.tensor([1, N], np.dtype(np.float16))
    #ref_ker.enqueue("gemmA_with_split_wei", [1, N],[1, RANK], tInput, tWeiA_Q, tWeiA_K, tWeiA_V, tC, RANK, K, N)
    # cl.finish()
    #compare(tOutputA_Ref.numpy(), tC.numpy())

    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DPART_NUM={WG_NUM} -DK_PER_WG={K_PER_WG} -DRANK={RANK} -DKV_PROJ_SZ={KV_PROJ_SZ} -DKV_GROUP_SZ={KV_GROUP_SZ} -DUSE_SLM_GEMMB={USE_SLM_GEMMB}")
    # N columns in one WG. The WG_NUM would separate K.
    opt_kernel.enqueue("gemmA", [N*WG_NUM],[N], tInput, tWeiA_Q, tWeiA_K, tWeiA_V, tOutputA, N, K)
    # ref_ker.enqueue("reduceA", [N],[N], tOutputA, toutputA_Reduce, N, WG_NUM)
    opt_kernel.enqueue("gemmB", [TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2],[GEMMB_LOCAL_SIZE], tOutputA, tWeiB_Q, tWeiB_K, tWeiB_V, tOutputB, N)
    cl.finish()
    # compare(tOutputA_Ref.numpy(), toutputA_Reduce.numpy())
    compare(tOutputB_Ref.numpy(), tOutputB.numpy())

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
# test_FMA_basic()
print("-------------------------")