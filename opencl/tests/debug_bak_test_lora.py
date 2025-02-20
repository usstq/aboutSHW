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
    __kernel void add(__global half * A, __global half *B,  __global half *C) {
        size_t id = get_global_linear_id();
        C[id] = A[id] + B[id];
    }
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx*K+i], B[i*N+n_idx], sum);
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
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m*K+i], B[i*RANK+sub_n], sum);

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
    __kernel void gemmB(__global half * input_ptr,  __global half *BQ, __global half *BK,__global half *BV,  __global half *CC) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int sg_idx = wg_id * sg_num  +  sg_id;

        __global half *C_ptr = CC + sg_idx * SG_SZ;
        // Default A, B for Q projection
        __global half *A_ptr = input_ptr;
        __global half *B_ptr = BQ + sg_idx * SG_SZ;
        int stride_B = KV_PROJ_SZ * KV_GROUP_SZ;

        if (sg_idx >= KV_PROJ_SZ * (KV_GROUP_SZ + 1) / SG_SZ) {
            // V projection
            A_ptr += RANK * 2;
            B_ptr = BV + sg_idx * SG_SZ - KV_PROJ_SZ * (KV_GROUP_SZ + 1);
            stride_B = KV_PROJ_SZ;
        } else if (sg_idx >=KV_PROJ_SZ * KV_GROUP_SZ / SG_SZ) {
            // K projection
            A_ptr += RANK;
            B_ptr = BK + sg_idx * SG_SZ - KV_PROJ_SZ * KV_GROUP_SZ;
            stride_B = KV_PROJ_SZ;
        }
        half sum = 0;
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                B_ptr += stride_B;
            }
            A_ptr += SG_SZ;
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
    np.random.seed(0)
    vRANGE = 10
    inputA = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    weiA = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)

    
    # inputA = np.random.rand(1, K).astype(np.float16)
    # weiA = np.random.rand(K, N).astype(np.float16)
    # C = np.matmul(A, B)
    ref_ker = kernel_cache(reference, options=f"-DTOKEN_HIDDEN_SZ={TOKEN_HIDDEN_SZ} -DKV_PROJ_SZ={KV_PROJ_SZ}")
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
    print("////////////////////////////")
    SG_SZ = 16
    # Run in parallel in 8 Xecores.
    WG_NUM = 2
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0 and RANK//SG_SZ >= 1, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    assert N <=256, f"'N'  {N} exceeds max value 256"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"

    N_SG_NUM = N // SG_SZ
    K_SG_NUM = K // SG_SZ
    K_PER_WG = ((K_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    tTempC = cl.tensor([WG_NUM, N], np.dtype(np.float16))
    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DK_PER_WG={K_PER_WG} -DRANK={RANK} -DKV_PROJ_SZ={KV_PROJ_SZ} -DKV_GROUP_SZ={KV_GROUP_SZ}")
    # N columns in one WG. The WG_NUM would separate K.
    opt_kernel.enqueue("gemmA", [N*WG_NUM],[N], tInput, tWeiA_Q, tWeiA_K, tWeiA_V, tTempC, N, K)
    cl.finish()
    C = np.zeros([1, N], dtype=np.float16)
    if 0: 
        tempC = tTempC.numpy()
        for i in range(N):
            for j in range(WG_NUM):
                C[0][i] += tempC[j][i]
                if i == 0:
                    print(f"tempC[0][{j}]:, {tempC[0][j]}, C: {C[0][i]}")
    else:
        tC = cl.tensor(C)
        tempC = tTempC.numpy()
        tTempC0 = cl.tensor(tempC[0, :])
        tTempC1 = cl.tensor(tempC[1, :])
        ref_ker.enqueue("add", [N],[N], tTempC0, tTempC1, tC)
        cl.finish()
        compare(tRefC.numpy(), tC.numpy())


    compare(tRefC.numpy(), C)
    print("#####################################")

    # GEMM B
    # inputB = tRefC.numpy()
    # inputB_Q = inputB[:, 0:RANK].flatten().reshape(1, RANK)
    # inputB_K = inputB[:, RANK:2*RANK].flatten().reshape(1, RANK)
    # inputB_V = inputB[:, 2*RANK:N].flatten().reshape(1, RANK)

    # weiB_Q = np.random.rand(RANK, TOKEN_HIDDEN_SZ).astype(np.float16)
    # weiB_K = np.random.rand(RANK, KV_PROJ_SZ).astype(np.float16)
    # weiB_V = np.random.rand(RANK, KV_PROJ_SZ).astype(np.float16)

    # ret_Q = np.matmul(inputB_Q, weiB_Q)
    # ret_K = np.matmul(inputB_K, weiB_K)
    # ret_V = np.matmul(inputB_V, weiB_V)
    # ret = np.concatenate((ret_Q, ret_K, ret_V), axis=1, dtype=np.float16)

    # tInputB = cl.tensor(inputB)
    # tWeiB_Q = cl.tensor(weiB_Q)
    # tWeiB_K = cl.tensor(weiB_K)
    # tWeiB_V = cl.tensor(weiB_V)
    # tRetRef = cl.tensor([1, TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2], np.dtype(np.float16))
    # tOutput = cl.tensor([1, TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2], np.dtype(np.float16))

    # ref_ker.enqueue("gemmB", [TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2],[KV_PROJ_SZ], tInputB, tWeiB_Q, tWeiB_K, tWeiB_V, tRetRef, RANK)
    # cl.finish()
    # compare(ret, tRetRef.numpy())

    # opt_kernel.enqueue("gemmB", [TOKEN_HIDDEN_SZ+KV_PROJ_SZ*2],[128], tInputB, tWeiB_Q, tWeiB_K, tWeiB_V, tOutput)
    # cl.finish()

    # compare(ret, tOutput.numpy())


# C = Matmul(A, B): A[1, 16], B [16, 16], C [1, 16]


def test_FMA_basic():
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *C, int N, int K) {
        int sg_id = get_sub_group_id();

        half sum = 0.f;
        ushort input = intel_sub_group_block_read_us((const __global ushort*)(A));
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < SG_SZ; j++) {
            half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B));
            half aa = as_half(intel_sub_group_broadcast(input, j));
            sum = fma(aa, bb, sum);
            B += N;
        }
        intel_sub_group_block_write_us((const __global ushort*)(C), as_short(sum));
    }
    '''
    N = 16
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
    kernel.enqueue("gemm", [N],[16], tA, tB, C, N, K)
    cl.finish()
    compare(C_ref, C.numpy())
    
def test_upping():
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *BQ, __global half *BK,__global half *BV,  __global half *C) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        unsigned int sg_idx = wg_id*8 +sg_id;
        
        __global half *C_ptr = C + sg_idx * SG_SZ;
        // Default A, B for Q projection
        __global half *A_ptr = A;
        __global half *B_ptr = BQ + sg_idx * SG_SZ;
        int stride_B = N1;

        if (sg_idx >= (N1+N2)/SG_SZ) {
            // V projection
            A_ptr = A + RANK*2;
            B_ptr = BV + (sg_idx * SG_SZ - (N1+N2));
            stride_B = N3;
        } else if (sg_idx >= N1/SG_SZ) {
            // K projection
            A_ptr = A + RANK;
            B_ptr = BK + (sg_idx * SG_SZ - N1);
            stride_B = N2;
        }
    
        half sum = 0;
        for (int i = 0; i < RANK/SG_SZ; i++) {
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr));
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                B_ptr += stride_B;
            }
            A_ptr += SG_SZ;
        }
        intel_sub_group_block_write_us((const __global ushort*)(C_ptr), as_short(sum));
    }
    
    __kernel void gemm_ref(__global half * input, __global half *B_Q,  __global half *B_K, __global half *B_V, __global half *CC) {
        int wgs = get_num_groups(0);
        int wg_id = get_group_id(0);
        int lid = get_local_id(0);
        int n_idx = get_global_id(0);

        __global half *A = input;
        __global half *BN = B_Q + n_idx;
        int stride = N1;

        if (wg_id ==  wgs - 2) {
            A += RANK;
            BN = B_K + lid;
            stride = N2;
        } else if (wg_id ==  wgs - 1) {
            A += RANK * 2;
            BN = B_V + lid;
            stride = N3;
        }

        half sum = 0.f;
        for (int i = 0; i < RANK; i++)
            //sum += A[i]*BN[i*stride];
            sum = fma(A[i], BN[i*stride], sum);
        CC[n_idx] = sum;
    }
    
    '''
    N1 = 1536
    N2 = 512
    N3 = 512
    RANK = 64
    vRANGE = 50
    np.random.seed(0)
    A = np.random.randint(-vRANGE, vRANGE+1, ([1, RANK*3])).astype(np.float16)
    A1 = A[:, 0:RANK]
    A2 = A[:, RANK:RANK*2]
    A3 = A[:, RANK*2:RANK*3]
    B1 = np.random.rand(RANK, N1).astype(np.float16)
    B2 = np.random.rand(RANK, N2).astype(np.float16)
    B3 = np.random.rand(RANK, N3).astype(np.float16)

    tA = cl.tensor(A)
    tB1 = cl.tensor(B1)
    tB2 = cl.tensor(B2)
    tB3 = cl.tensor(B3)
    SG_SZ = 16
    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DN={(N1+N2+N3)} -DN1={N1} -DN2={N2} -DN3={N3} -DRANK={RANK}")
    tRes = cl.tensor([1, (N1+N2+N3)], np.dtype(np.float16))
    kernel.enqueue("gemm", [(N1+N2+N3)],[128], tA, tB1, tB2, tB3, tRes)
    cl.finish()
    # compare(C, tRes.numpy())
    tRef = cl.tensor([1, (N1+N2+N3)], np.dtype(np.float16))

    kernel.enqueue("gemm_ref", [(N1+N2+N3)],[N3], tA, tB1, tB2, tB3, tRef)
    cl.finish()
    compare(tRef.numpy(), tRes.numpy())

def test_lowering():
    ref_src = r'''
    __kernel void gemm(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk) {
#if 1
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int n_idx = get_global_id(0);
        size_t blk_num = K / k_blk;
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
#else
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i], B[i*N+n_idx], sum);
        CC[n_idx] = sum;
#endif
    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }
    '''
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *temp_res, int N) {
        int sg_id = get_sub_group_id();
        int gid =  get_group_id(0);
        int k_start = gid * K_PER_WG;
        __global half *C_ptr = temp_res + N * gid;
        __global half *A_ptr = A + k_start;
        __global half *B_ptr = B + k_start * N;

        half sum = 0.f;
        for (int kk = 0;  kk < K_PER_WG; kk += SG_SZ) {
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
    N = 16
    K = 1536
    WG_NUM = 8
    SG_SZ = 16
    assert K %  WG_NUM == 0, f"'hidden_size' {K} is not multiple of WG_NUM {WG_NUM}"
    K_PER_WG = K // WG_NUM
    assert K_PER_WG %  SG_SZ == 0, f" 'kblock' {K_PER_WG} is not multiple of SG_SZ {SG_SZ}"
    vRANGE = 5
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    # A = np.random.rand(1, K).astype(np.float16)
    # B = np.random.rand(K, N).astype(np.float16)

    C_np = np.matmul(A, B)
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 16
    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DK_PER_WG={K_PER_WG}")
    kernel_ref =  kernel_cache(ref_src)
    tC_temp = cl.tensor([1*WG_NUM, N], np.dtype(np.float16))
    tC = cl.tensor([1, N], np.dtype(np.float16))

    tC_ref = cl.tensor([1, N], np.dtype(np.float16))
    kernel_ref.enqueue("gemm", [N],[N], tA, tB, tC_ref, N, K, K_PER_WG)
    cl.finish()

    kernel.enqueue("gemm", [N*WG_NUM],[N], tA, tB, tC_temp, N)
    cl.finish()
    kernel_ref.enqueue("reduce", [N],[N], tC_temp, tC, N, WG_NUM)
    cl.finish()
    compare(tC_ref.numpy(), tC.numpy())



cl.profiling(True)
# test_lora()
# test_FMA_basic()
# test_upping()
test_lowering()
print("-------------------------")