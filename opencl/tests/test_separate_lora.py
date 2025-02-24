#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

    
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


def test_separate_lora():
    ref_src = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk) {
#if 1
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
    __kernel void gemmA(__global half * A, __global half *B,  __global half *temp_res, int K) {
        int gid =  get_group_id(0);
        int sgid = get_sub_group_id();
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int k_start_grp = gid * K_PER_WG;
        int k_end_grp = (k_start_grp + K_PER_WG) > K ?  (K): (k_start_grp+K_PER_WG);

        // store each sg accumulation result into SLM. Will sum sg result into wg result. 
        __local half sgs_sum_buff[SG_NUM * N];
        __local half *sg_buffer_ptr = sgs_sum_buff + sgid * N;
        __global half *C_ptr = temp_res + N * gid;

        int k_start_sg = k_start_grp + sgid * K_PER_SG;
        __global half *A_ptr_sg = A + k_start_sg;
        __global half *B_ptr_sg = B + k_start_sg * N;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg diverge here.
        if (k_start_sg <  k_end_grp) {
            int klen_sg = ((k_start_sg + K_PER_SG) > k_end_grp) ? (k_end_grp-k_start_sg): K_PER_SG;

            for (int nblk = 0; nblk < N / SG_SZ; nblk++) {
                __global half *A_ptr = A_ptr_sg;
                __global half *B_ptr = B_ptr_sg + nblk * SG_SZ;
                half sum = 0.f;
                for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
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
                *(sg_buffer_ptr + nblk * SG_SZ + lid) = sum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int sg_offset = sgid * SG_SZ;
        //Each SG would try updating data.
         __attribute__((opencl_unroll_hint))
        for (int base = 0; (sg_offset  + base ) < N; base += SG_SZ * SG_NUM ) {
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < SG_NUM; i++) {
                sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(sgs_sum_buff + i * N + sg_offset)));
            }
            intel_sub_group_block_write_us((const __global ushort*)(C_ptr + sg_offset), as_short(sum));
        }
    }
    '''
    N = 64
    K = 1536
    WG_NUM = 7
    SG_NUM = 16
    SG_SZ = 16
    assert K % SG_SZ == 0, f"'input state' {K} is not multiple of SG_SZ {SG_SZ}"
    assert N % SG_SZ == 0, f"'RANK' {N} is not multiple of SG_SZ {SG_SZ}"

    K_SG_NUM = K // SG_SZ
    K_PER_WG = ((K_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    KBLKS_PER_WG = K_PER_WG / SG_SZ
    # assert( KBLKS_PER_WG > 0, f"Can't satisfy at least 1 KBLK(16) for one WG")
    
    # can't ensure each sg can be assigned the workload. sg maybe diverge.
    KBLKS_PER_SG = (KBLKS_PER_WG + SG_NUM - 1) // SG_NUM
    K_PER_SG = KBLKS_PER_SG * SG_SZ
    vRANGE = 2
    np.random.seed(0)
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)

    tA = cl.tensor(A)
    tB = cl.tensor(B)
    SG_SZ = 16
    kernel = kernel_cache(src, options=f"-DWG_NUM={WG_NUM} -DSG_SZ={SG_SZ} -DK_PER_WG={K_PER_WG} -DSG_NUM={SG_NUM} -DN={N} -DK_PER_SG={K_PER_SG}")
    kernel_ref =  kernel_cache(ref_src)
    tC_temp = cl.tensor([1*WG_NUM, N], np.dtype(np.float16))
    tC = cl.tensor([1, N], np.dtype(np.float16))

    tC_ref = cl.tensor([1, N], np.dtype(np.float16))
    kernel_ref.enqueue("gemmA", [N],[N], tA, tB, tC_ref, N, K, K_PER_WG)
    cl.finish()

    kernel.enqueue("gemmA", [SG_SZ * WG_NUM * SG_NUM],[SG_SZ * SG_NUM], tA, tB, tC_temp, K)
    cl.finish()
    kernel_ref.enqueue("reduce", [N],[N], tC_temp, tC, N, WG_NUM)
    cl.finish()
    compare(tC_ref.numpy(), tC.numpy())


cl.profiling(True)
# test_lora()
# test_FMA_basic()
# test_upping()
# test_lowering()
test_separate_lora()
print("-------------------------")