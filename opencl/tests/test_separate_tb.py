#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

    
def test_gemmA():
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

def test_gemmB():
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


def test_single_lora():
    ref_src = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, __global half * alpha, int N, int K, int k_blk) {
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
                //sub_sum = fma(A[k_idx], B[k_idx*N+n_idx], sub_sum);
                sub_sum = fma(A[k_idx], B[n_idx*K + k_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        //CC[n_idx] = sum * alpha[0];
        CC[n_idx] = sum;

    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }

    __kernel void gemmB(__global half * A, __global half *B,  __global half *CC,  int K) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i], B[n_idx*K + i], sum);
        CC[n_idx] = sum;
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
        __global half *C_ptr = temp_res + RANK * gid;
        if (k_start_grp > K)
            return;

#if GEMMA_USE_SLM
        //Workgroup share same input activation and each subgroup would access all the elements of the shared intput.Put it into SLM.
        __local half local_input[K_PER_WG];
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sgid * SG_SZ; offset < (k_end_grp - k_start_grp); offset += SG_SZ * GEMMA_SG_NUM ) {
            __global half *input_ptr = A + k_start_grp + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        __global half *A_ptr = A + k_start_grp;
        __global half *B_ptr = B + sgid * K + gid * K_PER_WG;
        half sum = 0.f;
        {
            __attribute__((opencl_unroll_hint))
            for (int kk = 0;  kk < (k_end_grp - k_start_grp); kk += SG_SZ * GEMMA_UNROLL_NUM) {
                __attribute__((opencl_unroll_hint))
                for (int i = 0; i < GEMMA_UNROLL_NUM; i++) {
                    half fma_res = 0.f;
                    int subk = i * SG_SZ;
    #if GEMMA_USE_SLM
                    ushort input = intel_sub_group_block_read_us((const __local ushort*)(local_input + kk + subk));
    #else
                    ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr + kk + subk));
    #endif
                    half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr + kk + subk)));
                    sum = fma(as_half(input), bb, sum);
                }
            }
        }
        sum = sub_group_reduce_add(sum);
        if (lid == 0)
            *(C_ptr + sgid) = sum;
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * A,  __global half *B,  __global half *CC, __global half * alpha, int N, __global half * reduce_res) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int sg_idx = wg_id * sg_num  +  sg_id;

        int id_sg_local = get_sub_group_local_id();
#if GEMMB_USE_SLM
        __local half reduce[RANK];
#else
        half reduce[RANK] = {0.f};
#endif

        __global half *C_ptr = CC + sg_idx;
        __global half *A_ptr = A;
        __global half *B_ptr = B + sg_idx * RANK;

        //1. Reduce
#if GEMMB_USE_SLM
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        //sg would diverge here. not all the sgs can satisfy 'offset < RANK'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sg_id * SG_SZ; offset < RANK; offset += local_sz) {
            __global half *subA_ptr = A_ptr + offset;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)subA_ptr));
                reduce[offset + id_sg_local] += partial_val;
                subA_ptr += RANK;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        
#else
        //Each work item would reduce the input activation.
        for (int i = 0; i < PART_NUM; i++) {
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < RANK; j++) {
                reduce[j] += A_ptr[i * RANK + j];
            }
        }
#endif

        if (get_local_linear_id () == 0) {
            for (int i = 0; i < RANK; i++)
                reduce_res[i] = reduce[i];
        }
        //2.  GEMMB
        if (sg_idx >= N)
            return;
        half sum = 0.f;
        __attribute__((opencl_unroll_hint))
        for (int subk = 0; subk < RANK; subk += SG_SZ) {
#if GEMMB_USE_SLM
            half aa = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce + subk)));
            //half aa = reduce[id_sg_local + subk];
#else
            half aa = reduce[id_sg_local + subk];
#endif
            //aa *= alpha[0];
            half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr + subk)));
            sum = fma(aa, bb, sum);
        }
        sum = sub_group_reduce_add(sum);
        if (id_sg_local == 0)
            CC[sg_idx] = sum;
    }
    '''
    RANK = 16
    INPUT_STATE = 1536
    OUTPUT_STATE = 1536

    #OUTPUT_STATE = INPUT_STATE
    GEMMA_SG_NUM = RANK
    GEMMA_USE_SLM = 1
    GEMMB_USE_SLM = 0
    WG_NUM = 5
    SG_SZ = 16
    GEMMB_LOCAL_SIZE = 128
    GEMMA_UNROLL_NUM = 8
    assert INPUT_STATE % (SG_SZ * GEMMA_UNROLL_NUM) == 0, f"'input state' {INPUT_STATE} is not multiple of SG_SZ {SG_SZ}"
    assert RANK % SG_SZ == 0, f"'lora_rank' {RANK} is not multiple of SG_SZ {SG_SZ}"
    GEMMB_UNROLL_NUM = RANK / SG_SZ

    INPUT_STATE_BLK_SZ = SG_SZ * GEMMA_UNROLL_NUM
    INPUT_STATE_BLK_NUM = INPUT_STATE // INPUT_STATE_BLK_SZ
    INPUT_STATE_PER_WG = ((INPUT_STATE_BLK_NUM + WG_NUM - 1) // WG_NUM) * INPUT_STATE_BLK_SZ
    # assert( KBLKS_PER_WG > 0, f"Can't satisfy at least 1 KBLK(16) for one WG")
    # can't ensure each sg can be assigned the workload. sg maybe diverge.
    vRANGE = 1
    np.random.seed(0)
    A = np.random.randint(-vRANGE, vRANGE, [1, INPUT_STATE]).astype(np.float16)
    weiA = np.random.randint(-vRANGE, vRANGE, [RANK, INPUT_STATE]).astype(np.float16)
    weiB = np.random.randint(-vRANGE, vRANGE, [OUTPUT_STATE, RANK]).astype(np.float16)
    # weiB = np.random.rand(OUTPUT_STATE, RANK).astype(np.float16)
    B_output = np.random.randint(-vRANGE, vRANGE, [1, OUTPUT_STATE]).astype(np.float16)
    alpha = np.random.rand(1).astype(np.float16)
    tAlpha = cl.tensor(alpha)

    tA = cl.tensor(A)
    tweiA = cl.tensor(weiA)
    tweiB = cl.tensor(weiB)
    tB_output = cl.tensor(B_output)


    kernel_opt = kernel_cache(src, options=f"-DWG_NUM={WG_NUM} -DSG_SZ={SG_SZ} -DK_PER_WG={INPUT_STATE_PER_WG}  -DRANK={RANK} -DGEMMA_SG_NUM={GEMMA_SG_NUM} -DGEMMA_UNROLL_NUM={GEMMA_UNROLL_NUM}\
                              -DGEMMA_USE_SLM={GEMMA_USE_SLM} -DGEMMB_USE_SLM={GEMMB_USE_SLM} -DPART_NUM={WG_NUM}")
    kernel_ref =  kernel_cache(ref_src)
    tA_output = cl.tensor([1*WG_NUM, RANK], np.dtype(np.float16))
    tA_reduce = cl.tensor([1, RANK], np.dtype(np.float16))
    tA_reduce_res = cl.tensor([1, RANK], np.dtype(np.float16))


    tC_ref = cl.tensor([1, RANK], np.dtype(np.float16))
    tResult_ref = cl.tensor([1, OUTPUT_STATE], np.dtype(np.float16))
    kernel_ref.enqueue("gemmA", [RANK],[RANK], tA, tweiA, tC_ref, tAlpha, RANK, INPUT_STATE, INPUT_STATE_PER_WG)
    REF_LOCAL_SIZE = 64
    assert OUTPUT_STATE % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {OUTPUT_STATE} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
    kernel_ref.enqueue("gemmB", [OUTPUT_STATE],[REF_LOCAL_SIZE], tC_ref, tweiB, tResult_ref, RANK)
    cl.finish()

    MAX_LWS = 1024
    if (OUTPUT_STATE * SG_SZ  > MAX_LWS):
        GEMMB_LWS = MAX_LWS
        GEMMB_GWS = (OUTPUT_STATE *SG_SZ + MAX_LWS - 1) // MAX_LWS * MAX_LWS
    else:
        GEMMB_LWS = OUTPUT_STATE * SG_SZ
        GEMMB_GWS = OUTPUT_STATE * SG_SZ
    print(f'GEMMB_LWS={GEMMB_LWS}, GEMMB_GWS={GEMMB_GWS}')
    kernel_opt.enqueue("gemmA", [SG_SZ * WG_NUM * GEMMA_SG_NUM],[SG_SZ * GEMMA_SG_NUM], tA, tweiA, tA_output, INPUT_STATE)
    kernel_ref.enqueue("reduce", [RANK],[RANK], tA_output, tA_reduce, RANK, WG_NUM)
    kernel_opt.enqueue("gemmB", [GEMMB_GWS],[GEMMB_LWS], tA_output, tweiB, tB_output, tAlpha, OUTPUT_STATE, tA_reduce_res)
    cl.finish()


    compare(tC_ref.numpy(), tA_reduce.numpy())
    # compare(tC_ref.numpy(), tA_reduce_res.numpy())

    compare(tResult_ref.numpy(), tB_output.numpy())
    



cl.profiling(True)
# test_gemmA()
# test_gemmB()
test_single_lora()
