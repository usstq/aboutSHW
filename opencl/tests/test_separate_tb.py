#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

def test_FMA_basic_TB():
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(16)))
    __kernel void gemm(__global half * A, __global half *B,  __global half *C,  int N, int K) {
        int wg_id = get_group_id(0);
        int sg_num = get_num_sub_groups();
        half sum = 0.f;
        int n_idx = wg_id * sg_num + get_sub_group_id();
        if (n_idx >= N)
            return;
        for (int k_idx = 0; k_idx < K; k_idx += 16) {
            half aa = as_half(intel_sub_group_block_read_us((const __global ushort*)(A + k_idx)));
            half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B +n_idx*K + k_idx)));
            sum = fma(aa, bb, sum);
        }
        sum = sub_group_reduce_add(sum);
        if (get_sub_group_local_id  () == 0)
            C[n_idx] = sum;
    }
    '''
    N = 3840
    K = 16
    np.random.seed(0)
    vRANGE = 5
    A = np.random.randint(-vRANGE, vRANGE+1, [1, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    C_ref = np.matmul(A, B.transpose(1,0))
    tA = cl.tensor(A)
    tB = cl.tensor(B)
    tC = cl.tensor([1, N], np.dtype(np.float16))
    SG_SZ = 16
    MAX_LWS = 1024
    if (N * SG_SZ  > MAX_LWS):
        GEMMB_LWS = MAX_LWS
        GEMMB_GWS = (N *SG_SZ + MAX_LWS - 1) // MAX_LWS * MAX_LWS
    else:
        GEMMB_LWS = N * SG_SZ
        GEMMB_GWS = N * SG_SZ
    kernel = cl.kernels(src, options=f"-DSG_SZ={SG_SZ}")
    kernel.enqueue("gemm", [GEMMB_GWS],[GEMMB_LWS], tA, tB, tC, N, K)
    cl.finish()
    compare(C_ref, tC.numpy())

def test_single_lora(rank, input_state, output_state, rep=3):
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
        __local half reduce[16*1024];
        __local bool sum_done;
#else
        half reduce[RANK] = {0.f};
#endif

        __global half *C_ptr = CC + sg_idx;
        __global half *A_ptr = A;
        __global half *B_ptr = B + sg_idx * RANK;

        //1. Reduce
#if GEMMB_USE_SLM
        if (!sum_done) {
            //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
            int local_sz = get_local_size(0);
            //sg would diverge here. not all the sgs can satisfy 'offset < RANK'.
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
            sum_done = true;
        }

        
#else
        //Each work item would reduce the input activation.
        for (int i = 0; i < PART_NUM; i++) {
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < RANK; j++) {
                reduce[j] += A_ptr[i * RANK + j];
            }
        }
#endif

#if 0
        //Debug the SLM sum works as expected
        if (get_local_linear_id () == 0) {
            for (int i = 0; i < RANK; i++)
                reduce_res[wg_id*RANK+i] = reduce[i];
        }
#endif
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
    cl.profiling(True)
    RANK = rank
    INPUT_STATE = input_state
    OUTPUT_STATE = output_state

    #OUTPUT_STATE = INPUT_STATE
    GEMMA_SG_NUM = RANK
    GEMMA_USE_SLM = 1
    GEMMB_USE_SLM = 1
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
        GEMMB_WGS = GEMMB_GWS // GEMMB_LWS
    else:
        GEMMB_LWS = OUTPUT_STATE * SG_SZ
        GEMMB_GWS = OUTPUT_STATE * SG_SZ
        GEMMB_WGS = GEMMB_GWS // GEMMB_LWS
    # tA_reduce_res = cl.tensor([GEMMB_WGS, RANK], np.dtype(np.float16))

    # print(f'GEMMB_LWS={GEMMB_LWS}, GEMMB_GWS={GEMMB_GWS}')
    kernel_opt.enqueue("gemmA", [SG_SZ * WG_NUM * GEMMA_SG_NUM],[SG_SZ * GEMMA_SG_NUM], tA, tweiA, tA_output, INPUT_STATE)
    # kernel_ref.enqueue("reduce", [RANK],[RANK], tA_output, tA_reduce, RANK, WG_NUM)
    kernel_opt.enqueue("gemmB", [GEMMB_GWS],[GEMMB_LWS], tA_output, tweiB, tB_output, tAlpha, OUTPUT_STATE, tA_reduce_res)
    profiling_data = cl.finish()
    ns_a = profiling_data[0]
    ns_b = profiling_data[1]
    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    print(f'| INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state} perf:')
    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')

    # compare(tC_ref.numpy(), tA_reduce.numpy())
    # for i in range (0, GEMMB_WGS):
    #     compare(tC_ref.numpy(), tA_reduce_res.numpy()[i,:].flatten().reshape(1, RANK))

    compare(tResult_ref.numpy(), tB_output.numpy())
    



cl.profiling(True)
test_single_lora(64, 1536, 1536)
test_single_lora(64, 1536, 512)
test_single_lora(64, 1536, 3840)


