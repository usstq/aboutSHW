#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

def test_single_lora():
    ref_src = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, __global half * alpha, int N, int K, int k_blk) {
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
        CC[n_idx] = sum * alpha[0];
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
    
    __kernel void gemmB(__global half * A, __global half *B,  __global half *CC, int N, int K) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i], B[i*N+n_idx], sum);
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

        // store each sg accumulation result into SLM. Will sum sg result into wg result. 
        __local half sgs_sum_buff[SG_NUM * RANK];
        __local half *sg_buffer_ptr = sgs_sum_buff + sgid * RANK;
        __global half *C_ptr = temp_res + RANK * gid;

#if GEMMA_USE_SLM
        //Workgroup share same input activation and each subgroup would access all the elements of the shared intput.Put it into SLM.
        __local half local_input[K_PER_WG];
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sgid * SG_SZ; offset < (k_end_grp - k_start_grp); offset += SG_SZ * SG_NUM ) {
            __global half *input_ptr = A + k_start_grp + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        int k_start_sg = k_start_grp + sgid * K_PER_SG;
        __global half *A_ptr_sg = A + k_start_sg;
        __global half *B_ptr_sg = B + k_start_sg * RANK;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg diverge here.
        if (k_start_sg <  k_end_grp) {
            int klen_sg = ((k_start_sg + K_PER_SG) > k_end_grp) ? (k_end_grp-k_start_sg): K_PER_SG;

            for (int nblk = 0; nblk < RANK / SG_SZ; nblk++) {
                __global half *A_ptr = A_ptr_sg;
                __global half *B_ptr = B_ptr_sg + nblk * SG_SZ;
                half sum = 0.f;
                for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
#if GEMMA_USE_SLM
                    ushort input = intel_sub_group_block_read_us((const __local ushort*)(local_input + sgid * (int)(K_PER_SG) + kk));
#else
                    ushort input = intel_sub_group_block_read_us((const __global ushort*)(A_ptr + kk));
#endif
                    __attribute__((opencl_unroll_hint))
                    for (int j = 0; j < SG_SZ; j++) {
                        half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
                        half aa = as_half(intel_sub_group_broadcast(input, j));
                        sum = fma(aa, bb, sum);
                        B_ptr += RANK;
                    }
                }
                *(sg_buffer_ptr + nblk * SG_SZ + lid) = sum;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int sg_offset = sgid * SG_SZ;
        //Each SG would try updating data.
         __attribute__((opencl_unroll_hint))
        for (int base = 0; (sg_offset  + base ) < RANK; base += SG_SZ * SG_NUM ) {
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < SG_NUM; i++) {
                sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(sgs_sum_buff + i * RANK + sg_offset)));
            }
            intel_sub_group_block_write_us((const __global ushort*)(C_ptr + sg_offset), as_short(sum));
        }
    }


    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * A,  __global half *B,  __global half *CC, __global half * alpha, int N) {
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

        __global half *C_ptr = CC + sg_idx * SG_SZ;
        __global half *A_ptr = A;
        __global half *B_ptr = B + sg_idx * SG_SZ;

        //1. Reduce
#if GEMMB_USE_SLM
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        //sg would diverge here. not all the sgs can satisfy 'offset < RANK'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sg_id * SG_SZ; offset < RANK; offset += local_sz) {
            __global half *subA_ptr = A_ptr + offset;
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)subA_ptr));
                sum += partial_val;
                subA_ptr += RANK;
            }
            reduce[offset + id_sg_local] = sum;
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
        //2.  GEMMB
        half sum = 0;
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
#if GEMMB_USE_SLM
            half input = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce + kk)));
#else
            half input = reduce[id_sg_local + kk];
#endif
            input *= alpha[0];
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += N;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }
    '''
    RANK = 64
    INPUT_STATE = 1536
    OUTPUT_STATE = INPUT_STATE
    WG_NUM = 4
    SG_NUM = 4
    SG_SZ = 16
    GEMMB_LOCAL_SIZE = 128
    assert INPUT_STATE % SG_SZ == 0, f"'input state' {INPUT_STATE} is not multiple of SG_SZ {SG_SZ}"
    assert RANK % SG_SZ == 0, f"'RANK' {RANK} is not multiple of SG_SZ {SG_SZ}"
    assert OUTPUT_STATE % SG_SZ == 0, f"'output state' {OUTPUT_STATE} is not multiple of SG_SZ {SG_SZ}"
    assert (OUTPUT_STATE) % GEMMB_LOCAL_SIZE == 0, f"GEMMB: 'global size' {OUTPUT_STATE} is not multiple of 'local size' {GEMMB_LOCAL_SIZE}"

    INPUT_STATE_SG_NUM = INPUT_STATE // SG_SZ
    INPUT_STATE_PER_WG = ((INPUT_STATE_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    INPUT_STATE_BLKS_PER_WG = INPUT_STATE_PER_WG / SG_SZ
    # assert( KBLKS_PER_WG > 0, f"Can't satisfy at least 1 KBLK(16) for one WG")
    # can't ensure each sg can be assigned the workload. sg maybe diverge.
    INPUT_STATE_BLKS_PER_SG = (INPUT_STATE_BLKS_PER_WG + SG_NUM - 1) // SG_NUM
    INPUT_STATE_PER_SG = INPUT_STATE_BLKS_PER_SG * SG_SZ
    vRANGE = 2
    np.random.seed(0)
    A = np.random.randint(-vRANGE, vRANGE+1, [1, INPUT_STATE]).astype(np.float16)
    weiA = np.random.randint(-vRANGE, vRANGE, [INPUT_STATE, RANK]).astype(np.float16)
    weiB = np.random.randint(-vRANGE, vRANGE, [RANK, OUTPUT_STATE]).astype(np.float16)
    B_output = np.random.randint(-vRANGE, vRANGE, [1, OUTPUT_STATE]).astype(np.float16)
    alpha = np.random.rand(1).astype(np.float16)
    tAlpha = cl.tensor(alpha)

    tA = cl.tensor(A)
    tweiA = cl.tensor(weiA)
    tweiB = cl.tensor(weiB)
    tB_output = cl.tensor(B_output)

    SG_SZ = 16
    GEMMA_USE_SLM = 1
    kernel_opt = kernel_cache(src, options=f"-DWG_NUM={WG_NUM} -DSG_SZ={SG_SZ} -DK_PER_WG={INPUT_STATE_PER_WG} -DSG_NUM={SG_NUM} -DRANK={RANK} -DK_PER_SG={INPUT_STATE_PER_SG} -DGEMMA_USE_SLM={GEMMA_USE_SLM} -DPART_NUM={WG_NUM}")
    kernel_ref =  kernel_cache(ref_src)
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    tA_output = cl.tensor(np.zeros([WG_NUM, RANK]).astype(np.float16))
    tA_reduce = cl.tensor([1, RANK], np.dtype(np.float16))

    tC_ref = cl.tensor([1, RANK], np.dtype(np.float16))
    tResult_ref = cl.tensor([1, OUTPUT_STATE], np.dtype(np.float16))
    kernel_ref.enqueue("gemmA", [RANK],[RANK], tA, tweiA, tC_ref, tAlpha, RANK, INPUT_STATE, INPUT_STATE_PER_WG)
    REF_LOCAL_SIZE = 128
    assert OUTPUT_STATE % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {OUTPUT_STATE} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
    kernel_ref.enqueue("gemmB", [OUTPUT_STATE],[REF_LOCAL_SIZE], tC_ref, tweiB, tResult_ref, OUTPUT_STATE, RANK)
    cl.finish()
    kernel_opt.enqueue("gemmA", [SG_SZ * WG_NUM * SG_NUM],[SG_SZ * SG_NUM], tA, tweiA, tA_output, INPUT_STATE)
    kernel_opt.enqueue("gemmB", [OUTPUT_STATE],[GEMMB_LOCAL_SIZE], tA_output, tweiB, tB_output, tAlpha, OUTPUT_STATE)
    cl.finish()
    # compare(tC_ref.numpy(), tA_reduce.numpy())
    compare(tResult_ref.numpy(), tB_output.numpy())


def test_single_lora_transposeB(rank, input_state, output_state, rep=3, check_acc=False):
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
        //CC[n_idx] = sum * alpha[n_idx];
        CC[n_idx] = sum;

    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }

    __kernel void gemmB(__global half *mainInput, __global half * A, __global half *B,  __global half *CC,  __global half * alpha, int K) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[i]*alpha[i], B[n_idx*K + i], sum);
        CC[n_idx] = sum + mainInput[n_idx];
    }
    '''

    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A, __global half *B,  __global half *temp_res, int K) {
        int gid_K =  get_group_id(0);
        int gid_N =  get_group_id(1);
        int sgid = get_sub_group_id();
        int n_idx = gid_N * get_num_sub_groups() + sgid;
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int k_start_grp = gid_K * K_PER_WG;
        int k_end_grp = (k_start_grp + K_PER_WG) > K ?  (K): (k_start_grp+K_PER_WG);
        __global half *C_ptr = temp_res + RANK * gid_K;
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
        __global half *B_ptr = B + n_idx * K + gid_K * K_PER_WG;
        half sum = 0.f;
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
        sum = sub_group_reduce_add(sum);
        if (lid == 0)
            *(C_ptr + n_idx) = sum;
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half *mainInput, __global half * A,  __global half *B,  __global half *CC, __global half * alpha, int N) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int sg_idx = wg_id * sg_num  +  sg_id;

        int id_sg_local = get_sub_group_local_id();
#if GEMMB_USE_SLM
        __local half reduce[16*1024];
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
        for (int offset = sg_id * SG_SZ; offset < RANK; offset += local_sz) {
            __global half *subA_ptr = A_ptr + offset;
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)subA_ptr));
                sum += partial_val;
                subA_ptr += RANK;
            }
            reduce[offset + id_sg_local] = sum;
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

#if 0
        //Debug the SLM sum works as expected
        if (get_local_linear_id () == 0) {
            for (int i = 0; i < RANK; i++)
                reduce_res[wg_id*RANK+i] = reduce[i];
        }
#endif
        //2.  GEMMB
        //if (sg_idx >= N)
            //return;
        if (sg_idx < N) {
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int subk = 0; subk < RANK; subk += SG_SZ) {
                half scale = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha + subk)));
#if GEMMB_USE_SLM
                half aa = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce + subk)));
                //half aa = reduce[id_sg_local + subk];
#else
                half aa = reduce[id_sg_local + subk];
#endif
                aa *= scale;
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr + subk)));
                sum = fma(aa, bb, sum);
            }
            sum = sub_group_reduce_add(sum);
            if (id_sg_local == 0)
                CC[sg_idx] = sum + mainInput[sg_idx];
        }
    }
    '''
    cl.profiling(True)
    RANK = rank
    INPUT_STATE = input_state
    OUTPUT_STATE = output_state
    SG_SZ = 16

    #OUTPUT_STATE = INPUT_STATE
    MAX_GEMMA_WG_SZ = 1024
    if (RANK * SG_SZ <= MAX_GEMMA_WG_SZ ):
        GEMMA_SG_NUM = RANK
        COLUMN_WGS = 1
    else:
        # It would be 
        assert (RANK) % (MAX_GEMMA_WG_SZ//SG_SZ) == 0, f"rank {RANK} is not multiple of MAX_GEMMA_WG_SZ {MAX_GEMMA_WG_SZ//SG_SZ}"
        GEMMA_SG_NUM = MAX_GEMMA_WG_SZ // SG_SZ
        COLUMN_WGS = RANK*SG_SZ // MAX_GEMMA_WG_SZ

    GEMMA_USE_SLM = 1
    GEMMB_USE_SLM = 1
    WG_NUM = 16
    GEMMB_LOCAL_SIZE = 128
    GEMMA_UNROLL_NUM = 4
    assert INPUT_STATE % (SG_SZ * GEMMA_UNROLL_NUM) == 0, f"'input state' {INPUT_STATE} is not multiple of SG_SZ {SG_SZ}"
    assert RANK % SG_SZ == 0, f"'lora_rank' {RANK} is not multiple of SG_SZ {SG_SZ}"
    GEMMB_UNROLL_NUM = RANK / SG_SZ

    INPUT_STATE_BLK_SZ = SG_SZ * GEMMA_UNROLL_NUM
    INPUT_STATE_BLK_NUM = INPUT_STATE // INPUT_STATE_BLK_SZ
    INPUT_STATE_PER_WG = ((INPUT_STATE_BLK_NUM + WG_NUM - 1) // WG_NUM) * INPUT_STATE_BLK_SZ
    # assert( KBLKS_PER_WG > 0, f"Can't satisfy at least 1 KBLK(16) for one WG")
    # can't ensure each sg can be assigned the workload. sg maybe diverge.
    vRANGE = 1
    # np.random.seed(0)
    main_input_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [1, OUTPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    lora_input_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [1, INPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    stateA_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [RANK, INPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    stateB_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [OUTPUT_STATE, RANK]).astype(np.float16)) for _ in range(rep)]
    res_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE, [1, OUTPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    alpha_list_np = [np.random.randint(-1, 2, [RANK]).astype(np.float16) for _ in range(rep)]
    alpha_list = [cl.tensor(np.multiply(alpha, 0.5)) for alpha in alpha_list_np]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    gemmA_output_list = [cl.tensor(np.zeros([WG_NUM, RANK]).astype(np.float16)) for _ in range(rep)]

    MAX_LWS = 512
    if (OUTPUT_STATE * SG_SZ  > MAX_LWS):
        GEMMB_LWS = MAX_LWS
        GEMMB_GWS = (OUTPUT_STATE *SG_SZ + MAX_LWS - 1) // MAX_LWS * MAX_LWS
        GEMMB_WGS = GEMMB_GWS // GEMMB_LWS
    else:
        GEMMB_LWS = OUTPUT_STATE * SG_SZ
        GEMMB_GWS = OUTPUT_STATE * SG_SZ
        GEMMB_WGS = GEMMB_GWS // GEMMB_LWS
    kernel_opt = cl.kernels(src, options=f"-DWG_NUM={WG_NUM} -DSG_SZ={SG_SZ} -DK_PER_WG={INPUT_STATE_PER_WG}  -DRANK={RANK} -DGEMMA_SG_NUM={GEMMA_SG_NUM} -DGEMMA_UNROLL_NUM={GEMMA_UNROLL_NUM}\
                              -DGEMMA_USE_SLM={GEMMA_USE_SLM} -DGEMMB_USE_SLM={GEMMB_USE_SLM} -DPART_NUM={WG_NUM}")
    kernel_ref =  cl.kernels(ref_src)
    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    print(f'| INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state}:')
    
    [WG_NUM, COLUMN_WGS, SG_SZ * GEMMA_SG_NUM], [1, 1, SG_SZ * GEMMA_SG_NUM],
    
    
    print(f'| [GEMMA]: GWS[{WG_NUM}, {COLUMN_WGS}, {SG_SZ * GEMMA_SG_NUM}], LWS:[1, 1 {SG_SZ * GEMMA_SG_NUM}] ')
    print(f'| [GEMMB]: GWS:[{GEMMB_GWS}], LWS:[{GEMMB_LWS}], ')
    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    flops_a = rank*input_state*2
    flops_b = rank*output_state*2
    # how many CPU memory loaded into GPU memory, stateA + input activation. duplicate input activation would be in GPU global cache?
    rd_bytes_a = (rank*input_state+input_state)*2
    # gemma output intermedia result should reside in GPU memory in assumption,stateB + alpha + main_input + output
    rd_bytes_b = (rank*output_state+output_state*2+rank)*2
    
    tA_reduce = cl.tensor([1, RANK], np.dtype(np.float16))
        
    for i in range(0, rep):
        tA_reduce = cl.tensor([1, RANK], np.dtype(np.float16))
        # print(f'GEMMB_LWS={GEMMB_LWS}, GEMMB_GWS={GEMMB_GWS}')
        kernel_opt.enqueue("gemmA", [WG_NUM, COLUMN_WGS, SG_SZ * GEMMA_SG_NUM], [1, 1, SG_SZ * GEMMA_SG_NUM], lora_input_list[i], stateA_list[i], gemmA_output_list[i], INPUT_STATE)
        kernel_opt.enqueue("gemmB", [GEMMB_GWS],[GEMMB_LWS], main_input_list[i], gemmA_output_list[i], stateB_list[i], res_list[i], alpha_list[i], OUTPUT_STATE)
    profiling_data = cl.finish()
    for i in range(0, rep):
        if check_acc:
            tA_output_ref = cl.tensor([1, RANK], np.dtype(np.float16))
            tResult_ref = cl.tensor([1, OUTPUT_STATE], np.dtype(np.float16))
            kernel_ref.enqueue("gemmA", [RANK],[RANK], lora_input_list[i], stateA_list[i], tA_output_ref, alpha_list[i], RANK, INPUT_STATE, INPUT_STATE_PER_WG)
            REF_LOCAL_SIZE = 64
            assert OUTPUT_STATE % REF_LOCAL_SIZE == 0, f"'OUTPUT_STATE' {OUTPUT_STATE} is not multiple of LOCAL_SIZE {REF_LOCAL_SIZE}"
            kernel_ref.enqueue("gemmB", [OUTPUT_STATE],[REF_LOCAL_SIZE], main_input_list[i],  tA_output_ref, stateB_list[i], tResult_ref, alpha_list[i],  RANK)
            kernel_ref.enqueue("reduce", [RANK],[RANK], gemmA_output_list[i], tA_reduce, RANK, WG_NUM)
            cl.finish()
            compare(tA_output_ref.numpy(), tA_reduce.numpy())
            compare(tResult_ref.numpy(), res_list[i].numpy())
            
        else:
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2+1]
            print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
 [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
 [total]: {(ns_a+ns_b)*1e-3:.1f}us")


cl.profiling(True)
# test_single_lora()
#Check accuray
# for rank in [16, 32, 64, 128, 192]:
#     for out_state in [512, 1536, 3840]:
#         test_single_lora_transposeB(rank, 1536, out_state, 1, True)
#Check perf
for rank in [192]:
    # for out_state in [1536]:
    for out_state in [2560]:
        test_single_lora_transposeB(rank, 1536, out_state, 40)
