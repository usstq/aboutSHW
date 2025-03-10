#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *


def test_single_lora_transposeB(batch, rank, input_state, output_state, rep=3, check_acc=False):
    ref_src = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk) {
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        size_t blk_num = (K + k_blk -1 )/ k_blk;
        half sum = 0.f;
        for (int i = 0; i < blk_num; i++) {
            half sub_sum = 0.f;
            int k_idx = i*k_blk;
            // incase K%k_blk != 0;
            if ((k_idx + k_blk) > K)
                k_blk = K - k_idx;
            for (size_t sub_k = 0; sub_k < k_blk; sub_k++) {
                sub_sum = fma(A[m_idx * K + k_idx], B[n_idx*K + k_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        CC[m_idx * N + n_idx] = sum;

    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[m_idx*cnt* N + i*N + n_idx];
        C[m_idx * N + n_idx] = sum;
    }

    __kernel void gemmB(__global half *mainInput, __global half * A, __global half *B,  __global half *CC,  __global half * alpha, int K, int N) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i]*alpha[i], B[n_idx*K + i], sum);
        CC[m_idx * N + n_idx] = sum + mainInput[n_idx];
    }
    '''

    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A, __global half *B,  __global half *temp_res, int K) {
        int gid_M =  get_group_id(0);
        int gid_K =  get_group_id(1);
        int gid_N =  get_group_id(2);

        int sgid = get_sub_group_id();
        int n_idx = gid_N * get_num_sub_groups() + sgid;
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int k_start_grp = gid_K * K_PER_WG;
        int k_end_grp = (k_start_grp + K_PER_WG) > K ?  (K): (k_start_grp+K_PER_WG);
        __global half *C_ptr = temp_res + gid_M * PART_NUM * RANK + RANK * gid_K;
        //Align up kbloks by workgroups. Maybe would cause some workgroups no workload. 
        if (k_start_grp > K)
            return;
#if GEMMA_USE_SLM
        //Workgroup share same input activation and each subgroup would access all the elements of the shared intput.Put it into SLM.
        __local half local_input[K_PER_WG];
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        int local_sz = get_local_size(2);
        __attribute__((opencl_unroll_hint))
        for (int offset = sgid * SG_SZ; offset < (k_end_grp - k_start_grp); offset += local_sz ) {
            __global half *input_ptr = A + gid_M * K + k_start_grp + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        //Align up n dimension with subgroups in wg. Maybe would cause some subgroups no workload in N dimesnion.
        if (n_idx >= RANK)
            return;
#if !GEMMA_USE_SLM
        __global half *A_ptr = A + gid_M * K + k_start_grp;
#endif
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
        int wgid_M = get_group_id(0);
        int wgid_N = get_group_id(1);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int n_idx = wgid_N * sg_num  +  sg_id;

        int id_sg_local = get_sub_group_local_id();
#if GEMMB_USE_SLM
        __local half reduce[RANK];
#else
        half reduce[RANK] = {0.f};
#endif

        __global half *C_ptr = CC + wgid_M * N + n_idx;
        __global half *A_ptr = A + wgid_M * RANK * PART_NUM;
        __global half *B_ptr = B + n_idx * RANK;

        //1. Reduce
#if GEMMB_USE_SLM
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(1);
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
        //2.  GEMMB
        //Aligning up n dimension with subgroups would cause some subgroups no workload in N dimesnion.
        if (n_idx >= N)
            return;
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
            *C_ptr = sum + mainInput[n_idx];
    }
    '''
    cl.profiling(True)
    RANK = rank
    INPUT_STATE = input_state
    OUTPUT_STATE = output_state
    SG_SZ = 16
    K_WGS = 0
    M_WGS = 0
    N_WGS = 0
    #MAX workgroup size = 1024 N_WGS si how many WGs needed along N dimension.
    MAX_GEMMA_SG_NUM = 64
    if (RANK <= MAX_GEMMA_SG_NUM ):
        GEMMA_SG_NUM = RANK
        N_WGS = 1
    else:
        # When RANK % GEMMA_SG_NUM != 0, maybe try to lower down `GEMMA_SG_NUM = GEMMA_SG_NUM / factor` to ensure  RANK % GEMMA_SG_NUM == 0?
        GEMMA_SG_NUM = MAX_GEMMA_SG_NUM
        N_WGS = (RANK + GEMMA_SG_NUM - 1) // GEMMA_SG_NUM
    GEMMA_USE_SLM = 1
    GEMMB_USE_SLM = 1
    # K_WGS is how many wgs are divided on the GEMMA K dimension.

    if batch == 1:
        K_WGS = 8
        GEMMA_UNROLL_NUM = 4
        M_WGS = 1
    else:
        K_WGS = 8
        GEMMA_UNROLL_NUM = 4
        M_WGS = batch

    # GEMMA micro kernel would accumulate GEMMA_UNROLL_NUM * SG_SZ in k dimension.
    assert INPUT_STATE % (SG_SZ * GEMMA_UNROLL_NUM) == 0, f"'input state' {INPUT_STATE} is not multiple of Kblocking {SG_SZ * GEMMA_UNROLL_NUM}"
    assert RANK % SG_SZ == 0, f"'lora_rank' {RANK} is not multiple of SG_SZ {SG_SZ}"
    # K blocks divided along WGs.
    INPUT_STATE_BLK_SZ = SG_SZ * GEMMA_UNROLL_NUM
    INPUT_STATE_BLK_NUM = INPUT_STATE // INPUT_STATE_BLK_SZ
    INPUT_STATE_PER_WG = ((INPUT_STATE_BLK_NUM + K_WGS - 1) // K_WGS) * INPUT_STATE_BLK_SZ
    # assert( KBLKS_PER_WG > 0, f"Can't satisfy at least 1 KBLK(16) for one WG")

    # GEMMB related setting
    MAX_GEMMB_SG_NUM = 64
    if (OUTPUT_STATE <= MAX_GEMMB_SG_NUM):
        GEMMB_LWS = OUTPUT_STATE * SG_SZ
        GEMMB_GWS = OUTPUT_STATE * SG_SZ
    elif OUTPUT_STATE == 512 or OUTPUT_STATE == 1536:
        GEMMB_LWS = MAX_GEMMB_SG_NUM * SG_SZ
        GEMMB_GWS = OUTPUT_STATE * SG_SZ
    elif OUTPUT_STATE == 3840:
        GEMMB_LWS = 60 * SG_SZ
        GEMMB_GWS = OUTPUT_STATE * SG_SZ
    else:
        GEMMB_LWS = MAX_GEMMB_SG_NUM * SG_SZ
        GEMMB_GWS = (OUTPUT_STATE + MAX_GEMMB_SG_NUM - 1) // MAX_GEMMB_SG_NUM * MAX_GEMMB_SG_NUM * SG_SZ

    vRANGE = 1
    # np.random.seed(0)
    main_input_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [batch, OUTPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    lora_input_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [batch, INPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    stateA_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [RANK, INPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    stateB_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [OUTPUT_STATE, RANK]).astype(np.float16)) for _ in range(rep)]
    res_list = [cl.tensor(np.random.randint(-vRANGE, vRANGE, [batch, OUTPUT_STATE]).astype(np.float16)) for _ in range(rep)]
    alpha_list_np = [np.random.randint(-1, 2, [RANK]).astype(np.float16) for _ in range(rep)]
    alpha_list = [cl.tensor(np.multiply(alpha, 0.5)) for alpha in alpha_list_np]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA.
    gemmA_output_list = [cl.tensor(np.zeros([batch, K_WGS, RANK]).astype(np.float16)) for _ in range(rep)]
    kernel_opt = cl.kernels(src, options=f"-DSG_SZ={SG_SZ} -DK_PER_WG={INPUT_STATE_PER_WG}  -DRANK={RANK} -DGEMMA_UNROLL_NUM={GEMMA_UNROLL_NUM}\
                            -DGEMMB_USE_SLM={GEMMB_USE_SLM} -DGEMMA_USE_SLM={GEMMA_USE_SLM} -DPART_NUM={K_WGS}")
    kernel_ref =  cl.kernels(ref_src)
    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    print(f'| INPUT_STATE:{input_state}, RANK:{rank}, OUPUT_STATE:{output_state}:')
    print(f'| [GEMMA]: GWS[{M_WGS}, {K_WGS}, {N_WGS*SG_SZ * GEMMA_SG_NUM}], LWS:[1, 1, {SG_SZ * GEMMA_SG_NUM}] ')
    print(f'| [GEMMB]: GWS:[{M_WGS}, {GEMMB_GWS}], LWS:[{1, GEMMB_LWS}], ')
    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    flops_a = (rank*input_state*2) * batch
    flops_b = (rank*output_state*2+ rank + output_state) * batch
    # how many CPU memory loaded into GPU memory, stateA + input activation. duplicate input activation would be in GPU global cache?
    rd_bytes_a = (rank*input_state+input_state*batch)*2
    # gemma output intermedia result should reside in GPU memory in assumption,stateB + alpha + main_input + output
    rd_bytes_b = (rank*output_state+output_state*2*batch+rank)*2
    # Invalid GPU cache.Ensure data is not in cache. Cache size 4MB.
    tempbuffer =  [cl.tensor(np.zeros([4*1024*1024]).astype(np.float16)) for _ in range(rep)]
    cl.finish()
    for i in range(0, rep):
        tA_reduce = cl.tensor([batch, RANK], np.dtype(np.float16))
        kernel_opt.enqueue("gemmA", [M_WGS, K_WGS, N_WGS * SG_SZ * GEMMA_SG_NUM], [1, 1, SG_SZ * GEMMA_SG_NUM], lora_input_list[i], stateA_list[i], gemmA_output_list[i], INPUT_STATE)
        kernel_opt.enqueue("gemmB", [M_WGS, GEMMB_GWS],[1, GEMMB_LWS], main_input_list[i], gemmA_output_list[i], stateB_list[i], res_list[i], alpha_list[i], OUTPUT_STATE)
    profiling_data = cl.finish()
    for i in range(0, rep):
        if check_acc:
            tA_output_ref = cl.tensor([batch, RANK], np.dtype(np.float16))
            tResult_ref = cl.tensor([batch, OUTPUT_STATE], np.dtype(np.float16))
            kernel_ref.enqueue("gemmA", [batch, RANK],[1, 1], lora_input_list[i], stateA_list[i], tA_output_ref, RANK, INPUT_STATE, INPUT_STATE_PER_WG)
            kernel_ref.enqueue("gemmB", [batch, OUTPUT_STATE],[1, 16], main_input_list[i],  tA_output_ref, stateB_list[i], tResult_ref, alpha_list[i],  RANK, OUTPUT_STATE)
            kernel_ref.enqueue("reduce", [batch, RANK],[1, RANK], gemmA_output_list[i], tA_reduce, RANK, K_WGS)
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
def test_transposeB_acc():
    # for rank in range(16//16, 256//16):
    #     for out_state in range(512//16, 3840//16):
    #         #unroll is 4, SG_AZ*4  = 64, input_state%64 == 0
    #         for input_state in (1024, 1536, 2048, 2560, 3072):
    #             test_single_lora_transposeB(1, rank*16, input_state, out_state*16, 1, True)
    for batch in (2, 5, 7, 8, 128, 213):
        for rank in range(16//16, 128//16):
            for out_state in range(512//512, 3840//512):
                #unroll is 4, SG_AZ*4  = 64, input_state%64 == 0
                for input_state in (1024, 1536, 2048, 2560, 3072):
                    test_single_lora_transposeB(batch, rank*16, input_state, out_state*256, 1, True)
    # test_single_lora_transposeB(2, 64, 1536, 512, 1, True)


def test_transposeB_perf():
    #Check perf of cusmtomer minicpm
    for rank in [64]:
        for out_state in [512, 1536, 3840]:
            for input_state in [1536]:
                test_single_lora_transposeB(1024, rank, input_state, out_state, 20)
    #Increase workload to ensure memory bound. 
    for rank in [1536*2]:
        for out_state in [2048*4]:
            for input_state in [1536*2]:
                test_single_lora_transposeB(1, rank, input_state, out_state, 20)

# tranposeB = False test
# test_single_lora()

# tranposeB = True test
# Check accuray, will take a while.
# test_transposeB_acc()
test_transposeB_perf()
