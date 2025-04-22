#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

def ALIGN_UP(a, b):
    return ((a + (b -1)) // b *b)
def DIV_UP(a, b):
    return ((a + (b -1)) // b)


def test_lora_fused_qkv(input_state, kv_state, rank):
    reference =  r'''
    __kernel void reduceA(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }

        __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk, int M) {
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
                sub_sum = fma(A[m_idx * K + k_idx], B[k_idx*N+n_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        CC[m_idx * N + n_idx] = sum;
    }

    __kernel void gemmB(__global half * main_input, __global half * lora_input, __global half *B_0,  __global half *B_1, __global half *B_2, __global half *CC, __global half * alpha, int rank) {
        const int m_idx = get_global_id(0);
        const int n_idx = get_global_id(1);
        int n_off = n_idx;
        __global half *A_ptr = lora_input + m_idx * rank * 3;
        __global half* scale_ptr = alpha;
        __global half *B_ptr = B_0 + n_off;
        int stride_B = N_0;
        int stride_C = (N_0 + N_1_2 + N_1_2);

        if (n_idx >= (N_0+N_1_2)) {
            // Use B_2
            A_ptr += rank *2;
            scale_ptr += rank * 2;
            n_off -= (N_0+N_1_2);
            B_ptr = B_2 + n_off;
            stride_B = N_1_2;

        } else if (n_idx >= N_0) {
            //Use B_1
            A_ptr += rank;
            scale_ptr += rank;
            n_off -= N_0;
            B_ptr = B_1 + n_off;
            stride_B = N_1_2;
        }
        
        half sum = 0.f;
        for (int k = 0; k < rank; k++)
            sum = fma(A_ptr[k]*scale_ptr[k], B_ptr[k*stride_B], sum);
        CC[m_idx *stride_C + n_idx] = sum + main_input[m_idx *stride_C + n_idx];
    }
    '''
    opt_src =  r'''
#define MAX_GEMMA_SGK (64/3)
#define MAX_LORA_RANK (256)
#define MAX_N (MAX_LORA_RANK * 3)
#define MAX_GEMMA_SG_BK 256
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A, __global half *B0, __global half *B1, __global half *B2,  __global half *C, int K) {

        int gid0 =  get_group_id(0);
        int sgid = get_sub_group_id();
        // For 2nd token, rank is small. sg in one wg would be divided by 2 dimensions to increase threads number in wg.
        int sgN = RANK*3 / SG_SZ;
        int sgid_k = sgid / sgN;
        int n_idx = sgid % sgN * SG_SZ;
        int n_off = n_idx % RANK;
        int B_idx = n_idx / RANK;
        __global half *B = B_idx == 0 ? B0 : ((B_idx == 1) ? B1: B2);
        B += n_off;
        
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int bk_wg = GEMMA_SGK * GEMMA_SG_BK;
        int k_start_wg = gid0 * bk_wg;
        int wg_k_len = (k_start_wg + bk_wg) > K ?  (K - k_start_wg) : bk_wg;
        int sgK = (wg_k_len + GEMMA_SG_BK - 1) / GEMMA_SG_BK;

        // store each sg accumulation result into SLM. Will reduce sg result into wg result. 
        __local half fma_buff[MAX_GEMMA_SGK * MAX_N];
        __local half *sg_fma_buff = fma_buff + sgid_k * MAX_N;

        //put all need input activation into SLM. 'sgN' sgs would share same input.
        __local half local_input[MAX_GEMMA_SGK * MAX_GEMMA_SG_BK];
        int local_sz = get_num_sub_groups() * SG_SZ;
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        for (int offset = sgid * SG_SZ; offset < wg_k_len; offset += local_sz) {
            __global half *input_ptr = A + k_start_wg + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int k_offset = sgid_k * GEMMA_SG_BK;
        int k_idx = k_start_wg + k_offset;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
        if (sgid_k * GEMMA_SG_BK  <  wg_k_len) {
            int klen_sg =  (k_offset + GEMMA_SG_BK) > wg_k_len ? (wg_k_len - k_offset) : GEMMA_SG_BK;
            __global half *B_ptr = B + k_idx * RANK;
            __local half *A_ptr = local_input + k_offset;
            half sum = 0.f;
            for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
                ushort input = intel_sub_group_block_read_us((const __local ushort*)(A_ptr + kk));
                __attribute__((opencl_unroll_hint))
                for (int j = 0; j < SG_SZ; j++) {
                    half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
                    half aa = as_half(intel_sub_group_broadcast(input, j));
                    sum = fma(aa, bb, sum);
                    B_ptr += RANK;
                }
            }
            *(sg_fma_buff + n_idx + lid) = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __global half *C_ptr = C + RANK * 3 * gid0;
        //only need sg on N dimenion to update data.
        if (sgid_k != 0)
            return;
        half sum = 0.f;
        if (sgK == GEMMA_SGK) {
            //unroll
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < GEMMA_SGK; i++) {
                sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_N + n_idx)));
            }
        } else {
            //can't unroll, tail handling
            for (int i = 0;  i < sgK; i++)
            sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_N + n_idx)));
        }
        intel_sub_group_block_write_us((const __global ushort*)(C_ptr + n_idx), as_short(sum));
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * main_input, __global half *A,  __global half *B0, __global half *B1,__global half *B2,  __global half *C, __global half * alpha, int N) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int n_idx = (wg_id * sg_num  +  sg_id) * SG_SZ;
        int id_sg_local = get_sub_group_local_id();
        __local half reduce[MAX_N];
        int slm_offset = 0;
        __global half *B_ptr = B0 + n_idx;
        int B_stride = N0;
        if (n_idx >= (N0 + N1_2)) {
            // V projection
            B_ptr = B2 + n_idx - N0 - N1_2;
            B_stride = N1_2;
            alpha +=  RANK * 2;
            slm_offset = RANK *2;
        } else if (n_idx >= N0) {
            // K projection
            B_ptr = B1 + n_idx - N0;
            B_stride = N1_2;
            alpha += RANK;
            slm_offset = RANK;
        }
    
        //1. Reduce
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        for (int offset = sg_id * SG_SZ; offset < MAX_N; offset += local_sz) {
            __global half *A_ptr = A + offset;
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < GEMMB_PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)A_ptr));
                sum += partial_val;
                A_ptr += RANK*3;
            }
            reduce[offset + id_sg_local] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_idx >= N)
            return;
        //2.  GEMMB
        __global half *C_ptr = C + n_idx;
        __local half* reduce_ptr = reduce + slm_offset;
        half sum = 0;
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            half scale = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha + kk)));
            half input = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce_ptr + kk)));

            input *= scale;
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += B_stride;
            }
        }
        __global half *main_input_ptr = main_input + n_idx;
        half m_input = as_half(intel_sub_group_block_read_us((const __global ushort*)main_input_ptr));
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum+m_input));
    }
    '''
    cl.profiling(True)
    REPEAT = 20
    CHECK_RES = 1
    INPUT_STATE = input_state
    KV_OUTPUT_STATE = kv_state
    Q_OUTPUT_STATE = INPUT_STATE
    QKV_OUTPUT_STATE = (Q_OUTPUT_STATE+KV_OUTPUT_STATE*2)
    RANK = rank
    batch = 1
    N = RANK * 3
    K = INPUT_STATE
    SG_SZ = 16

    MAX_WG_SZ = 1024
    gemma_sgK = MAX_WG_SZ//(RANK*3)
    gemma_sg_BK = 32
    gemma_wg_BK = gemma_sg_BK * gemma_sgK
    gemma_wgs = DIV_UP(INPUT_STATE, gemma_sg_BK *gemma_sgK)
    PART_NUM = DIV_UP(INPUT_STATE, gemma_sg_BK *gemma_sgK)
    gemma_lws = [1 , gemma_sgK, RANK*3]
    gemma_gws = [gemma_wgs, gemma_sgK, RANK*3]
    
    gemmb_sgN = MAX_WG_SZ//SG_SZ
    gemmb_wg_sz = gemmb_sgN * SG_SZ
    gemmb_lws = [gemmb_wg_sz]
    gemmb_gws = [ALIGN_UP(QKV_OUTPUT_STATE, gemmb_wg_sz)]

    # Run in parallel in 8 Xecores.
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0 and RANK//SG_SZ >= 1, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"

    USE_RANDINT = 0

    # np.random.seed(0)
    if USE_RANDINT:
        vRANGE = 2
        inputA = np.random.randint(-vRANGE, vRANGE, [1, K]).astype(np.float16)
        weiA = np.random.randint(-vRANGE, vRANGE, [K, N]).astype(np.float16)
        weiB_Q = np.random.randint(-vRANGE, vRANGE, [RANK, Q_OUTPUT_STATE]).astype(np.float16)
        weiB_K = np.random.randint(-vRANGE, vRANGE, [RANK, KV_OUTPUT_STATE]).astype(np.float16)
        weiB_V = np.random.randint(-vRANGE, vRANGE, [RANK, KV_OUTPUT_STATE]).astype(np.float16)
        mainInput = np.random.randint(-vRANGE, vRANGE, [1, QKV_OUTPUT_STATE]).astype(np.float16)
    else :
        inputA = np.random.rand(1, K).astype(np.float16)
        weiA = np.random.rand(K, N).astype(np.float16)
        weiB_Q = np.random.rand(RANK, Q_OUTPUT_STATE).astype(np.float16)
        weiB_K = np.random.rand(RANK, KV_OUTPUT_STATE).astype(np.float16)
        weiB_V = np.random.rand(RANK, KV_OUTPUT_STATE).astype(np.float16)
        mainInput = np.random.rand(1, QKV_OUTPUT_STATE).astype(np.float16)
    outputA = np.zeros([gemma_wgs, RANK*3]).astype(np.float16)
    outputB = np.random.rand(1, QKV_OUTPUT_STATE).astype(np.float16)

    # Split [K, N]  to  3 [K, RANK]. Slicing would keep the original stride.So flatten and reshape.
    alpha = np.random.rand(3, RANK).astype(np.float16)
    weiA_Q = weiA[:, 0:RANK].flatten().reshape(K, RANK)
    weiA_K = weiA[:, RANK:2*RANK].flatten().reshape(K, RANK)
    weiA_V = weiA[:, 2*RANK:N].flatten().reshape(K, RANK)
    
    stateA0_list= [cl.tensor(weiA_Q) for _ in range(REPEAT)]
    stateA1_list= [cl.tensor(weiA_K) for _ in range(REPEAT)]
    stateA2_list= [cl.tensor(weiA_V) for _ in range(REPEAT)]

    stateB0_list= [cl.tensor(weiB_Q) for _ in range(REPEAT)]
    stateB1_list= [cl.tensor(weiB_K) for _ in range(REPEAT)]
    stateB2_list= [cl.tensor(weiB_V) for _ in range(REPEAT)]

    # todo needs to split when developing GEMMB
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]

    loraInput_list = [cl.tensor(inputA)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    outputA_list =  [cl.tensor(outputA)for _ in range(REPEAT)]
    outputB_list =  [cl.tensor(outputB)for _ in range(REPEAT)]

    
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 

    print(f'----------------------------------------------------------------------------------------------------------------------------------')
    print(f'| BATCH = 1 Q_STATE:{input_state}, RANK:{rank}, KV_STATE:{kv_state}:')
    print(f'| [2ND_GEMMA]: GWS:{gemma_gws}, LWS:{gemma_lws} SG_BK:{gemma_sg_BK} SGK:{gemma_sgK}')
    print(f'| [2ND_GEMMB]: GWS:{gemmb_gws}, LWS:{gemmb_lws} SGN:{gemmb_sgN}')
    print(f'----------------------------------------------------------------------------------------------------------------------------------')

    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DGEMMB_PART_NUM={PART_NUM} -DRANK={RANK} -DN1_2={KV_OUTPUT_STATE} -DN0={Q_OUTPUT_STATE}\
                                -DGEMMA_SGK={gemma_sgK} -DGEMMA_SG_BK={gemma_sg_BK}")

    for i in range(0, REPEAT):
        opt_kernel.enqueue("gemmA", gemma_gws, gemma_lws, loraInput_list[i], stateA0_list[i], stateA1_list[i], stateA2_list[i], outputA_list[i], INPUT_STATE)
        opt_kernel.enqueue("gemmB", gemmb_gws ,gemmb_lws, mainInput_list[i], outputA_list[i],  stateB0_list[i], stateB1_list[i], stateB2_list[i], outputB_list[i], alpha_list[i], QKV_OUTPUT_STATE)
    print(cl.finish())
    if CHECK_RES:
        tOutputA_Ref = cl.tensor([1, RANK*3], np.dtype(np.float16))
        tOutputB_Ref = cl.tensor([1, QKV_OUTPUT_STATE], np.dtype(np.float16))
        tWeiA = cl.tensor(weiA)
        toutputA_Reduce = cl.tensor([1, N], np.dtype(np.float16))
        ref_ker = kernel_cache(reference, options=f"-DN_0={Q_OUTPUT_STATE} -DN_1_2={KV_OUTPUT_STATE}")
        # Each WG would update `RANK` columns. 3 WGs needed.
        ref_ker.enqueue("gemmA", [batch, N],[1, RANK], loraInput_list[0], tWeiA, tOutputA_Ref, N, K, gemma_wg_BK, 1)
        ref_ker.enqueue("gemmB", [batch, QKV_OUTPUT_STATE],[1, min(QKV_OUTPUT_STATE, 1024)], mainInput_list[0], tOutputA_Ref, stateB0_list[0], stateB1_list[0], stateB2_list[0], tOutputB_Ref, alpha_list[0], RANK)
        compare(tOutputB_Ref.numpy(), outputB_list[0].numpy())
        print(" ACC OK!")

cl.profiling(True)
# test_lora_fused_qkv(1536, 128, 64)
# test_lora_fused_qkv(1536, 256, 64)
# for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*16, 11*16, 13*16, 15*16, 12*16,17*16):
#     for rank in (16, 32, 64, 128, 256):
#         test_lora_fused_qkv(input_state, input_state, rank)
#         for kv_groups in (3, 4, 5, 6, 7, 8, 9):
#             if input_state % kv_groups == 0 and input_state//kv_groups%16 == 0:
#                 test_lora_fused_qkv(input_state, input_state//kv_groups, rank)

lora_qkv_ref_kernel =  r'''
    __kernel void reduceA(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }

        __kernel void gemmA(__global half * A, __global half *B,  __global half *CC, int N, int K, int k_blk, int M) {
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
                sub_sum = fma(A[m_idx * K + k_idx], B[k_idx*N+n_idx], sub_sum);
                k_idx++;
            }
            sum += sub_sum;
        }
        CC[m_idx * N + n_idx] = sum;
    }

    __kernel void gemmB(__global half * main_input, __global half * lora_input, __global half *B_0,  __global half *B_1, __global half *B_2, __global half *CC, __global half * alpha, int rank) {
        const int m_idx = get_global_id(0);
        const int n_idx = get_global_id(1);
        int n_off = n_idx;
        __global half *A_ptr = lora_input + m_idx * rank * 3;
        __global half* scale_ptr = alpha;
        __global half *B_ptr = B_0 + n_off;
        int stride_B = N_0;
        int stride_C = (N_0 + N_1_2 + N_1_2);

        if (n_idx >= (N_0+N_1_2)) {
            // Use B_2
            A_ptr += rank *2;
            scale_ptr += rank * 2;
            n_off -= (N_0+N_1_2);
            B_ptr = B_2 + n_off;
            stride_B = N_1_2;

        } else if (n_idx >= N_0) {
            //Use B_1
            A_ptr += rank;
            scale_ptr += rank;
            n_off -= N_0;
            B_ptr = B_1 + n_off;
            stride_B = N_1_2;
        }
        
        half sum = 0.f;
        for (int k = 0; k < rank; k++)
            sum = fma(A_ptr[k]*scale_ptr[k], B_ptr[k*stride_B], sum);
        CC[m_idx *stride_C + n_idx] = sum + main_input[m_idx *stride_C + n_idx];
    }
    '''
lora_qkv_opt =  r'''
#define MAX_GEMMA_SGK (64/3)
#define MAX_LORA_RANK (256)
#define MAX_N (MAX_LORA_RANK * 3)
#define MAX_GEMMA_SG_BK 256
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global half * A, __global half *B0, __global half *B1, __global half *B2,  __global half *C, int K) {

        int gid0 =  get_group_id(0);
        int sgid = get_sub_group_id();
        // For 2nd token, rank is small. sg in one wg would be divided by 2 dimensions to increase threads number in wg.
        int sgN = RANK*3 / SG_SZ;
        int sgid_k = sgid / sgN;
        int n_idx = sgid % sgN * SG_SZ;
        int n_off = n_idx % RANK;
        int B_idx = n_idx / RANK;
        __global half *B = B_idx == 0 ? B0 : ((B_idx == 1) ? B1: B2);
        B += n_off;
        
        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int bk_wg = GEMMA_SGK * GEMMA_SG_BK;
        int k_start_wg = gid0 * bk_wg;
        int wg_k_len = (k_start_wg + bk_wg) > K ?  (K - k_start_wg) : bk_wg;
        int sgK = (wg_k_len + GEMMA_SG_BK - 1) / GEMMA_SG_BK;

        // store each sg accumulation result into SLM. Will reduce sg result into wg result. 
        __local half fma_buff[MAX_GEMMA_SGK * MAX_N];
        __local half *sg_fma_buff = fma_buff + sgid_k * MAX_N;

        //put all need input activation into SLM. 'sgN' sgs would share same input.
        __local half local_input[MAX_GEMMA_SGK * MAX_GEMMA_SG_BK];
        int local_sz = get_num_sub_groups() * SG_SZ;
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        for (int offset = sgid * SG_SZ; offset < wg_k_len; offset += local_sz) {
            __global half *input_ptr = A + k_start_wg + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + lid] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int k_offset = sgid_k * GEMMA_SG_BK;
        int k_idx = k_start_wg + k_offset;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
        if (sgid_k * GEMMA_SG_BK  <  wg_k_len) {
            int klen_sg =  (k_offset + GEMMA_SG_BK) > wg_k_len ? (wg_k_len - k_offset) : GEMMA_SG_BK;
            __global half *B_ptr = B + k_idx * RANK;
            __local half *A_ptr = local_input + k_offset;
            half sum = 0.f;
            for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
                ushort input = intel_sub_group_block_read_us((const __local ushort*)(A_ptr + kk));
                __attribute__((opencl_unroll_hint))
                for (int j = 0; j < SG_SZ; j++) {
                    half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)(B_ptr)));
                    half aa = as_half(intel_sub_group_broadcast(input, j));
                    sum = fma(aa, bb, sum);
                    B_ptr += RANK;
                }
            }
            *(sg_fma_buff + n_idx + lid) = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __global half *C_ptr = C + RANK * 3 * gid0;
        //only need sg on N dimenion to update data.
        if (sgid_k != 0)
            return;
        half sum = 0.f;
        if (sgK == GEMMA_SGK) {
            //unroll
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < GEMMA_SGK; i++) {
                sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_N + n_idx)));
            }
        } else {
            //can't unroll, tail handling
            for (int i = 0;  i < sgK; i++)
            sum += as_half(intel_sub_group_block_read_us((const __local ushort*)(fma_buff + i * MAX_N + n_idx)));
        }
        intel_sub_group_block_write_us((const __global ushort*)(C_ptr + n_idx), as_short(sum));
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * main_input, __global half *A,  __global half *B0, __global half *B1,__global half *B2,  __global half *C, __global half * alpha, int N) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int n_idx = (wg_id * sg_num  +  sg_id) * SG_SZ;
        int id_sg_local = get_sub_group_local_id();
        __local half reduce[MAX_N];
        int slm_offset = 0;
        __global half *B_ptr = B0 + n_idx;
        int B_stride = N0;
        if (n_idx >= (N0 + N1_2)) {
            // V projection
            B_ptr = B2 + n_idx - N0 - N1_2;
            B_stride = N1_2;
            alpha +=  RANK * 2;
            slm_offset = RANK *2;
        } else if (n_idx >= N0) {
            // K projection
            B_ptr = B1 + n_idx - N0;
            B_stride = N1_2;
            alpha += RANK;
            slm_offset = RANK;
        }
    
        //1. Reduce
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        for (int offset = sg_id * SG_SZ; offset < MAX_N; offset += local_sz) {
            __global half *A_ptr = A + offset;
            half sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < GEMMB_PART_NUM; part_idx++) {
                half partial_val = as_half(intel_sub_group_block_read_us((const __global ushort*)A_ptr));
                sum += partial_val;
                A_ptr += RANK*3;
            }
            reduce[offset + id_sg_local] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_idx >= N)
            return;
        //2.  GEMMB
        __global half *C_ptr = C + n_idx;
        __local half* reduce_ptr = reduce + slm_offset;
        half sum = 0;
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            half scale = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha + kk)));
            half input = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce_ptr + kk)));

            input *= scale;
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += B_stride;
            }
        }
        __global half *main_input_ptr = main_input + n_idx;
        half m_input = as_half(intel_sub_group_block_read_us((const __global ushort*)main_input_ptr));
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum+m_input));
    }
    '''
    

# GEMMA has a big K but small N(rank), GEMMA kernel  would divide K by WGs and K diemsnion SGs.
# GEMMB only divide N by WGs and SGs because N is big and K(Rank) is small..
# gemma_sg_BK: the Number of K accumuated in one sg for GEMMA.
# gemma_sgK: the number of sg in K dimension for GEMMA.
# gemmb_sgN: the number of sg in N dimension for GEMMB
class LORA_QKV_2ND:
    def __init__(self, rank, input_state, kv_state, gemma_sg_BK, gemma_sgK, gemmb_sgN, use_ref = False):
        self.rank = rank
        self.input_state = input_state
        self.q_state = input_state
        self.kv_state = kv_state
        self.sg_sz = 16
        assert gemmb_sgN <=64, f'gemmb_sgN:{gemmb_sgN} bigger than 64 limitation'
        self.gemmb_wg_sz = gemmb_sgN * self.sg_sz
        self.use_ref = use_ref
        # WGs would divide K dimension. N would not divided by WGs
        self.gemma_wgs = DIV_UP(input_state, gemma_sg_BK *gemma_sgK)
        self.gemma_sgK = gemma_sgK

        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.kv_state % self.sg_sz == 0, f"'input kv_state' {self.kv_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.rank % self.sg_sz == 0, f"'RANK' {self.rank} is not multiple of SG_SZ {self.sg_sz}"
        assert gemma_sg_BK % self.sg_sz == 0, f"'gemma_sg_BK' { gemma_sg_BK} is not multiple of SG_SZ {self.sg_sz}"
        assert self.gemmb_wg_sz % self.sg_sz == 0, f"'gemmb_wg_sz' {self.gemmb_wg_sz} is not multiple of SG_SZ {self.sg_sz}"
        self.gemma_wg_BK = gemma_sg_BK * gemma_sgK
        
        options = f'-DSG_SZ={self.sg_sz}  -DGEMMA_SGK={gemma_sgK} -DRANK={rank} -DGEMMA_SG_BK={gemma_sg_BK} -DGEMMB_PART_NUM={self.gemma_wgs}\
                        -DN1_2={self.kv_state} -DN0={self.q_state}'
        if use_ref:
            self.cl_kernels_ref = kernel_cache(lora_qkv_ref_kernel, options=f"-DN_0={self.q_state} -DN_1_2={self.kv_state}")
        else:
            self.cl_kernels_opt = kernel_cache(lora_qkv_opt, options)
        self.gemma_lws = [1 , gemma_sgK, self.rank*3]
        self.gemma_gws = [self.gemma_wgs, gemma_sgK, self.rank*3]
        self.gemmb_lws = [self.gemmb_wg_sz]
        self.gemmb_gws = [ALIGN_UP(self.q_state + 2*self.kv_state, self.gemmb_wg_sz)]
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| BATCH = 1 Q_STATE:{input_state}, KV_STATE:{kv_state}, RANK:{rank}:')
            print(f'| [2ND_GEMMA]: GWS:{self.gemma_gws}, LWS:{self.gemma_lws} SG_BK:{gemma_sg_BK} SGK:{gemma_sgK}')
            print(f'| [2ND_GEMMB]: GWS:{self.gemmb_gws}, LWS:{self.gemmb_lws} SGN:{gemmb_sgN}')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')

    def __call__(self, mainInput, loraInput, stateA0, stateA1, stateA2, stateA, stateAlpha, stateB0, stateB1, stateB2, Aoutput, result):

        if self.use_ref:
            tA_output_ref = cl.tensor([1, self.rank*3], np.dtype(np.float16))
            self.cl_kernels_ref.enqueue("gemmA", [1, self.rank*3],[1, self.rank], loraInput, stateA,
                                        tA_output_ref, self.rank*3, self.input_state, self.gemma_wg_BK, 1)
            REF_LOCAL_SIZE = 16
            self.cl_kernels_ref.enqueue("gemmB", [1, self.input_state + 2*self.kv_state],[1, min(self.input_state + 2*self.kv_state, 1024)],
                                        mainInput, tA_output_ref, stateB0, stateB1, stateB2, result, stateAlpha, self.rank)
            # print("======================ref:")
            # print(tA_output_ref.numpy())
        else:
            cl_kernels = self.cl_kernels_opt
            # GEMMA: ONE WG would has {self.gemma_sgK*self.rank/SG_SZ}subgroups. self.rank/SG_SZ subgroups on N dimension, self.gemma_sgK on K dimension
            # Total {self.gemma_wg} WGS
            cl_kernels.enqueue("gemmA", self.gemma_gws, self.gemma_lws, 
                                        loraInput, stateA0, stateA1, stateA2, Aoutput, self.input_state)
            # GEMMB: ONE WG would has {gemmb_wg_sz/SG_SZ}subgroups.
            # Total {(self.q_state+2*self.kv_state)/gemmb_wg_sz} WGS
            cl_kernels.enqueue("gemmB", self.gemmb_gws, self.gemmb_lws, 
                                        mainInput, Aoutput, stateB0, stateB1, stateB2, result, stateAlpha, self.q_state+2*self.kv_state)
            # print("======================opt:")
            # print(Aoutput.numpy())
        # cl.finish()
        return result


def test_lora_qkv_2nd(input_state, rank, kv_state, gemma_sgK = 8, gemma_sg_BK = 32, gemmb_sgN = 16, check_acc = False):
    VERBOSE=1
    cl.profiling(True)
    SG_SZ = 16
    vRANGE = 1
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 200
    np.random.seed(0)

    # for GEMMA, K decides how many WGs are needed.
    gemma_wgs = DIV_UP(input_state, gemma_sg_BK *gemma_sgK)
    stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank*3]).astype(np.float16)
    alpha = np.random.rand(3, rank).astype(np.float16)
    qkv_state = input_state + kv_state*2
    stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]

    stateA_0 = stateA[:, 0:rank].flatten().reshape(input_state, rank)
    stateA_1 = stateA[:, rank:2*rank].flatten().reshape(input_state, rank)
    stateA_2 = stateA[:, 2*rank:3*rank].flatten().reshape(input_state, rank)
    stateB0 = np.random.randint(-vRANGE, vRANGE+1, [rank, input_state]).astype(np.float16)
    stateB1 = np.random.randint(-vRANGE, vRANGE+1, [rank, kv_state]).astype(np.float16)
    stateB2 = np.random.randint(-vRANGE, vRANGE+1, [rank, kv_state]).astype(np.float16)

    loraInput = np.random.randint(-vRANGE, vRANGE+1, [1, input_state]).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [1, qkv_state]).astype(np.float16)
    Aoutput = np.zeros([gemma_wgs, rank*3]).astype(np.float16) 
    
    stateA0_list = [cl.tensor(stateA_0)for _ in range(REPEAT)]
    stateA1_list = [cl.tensor(stateA_1)for _ in range(REPEAT)]
    stateA2_list = [cl.tensor(stateA_2)for _ in range(REPEAT)]

    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    stateB0_list = [cl.tensor(stateB0)for _ in range(REPEAT)]
    stateB1_list = [cl.tensor(stateB1)for _ in range(REPEAT)]
    stateB2_list = [cl.tensor(stateB2)for _ in range(REPEAT)]

    loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([1, qkv_state], np.dtype(np.float16))for _ in range(REPEAT)]
    ref_list = [cl.tensor([1, qkv_state], np.dtype(np.float16))for _ in range(REPEAT)]
    ref = LORA_QKV_2ND(rank, input_state, kv_state, gemma_sg_BK, gemma_sgK, gemmb_sgN,True)
    opt = LORA_QKV_2ND(rank, input_state, kv_state, gemma_sg_BK, gemma_sgK, gemmb_sgN,False)
    
    if check_acc:
        opt(mainInput_list[0], loraInput_list[0], stateA0_list[0], stateA1_list[0], stateA2_list[0], None, alpha_list[0],
                stateB0_list[0],stateB1_list[0], stateB2_list[0], A_output_list[0], res_list[0])
        ref(mainInput_list[0], loraInput_list[0], None, None, None, stateA_list[0], alpha_list[0],
                stateB0_list[0],stateB1_list[0], stateB2_list[0], None, ref_list[0])
        cl.finish()
        compare(ref_list[0].numpy(), res_list[0].numpy())
        print(f'INPUT_STATE:{input_state}, RANK:{rank}, KV_STATE:{kv_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[0], loraInput_list[0], stateA0_list[0], stateA1_list[0], stateA2_list[0], stateA_list[0], alpha_list[0],
                stateB0_list[0],stateB1_list[0], stateB2_list[0], A_output_list[0], res_list[0])
        profiling_data  = cl.finish()
        # FMA
        flops_a = rank*3*input_state*2
        # FMA + multiply alpha + add main
        flops_b = rank*qkv_state*2 + rank*3 +qkv_state
        # lora input + weightA
        rd_bytes_a = (rank*3*input_state+input_state)*2
        # A output + main input + weightB  + scale
        rd_bytes_b = (rank*qkv_state+qkv_state*2+rank*3)*2

        for i in range(1, REPEAT):
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2 + 1]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
    [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
    [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')


def blocking_2nd(rank, input_state, qkv_state):
    # 512 or 1024? First try 512
    MAX_WG_SZ = 1024
    gemma_sg_BK = 32
    gemma_sgK = MAX_WG_SZ//(rank*3)
    gemmb_sgN = min(qkv_state, MAX_WG_SZ)//16
    return [gemma_sg_BK, gemma_sgK, gemmb_sgN]

cl.profiling(True)
# rank=64
# input_state=1536
# kv_state=512
# gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, input_state+2*kv_state)
# test_lora_qkv_2nd(input_state, rank, kv_state, gemma_sgK, gemma_sg_BK, gemmb_sgN, check_acc = True)

#2nd acc
if 0:
    for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*16, 11*16, 13*16, 15*16, 12*16,17*16):
        for rank in (16, 32, 64, 128, 256):
            kv_state = input_state
            gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, input_state+2*kv_state)
            test_lora_qkv_2nd(input_state, rank, kv_state, gemma_sgK, gemma_sg_BK, gemmb_sgN, check_acc = True)
            for kv_groups in (3, 4, 5, 6, 7, 8, 9):
                if input_state % kv_groups == 0 and input_state//kv_groups%16 == 0:
                    kv_state = input_state // kv_groups
                    gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, input_state+2*kv_state)
                    test_lora_qkv_2nd(input_state, rank, kv_state, gemma_sgK, gemma_sg_BK, gemmb_sgN, check_acc = True)
#2nd perf based on qwen QKV,
if 1:
    rank=64
    input_state=1536
    kv_state=256
    gemma_sg_BK, gemma_sgK, gemmb_sgN = blocking_2nd(rank, input_state, input_state+2*kv_state)
    test_lora_qkv_2nd(input_state, rank, kv_state, gemma_sgK, gemma_sg_BK, gemmb_sgN)

print("-------------------------")