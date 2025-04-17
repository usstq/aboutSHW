#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *
def test_lora_fused_qkv():
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
        __global half* subA_ptr= A + k_start;
#if GEMMA_USE_SLM
        //Workgroup share same input activation and each subgroup would access all the elements of the shared intput.Put it into SLM.
        __local half local_input[K_PER_WG];
        int local_sz = get_local_size(0);
        int id_sg_local = get_sub_group_local_id();
        //sg could diverge here. not all the sgs can satisfy 'offset < k_len'.
        __attribute__((opencl_unroll_hint))
        for (int offset = sgid * SG_SZ; offset < k_len; offset += local_sz) {
            __global half *input_ptr = subA_ptr + offset;
            half copy_val = as_half(intel_sub_group_block_read_us((const __global ushort*)input_ptr));
            local_input[offset + id_sg_local] = copy_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
#endif
        half sum = 0;
        for (int subk = 0;  subk < k_len; subk += SG_SZ) {
#if GEMMA_USE_SLM
            //ushort input = as_ushort(local_input[subk + id_sg_local]);
            ushort input = intel_sub_group_block_read_us((const __local ushort*)(local_input + subk));
#else
            ushort input = intel_sub_group_block_read_us((const __global ushort*)(subA_ptr + subk));
#endif
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)weight_ptr));
                half aa = as_half(intel_sub_group_broadcast(input, j));
                sum = fma(aa, bb, sum);
                weight_ptr += RANK;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum));
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global half * main_input, __global half * input_ptr,  __global half *BQ, __global half *BK,__global half *BV, __global half *CC, __global half * alpha, int A_stride) {
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

        // Default A, B for Q projection
        __global half *A_ptr = input_ptr;
        __global half *B_ptr = BQ + sg_idx * SG_SZ;
        int B_stride = KV_OUTPUT_STATE * KV_GROUP_SZ;
        half* sub_alpha = alpha;

        if (sg_idx >= KV_OUTPUT_STATE * (KV_GROUP_SZ + 1) / SG_SZ) {
            // V projection
            A_ptr += RANK * 2;
            B_ptr = BV + sg_idx * SG_SZ - KV_OUTPUT_STATE * (KV_GROUP_SZ + 1);
            B_stride = KV_OUTPUT_STATE;
            sub_alpha = alpha + RANK * 2;
        } else if (sg_idx >=KV_OUTPUT_STATE * KV_GROUP_SZ / SG_SZ) {
            // K projection
            A_ptr += RANK;
            B_ptr = BK + sg_idx * SG_SZ - KV_OUTPUT_STATE * KV_GROUP_SZ;
            B_stride = KV_OUTPUT_STATE;
            sub_alpha = alpha + RANK * 1;
        }

        //1. Reduce
#if GEMMB_USE_SLM
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        // Q, K, V has its separate input activation(let us call input_Q, input_K, input_V) and weight(BQ, BK, BV).
        // We must ensure Q, K, V subgroups can't be mixed inot same WG. See  `GEMMB_USE_SLM` initializing check.
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
                subA_ptr += A_stride;
            }
            reduce[offset + id_sg_local] = sum;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
#else
        //Each work item would reduce the input activation.
        for (int i = 0; i < PART_NUM; i++) {
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < RANK; j++) {
                reduce[j] += A_ptr[i * A_stride + j];
            }
        }
#endif
        //2.  GEMMB
        __global half *C_ptr = CC + sg_idx * SG_SZ;
        __global half *main_input_ptr = main_input + sg_idx * SG_SZ;
        half m_input = as_half(intel_sub_group_block_read_us((const __global ushort*)main_input_ptr));
        half sum = 0;
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            half scale = as_half(intel_sub_group_block_read_us((const __global ushort*)(sub_alpha + kk)));
#if GEMMB_USE_SLM
            half input = as_half(intel_sub_group_block_read_us((const __local ushort*)(reduce + kk)));
#else
            half input = reduce[id_sg_local + kk];
#endif
            input *= scale;
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                half bb = as_half(intel_sub_group_block_read_us((const __global ushort*)B_ptr));
                half aa = as_half(intel_sub_group_broadcast(as_ushort(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += B_stride;
            }
        }
        intel_sub_group_block_write_us((const __global ushort*)C_ptr, as_short(sum+m_input));
    }
    '''
    cl.profiling(True)
    CHECK_RES = 1
    INPUT_STATE = 1536
    KV_OUTPUT_STATE = 512
    Q_OUTPUT_STATE = INPUT_STATE
    KV_GROUP_SZ = Q_OUTPUT_STATE // KV_OUTPUT_STATE
    QKV_OUTPUT_STATE = (Q_OUTPUT_STATE+KV_OUTPUT_STATE*2)
    RANK = 64
    batch = 1
    N = RANK * 3
    K = INPUT_STATE
    WG_NUM = 8
    SG_SZ = 8
    # Run in parallel in 8 Xecores.
    # RANK and input hidden_size should be multiple of subgroup_size
    assert RANK % SG_SZ == 0 and RANK//SG_SZ >= 1, f"'Rank'  {RANK} is not multiple of SG_SIZE {SG_SZ}"
    # 256 is max number of each EU run with 1 thread. Maybe 1024? keep 256 for now.
    assert N <=256, f"'N'  {N} exceeds max value 256"
    assert K % SG_SZ == 0, f"'hidden_size' {K} is not multiple of SG_SIZE {SG_SZ}"

    N_SG_NUM = N // SG_SZ
    K_SG_NUM = K // SG_SZ
    K_PER_WG = ((K_SG_NUM + WG_NUM - 1) // WG_NUM) * SG_SZ
    USE_RANDINT = 0
    # gemmb used 8 EUs per Xecore to balance the workload between Xecores(3, 4, 7, 8 Xecores can has)
    GEMMB_LOCAL_SIZE = 128
    assert (QKV_OUTPUT_STATE) % GEMMB_LOCAL_SIZE == 0, f"GEMMB: 'global size' {(QKV_OUTPUT_STATE)} is not multiple of 'local size' {GEMMB_LOCAL_SIZE}"

    # Each Xecore should share the same [1,RANK] input for gemmB.If  KV_OUTPUT_STATE % GEMMB_LOCAL_SIZE, means  Xecore  would use 2 different inputs(such as `RANK` input elements across input_Q and input_K).
    GEMMB_USE_SLM = 1 if KV_OUTPUT_STATE % GEMMB_LOCAL_SIZE == 0 else 0
    GEMMA_USE_SLM = 1
    # np.random.seed(0)
    print(f"WG_NUM for K: {WG_NUM} , GEMMB_USE_SLM: {GEMMB_USE_SLM}, GEMMA_USE_SLM: {GEMMA_USE_SLM}")
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
    # Split [K, N]  to  3 [K, RANK]. Slicing would keep the original stride.So flatten and reshape.
    alpha = np.random.rand(3, RANK).astype(np.float16)
    tAlpha = cl.tensor(alpha)
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
    tMainInput = cl.tensor(mainInput)

    tOutputB_Ref = cl.tensor([1, QKV_OUTPUT_STATE], np.dtype(np.float16))
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    tOutputA = cl.tensor(np.zeros([WG_NUM, N]).astype(np.float16))
    toutputA_Reduce = cl.tensor([1, N], np.dtype(np.float16))
    tOutputB = cl.tensor([1, QKV_OUTPUT_STATE], np.dtype(np.float16))
    ref_ker = kernel_cache(reference, options=f"-DN_0={Q_OUTPUT_STATE} -DN_1_2={KV_OUTPUT_STATE}")
    # Each WG would update `RANK` columns. 3 WGs needed.
    ref_ker.enqueue("gemmA", [batch, N],[1, RANK], tInput, tWeiA, tOutputA_Ref, N, K, K_PER_WG, 1)
    # Each WG would update `KV_OUTPUT_STATE` columns. (KV_GROUP_SZ + 2) WGs needed.
    ref_ker.enqueue("gemmB", [batch, QKV_OUTPUT_STATE],[1, min(QKV_OUTPUT_STATE, 1024)], tMainInput, tOutputA_Ref, tWeiB_Q, tWeiB_K, tWeiB_V, tOutputB_Ref, tAlpha, RANK)
    cl.finish()

    opt_kernel = kernel_cache(opt_src, options=f"-DSG_SZ={SG_SZ} -DPART_NUM={WG_NUM} -DK_PER_WG={K_PER_WG} -DRANK={RANK} -DKV_OUTPUT_STATE={KV_OUTPUT_STATE} -DKV_GROUP_SZ={KV_GROUP_SZ} -DGEMMB_USE_SLM={GEMMB_USE_SLM} -DGEMMA_USE_SLM={GEMMA_USE_SLM}")
    # N columns in one WG. K is divided by WG_NUM. K/WG_NUM accumulated  in each WG.
    opt_kernel.enqueue("gemmA", [N*WG_NUM],[N], tInput, tWeiA_Q, tWeiA_K, tWeiA_V, tOutputA, N, K)
    # `K` dimension is small. So divide columns across all the WGs. local size can be enlarged based on input hidden size and groups.
    opt_kernel.enqueue("gemmB", [QKV_OUTPUT_STATE],[GEMMB_LOCAL_SIZE], tMainInput, tOutputA, tWeiB_Q, tWeiB_K, tWeiB_V, tOutputB, tAlpha, N)
    print(cl.finish())
    if CHECK_RES:
        # reduce A kernel is used to check gemmA result.
        # ref_ker.enqueue("reduceA", [N],[N], tOutputA, toutputA_Reduce, N, WG_NUM)
        # cl.finish()
        # multiplying scale make these comparing failing.ref mulitply in gemma, opt multiply in gemmb.
        # compare(tOutputA_Ref.numpy(), toutputA_Reduce.numpy())
        # print(" GEMMA OK!")
        compare(tOutputB_Ref.numpy(), tOutputB.numpy())
        print(" GEMMB OK!")


cl.profiling(True)
test_lora_fused_qkv()
print("-------------------------")