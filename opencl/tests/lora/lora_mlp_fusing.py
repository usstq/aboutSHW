#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *


VERBOSE = True

def ALIGN_UP(a, b):
    return ((a + (b -1)) // b *b)
def DIV_UP(a, b):
    return ((a + (b -1)) // b)
def get_max_wg_sz():
    if os.environ.get('MAX_WG_SZ_512') != None:
        return 512
    return 1024
"""
------------------------------------------------------------------------------------------------------------------
qkv lora reference kernel
------------------------------------------------------------------------------------------------------------------
"""
mlp_lora_ref_kernel =  r'''
        __kernel void gemmA(__global float * A, __global float *B,  __global float *CC, int N, int K, int k_blk, int M) {
        // Seems kblocks accumulation would introduce error. So ref kernel accumulation should behave same with target.
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        size_t blk_num = (K + k_blk -1 )/ k_blk;
        float sum = 0.f;
        for (int i = 0; i < blk_num; i++) {
            float sub_sum = 0.f;
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

    __kernel void gemmB(__global float * main_input, __global float * lora_input, __global float *B_0,  __global float *B_1, __global float *CC, __global float * alpha, int rank) {
        const int m_idx = get_global_id(0);
        const int n_idx = get_global_id(1);
        __global float *A_ptr = lora_input + m_idx * rank * 2;
        __global float* scale_ptr = alpha;
        __global float *B_ptr = B_0 + n_idx;
        int stride_B = N_0;
        int stride_C = N_0 * 2;
        if (n_idx >= N_0) {
            //Use B_1
            A_ptr += rank;
            scale_ptr += rank;
            B_ptr = B_1 + n_idx - N_0;
        }

        float sum = 0.f;
        for (int k = 0; k < rank; k++)
            sum = fma(A_ptr[k]*scale_ptr[k], B_ptr[k*stride_B], sum);
        CC[m_idx *stride_C + n_idx] = sum + main_input[m_idx *stride_C + n_idx];
    }
    '''
"""
------------------------------------------------------------------------------------------------------------------
qkv lora 2nd token
------------------------------------------------------------------------------------------------------------------
"""
mlp_lora_opt =  r'''
//SLM :32KB for GEMMA,  1KB for GEMMB.
#define LORA_CNT 2
#define MAX_GEMMA_SGK (64/LORA_CNT)
#define MAX_LORA_RANK (256)
#define MAX_GEMMA_N (MAX_LORA_RANK*LORA_CNT)
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmA(__global float * A, __global float *B0, __global float *B1,  __global float *C, int K) {

        int gid0 =  get_group_id(0);
        int gid2 =  get_group_id(2);
        int sgid = get_sub_group_id();
        int sgid_k = sgid / sgN;
        int n_idx = (gid2 * sgN +  sgid % sgN) * SG_SZ;
        int n_off = n_idx % RANK;
        int B_idx = n_idx / RANK;
        __global float *B = B_idx == 0 ? B0 : B1;
        B += n_off;

        int lid =  get_sub_group_local_id();
        //How many K is accumulated in the WG.
        int bk_wg = GEMMA_SGK * GEMMA_SG_BK;
        int k_start_wg = gid0 * bk_wg;
        int wg_k_len = (k_start_wg + bk_wg) > K ?  (K - k_start_wg) : bk_wg;
        int sgK = (wg_k_len + GEMMA_SG_BK - 1) / GEMMA_SG_BK;

        // store each sg accumulation result into SLM. Will reduce sg result into wg result.
        __local float fma_buff[MAX_GEMMA_SGK * MAX_GEMMA_N];
        __local float *sg_fma_buff = fma_buff + sgid_k * MAX_GEMMA_N;
        int k_offset = sgid_k * GEMMA_SG_BK;
        int k_idx = k_start_wg + k_offset;

        //The sg is needs to accumulate. Otherwise, sg not needed. sg_k diverge here.
        if (sgid_k * GEMMA_SG_BK  <  wg_k_len) {
            int klen_sg =  (k_offset + GEMMA_SG_BK) > wg_k_len ? (wg_k_len - k_offset) : GEMMA_SG_BK;
            __global float *B_ptr = B + k_idx * RANK;
            __global float *A_ptr = A + k_start_wg + k_offset;
            float sum = 0.f;
            for (int kk = 0;  kk < klen_sg; kk += SG_SZ) {
                uint input = intel_sub_group_block_read((const __global uint*)(A_ptr + kk));
                __attribute__((opencl_unroll_hint))
                for (int j = 0; j < SG_SZ; j++) {
                    float bb = as_float(intel_sub_group_block_read((const __global uint*)(B_ptr)));
                    float aa = as_float(sub_group_broadcast(input, j));
                    sum = fma(aa, bb, sum);
                    B_ptr += RANK;
                }
            }
            *(sg_fma_buff + n_idx + lid) = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        __global float *C_ptr = C + RANK * LORA_CNT * gid0;
        //only need sg on N dimenion to update data.
        if (sgid_k != 0)
            return;
        float sum = 0.f;
        if (sgK == GEMMA_SGK) {
            //unroll
            __attribute__((opencl_unroll_hint))
            for (int i = 0;  i < GEMMA_SGK; i++) {
                sum += as_float(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_GEMMA_N + n_idx)));
            }
        } else {
            //can't unroll, tail handling
            for (int i = 0;  i < sgK; i++)
            sum += as_float(intel_sub_group_block_read((const __local uint*)(fma_buff + i * MAX_GEMMA_N + n_idx)));
        }
        intel_sub_group_block_write((const __global uint*)(C_ptr + n_idx), as_int(sum));
    }

    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void gemmB(__global float * main_input, __global float *A,  __global float *B0, __global float *B1,  __global float *C, __global float * alpha0, __global float * alpha1, int N) {
        int wg_id = get_group_id(0);
        int sg_id = get_sub_group_id();
        int sg_num = get_num_sub_groups();
        int n_idx = (wg_id * sg_num  +  sg_id) * SG_SZ;
        int id_sg_local = get_sub_group_local_id();
        //ONE WG maybe output both  MLP GATE and MLP UP output, So reduce the 2 A matrice in each WG for easiness.

        __local float reduce[MAX_GEMMA_N];
        int slm_offset = 0;
        __global float *B_ptr = B0 + n_idx;
         __global float* alpha = alpha0;
        int B_stride = N0;
        if (n_idx >= N0) {
            // K projection
            B_ptr = B1 + n_idx - N0;
            alpha = alpha1;
            slm_offset = RANK;
        }

        //1. Reduce
        //EACH WG would reduce input activation and save into local memory `reduce[RANK]`.
        int local_sz = get_local_size(0);
        for (int offset = sg_id * SG_SZ; offset < MAX_GEMMA_N; offset += local_sz) {
            __global float *A_ptr = A + offset;
            float sum = 0.f;
            __attribute__((opencl_unroll_hint))
            for (int part_idx = 0; part_idx < GEMMB_PART_NUM; part_idx++) {
                float partial_val = as_float(intel_sub_group_block_read((const __global uint*)A_ptr));
                sum += partial_val;
                A_ptr += RANK*LORA_CNT;
            }
            reduce[offset + id_sg_local] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_idx >= N)
            return;
        //2.  GEMMB
        __global float *C_ptr = C + n_idx;
        __local float* reduce_ptr = reduce + slm_offset;
        float sum = 0;
        __attribute__((opencl_unroll_hint))
        for (int kk = 0; kk < RANK; kk += SG_SZ) {
            float scale = as_float(intel_sub_group_block_read((const __global uint*)(alpha + kk)));
            float input = as_float(intel_sub_group_block_read((const __local uint*)(reduce_ptr + kk)));

            input *= scale;
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SG_SZ; j++) {
                float bb = as_float(intel_sub_group_block_read((const __global uint*)B_ptr));
                float aa = as_float(sub_group_broadcast(as_uint(input), j));
                sum = fma(aa, bb, sum);
                B_ptr += B_stride;
            }
        }
        __global float *main_input_ptr = main_input + n_idx;
        float m_input = as_float(intel_sub_group_block_read((const __global uint*)main_input_ptr));
        intel_sub_group_block_write((const __global uint*)C_ptr, as_int(sum+m_input));
    }
    '''


# GEMMA has a big K but small N(rank*2), GEMMA kernel  would divide K by WGs and K diemsnion SGs.
# GEMMB only divide N by WGs and SGs because N is big and K(Rank) is small.
# output_state: MLP intermediate_dim
# gemma_sg_BK: the Number of K accumuated in one sg for GEMMA.
# gemma_sgK: the number of sg in K dimension for GEMMA.
# gemmb_sgN: the number of sg in N dimension for GEMMB
class MLP_LORA_2ND:
    def __init__(self, rank, input_state, output_state, gemma_sgN, gemma_sg_BK, gemma_sgK, gemmb_sgN, use_ref = False):
        self.rank = rank
        self.input_state = input_state
        self.output_state = output_state
        self.fused_out_state = output_state*2

        self.sg_sz = 8
        self.gemmb_wg_sz = gemmb_sgN * self.sg_sz
        self.use_ref = use_ref
        mutlwgs_N = True if (get_max_wg_sz() <  rank*2) else False
        self.gemma_wgs_k  = DIV_UP(input_state, gemma_sg_BK *gemma_sgK)
        self.gemma_sgK = gemma_sgK
        assert gemmb_sgN <=1024//self.sg_sz, f'gemmb_sgN:{gemmb_sgN} bigger than {1024//self.sg_sz} limitation'
        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.output_state % self.sg_sz == 0, f"'output_state' {self.output_state} is not multiple of SG_SZ {self.sg_sz}"
        assert self.rank % self.sg_sz == 0, f"'RANK' {self.rank} is not multiple of SG_SZ {self.sg_sz}"
        assert gemma_sg_BK % self.sg_sz == 0, f"'gemma_sg_BK' { gemma_sg_BK} is not multiple of SG_SZ {self.sg_sz}"
        assert self.gemmb_wg_sz % self.sg_sz == 0, f"'gemmb_wg_sz' {self.gemmb_wg_sz} is not multiple of SG_SZ {self.sg_sz}"
        # gemma_wg_BK only used for refrence kernel
        self.gemma_wg_BK = gemma_sg_BK * gemma_sgK

        options = f'-DSG_SZ={self.sg_sz}  -DGEMMA_SGK={gemma_sgK} -DRANK={rank} -DGEMMA_SG_BK={gemma_sg_BK} -DGEMMB_PART_NUM={self.gemma_wgs_k } -DN0={self.output_state}\
                    -DsgN={gemma_sgN}'
        if use_ref:
            self.cl_kernels_ref = kernel_cache(mlp_lora_ref_kernel, options=f"-DN_0={self.output_state}")
        else:
            self.cl_kernels_opt = kernel_cache(mlp_lora_opt, options)
        self.gemma_lws = [1 , gemma_sgK, self.rank if mutlwgs_N  else self.rank*2]
        self.gemma_gws = [self.gemma_wgs_k , gemma_sgK, self.rank*2]
        self.gemmb_lws = [self.gemmb_wg_sz]
        self.gemmb_gws = [ALIGN_UP(self.fused_out_state, self.gemmb_wg_sz)]
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| BATCH = 1 INPUT_STATE:{input_state}, FUSED_OUT_STATE:{self.fused_out_state}, RANK:{rank}:')
            print(f'| [2ND_GEMMA]: GWS:{self.gemma_gws}, LWS:{self.gemma_lws} SG_BK:{gemma_sg_BK} SGK:{gemma_sgK}')
            print(f'| [2ND_GEMMB]: GWS:{self.gemmb_gws}, LWS:{self.gemmb_lws} SGN:{gemmb_sgN}')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')

    def __call__(self, mainInput, loraInput, stateA0, stateA1, stateA, stateAlpha0, stateAlpha1, stateAlpha, stateB0, stateB1, Aoutput, result):

        if self.use_ref:
            tA_output_ref = cl.tensor([1, self.rank*2], np.dtype(np.float32))
            self.cl_kernels_ref.enqueue("gemmA", [1, self.rank*2],[1, self.rank], loraInput, stateA,
                                        tA_output_ref, self.rank*2, self.input_state, self.gemma_wg_BK, 1)
            self.cl_kernels_ref.enqueue("gemmB", [1, self.fused_out_state],[1, min(self.fused_out_state, 1024)],
                                        mainInput, tA_output_ref, stateB0, stateB1, result, stateAlpha, self.rank)

        else:
            cl_kernels = self.cl_kernels_opt

            cl_kernels.enqueue("gemmA", self.gemma_gws, self.gemma_lws,
                                        loraInput, stateA0, stateA1, Aoutput, self.input_state)
            cl_kernels.enqueue("gemmB", self.gemmb_gws, self.gemmb_lws,
                                        mainInput, Aoutput, stateB0, stateB1, result, stateAlpha0, stateAlpha1, self.fused_out_state)
        return result

def mlp_blocking_2nd(rank, input_state, output_state):
    SG_SZ = 8
    lora_cnt = 2
    fused_output = output_state*2
    MAX_WG_SZ =  get_max_wg_sz()
    gemma_sg_BK = 32
    if MAX_WG_SZ < rank*lora_cnt:
        gemma_sgN = rank // SG_SZ
        gemma_sgK = MAX_WG_SZ//(rank)
    else:
        gemma_sgN = (rank*lora_cnt) // SG_SZ
        gemma_sgK = MAX_WG_SZ//(rank*lora_cnt)
    gemmb_sgN = min(fused_output, MAX_WG_SZ)//SG_SZ
    return [gemma_sgN, gemma_sg_BK, gemma_sgK, gemmb_sgN]

def test_mlp_lora_2nd(input_state, rank, output_state, gemma_sgN, gemma_sgK = 8, gemma_sg_BK = 32, gemmb_sgN = 16, check_acc = False):
    cl.profiling(True)
    SG_SZ = 8
    vRANGE = 1
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 200
    # np.random.seed(0)

    # for GEMMA, K decides how many WGs are needed.
    gemma_wgs_k  = DIV_UP(input_state, gemma_sg_BK *gemma_sgK)
    stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank*2]).astype(np.float32)
    alpha = np.random.rand(2, rank).astype(np.float32)
    stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]

    alpha0 = alpha[0:1, :].flatten().reshape(1, rank)
    alpha1 = alpha[1:2, :].flatten().reshape(1, rank)

    stateA_0 = stateA[:, 0:rank].flatten().reshape(input_state, rank)
    stateA_1 = stateA[:, rank:2*rank].flatten().reshape(input_state, rank)

    stateB0 = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float32)
    stateB1 = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float32)

    loraInput = np.random.randint(-vRANGE, vRANGE+1, [1, input_state]).astype(np.float32)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [1, output_state*2]).astype(np.float32)
    Aoutput = np.zeros([gemma_wgs_k , rank*2]).astype(np.float32)

    stateA0_list = [cl.tensor(stateA_0)for _ in range(REPEAT)]
    stateA1_list = [cl.tensor(stateA_1)for _ in range(REPEAT)]

    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    alpha0_list = [cl.tensor(alpha0) for _ in range(REPEAT)]
    alpha1_list = [cl.tensor(alpha1) for _ in range(REPEAT)]

    stateB0_list = [cl.tensor(stateB0)for _ in range(REPEAT)]
    stateB1_list = [cl.tensor(stateB1)for _ in range(REPEAT)]

    loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA.
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([1, output_state*2], np.dtype(np.float32))for _ in range(REPEAT)]
    ref_list = [cl.tensor([1, output_state*2], np.dtype(np.float32))for _ in range(REPEAT)]
    ref = MLP_LORA_2ND(rank, input_state, output_state, gemma_sgN, gemma_sg_BK, gemma_sgK, gemmb_sgN, True)
    opt = MLP_LORA_2ND(rank, input_state, output_state, gemma_sgN, gemma_sg_BK, gemma_sgK, gemmb_sgN, False)

    if check_acc:
        opt(mainInput_list[0], loraInput_list[0], stateA0_list[0], stateA1_list[0], None, alpha0_list[0], alpha1_list[0], None,
                stateB0_list[0],stateB1_list[0], A_output_list[0], res_list[0])
        ref(mainInput_list[0], loraInput_list[0], None, None, stateA_list[0], None, None, alpha_list[0],
                stateB0_list[0],stateB1_list[0], None, ref_list[0])
        cl.finish()
        compare(ref_list[0].numpy(), res_list[0].numpy())
        print(f'INPUT_STATE:{input_state}, RANK:{rank}, OUTPUT_STATE:{output_state}ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[0], loraInput_list[0], stateA0_list[0], stateA1_list[0], None, alpha0_list[0], alpha1_list[0], None,
                stateB0_list[0],stateB1_list[0], A_output_list[0], res_list[0])
        profiling_data  = cl.finish()
        # FMA
        flops_a = rank*2*input_state*2
        # FMA + multiply alpha + add main
        flops_b = rank*output_state*2*2 + rank*2 +output_state*2
        # lora input + weightA
        rd_bytes_a = (rank*2*input_state+input_state)*2
        # B output + main input + weightB  + scale + A output
        rd_bytes_b = (rank*output_state*2+output_state*2*2+rank*2)*2

        for i in range(1, REPEAT):
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2 + 1]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
    [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
    [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')


"""
------------------------------------------------------------------------------------------------------------------
mlp lora 1st token
------------------------------------------------------------------------------------------------------------------
"""
def mlp_generate_store_C(regM, regN, withscale, withSum):
    src = ""
    if withscale:
        # read all needed scale.
        src += r'''
        ''' + "\n\t".join([f'float alpha_{n} = as_float(intel_sub_group_block_read((const __global uint*)(alpha_ptr + SG_SZ * {n})));' for n in range(regN)]) + r'''
        '''
    elif withSum:
        src += r'''
        __global float *main_ptr = mainInput + m_idx * N + n_idx;
        ''' + "\n\t".join([f'float main_N{n} = 0;' for n in range(regN)]) + r'''
        '''

    for m in range(regM):
        # read out `regN` mainInput
        if withSum:
            src += "\n\t".join([f'main_N{n} = as_float(intel_sub_group_block_read((const __global uint*)(main_ptr + SG_SZ * {n})));' for n in range(regN)]) + r'''
                    '''
        for n in range(regN):
            if withscale:
                src +=  f"\n\tintel_sub_group_block_write((__global uint*)(ptrC + SG_SZ * {n}), as_int(sum{m}_{n}*alpha_{n}));"
            elif withSum:
                src +=  f"\n\tintel_sub_group_block_write((__global uint*)(ptrC + SG_SZ * {n}), as_int(sum{m}_{n}+main_N{n}));"
            else:
                src +=  f"\n\tintel_sub_group_block_write((__global uint*)(ptrC + SG_SZ * {n}), as_int(sum{m}_{n}));"
        if withSum:
            src += f"\n\tmain_ptr+= N;"
        src += f"\n\tptrC += N;\n\n"

    return src

def mlp_generate_A_B(withscale, withSum):
    src = ''
    if withscale:
        src = r'''
        int rank = N / 2;
        int strideA=K;
        int strideB=rank;
        __global float *ptrA = A + m_idx * strideA;
        __global float *ptrB = B0 + n_idx;
        __global float *alpha_ptr = alpha0 + n_idx;
        if (n_idx >= rank) {
            ptrB = B1 + n_idx-rank;
            alpha_ptr = alpha1 + n_idx - rank;
        }
        '''
    elif withSum:
        src = r'''
        //rank*2
        int strideA=K*2;
        int strideB=N0;
        __global float *ptrA = A + m_idx * strideA;
        __global float *ptrB = B0 + n_idx;
        if (n_idx >= N0) {
            // K projection
            ptrB = B1 + n_idx - N0;
            strideB = N0;
            ptrA += K;
        }
        '''
    return src

# This function would generate 1st token kernel based on 'regM, regN, withscale, withSum'.
# regM, regN would be the register  blocking.
# withscale = true for LORA GEMMA kernel to genereate multiply post ops
# withsum =  true for LORA GEMMB kernel to genereate add post ops
def mlp_generate_gemm_src(regM, regN, withscale = False, withSum=False):
    assert (withscale and withSum) == False, f"not support both LORA alpha and sum in one gemm kernel"
    if withscale:
        func = f'gemmA_rM{regM}_rN{regN}'
    elif withSum:
        func = f'gemmB_rM{regM}_rN{regN}'
    else:
        func = f'gemm_rM{regM}_rN{regN}'

    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void
    ''' + f'{func}' + r'''(__global float * A, __global float *B0, __global float *B1, __global float *C, __global float *alpha0, __global float *alpha1,   __global float *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgN = get_local_size(1) / SG_SZ;
        int sgM = get_local_size(0);
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''
        int BM = regM * sgM;
        int BN = get_local_size(1) * regN;
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;
        if (m_idx >= M || n_idx >=N )
            return;
        if (m_idx + regM > M)
            m_idx = M - regM;
        if (n_idx + regN * SG_SZ > N)
            n_idx = N - regN * SG_SZ;

        __global float *ptrC = C + m_idx * N + n_idx;
        ''' + mlp_generate_A_B(withscale, withSum) + r'''

        '''  + "\n\t ".join([f"float sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int i = 0; i < K; i += SG_SZ) {

                '''  + "\n\t\t ".join([f"uint input{m} = intel_sub_group_block_read((const __global uint*)(ptrA + {m} * strideA));" for m in range(regM)]) + r'''

                //__attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {

                     '''  + "\n\t\t\t ".join([f"float bb{n} = as_float(intel_sub_group_block_read((const __global uint*)(ptrB + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                     '''  + "\n\t\t\t ".join([f"float aa{m} = as_float(sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                     ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    ptrB += strideB;
                }
                ptrA +=SG_SZ;
        }

        ''' +  mlp_generate_store_C(regM, regN, withscale, withSum) + r'''
    }
    '''
    return [func, src]

#A_regM, A_regN, A_sgM, A_sgN: GEMMA register blocking in one sg and sg number in M and N dimesnion.
#B_regM, B_regN, B_sgM, B_sgN: GEMMB register blocking in one sg and sg number in M and N dimesnion.
#output_state: MLP intermediate_dim

class MLP_LORA_1ST:
    def __init__(self, batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, use_ref = False):
        self.batch = batch
        self.rank = rank
        self.input_state = input_state
        self.q_state = input_state
        self.output_state = output_state
        self.fused_output_state = output_state*2
        self.sg_sz = 8
        self.lora_cnt=2

        assert batch >= A_regM and batch >=B_regM , f'batch:{batch} is smaller than A_regM/B_regM:{A_regM}/{B_regM}'
        assert rank % self.sg_sz == 0 , f'rank:{rank} is not multiple of SG_SZ:{self.sg_sz}'
        assert output_state % (self.sg_sz*B_regN) == 0 , f'output_state:{output_state} is not multiple of SG_SZ*B_regN:{self.sg_sz*B_regN}'
        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert rank >= A_regN * self.sg_sz, f'rank:{rank} is smaller than :A_regN * SG_SZ {A_regN*self.sg_sz}'
        assert output_state >= B_regN * self.sg_sz, f'kv_state:{output_state} is smaller than :B_regN * SG_SZ {B_regN*self.sg_sz}'
        ##one SG calculation in N dimension can't cross gate and up outpu. For both GEMMA and GEMMB.
        ##the mlp_blocking_1st() would ensure alignment. Here just double check.
        assert (rank%(self.sg_sz*A_regN)) == 0
        assert self.output_state %(B_regN*self.sg_sz) == 0, f'kvstate:{output_state}, B_regN:{B_regN}'

        A_BM = A_regM*A_sgM
        A_BN = A_regN*A_sgN*self.sg_sz
        B_BM = B_regM*B_sgM
        B_BN = B_regN*B_sgN*self.sg_sz

        self.gemma_GWS = [ALIGN_UP(batch, A_BM)//A_regM , ALIGN_UP(rank*self.lora_cnt, A_BN)//(A_regN)]
        self.gemma_LWS = [A_sgM, A_sgN * self.sg_sz]
        assert A_sgM *A_sgN * self.sg_sz <= 1024, f" A_LWS:{self.gemma_LWS} exceed 1024 limitation"

        self.gemmb_GWS = [ALIGN_UP(batch, B_BM)//B_regM , ALIGN_UP(self.fused_output_state, B_BN)//(B_regN)]
        self.gemmb_LWS = [B_sgM, B_sgN *  self.sg_sz]
        assert B_sgM *B_sgN *  self.sg_sz <= 1024, f" B_LWS:{self.gemmb_LWS} exceed 1024 limitation"

        if use_ref:
            self.cl_kernels_ref = kernel_cache(mlp_lora_ref_kernel,  options=f"-DN_0={self.output_state}")
            self.tA_output_ref = cl.tensor([batch, self.rank*3], np.dtype(np.float32))

        else:
            gemma_func, gemma_kernel_src = mlp_generate_gemm_src(A_regM, A_regN,  True, False)
            gemmb_func, gemmb_kernel_src = mlp_generate_gemm_src(B_regM, B_regN,  False, True)
            self.gemma_func = gemma_func
            self.gemmb_func = gemmb_func
            self.kernel_opt_gemma = kernel_cache(gemma_kernel_src, options=f"-DSG_SZ={self.sg_sz}")
            self.kernel_opt_gemmb = kernel_cache(gemmb_kernel_src, options=f"-DSG_SZ={self.sg_sz} -DN0={self.output_state}")
        self.use_ref = use_ref
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| BATCH = {batch} INPUT_STATE:{input_state}, FUSED_OUTPUT_STATE:{self.fused_output_state}, RANK:{rank}:')
            print(f'[1ST_GEMMA] GWS:{self.gemma_GWS}, LWS:{self.gemma_LWS}, M:{batch}/{A_BM}, N:{rank}/{A_BN}, SGM:{A_sgM} SGN:{A_sgN} REGM:{A_regM} REGN:{A_regN}')
            print(f'[1st_GEMMB] GWS:{self.gemmb_GWS}, LWS:{self.gemma_LWS}, M:{batch}/{B_BM}, N:{rank}/{B_BN}, SGM:{B_sgM} SGN:{B_sgN} REGM:{B_regM} REGN:{B_regN}')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')


    def __call__(self, mainInput, loraInput, stateA0, stateA1, stateA, stateAlpha0, stateAlpha1, stateAlpha, stateB0, stateB1, Aoutput, result):
        if self.use_ref:
            self.cl_kernels_ref.enqueue("gemmA", [self.batch, self.rank*self.lora_cnt],[1, self.rank], loraInput, stateA,
                                        self.tA_output_ref, self.rank*self.lora_cnt, self.input_state, 1, self.batch)
            self.cl_kernels_ref.enqueue("gemmB", [self.batch, self.fused_output_state],[1, min(self.fused_output_state, 1024)],
                                        mainInput, self.tA_output_ref, stateB0, stateB1, result, stateAlpha, self.rank)
            return self.tA_output_ref

        else:
            self.kernel_opt_gemma.enqueue(self.gemma_func, self.gemma_GWS, self.gemma_LWS,
                                         loraInput, stateA0, stateA1, Aoutput, stateAlpha0, stateAlpha1, mainInput,self.batch, self.rank*self.lora_cnt, self.input_state)
            self.kernel_opt_gemmb.enqueue(self.gemmb_func, self.gemmb_GWS, self.gemmb_LWS,
                                          Aoutput, stateB0, stateB1, result, stateAlpha0, stateAlpha1, mainInput, self.batch, self.fused_output_state, self.rank)
            return Aoutput


def test_mlp_lora_1st(batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, check_acc = False):
    cl.profiling(True)
    SG_SZ = 8
    vRANGE = 1
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 100
    # np.random.seed(0)
    stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank*2]).astype(np.float32)
    alpha = np.random.rand(2, rank).astype(np.float32)
    stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]

    stateA_0 = stateA[:, 0:rank].flatten().reshape(input_state, rank)
    stateA_1 = stateA[:, rank:2*rank].flatten().reshape(input_state, rank)

    alpha0 = alpha[0:1, :].flatten().reshape(1, rank)
    alpha1 = alpha[1:2, :].flatten().reshape(1, rank)

    stateB0 = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float32)
    stateB1 = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float32)

    loraInput = np.random.randint(-vRANGE, vRANGE+1, [batch, input_state]).astype(np.float32)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [batch, output_state*2]).astype(np.float32)
    Aoutput = np.ones([batch, rank*2]).astype(np.float32)

    stateA0_list = [cl.tensor(stateA_0)for _ in range(REPEAT)]
    stateA1_list = [cl.tensor(stateA_1)for _ in range(REPEAT)]

    alpha0_list = [cl.tensor(alpha0) for _ in range(REPEAT)]
    alpha1_list = [cl.tensor(alpha1) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]


    stateB0_list = [cl.tensor(stateB0)for _ in range(REPEAT)]
    stateB1_list = [cl.tensor(stateB1)for _ in range(REPEAT)]

    loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA.
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([batch, output_state*2], np.dtype(np.float32))for _ in range(REPEAT)]
    ref_result = cl.tensor([batch, output_state*2], np.dtype(np.float32))


    ref = MLP_LORA_1ST(batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN,True)
    opt = MLP_LORA_1ST( batch, rank, input_state, output_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN,False)


    if check_acc:
        tmp_opt=opt(mainInput_list[0], loraInput_list[0], stateA0_list[0], stateA1_list[0],  None, alpha0_list[0], alpha1_list[0], None, stateB0_list[0], stateB1_list[0],  A_output_list[0], res_list[0])
        tmp_ref=ref(mainInput_list[0], loraInput_list[0], None, None, stateA_list[0], None, None, alpha_list[0], stateB0_list[0], stateB1_list[0],  A_output_list[0], ref_result)
        cl.finish()
        # compare(tmp_ref.numpy(), tmp_opt.numpy())
        compare(ref_result.numpy(), res_list[0].numpy())
        print(f'BATCH:{batch} INPUT_STATE:{input_state}, RANK:{rank}, OUTPUT_STATE:{output_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[i], loraInput_list[i], stateA0_list[i], stateA1_list[i],  None, alpha0_list[i], alpha1_list[i], None, stateB0_list[i], stateB1_list[i],  A_output_list[i], res_list[i])
        profiling_data  = cl.finish()
        # FMA + multiply scale
        flops_a = batch*rank*2*input_state*2 + batch*rank*2
        # FMA + add
        fused_output = output_state * 2
        flops_b = batch*rank*fused_output*2 + batch*fused_output
        # loarinput + stateA + scale
        rd_bytes_a = (rank*input_state*2 + input_state*batch + rank*2)*2
        # maininput + Aoutput + stateB
        rd_bytes_b = (rank*fused_output + fused_output*batch + rank*2*batch)*2

        for i in range(1, REPEAT):
            ns_a = profiling_data[i*2]
            ns_b = profiling_data[i*2 + 1]
            if VERBOSE:
                print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
 [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
 [total]: {(ns_a+ns_b)*1e-3:.1f}us")
            else:
                print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')

def mlp_blocking_1st(batch, rank, input_state, outstate):
    max_wg_sz = get_max_wg_sz()
    sg_sz = 8
    max_sg_num = max_wg_sz//sg_sz
    lora_cnt = 2
    assert batch >= 8, f"batch:{batch} too small in 1st token, not support in opt kernel"
# GEMMA:
    if batch >= 1000:
        if rank ==16:
            A_regM, A_regN = [8, 1]
        else:
            A_regM, A_regN = [8, 2]
    else:
        A_regM, A_regN = [8, 1]
    if lora_cnt * rank >  max_wg_sz and A_regN == 1:
        A_regN = 2
    A_sgN = rank*lora_cnt//(sg_sz*A_regN)
    A_sgM = max_sg_num // A_sgN
# GEMMB:
    # check whether kv state can use 2 registers in N dim.
    if outstate % (2*sg_sz) == 0:
        if batch < 256:
            B_regM, B_regN = [8, 2]
            B_sgM = 16
            B_sgN = max_sg_num // B_sgM
        else:
            B_regM, B_regN = [8, 2]
            B_sgM, B_sgN = [8, 4]
    else:
        B_regM, B_regN = [8, 1]
        B_sgM = 8
        B_sgN = max_sg_num // B_sgM
    ##one SG calculation can't cross Gate projection and Up projecion. For both GEMMA and GEMMB.
    assert (rank%(sg_sz*A_regN)) == 0
    assert outstate %(B_regN*sg_sz) == 0, f'outstate:{outstate}, B_regN:{B_regN}'
    return [A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN]


if __name__ == '__main__':
    #2nd acc:
    if 1:
        for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*16, 11*16, 13*16, 15*16, 12*16,17*16):
            for rank in (16, 32, 64, 128, 256):
                for output_state in (1024, 1536, 3840, 8960, input_state*8,):
                    gemma_sgN, gemma_sg_BK, gemma_sgK, gemmb_sgN = mlp_blocking_2nd(rank, input_state, output_state)
                    test_mlp_lora_2nd(input_state, rank, output_state, gemma_sgN, gemma_sgK, gemma_sg_BK, gemmb_sgN, check_acc = True)
    #1st acc:
    if 1:
        for batch in range(2055, 2069):
            for input_state in (7*16, 11*16, 13*16, 15*16, 12*16,17*16, 1024, 1536, 2048, 2560, 3072, 3840, 4096,):
                for rank in (16, 32, 64, 128, 256):
                    for output_state in (7*16, 11*16, 13*16, 15*16, 12*16,17*16, 1024, 1536, 3840, 8960,input_state*8,input_state*16,):
                        A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = mlp_blocking_1st(batch, rank, input_state, output_state)
                        test_mlp_lora_1st(batch, rank, input_state, output_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
                                    B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)
    #2nd perf based on qwen MLP
    if 0:
        rank=64
        input_state=1536
        output_state=8960
        gemma_sgN, gemma_sg_BK, gemma_sgK, gemmb_sgN = mlp_blocking_2nd(rank, input_state,output_state)
        test_mlp_lora_2nd(input_state, rank, output_state, gemma_sgN, gemma_sgK, gemma_sg_BK, gemmb_sgN)
    #2nd perf based on qwen MLP
    if 0:
        for batch in(3192,):
            A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = mlp_blocking_1st(batch, 64, 1536, 8960)
            test_mlp_lora_1st(batch, 64, 1536, 8960, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
                                B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN)


    print("-------------------------")