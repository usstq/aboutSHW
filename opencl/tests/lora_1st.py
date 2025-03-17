
#!/usr/bin/python3
import os
# os.environ['cl_intel_driver_diagnostics'] = "0"  # to enable cl_intel_driver_diagnostics

from clops import cl
import numpy as np
import sys
from clops import compare
from clops.utils import *

# BM = regM*sgM
# BN = regN*sgN*SG_SZ
# GWS = [M//regM, N//(regN)]
# LWS = [sgM, sgN * SG_SZ]
# GWS:[256, 16], LWS:[16, 16], M:1024/64, N:64/64, K:1536/128
def generate_store_C(regM, regN, withscale, withSum):
    src = ""
    if withscale:
        # read all needed scale.
        src += r'''
        __global half *alpha_ptr = alpha + n_idx;
        ''' + "\n\t".join([f'half alpha_{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha_ptr + SG_SZ * {n})));' for n in range(regN)]) + r'''
        '''
    elif withSum:
        src += r'''
        __global half *main_ptr = mainInput + m_idx * N + n_idx;
        ''' + "\n\t".join([f'half main_N{n} = 0;' for n in range(regN)]) + r'''
        '''

    for m in range(regM):
        # read out `regN` mainInput
        if withSum:
            src += "\n\t".join([f'main_N{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * {n})));' for n in range(regN)]) + r'''
                    ''' 
        for n in range(regN):
            if withscale:
                src +=  f"\n\tintel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * {n}), as_short(sum{m}_{n}*alpha_{n}));"
            elif withSum:
                src +=  f"\n\tintel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * {n}), as_short(sum{m}_{n}+main_N{n}));"
            else:
                src +=  f"\n\tintel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * {n}), as_short(sum{m}_{n}));"
        if withSum:
            src += f"\n\tmain_ptr+= N;"
        src += f"\n\tptrC += N;\n\n"


            
    return src

def ALIGN_UP(a, b):
    return ((a + (b -1)) // b *b)

def generate_gemm_src(M, N, K,regM, regN, sgM, sgN, withscale = False, withSum=False):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''
        //__local half lA[BM*BK];
        //__local half lB[BK*BN];
        //__local half* ptrA = lA;
        //__local half* ptrB = lB;
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;
        if (m_idx >= M || n_idx >=N )
            return;
        if (m_idx + regM > M)
            m_idx = M - regM;
        if (n_idx + regN * SG_SZ > N)
            n_idx = N - regN * SG_SZ;
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        //for(int i = 0; i < K; i += BK) {
        for(int i = 0; i < K; i += SG_SZ) {

            // copy B & A into shared local mem
            //for(int j = 0; j < BK; j+=SG_SZ) {
                
                '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * K));" for m in range(regM)]) + r'''

                //__attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {
            
                     '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                     '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                     ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    ptrB += N;
                }
                ptrA +=SG_SZ;
            //}
        }

        ''' +  generate_store_C(regM, regN, withscale, withSum) + r'''
    }
    '''
    return [func, src]

def test_lora_1st(batch, input_state, rank , output_state, A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN):
    gemma_func, gemma_kernel_src = generate_gemm_src(batch, rank, input_state, A_regM, A_regN, A_sgM, A_sgN, True, False)
    gemmb_func, gemmb_kernel_src = generate_gemm_src(batch, output_state, rank, B_regM, B_regN, B_sgM, B_sgN, False, True)

    cl_kernel_sources_ref = r'''
    __kernel void gemmA(__global half * A, __global half *B,  __global half *CC,  __global half * alpha, int N, int K, int k_blk, int M) {
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
        CC[m_idx * N + n_idx] = sum * alpha[n_idx];
    }

    __kernel void gemmB(__global half * main_input, __global half * A, __global half *B,  __global half *CC, int N, int K, int M) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i] , B[i*N+n_idx], sum);
        CC[m_idx * N + n_idx] = sum + main_input[m_idx * N + n_idx];
    }
'''
    # print(src)
    cl.profiling(True)

    # np.random.seed(0)
    vRANGE = 1
    REPEAT = 1
    stateA = np.random.randint(-vRANGE, vRANGE+1, [input_state, rank]).astype(np.float16)
    alpha = np.random.rand(rank).astype(np.float16)
    stateB = np.random.randint(-vRANGE, vRANGE+1, [rank, output_state]).astype(np.float16)
    loraInput = np.random.randint(-vRANGE, vRANGE+1, [batch, input_state]).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [batch, output_state]).astype(np.float16)
    Aoutput = np.zeros([batch, rank]).astype(np.float16) 
    
    stateA_list= [cl.tensor(stateA) for _ in range(REPEAT)]
    alpha_list = [cl.tensor(alpha) for _ in range(REPEAT)]
    stateB_list = [cl.tensor(stateB)for _ in range(REPEAT)]
    loraInput_list = [cl.tensor(loraInput)for _ in range(REPEAT)]
    mainInput_list = [cl.tensor(mainInput)for _ in range(REPEAT)]
    #Must set the output to be zeros to avoid not all the data is updated in GEMMA. 
    A_output_list = [cl.tensor(Aoutput)for _ in range(REPEAT)]
    res_list = [cl.tensor([batch, output_state], np.dtype(np.float16))for _ in range(REPEAT)]

    SG_SZ = 16
    A_BM = A_regM*A_sgM
    A_BN = A_regN*A_sgN*SG_SZ
    
    B_BM = B_regM*B_sgM
    B_BN = B_regN*B_sgN*SG_SZ
    
    assert batch >= A_regM and batch >=B_regM , f'batch:{batch} is smaller than A_BM/B_BM:{A_regM}/{B_regM}'
    # assert M % BM == 0 , f'M:{M} is not multiple of BM:{BM}'
    assert rank % SG_SZ == 0 , f'rank:{rank} is not multiple of A_BN:{SG_SZ}'
    assert output_state % SG_SZ == 0 , f'output_state:{output_state} is not multiple of B_BN:{SG_SZ}'
    assert rank >= A_regN * SG_SZ, f'rank:{rank} is smaller than :A_regN * SG_SZ {A_regN*SG_SZ}'
    assert output_state >= B_regN * SG_SZ, f'output_state:{output_state} is smaller than :B_regN * SG_SZ {B_regN*SG_SZ}'


    ref_res = cl.tensor([batch, output_state], np.dtype(np.float16))
    kernel_ref = kernel_cache(cl_kernel_sources_ref)
    t_Aoutput_ref = cl.tensor(np.zeros([batch, rank]).astype(np.float16)) 

    REF_LOCAL_SIZE = 16
    kernel_ref.enqueue("gemmA", [batch, rank],[1, REF_LOCAL_SIZE], loraInput_list[0], stateA_list[0],
                                t_Aoutput_ref, alpha_list[0], rank, input_state, SG_SZ, batch)
    kernel_ref.enqueue("gemmB", [batch, output_state],[1, REF_LOCAL_SIZE],
                                mainInput_list[0], t_Aoutput_ref, stateB_list[0], ref_res, output_state, rank, batch)
    cl.finish()

    kernel_opt_gemma = kernel_cache(gemma_kernel_src, options=f"-DSG_SZ={SG_SZ}  -DBM={A_BM} -DBN={A_BN} -DsgM={A_sgM} -DsgN={A_sgN}")
    A_GWS = [ALIGN_UP(batch, A_BM)//A_regM , ALIGN_UP(rank, A_BN)//(A_regN)]
    A_LWS = [A_sgM, A_sgN * SG_SZ]
    assert A_sgM *A_sgN * SG_SZ <= 1024, f" A_LWS:{A_LWS} exceed 1024 limitation"

    kernel_opt_gemmb = kernel_cache(gemmb_kernel_src, options=f"-DSG_SZ={SG_SZ}  -DBM={B_BM} -DBN={B_BN} -DsgM={B_sgM} -DsgN={B_sgN}")
    B_GWS = [ALIGN_UP(batch, B_BM)//B_regM , ALIGN_UP(output_state, B_BN)//(B_regN)]
    B_LWS = [B_sgM, B_sgN * SG_SZ]
    assert B_sgM *B_sgN * SG_SZ <= 1024, f" B_LWS:{B_LWS} exceed 1024 limitation"

    for i in range(0, REPEAT):
        kernel_opt_gemma.enqueue(gemma_func, A_GWS, A_LWS, loraInput_list[i], stateA_list[i], A_output_list[i], alpha_list[i], mainInput_list[i], batch, rank, input_state)
        kernel_opt_gemmb.enqueue(gemmb_func, B_GWS, B_LWS, A_output_list[i], stateB_list[i], res_list[i], alpha_list[i], mainInput_list[i], batch, output_state, rank)
    profiling_data = cl.finish()
    flops_a = batch*rank*input_state*2
    flops_b = batch*rank*output_state*2
    rd_bytes_a = (rank*input_state+input_state*batch+rank)*2
    rd_bytes_b = (rank*output_state+output_state*batch+rank*batch)*2
    print("----------------------------------------------------")
    for i in range(0, REPEAT):
        ns_a = profiling_data[i*2]
        ns_b = profiling_data[i*2 + 1]
        print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
        [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
        [total]: {(ns_a+ns_b)*1e-3:.1f}us")
    print("----------------------------------------------------")
    print(f'[GEMMA] GWS:{A_GWS}, LWS:{A_LWS}, M:{batch}/{A_BM}, N:{rank}/{A_BN}')
    print(f'[GEMMB] GWS:{B_GWS}, LWS:{B_LWS}, M:{batch}/{B_BM}, N:{rank}/{B_BN}')
    compare(ref_res.numpy(), res_list[0].numpy())
    print("ACC OK!!!")
    print("----------------------------------------------------")



# GEMMA will not divide N across WGs. 1. choose[regM, regN] based on rank and m, 2, choose[sgN], 3 sgM = 16//sgN or sgM = 8//sgN
# seems regM and regN influces FPS more than sgM, sgN. 
# If rank is 128, 192, and M >= 2048, can try[16, 2]
# GEMMA rank == 64, use regM, regN for  [8,2] , sgM = 16, sgN = 2, when M >= 2048. 
# other wise,  choose regM = 4, regN = 1, sgM = ?, sgN = 1/2/8.

# GEMMB: M < 256 regM=8, regN=2, sgM=16, sgN=4, small M() fits into
#        M >= 256, regM=16, regN=2, sgM=8, sgN=4

for batch in range(8, 1024):
    for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*16, 11*16, 13*16, 15*16, 12*16,17*16):
        for output_state in range(256//16, 1024//16):
            for rank in (16, 32 ,64, 128):
                test_lora_1st(batch, input_state, rank , output_state*16, A_regM = 8, A_regN = 1, A_sgM=16, A_sgN = 2, B_regM=8, B_regN=1, B_sgM=8, B_sgN=4)    