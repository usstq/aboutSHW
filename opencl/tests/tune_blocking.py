
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

def ALIGN_DOWN(a, b):
    return (a // b * b)

cl_kernel_sources_ref = r'''
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
        //CC[m_idx * N + n_idx] = sum * alpha[n_idx];
        CC[m_idx * N + n_idx] = sum;
    }
    __kernel void reduce(__global half * temp_C, __global half *C, int N, int cnt) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[m_idx*cnt*N + i*N + n_idx];
        C[m_idx * N + n_idx] = sum;
    }
    
    __kernel void gemmB(__global half * main_input, __global half * A, __global half *B,  __global half *CC, __global half * alpha, int N, int K, int M) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i] * alpha[i], B[i*N+n_idx], sum);
        //CC[m_idx * N + n_idx] = sum + main_input[m_idx * N + n_idx];
        CC[m_idx * N + n_idx] = sum + main_input[m_idx * N + n_idx];
    }
'''

# BM = regM*sgM
# BN = regN*sgN*SG_SZ
# GWS = [M//regM, N//(regN)]
# LWS = [sgM, sgN * SG_SZ]
# GWS:[256, 16], LWS:[16, 16], M:1024/64, N:64/64, K:1536/128
def gen_store_C(regM, regN, withscale, withSum):
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

def test_FMA_blocking(M, N, K,regM, regN, sgM, sgN, withscale = False, withSum=False):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    gen_src =  r'''
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

        ''' +  gen_store_C(regM, regN, withscale, withSum) + r'''
    }
    '''
    # print(gen_src)
    cl.profiling(True)

    # np.random.seed(0)
    vRANGE = 1
    REPEAT = 100
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    alpha = np.random.rand(N).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)


    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tAlpha_list =  [cl.tensor(alpha) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]
    tMaininput_list = [cl.tensor(mainInput) for _ in range(REPEAT)]

    SG_SZ = 16
    BK= 64
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    
    assert M >= regM , f'M:{M} is smaller than regM:{regM}'
    assert N >= regN*SG_SZ , f'N:{N} is smaller than :regN*SG_SZ {regN*SG_SZ}'
    assert N % SG_SZ == 0 , f'N:{N} is not multiple of SG_SZ:{SG_SZ}'

    # assert BK % SG_SZ == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'
    # assert K % BK == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'

    kernel = kernel_cache(gen_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    GWS = [ALIGN_UP(M, BM)//regM , ALIGN_UP(N, BN)//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"

    for i in range(0, REPEAT):
        kernel.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time_opt in ns:
        print(f'TPUT: [W/O SLM]:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    if 0: 
        if withscale:
            C_ref = np.multiply(np.matmul(A, B), alpha)
        elif withSum:
            C_ref = np.add(np.matmul(A, B), mainInput)
        else:
            C_ref = np.matmul(A, B)
        compare(C_ref, tC_list[0].numpy())


def test_SLM_FMA(M, N, K,regM, regN, sgM, sgN, withscale = False, withSum=False):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    gen_slm_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''
    
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;
        // Some hack when rows needed to handle in `sg` is not regM.
        //Less than regM rows to handle in this sg. Move up to padding. Duplicated FMA and storing with same result between  m_sgs.
        if (m_idx < M && (m_idx + regM) > M)
            m_idx = M - regM;
        // Some hack when columns  needed to handle in `sg` is not regN*SG_SZ.
        //Less than regN*SG_SZ columns to handle in this sg. Move left to padding. Duplicated FMA and storing with same result between  n_sgs.
        if (n_idx < N && (n_idx + regN * SG_SZ) > N)
            n_idx = N - regN * SG_SZ;
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;
        
        __local half lA[BM*BK];
        __local half lB[BK*BN];

        __local half* lA_ptr_dest = lA + sgid_M * regM *BK;
        __local half* lB_ptr_dest = lB + sgid_N * regN * SG_SZ;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int j = 0; j < K; j += BK) {
            int k_len = (j+BK) > K ? (K - j) : BK;
            //copy A to SLM.
            if (m_idx < M) {
                __attribute__((opencl_unroll_hint))
                for (int offset_K = sgid_N * SG_SZ; offset_K < k_len; offset_K += sgN * SG_SZ) {
                    ''' + "\n\t\t ".join([f"ushort tmpA_{m} = intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * K + offset_K));" for m in range(regM)]) + r'''
                    ''' + "\n\t\t ".join([f"intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + {m} * BK + offset_K), tmpA_{m});" for m in range(regM)])+ r'''                    
                }
            }
           if (n_idx < N) {
                //copy B to SLM.
                __attribute__((opencl_unroll_hint))
                for (int offset_K = sgid_M; offset_K < k_len; offset_K += sgM) {
                    ''' + "\n\t\t ".join([f"ushort tmpB_{n} = intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * SG_SZ + offset_K*N));" for n in range(regN)]) + r'''
                    ''' + "\n\t\t ".join([f"intel_sub_group_block_write_us((const __local ushort*)(lB_ptr_dest + {n} * SG_SZ + offset_K*BN), tmpB_{n});" for n in range(regN)])+ r'''  
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            __local half* lA_ptr = lA_ptr_dest;
            __local half* lB_ptr = lB_ptr_dest;

            if (m_idx < M && n_idx < N) {
                // FMA Matmul([BM, BK], [BK, BN]) = [BM, BN]
                for(int i = 0; i < k_len; i+=SG_SZ) {
                    
                    '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + {m} * BK));" for m in range(regM)]) + r'''

                    //__attribute__((opencl_unroll_hint))
                    for (int kk = 0; kk < SG_SZ; kk++) {
                
                        '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __local ushort*)(lB_ptr + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                        '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                        ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                        lB_ptr += BN;
                    }
                    lA_ptr +=SG_SZ;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            ptrA += BK;
            ptrB += BK * N;
        }
        if (m_idx >= M || n_idx >= N)
            return;
        ''' +  gen_store_C(regM, regN, withscale, withSum) + r'''
    }
    '''
    print(gen_slm_src)
    cl.profiling(True)

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 100
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    alpha = np.random.rand(N).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)


    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tAlpha_list =  [cl.tensor(alpha) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]

    tMaininput_list = [cl.tensor(mainInput) for _ in range(REPEAT)]

    SG_SZ = 16
    SLM_SZ = 64*1024
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    BK = min(K, ALIGN_DOWN(SLM_SZ // ((BM + BN)*2), SG_SZ))
    BK = 80
        
    assert M >= regM , f'M:{M} is smaller than regM:{regM}'
    assert N >= regN*SG_SZ , f'N:{N} is smaller than :regN*SG_SZ {regN*SG_SZ}'
    assert N % SG_SZ == 0 , f'N:{N} is not multiple of SG_SZ:{SG_SZ}'

    # assert BK % SG_SZ == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'
    # assert K % BK == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'

    kernel_slm = kernel_cache(gen_slm_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    
    GWS = [ALIGN_UP(M, BM)//regM , ALIGN_UP(N, BN)//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        kernel_slm.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for i in range(0, REPEAT):
        time_slm = ns[i]
        print(f'TPUT: [SLM]:{flops/time_slm:.1f} GFLOPS, us: {time_slm*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    if 0:
        if withscale:
            C_ref = np.multiply(np.matmul(A, B), alpha)
        elif withSum:
            C_ref = np.add(np.matmul(A, B), mainInput)
        else:
            C_ref = np.matmul(A, B)
        compare(C_ref, tC_slm_list[0].numpy())

# GEMMA will not divide N across WGs. 1. choose[regM, regN] based on rank and m, 2, choose[sgN], 3 sgM = 16//sgN or sgM = 8//sgN
# seems regM and regN influces FPS more than sgM, sgN. 
# If rank is 128, 192, and M >= 2048, can try[16, 2]
# GEMMA rank == 64, use regM, regN for  [8,2] , sgM = 16, sgN = 2, when M >= 2048. 
# other wise,  choose regM = 4, regN = 1, sgM = ?, sgN = 1/2/8.

# GEMMB: M < 256 regM=8, regN=2, sgM=16, sgN=4, small M() fits into
#        M >= 256, regM=16, regN=2, sgM=8, sgN=4


# test acc of M/:
# for batch in range(8, 256, 3):
#     for n_idx in range(2, 32):
#         test_SLM_FMA(batch, N=n_idx*16, K=64, regM=8, regN=2, sgM=16, sgN=4)
#         test_SLM_FMA(batch, N=n_idx*16, K=64, regM=8, regN=2, sgM=16, sgN=4, withscale=True)
#         test_SLM_FMA(batch, N=n_idx*16, K=64, regM=8, regN=2, sgM=16, sgN=4, withscale=False, withSum=True)


# for batch in range(1024, 2048, 57):
#     test_SLM_FMA(batch, N=64, K=1536, regM=16, regN=2, sgM=8, sgN=2)
#     test_SLM_FMA(batch, N=64, K=1536, regM=16, regN=2, sgM=8, sgN=2, withscale=True)
#     test_SLM_FMA(batch, N=64, K=1536, regM=16, regN=2, sgM=8, sgN=2, withscale=False, withSum=True)

# for batch in range(256, 512, 17):
#     test_SLM_FMA(batch, N=1536, K=64, regM=4, regN=1, sgM=16, sgN=4)
#     test_SLM_FMA(batch, N=1536, K=64, regM=4, regN=1, sgM=16, sgN=4, withscale=True)
#     test_SLM_FMA(batch, N=1536, K=64, regM=4, regN=1, sgM=16, sgN=4,  withscale=False, withSum=True)

# test perf:
# for k in (64,):
#      for m in (3072,):
#          for n in (512,1536,3840,):
#              test_SLM_FMA(m, n, k, regM=16, regN=2, sgM=8, sgN=4)

test_SLM_FMA(2048, N=2048, K=1536, regM=8, regN=2, sgM=16, sgN=2)
test_FMA_blocking(2048, N=2048, K=1536, regM=8, regN=2, sgM=16, sgN=2)

# for batch in range(8, 256):
#     for n_idx in range(2, 32):
#         test_SLM_FMA(batch, N=n_idx*16, K=256, regM=4, regN=2, sgM=8, sgN=4)

# test_SLM_FMA(65, 272, K=256, regM=4, regN=2, sgM=8, sgN=4)