
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

        ''' +  gen_store_C(regM, regN, withscale, withSum) + r'''
    }
    '''
    

    # print(src)
    cl.profiling(True)

    # np.random.seed(0)
    vRANGE = 1
    REPEAT = 100
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    alpha = np.random.rand(N).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)
    if withscale:
        C_ref = np.multiply(np.matmul(A, B), alpha)
    elif withSum:
        C_ref = np.add(np.matmul(A, B), mainInput)
    else:
        C_ref = np.matmul(A, B)

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

    kernel = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    GWS = [ALIGN_UP(M, BM)//regM , ALIGN_UP(N, BN)//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"

    for i in range(0, REPEAT):
        kernel.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time in ns:
        print(f'TPUT: {flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    compare(C_ref, tC_list[0].numpy())


def test_SLM_FMA(M, N, K,regM, regN, sgM, sgN, withscale = False, withSum=False):
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


    src = r'''
        __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        int regM = 4;
        int regN = 2;

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

         half sum0_0 = 0;
         half sum0_1 = 0;
         half sum1_0 = 0;
         half sum1_1 = 0;
         half sum2_0 = 0;
         half sum2_1 = 0;
         half sum3_0 = 0;
         half sum3_1 = 0;;

        for(int j = 0; j < K; j += BK) {
            //copy A to SLM.
            if (m_idx < M) {
                __attribute__((opencl_unroll_hint))
                for (int offset_K = sgid_N * SG_SZ; offset_K < BK; offset_K += sgN * SG_SZ) {
                    ushort tmpA_0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0 * K + offset_K));
                    ushort tmpA_1 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 1 * K + offset_K));
                    ushort tmpA_2 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 2 * K + offset_K));
                    ushort tmpA_3 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 3 * K + offset_K));
                    
                    intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + 0 * BK + offset_K), tmpA_0);
                    intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + 1 * BK + offset_K), tmpA_1);
                    intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + 2 * BK + offset_K), tmpA_2);
                    intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + 3 * BK + offset_K), tmpA_3);
                }
            }
            if (n_idx < N) {
                //copy B to SLM.
                __attribute__((opencl_unroll_hint))
                for (int offset_K = sgid_M; offset_K < BK; offset_K += sgM) {
                    ushort tmpB_0 = intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SG_SZ + offset_K*N));
                    ushort tmpB_1 = intel_sub_group_block_read_us((const __global ushort*)(ptrB + 1 * SG_SZ + offset_K*N));
                    intel_sub_group_block_write_us((const __local ushort*)(lB_ptr_dest + 0 * SG_SZ + offset_K*BN), tmpB_0);
                    intel_sub_group_block_write_us((const __local ushort*)(lB_ptr_dest + 1 * SG_SZ + offset_K*BN), tmpB_1);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            __local half* lA_ptr = lA_ptr_dest;
            __local half* lB_ptr = lB_ptr_dest;
            if (m_idx < M && n_idx < N) {
                // FMA Matmul([BM, BK], [BK, BN]) = [BM, BN]
                for(int i = 0; i < BK; i+=SG_SZ) {
                    ushort input0 = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + 0 * BK));
                    ushort input1 = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + 1 * BK));
                    ushort input2 = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + 2 * BK));
                    ushort input3 = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + 3 * BK));

                    //__attribute__((opencl_unroll_hint))
                    for (int kk = 0; kk < SG_SZ; kk++) {
                
                        half bb0 = as_half(intel_sub_group_block_read_us((const __local ushort*)(lB_ptr + 0 * SG_SZ)));
                            half bb1 = as_half(intel_sub_group_block_read_us((const __local ushort*)(lB_ptr + 1 * SG_SZ)));
                        half aa0 = as_half(intel_sub_group_broadcast(input0, kk));
                            half aa1 = as_half(intel_sub_group_broadcast(input1, kk));
                            half aa2 = as_half(intel_sub_group_broadcast(input2, kk));
                            half aa3 = as_half(intel_sub_group_broadcast(input3, kk));
                        sum0_0 = fma(aa0, bb0, sum0_0);
                            sum0_1 = fma(aa0, bb1, sum0_1);
                            sum1_0 = fma(aa1, bb0, sum1_0);
                            sum1_1 = fma(aa1, bb1, sum1_1);
                            sum2_0 = fma(aa2, bb0, sum2_0);
                            sum2_1 = fma(aa2, bb1, sum2_1);
                            sum3_0 = fma(aa3, bb0, sum3_0);
                            sum3_1 = fma(aa3, bb1, sum3_1);
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
        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum0_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum0_1));
        ptrC += N;

        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum1_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum1_1));
        ptrC += N;

        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum2_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum2_1));
        ptrC += N;

        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum3_0));
        intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum3_1));
        ptrC += N;
    }
    
    '''
    # print(gen_slm_src)
    cl.profiling(True)

    # np.random.seed(0)
    vRANGE = 1
    REPEAT = 100
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    alpha = np.random.rand(N).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)
    if withscale:
        C_ref = np.multiply(np.matmul(A, B), alpha)
    elif withSum:
        C_ref = np.add(np.matmul(A, B), mainInput)
    else:
        C_ref = np.matmul(A, B)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tAlpha_list =  [cl.tensor(alpha) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]

    tMaininput_list = [cl.tensor(mainInput) for _ in range(REPEAT)]

    SG_SZ = 16
    SLM_SZ = 64*1024
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    MAX_BK = min(K, ALIGN_DOWN(SLM_SZ // ((BM + BN)*2), SG_SZ))
    
    BK =64
    
    assert M >= regM , f'M:{M} is smaller than regM:{regM}'
    assert N >= regN*SG_SZ , f'N:{N} is smaller than :regN*SG_SZ {regN*SG_SZ}'
    assert N % SG_SZ == 0 , f'N:{N} is not multiple of SG_SZ:{SG_SZ}'

    # assert BK % SG_SZ == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'
    # assert K % BK == 0 , f'BK:{BK} is not multiple of SG_SZ:{SG_SZ}'

    kernel_opt = kernel_cache(gen_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    kernel_slm = kernel_cache(gen_slm_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")
    
    GWS = [ALIGN_UP(M, BM)//regM , ALIGN_UP(N, BN)//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"

    for i in range(0, REPEAT):
        kernel_opt.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
        kernel_slm.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], tAlpha_list[i], tMaininput_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for i in range(0, REPEAT):
        time_opt = ns[2*i]
        time_slm = ns[2*i + 1]
        print(f'TPUT: [OPT]:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}, [SLM]:{flops/time_slm:.1f} GFLOPS, us: {time_slm*1e-3:.1f}, [GAIN]:{time_opt/time_slm:.2f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    compare(C_ref, tC_slm_list[0].numpy())
    compare(C_ref, tC_list[0].numpy())

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

test_SLM_FMA(3051, N=64, K=1536, regM=8, regN=2, sgM=32, sgN=2)

# for batch in range(8, 256):
#     for n_idx in range(2, 32):
#         test_SLM_FMA(batch, N=n_idx*16, K=256, regM=4, regN=2, sgM=8, sgN=4)

# test_SLM_FMA(65, 272, K=256, regM=4, regN=2, sgM=8, sgN=4)