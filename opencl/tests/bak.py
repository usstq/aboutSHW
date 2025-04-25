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

"""
------------------------------------------------------------------------------------------------------------------
opt lora 1st token
------------------------------------------------------------------------------------------------------------------
"""

lora_qkv_ref_kernel =  r'''
    __kernel void reduceA(__global half * temp_C, __global half *C, int N, int cnt) {
        int n_idx = get_global_id(0);
        half sum = 0.f;
        for (int i = 0; i < cnt; i++)
            sum += temp_C[i*N + n_idx];
        C[n_idx] = sum;
    }

        __kernel void gemmA(__global half * A, __global half *B,  __global half *CC,  __global half *alpha, int N, int K, int k_blk, int M) {
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
        CC[m_idx * N + n_idx] = sum*alpha[n_idx];
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
            //sum = fma(A_ptr[k]*scale_ptr[k], B_ptr[k*stride_B], sum);
            sum = fma(A_ptr[k], B_ptr[k*stride_B], sum);

        CC[m_idx *stride_C + n_idx] = sum + main_input[m_idx *stride_C + n_idx];
    }
    '''

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
# This function would generate 1st token kernel based on 'regM, regN, withscale, withSum'.
# regM, regN would be the register  blocking.
# withscale = true for LORA GEMMA kernel to genereate multiply post ops
# withsum =  true for LORA GEMMB kernel to genereate add post ops
def generate_gemm_src(regM, regN, withscale = False, withSum=False):
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
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K) {
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
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int i = 0; i < K; i += SG_SZ) {
                
                '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * K));" for m in range(regM)]) + r'''

                //__attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {
            
                     '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                     '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                     ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    ptrB += N;
                }
                ptrA +=SG_SZ;
        }

        ''' +  generate_store_C(regM, regN, withscale, withSum) + r'''
    }
    '''
    return [func, src]

gemma_kernel_src=r'''
   __kernel void  gemmA_rM8_rN2(__global half * A, __global half *B0, __global half *B1, __global half *B2,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K, int rank) {
        int sgid = get_sub_group_id();
        int sgN = get_local_size(1) / SG_SZ;
        int sgM = get_local_size(0);
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        int regM = 8;
        int regN = 2;
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
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B0 + n_idx;

        if (n_idx >= rank*2) {
            // V projection
            ptrB = B2 + n_idx-rank*2;

        } else if (n_idx >= rank) {
            // K projection
            ptrB = B1 + n_idx-rank;

        }
        __global half *ptrC = C + m_idx * N + n_idx;

        half sum0_0 = 0;
	 half sum0_1 = 0;
	 half sum1_0 = 0;
	 half sum1_1 = 0;
	 half sum2_0 = 0;
	 half sum2_1 = 0;
	 half sum3_0 = 0;
	 half sum3_1 = 0;
	 half sum4_0 = 0;
	 half sum4_1 = 0;
	 half sum5_0 = 0;
	 half sum5_1 = 0;
	 half sum6_0 = 0;
	 half sum6_1 = 0;
	 half sum7_0 = 0;
	 half sum7_1 = 0;;

        for(int i = 0; i < K; i += SG_SZ) {
                
                ushort input0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0 * K));
		 ushort input1 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 1 * K));
		 ushort input2 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 2 * K));
		 ushort input3 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 3 * K));
		 ushort input4 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 4 * K));
		 ushort input5 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 5 * K));
		 ushort input6 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 6 * K));
		 ushort input7 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 7 * K));

                //__attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {
            
                     half bb0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SG_SZ)));
			 half bb1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 1 * SG_SZ)));
                     half aa0 = as_half(intel_sub_group_broadcast(input0, kk));
			 half aa1 = as_half(intel_sub_group_broadcast(input1, kk));
			 half aa2 = as_half(intel_sub_group_broadcast(input2, kk));
			 half aa3 = as_half(intel_sub_group_broadcast(input3, kk));
			 half aa4 = as_half(intel_sub_group_broadcast(input4, kk));
			 half aa5 = as_half(intel_sub_group_broadcast(input5, kk));
			 half aa6 = as_half(intel_sub_group_broadcast(input6, kk));
			 half aa7 = as_half(intel_sub_group_broadcast(input7, kk));
                     sum0_0 = fma(aa0, bb0, sum0_0);
			sum0_1 = fma(aa0, bb1, sum0_1);
			sum1_0 = fma(aa1, bb0, sum1_0);
			sum1_1 = fma(aa1, bb1, sum1_1);
			sum2_0 = fma(aa2, bb0, sum2_0);
			sum2_1 = fma(aa2, bb1, sum2_1);
			sum3_0 = fma(aa3, bb0, sum3_0);
			sum3_1 = fma(aa3, bb1, sum3_1);
			sum4_0 = fma(aa4, bb0, sum4_0);
			sum4_1 = fma(aa4, bb1, sum4_1);
			sum5_0 = fma(aa5, bb0, sum5_0);
			sum5_1 = fma(aa5, bb1, sum5_1);
			sum6_0 = fma(aa6, bb0, sum6_0);
			sum6_1 = fma(aa6, bb1, sum6_1);
			sum7_0 = fma(aa7, bb0, sum7_0);
			sum7_1 = fma(aa7, bb1, sum7_1);
                    ptrB += rank;
                }
                ptrA +=SG_SZ;
        }

        
        __global half *alpha_ptr = alpha + n_idx;
        half alpha_0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha_ptr + SG_SZ * 0)));
	half alpha_1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(alpha_ptr + SG_SZ * 1)));
        
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum0_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum0_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum1_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum1_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum2_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum2_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum3_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum3_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum4_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum4_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum5_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum5_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum6_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum6_1*alpha_1));
	ptrC += N;


	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum7_0*alpha_0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum7_1*alpha_1));
	ptrC += N;
    }
'''

gemmb_kernel_src=r'''
  __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void 
    gemmB_rM16_rN2(__global half * A,  __global half *B0, __global half *B1, __global half *B2,  __global half *C, __global half *alpha,  __global half *mainInput, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgN = get_local_size(1) / SG_SZ;
        int sgM = get_local_size(0);
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        int regM = 16;
int regN = 2;
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
        int strideA = K*3;
        __global half *ptrC = C + m_idx * N + n_idx;
        __global half *ptrA = A + m_idx * strideA;
        __global half *ptrB = B0 + n_idx;
        int strideB = N0;

        if (n_idx >= (N0 + N1_2)) {
            // V projection
            ptrB = B2 + n_idx - N0 - N1_2;
            strideB = N1_2;
            ptrA += K*2;

        } else if (n_idx >= N0) {
            // K projection
            ptrB = B1 + n_idx - N0;
            strideB = N1_2;
            ptrA += K;
        }
        
        half sum0_0 = 0;
	 half sum0_1 = 0;
	 half sum1_0 = 0;
	 half sum1_1 = 0;
	 half sum2_0 = 0;
	 half sum2_1 = 0;
	 half sum3_0 = 0;
	 half sum3_1 = 0;
	 half sum4_0 = 0;
	 half sum4_1 = 0;
	 half sum5_0 = 0;
	 half sum5_1 = 0;
	 half sum6_0 = 0;
	 half sum6_1 = 0;
	 half sum7_0 = 0;
	 half sum7_1 = 0;
	 half sum8_0 = 0;
	 half sum8_1 = 0;
	 half sum9_0 = 0;
	 half sum9_1 = 0;
	 half sum10_0 = 0;
	 half sum10_1 = 0;
	 half sum11_0 = 0;
	 half sum11_1 = 0;
	 half sum12_0 = 0;
	 half sum12_1 = 0;
	 half sum13_0 = 0;
	 half sum13_1 = 0;
	 half sum14_0 = 0;
	 half sum14_1 = 0;
	 half sum15_0 = 0;
	 half sum15_1 = 0;;

        for(int i = 0; i < K; i += SG_SZ) {
                
                ushort input0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 0 * K));
		 ushort input1 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 1 * strideA));
		 ushort input2 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 2 * strideA));
		 ushort input3 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 3 * strideA));
		 ushort input4 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 4 * strideA));
		 ushort input5 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 5 * strideA));
		 ushort input6 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 6 * strideA));
		 ushort input7 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 7 * strideA));
		 ushort input8 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 8 * strideA));
		 ushort input9 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 9 * strideA));
		 ushort input10 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 10 * strideA));
		 ushort input11 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 11 * strideA));
		 ushort input12 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 12 * strideA));
		 ushort input13 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 13 * strideA));
		 ushort input14 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 14 * strideA));
		 ushort input15 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + 15 * strideA));

                //__attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {
            
                     half bb0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 0 * SG_SZ)));
			 half bb1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + 1 * SG_SZ)));
                     half aa0 = as_half(intel_sub_group_broadcast(input0, kk));
			 half aa1 = as_half(intel_sub_group_broadcast(input1, kk));
			 half aa2 = as_half(intel_sub_group_broadcast(input2, kk));
			 half aa3 = as_half(intel_sub_group_broadcast(input3, kk));
			 half aa4 = as_half(intel_sub_group_broadcast(input4, kk));
			 half aa5 = as_half(intel_sub_group_broadcast(input5, kk));
			 half aa6 = as_half(intel_sub_group_broadcast(input6, kk));
			 half aa7 = as_half(intel_sub_group_broadcast(input7, kk));
			 half aa8 = as_half(intel_sub_group_broadcast(input8, kk));
			 half aa9 = as_half(intel_sub_group_broadcast(input9, kk));
			 half aa10 = as_half(intel_sub_group_broadcast(input10, kk));
			 half aa11 = as_half(intel_sub_group_broadcast(input11, kk));
			 half aa12 = as_half(intel_sub_group_broadcast(input12, kk));
			 half aa13 = as_half(intel_sub_group_broadcast(input13, kk));
			 half aa14 = as_half(intel_sub_group_broadcast(input14, kk));
			 half aa15 = as_half(intel_sub_group_broadcast(input15, kk));
                     sum0_0 = fma(aa0, bb0, sum0_0);
			sum0_1 = fma(aa0, bb1, sum0_1);
			sum1_0 = fma(aa1, bb0, sum1_0);
			sum1_1 = fma(aa1, bb1, sum1_1);
			sum2_0 = fma(aa2, bb0, sum2_0);
			sum2_1 = fma(aa2, bb1, sum2_1);
			sum3_0 = fma(aa3, bb0, sum3_0);
			sum3_1 = fma(aa3, bb1, sum3_1);
			sum4_0 = fma(aa4, bb0, sum4_0);
			sum4_1 = fma(aa4, bb1, sum4_1);
			sum5_0 = fma(aa5, bb0, sum5_0);
			sum5_1 = fma(aa5, bb1, sum5_1);
			sum6_0 = fma(aa6, bb0, sum6_0);
			sum6_1 = fma(aa6, bb1, sum6_1);
			sum7_0 = fma(aa7, bb0, sum7_0);
			sum7_1 = fma(aa7, bb1, sum7_1);
			sum8_0 = fma(aa8, bb0, sum8_0);
			sum8_1 = fma(aa8, bb1, sum8_1);
			sum9_0 = fma(aa9, bb0, sum9_0);
			sum9_1 = fma(aa9, bb1, sum9_1);
			sum10_0 = fma(aa10, bb0, sum10_0);
			sum10_1 = fma(aa10, bb1, sum10_1);
			sum11_0 = fma(aa11, bb0, sum11_0);
			sum11_1 = fma(aa11, bb1, sum11_1);
			sum12_0 = fma(aa12, bb0, sum12_0);
			sum12_1 = fma(aa12, bb1, sum12_1);
			sum13_0 = fma(aa13, bb0, sum13_0);
			sum13_1 = fma(aa13, bb1, sum13_1);
			sum14_0 = fma(aa14, bb0, sum14_0);
			sum14_1 = fma(aa14, bb1, sum14_1);
			sum15_0 = fma(aa15, bb0, sum15_0);
			sum15_1 = fma(aa15, bb1, sum15_1);
                    ptrB += strideB;
                }
                ptrA +=SG_SZ;
        }

        
        __global half *main_ptr = mainInput + m_idx * N + n_idx;
        half main_N0 = 0;
	half main_N1 = 0;
        main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum0_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum0_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum1_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum1_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum2_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum2_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum3_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum3_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum4_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum4_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum5_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum5_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum6_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum6_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum7_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum7_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum8_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum8_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum9_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum9_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum10_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum10_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum11_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum11_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum12_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum12_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum13_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum13_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum14_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum14_1+main_N1));
	main_ptr+= N;
	ptrC += N;

main_N0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 0)));
	main_N1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(main_ptr + SG_SZ * 1)));
                    
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 0), as_short(sum15_0+main_N0));
	intel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * 1), as_short(sum15_1+main_N1));
	main_ptr+= N;
	ptrC += N;


    }
    
'''

 #A_regM, A_regN, A_sgM, A_sgN: GEMMA register blocking in one sg and sg number in M and N dimesnion.
 #B_regM, B_regN, B_sgM, B_sgN: GEMMB register blocking in one sg and sg number in M and N dimesnion.
class LORA_1ST:
    def __init__(self, batch, rank, input_state, kv_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, use_ref = False):
        self.batch = batch
        self.rank = rank
        self.input_state = input_state
        self.q_state = input_state
        self.kv_state = kv_state
        self.sg_sz = 16
        self.qkv_state = input_state + kv_state*2

        
        assert batch >= A_regM and batch >=B_regM , f'batch:{batch} is smaller than A_regM/B_regM:{A_regM}/{B_regM}'
        assert rank % self.sg_sz == 0 , f'rank:{rank} is not multiple of SG_SZ:{self.sg_sz}'
        assert kv_state % (self.sg_sz*B_regN) == 0 , f'output_state:{kv_state} is not multiple of SG_SZ*B_regN:{self.sg_sz*B_regN}'
        assert self.input_state % self.sg_sz == 0, f"'input state' {self.input_state} is not multiple of SG_SZ {self.sg_sz}"
        assert rank >= A_regN * self.sg_sz, f'rank:{rank} is smaller than :A_regN * SG_SZ {A_regN*self.sg_sz}'
        assert kv_state >= B_regN * self.sg_sz, f'kv_state:{kv_state} is smaller than :B_regN * SG_SZ {B_regN*self.sg_sz}'
        
        
        A_BM = A_regM*A_sgM
        A_BN = A_regN*A_sgN*self.sg_sz
        B_BM = B_regM*B_sgM
        B_BN = B_regN*B_sgN*self.sg_sz
        
        
        # gemma_func, gemma_kernel_src = generate_gemm_src(A_regM, A_regN,  True, False)
        # gemmb_func, gemmb_kernel_src = generate_gemm_src(B_regM, B_regN,  False, True)
        # self.gemma_func = gemma_func
        # self.gemmb_func = gemmb_func

        self.gemma_GWS = [ALIGN_UP(batch, A_BM)//A_regM , ALIGN_UP(rank*3, A_BN)//(A_regN)]
        self.gemma_LWS = [A_sgM, A_sgN * self.sg_sz]
        assert A_sgM *A_sgN * self.sg_sz <= 1024, f" A_LWS:{self.gemma_LWS} exceed 1024 limitation"
        
        self.gemmb_GWS = [ALIGN_UP(batch, B_BM)//B_regM , ALIGN_UP(self.qkv_state, B_BN)//(B_regN)]
        self.gemmb_LWS = [B_sgM, B_sgN *  self.sg_sz]
        assert B_sgM *B_sgN *  self.sg_sz <= 1024, f" B_LWS:{self.gemmb_LWS} exceed 1024 limitation"
        
        if use_ref:
            self.cl_kernels_ref = kernel_cache(lora_qkv_ref_kernel,  options=f"-DN_0={self.q_state} -DN_1_2={self.kv_state}")
            self.tA_output_ref = cl.tensor([batch, self.rank*3], np.dtype(np.float16))

        else:
            self.kernel_opt_gemma = kernel_cache(gemma_kernel_src, options=f"-DSG_SZ={self.sg_sz}")
            self.kernel_opt_gemmb = kernel_cache(gemmb_kernel_src, options=f"-DSG_SZ={self.sg_sz} -DN0={self.q_state} -DN1_2={self.kv_state}")

        self.use_ref = use_ref
        if use_ref == False:
            print(f'----------------------------------------------------------------------------------------------------------------------------------')
            print(f'| BATCH = {batch} Q_STATE:{input_state}, KV_STATE:{kv_state}, RANK:{rank}:')
            print(f'[1ST_GEMMA] GWS:{self.gemma_GWS}, LWS:{self.gemma_LWS}, M:{batch}/{A_BM}, N:{rank}/{A_BN}, SGM:{A_sgM} SGN:{A_sgN} REGM:{A_regM} REGN:{A_regN}')
            print(f'[1st_GEMMB] GWS:{self.gemmb_GWS}, LWS:{self.gemmb_GWS}, M:{batch}/{B_BM}, N:{rank}/{B_BN}, SGM:{B_sgM} SGN:{B_sgN} REGM:{B_regM} REGN:{B_regN}')
            print(f'----------------------------------------------------------------------------------------------------------------------------------')



    def __call__(self, mainInput, loraInput, stateA0, stateA1, stateA2, stateA, stateAlpha, stateB0, stateB1, stateB2, Aoutput, result):

        if self.use_ref:
            self.cl_kernels_ref.enqueue("gemmA", [self.batch, self.rank*3],[1, self.rank], loraInput, stateA,
                                        self.tA_output_ref, stateAlpha, self.rank*3, self.input_state, 1, self.batch)
            # REF_LOCAL_SIZE = 16
            self.cl_kernels_ref.enqueue("gemmB", [self.batch, self.input_state + 2*self.kv_state],[1, min(self.qkv_state, 1024)],
                                        mainInput, self.tA_output_ref, stateB0, stateB1, stateB2, result, stateAlpha, self.rank)
            return self.tA_output_ref

        else:
            # GEMMA: ONE WG would has {self.gemma_sgK*self.rank/SG_SZ}subgroups. self.rank/SG_SZ subgroups on N dimension, self.gemma_sgK on K dimension
            # Total {self.gemma_wg} WGS
            self.kernel_opt_gemma.enqueue("gemmA_rM8_rN2", self.gemma_GWS, self.gemma_LWS, 
                                        loraInput, stateA0, stateA1, stateA2, Aoutput, stateAlpha, mainInput,self.batch, self.rank*3, self.input_state, self.rank)
            # GEMMB: ONE WG would has {gemmb_wg_sz/SG_SZ}subgroups.
            # Total {(self.q_state+2*self.kv_state)/gemmb_wg_sz} WGS
            self.kernel_opt_gemmb.enqueue("gemmB_rM16_rN2", self.gemmb_GWS, self.gemmb_LWS, 
                                          Aoutput, stateB0, stateB1, stateB2, result, stateAlpha, mainInput, self.batch, self.qkv_state, self.rank)
            return Aoutput

        # print("======================opt:")


        # cl.finish()
        # return result



def test_lora_1st(batch, rank, input_state, kv_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN, check_acc = False):
    cl.profiling(True)
    SG_SZ = 16
    vRANGE = 1
    if check_acc:
        REPEAT = 1
    else:
        REPEAT = 100
    # np.random.seed(0)
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

    loraInput = np.random.randint(-vRANGE, vRANGE+1, [batch, input_state]).astype(np.float16)
    mainInput = np.random.randint(-vRANGE, vRANGE+1, [batch, qkv_state]).astype(np.float16)
    Aoutput = np.ones([batch, rank*3]).astype(np.float16) 
    
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
    res_list = [cl.tensor([batch, qkv_state], np.dtype(np.float16))for _ in range(REPEAT)]
    ref_result = cl.tensor([batch, qkv_state], np.dtype(np.float16))


    ref = LORA_1ST(batch, rank, input_state, kv_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN,True)
    opt = LORA_1ST( batch, rank, input_state, kv_state,  A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN,False)
    
    
    if check_acc:
        tmp_opt=opt(mainInput_list[0], loraInput_list[0], stateA0_list[0], stateA1_list[0], stateA2_list[0], None, alpha_list[0], stateB0_list[0], stateB1_list[0], stateB2_list[0], A_output_list[0], res_list[0])
        # print(cl.finish())
        tmp_ref=ref(mainInput_list[0], loraInput_list[0], None, None, None, stateA_list[0], alpha_list[0], stateB0_list[0], stateB1_list[0], stateB2_list[0], A_output_list[0], ref_result)
        cl.finish()
        compare(tmp_ref.numpy(), tmp_opt.numpy())

        compare(ref_result.numpy(), res_list[0].numpy())
        print(f'BATCH:{batch} INPUT_STATE:{input_state}, RANK:{rank}, KV_STATE:{kv_state} ACC PASS!')
    else:
        for i in range(0, REPEAT):
            opt(mainInput_list[i], loraInput_list[i], stateA0_list[i], stateA1_list[i], stateA2_list[i], None, alpha_list[i], stateB0_list[i], stateB1_list[i], stateB2_list[i], A_output_list[i], res_list[i])
        print(cl.finish())
#         profiling_data  = cl.finish()
#         # FMA + multiply scale
#         flops_a = batch*rank*input_state*2 + batch*rank
#         # FMA + add
#         flops_b = batch*rank*output_state*2 + batch*output_state
#         # loarinput + stateA + scale
#         rd_bytes_a = (rank*input_state + input_state*batch + rank)*2
#         # maininput + Aoutput + stateB 
#         rd_bytes_b = (rank*output_state + output_state*batch + rank*batch)*2

#         for i in range(1, REPEAT):
#             ns_a = profiling_data[i*2]
#             ns_b = profiling_data[i*2 + 1]
#             if VERBOSE:
#                 print(f"[GEMMA]: {flops_a*1e-6:.1f} Mflop/{ns_a*1e-3:.1f} us = {flops_a/ns_a:.3f} GFlops/s  {rd_bytes_a/1e6:.1f} MB/{ns_a*1e-3:.1f} us = {rd_bytes_a/ns_a:.1f} GB/s ,\
#  [GEMMB]: {flops_b*1e-6:.1f} Mflop/{ns_b*1e-3:.1f} us = {flops_b/ns_b:.3f} GFlops/s  {rd_bytes_b/1e6:.1f} MB/{ns_b*1e-3:.1f} us = {rd_bytes_b/ns_b:.1f} GB/s ,\
#  [total]: {(ns_a+ns_b)*1e-3:.1f}us")
#             else:
#                 print(f'[latency]: {ns_a*1e-3:.1f} + {ns_b*1e-3:.1f} = {(ns_a+ns_b)*1e-3:.1f}us')

def blocking_1nd(batch, rank, input_state, kvstate):
    assert batch >= 8, f"batch:{batch} too small in 1st token, not support in opt kernel"
# GEMMA will not divide N across WGs. 1. choose[regM, regN] based on rank and m, 
#                                     2, choose[sgN], 
#                                     3  sgM = 16//sgN or sgM = 8//sgN
# seems regM and regN influences FPS more than sgM, sgN. 
# If rank is 128, 256, and M >= 2000, can try[16, 2]??
# GEMMA rank == 64, use regM, regN for  [8,2] , sgM = 16, sgN = 2, when M >= 2048. 
# other wise,  choose regM = 4, regN = 1, sgM = ?, sgN = 1/2/8.
    sg_sz = 16
    if batch >= 2000:
        if rank == 64:
            A_regM, A_regN = [8, 2]
        elif rank == 128 or rank == 256:
            A_regM, A_regN = [8, 2]
        else:
            A_regM, A_regN = [4, 1]
        A_sgN = rank*3//(sg_sz*A_regN)
        A_sgM = DIV_UP(48, A_sgN)
    else:
        A_regM, A_regN = [4, 1]
        A_sgN = rank*3//(sg_sz*A_regN)
        A_sgM = DIV_UP(48, A_sgN)
# GEMMB: M < 256 regM=8, regN=2, sgM=16, sgN=4, small M() fits into
#        M >= 256, regM=16, regN=2, sgM=8, sgN=4
    if batch < 256:
        B_regM, B_regN = [8, 2]
        B_sgM, B_sgN = [16, 4]

    else:
        B_regM, B_regN = [16, 2]
        B_sgM, B_sgN = [8, 4]
    ##one SG calculation can't cross Q,K,V should only one of Q,K,V . For both GEMMA and GEMMB.
    assert (rank%(sg_sz*A_regN)) == 0
    assert kvstate %(B_regN*sg_sz) == 0, f'kvstate:{kvstate}, B_regN:{B_regN}'
    return [A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN]

cl.profiling(True)

# A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(3172, 64, 1536, 1536)
# test_lora_1st(3172, 64, 1536, 256, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
#                     B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)

# test_lora_1st(3180, 64, 1536, 256, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
#                     B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)

# test_lora_1st(3172, 64, 1536, 512, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
#                     B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)

for batch in range(2054, 2077):              
    for input_state in (1024, 1536, 2048, 2560, 3072, 3840, 4096, 7*32, 11*32, 13*32, 15*32, 12*32,17*32):
        for rank in (32, 64, 128, 256):
            kv_state = input_state
            A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, kv_state)
            test_lora_1st(batch, rank, input_state, kv_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
                        B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)
            for kv_groups in (3, 4, 5, 6, 7, 8, 9):
                if (input_state % kv_groups == 0) and ((input_state//kv_groups)%32 == 0):
                    kv_state = input_state // kv_groups
                    A_regM, A_regN, A_sgM, A_sgN, B_regM, B_regN, B_sgM, B_sgN = blocking_1nd(batch, rank, input_state, kv_state)
                    test_lora_1st(batch, rank, input_state, kv_state, A_regM = A_regM, A_regN = A_regN, A_sgM=A_sgM, A_sgN = A_sgN,
                        B_regM=B_regM, B_regN=B_regN, B_sgM=B_sgM, B_sgN=B_sgN, check_acc=True)
print("-------------------------")