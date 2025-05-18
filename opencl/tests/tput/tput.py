
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
    __kernel void gemm(__global half * A, __global half *B,  __global half *CC, int M,  int N, int K) {
        int m_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        half sum = 0.f;
        for (int i = 0; i < K; i++)
            sum = fma(A[m_idx * K + i], B[i*N+n_idx], sum);
        CC[m_idx * N + n_idx] = sum;
    }
'''

# BM = regM*sgM
# BN = regN*sgN*SG_SZ
# GWS = [M//regM, N//(regN)]
# LWS = [sgM, sgN * SG_SZ]
# GWS:[256, 16], LWS:[16, 16], M:1024/64, N:64/64, K:1536/128
def gen_store_C(regM, regN):
    src = ""
    for m in range(regM):
        for n in range(regN):
            src +=  f"\n\tintel_sub_group_block_write_us((__global ushort*)(ptrC + SG_SZ * {n}), as_short(sum{m}_{n}));"
        src += f"\n\tptrC += N;\n\n"
    return src

# Use block read without SLM
def test_FMA(M, N, K,regM, regN, sgM, sgN):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    gen_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C, int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''
        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;
        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int i = 0; i < K; i += SG_SZ) {

            // copy B & A into shared local mem

                '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * K));" for m in range(regM)]) + r'''

                __attribute__((opencl_unroll_hint()))
                for (int kk = 0; kk < SG_SZ; kk++) {

                     '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                     '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                     ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    ptrB += N;
                }
                ptrA +=SG_SZ;
            //}
        }

        ''' +  gen_store_C(regM, regN) + r'''
    }
    '''
    cl.profiling(True)

    SG_SZ = 16
    BM = regM*sgM
    BN = regN*sgN*SG_SZ

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 50
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor(C) for _ in range(REPEAT)]

    kernel_opt = kernel_cache(gen_src, options=f"-DSG_SZ={SG_SZ}  -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}", dump="./dump/")

    GWS = [M//regM , N//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        kernel_opt.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time in ns:
        print(f'TPUT: [NO_SLM]:{flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}')
    print("----------------------------------------------------")
    if 1:
        tC_ref = cl.tensor(C)
        kernel_ref = kernel_cache(cl_kernel_sources_ref, options=f"-DSG_SZ={SG_SZ}")
        kernel_ref.enqueue("gemm", [M, N], [8 , 8], tA_list[0], tB_list[0],tC_ref, M, N, K)
        cl.finish()
        compare(tC_ref.numpy(), tC_slm_list[0].numpy())

# Use general block read into SLM.
def test_SLM_FMA(M, N, K,regM, regN, sgM, sgN, BK):
    func = f'gemm_SLM_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    gen_slm_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C,  int M, int N, int K) {
#define READ_DATA 1
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''

        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;

        __global half *ptrA = A + m_idx * K;
        __global half *ptrB = B + n_idx;
        __global half *ptrC = C + m_idx * N + n_idx;

        __local half lA[BM*BK];
        __local half lB[BK*BN];

        __local half* lA_ptr_dest = lA + sgid_M * regM *BK;
        __local half* lB_ptr_dest = lB + sgid_N * regN * SG_SZ;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int j = 0; j < K; j += BK) {
#if READ_DATA
            int k_len = BK;
            //copy A to SLM.
            for (int offset_K = sgid_N * SG_SZ; offset_K < k_len; offset_K += sgN * SG_SZ) {
                ''' + "\n\t\t ".join([f"ushort tmpA_{m} = intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * K + offset_K));" for m in range(regM)]) + r'''
                ''' + "\n\t\t ".join([f"intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + {m} * BK + offset_K), tmpA_{m});" for m in range(regM)])+ r'''
            }
            //copy B to SLM.
            for (int offset_K = sgid_M; offset_K < k_len; offset_K += sgM) {
                ''' + "\n\t\t ".join([f"ushort tmpB_{n} = intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * SG_SZ + offset_K*N));" for n in range(regN)]) + r'''
                ''' + "\n\t\t ".join([f"intel_sub_group_block_write_us((const __local ushort*)(lB_ptr_dest + {n} * SG_SZ + offset_K*BN), tmpB_{n});" for n in range(regN)])+ r'''
            }
            barrier(CLK_LOCAL_MEM_FENCE);
#endif
            __local half* lA_ptr = lA_ptr_dest;
            __local half* lB_ptr = lB_ptr_dest;

            // FMA Matmul([BM, BK], [BK, BN]) = [BM, BN]
            for(int i = 0; i < BK; i+=SG_SZ) {

                '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + {m} * BK));" for m in range(regM)]) + r'''

                __attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {

                    '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __local ushort*)(lB_ptr + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                    '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                    ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    lB_ptr += BN;
                }
                lA_ptr +=SG_SZ;
            }
#if READ_DATA
            barrier(CLK_LOCAL_MEM_FENCE);
#endif
            ptrA += BK;
            ptrB += BK * N;
        }
        ''' +  gen_store_C(regM, regN) + r'''
    }
    '''
    # print(gen_slm_src)
    cl.profiling(True)

    SG_SZ = 16
    SLM_SZ = 64*1024
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    print("#####################################################")
    print(f'SLM:{(BM*BK*2+BN*BK*2)/1024}KB')
    print("#####################################################")
    np.random.seed(0)
    vRANGE = 1
    REPEAT = 50
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor(C) for _ in range(REPEAT)]

    kernel_slm = kernel_cache(gen_slm_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")

    GWS = [M//regM , N//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        kernel_slm.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time in ns:
        print(f'TPUT: [SLM]:{flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    if 1:
        tC_ref = cl.tensor(C)
        kernel_ref = kernel_cache(cl_kernel_sources_ref, options=f"-DSG_SZ={SG_SZ}")
        kernel_ref.enqueue("gemm", [M, N], [8 , 8], tA_list[0], tB_list[0],tC_ref, M, N, K)
        cl.finish()
        compare(tC_ref.numpy(), tC_slm_list[0].numpy())

def test_SLM_FMA_COPY_OPT(M, N, K,regM, regN, sgM, sgN, BK):
    func = f'gemm_SLM_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
     # A: BK is contineous read, how many continious block reading A into SLM in one sg. each block is BK
    A_BLK_SZ = BK // 16
    A_blks = regM // sgN

    # B: BN is memory contineous,  how many continious block reading B into SLM in one sg.each block is regN*16
    B_BLK_SZ = regN
    B_blks = BK // sgM

    gen_slm_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C,  int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''

        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;

        __global half *ptrA = A + m_idx * K + K * A_BLKs * sgid_N;
        __global half *ptrB =  B + n_idx + N * B_BLKs * sgid_M;

        __global half *ptrC = C + m_idx * N + n_idx;

        __local half lA[BM*BK];
        __local half lB[BK*BN];

        __local half* lA_ptr = lA + sgid_M * regM *BK;
        __local half* lB_ptr = lB + sgid_N * regN * SG_SZ;

        __local half* lA_ptr_dest = lA + sgid_M * regM *BK + A_BLKs * sgid_N * BK;
        __local half* lB_ptr_dest = lB + sgid_N * regN * SG_SZ +  B_BLKs * sgid_M * BN;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int j = 0; j < K; j += BK) {
#if A_VEC_READ

            ''' + f'ushort{A_BLK_SZ} tmpA_0;' + r'''
#else
            ''' + f'ushort tmpA_0;' + r'''
#endif

# if B_VEC_READ
            ''' + f'ushort{B_BLK_SZ} tmpB_0;' + r'''
#else
            ''' + f'ushort tmpB_0;' + r'''
#endif
             __attribute__((opencl_unroll_hint))
            for (int blkA_idx = 0; blkA_idx < A_BLKs; blkA_idx++) {
#if A_VEC_READ
                ''' + f"tmpA_0 = intel_sub_group_block_read_us{A_BLK_SZ}((const __global ushort*)(ptrA + blkA_idx * K));"  + r'''
                ''' + f"intel_sub_group_block_write_us{A_BLK_SZ}((const __local ushort*)(lA_ptr_dest + blkA_idx * BK), tmpA_0);"+ r'''
#else
                __attribute__((opencl_unroll_hint))
                for (int idx = 0; idx < A_BLK_SZ ; idx++) {
                    tmpA_0 = intel_sub_group_block_read_us((const __global ushort*)(ptrA + blkA_idx * K + idx * SG_SZ));
                    intel_sub_group_block_write_us((const __local ushort*)(lA_ptr_dest + blkA_idx * BK + idx * SG_SZ), tmpA_0);
                }
#endif
            }
            __attribute__((opencl_unroll_hint))
            for (int blkB_idx = 0; blkB_idx < B_BLKs; blkB_idx++) {
#if B_VEC_READ
                ''' + f"tmpB_0 = intel_sub_group_block_read_us{B_BLK_SZ}((const __global ushort*)(ptrB + blkB_idx * N));"  + r'''
                ''' + f"intel_sub_group_block_write_us{B_BLK_SZ}((const __local ushort*)(lB_ptr_dest + blkB_idx * BN), tmpB_0);"+ r'''
#else
                __attribute__((opencl_unroll_hint))
                for (int idx = 0; idx < B_BLK_SZ; idx++) {
                    tmpB_0 = intel_sub_group_block_read_us((const __global ushort*)(ptrB + blkB_idx * N + idx * SG_SZ));
                    intel_sub_group_block_write_us((const __local ushort*)(lB_ptr_dest + blkB_idx * BN + + idx * SG_SZ), tmpB_0);
                }
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // FMA Matmul([BM, BK], [BK, BN]) = [BM, BN]
            for(int i = 0; i < BK; i+=SG_SZ) {

                '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __local ushort*)(lA_ptr + {m} * BK));" for m in range(regM)]) + r'''

                __attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {

                    '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __local ushort*)(lB_ptr + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                    '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                    ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    lB_ptr += BN;
                }
                lA_ptr +=SG_SZ;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            ptrA += BK;
            ptrB += BK * N;
            lA_ptr -= BK;
            lB_ptr -= BN*BK;
        }
        ''' +  gen_store_C(regM, regN) + r'''
    }
    '''
    # print(gen_slm_src)
    cl.profiling(True)

    SG_SZ = 16
    SLM_SZ = 64*1024
    BM = regM*sgM
    BN = regN*sgN*SG_SZ
    print("#####################################################")
    print(f'SLM:{(BM*BK*2+BN*BK*2)/1024}KB')
    print("#####################################################")
    np.random.seed(0)
    vRANGE = 1
    REPEAT = 50
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)

    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor(C) for _ in range(REPEAT)]
    if BK == 32 or BK ==64 or BK ==128:
        A_VEC_READ = 1
    else:
        A_VEC_READ = 0

    if  regN == 2 or  regN == 4 or  regN == 8:
        B_VEC_READ = 1
    else:
        B_VEC_READ = 0

    A_VEC_READ = 0
    B_VEC_READ = 0
    kernel_slm = kernel_cache(gen_slm_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN} \
                              -DA_BLKs={A_blks} -DB_BLKs={B_blks} -DA_BLK_SZ={A_BLK_SZ} -DB_BLK_SZ={B_BLK_SZ} -DA_VEC_READ={A_VEC_READ} -DB_VEC_READ={B_VEC_READ}")

    GWS = [M//regM , N//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        kernel_slm.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    for time in ns:
        print(f'TPUT: [SLM_OPT]:{flops/time:.1f} GFLOPS, us: {time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK}')
    print("----------------------------------------------------")
    if 1:
        tC_ref = cl.tensor(C)
        kernel_ref = kernel_cache(cl_kernel_sources_ref, options=f"-DSG_SZ={SG_SZ}")
        kernel_ref.enqueue("gemm", [M, N], [8 , 8], tA_list[0], tB_list[0],tC_ref, M, N, K)
        cl.finish()
        compare(tC_ref.numpy(), tC_slm_list[0].numpy())

SG_SZ =16
regM=24
sgM=8
regN=2
sgN=8

WGS_M = 16
WGS_N = 16

bk=64
K = bk*100
M = regM * sgM * WGS_M
N = regN * sgN * SG_SZ * WGS_N

assert K%bk == 0
# A770 FMA tput: 512 XVE *16 lanes *2 FMA_OPS *2.4G = 39.3 TFLOPS

test_FMA(M, N, K,regM, regN, sgM, sgN)
test_SLM_FMA(M, N, K, regM, regN, sgM=sgM, sgN=sgN,BK=bk)
test_SLM_FMA_COPY_OPT(M, N, K, regM, regN, sgM=sgM, sgN=sgN,BK=bk)


# TPUT: [SLM]:21200.6 GFLOPS, us: 1215.5
