
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
    //prepack A from [m,k] to [M, K, bk, bm]
    __kernel void repackA(__global half * A, __global half *repackA, int M, int K)
    {
        int m_idx = get_global_id(0);
        int k_idx = get_global_id(1);
        int offset = m_idx / BM * K * BM + k_idx / BK * BM * BK + m_idx % BM * BK + k_idx % BK;
        repackA[offset] = A[m_idx * K + k_idx];
    }
    //prepack B from [k,n] to [N, K, bk, bn]
    __kernel void repackB(__global half * B, __global half *repackB, int K, int N)
    {
        int k_idx = get_global_id(0);
        int n_idx = get_global_id(1);
        int offset = n_idx / BN * BN * K + k_idx * BN + n_idx % BN;
        repackB[offset] = B[k_idx * N + n_idx];
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


def gen_SLMA(blockA):
    src = ''

    if  blockA == 24:
        src = r'''
        ushort8 tmpA_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest), tmpA_0);
        ushort8 tmpA_1 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA+128));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest+128), tmpA_1);
        ushort8 tmpA_2 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA+256));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest+256), tmpA_2);
        '''
    elif  blockA == 32:
        src = r'''
        ushort8 tmpA_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest), tmpA_0);
        ushort8 tmpA_1 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA+128));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest+128), tmpA_1);
        ushort8 tmpA_2 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA+256));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest+256), tmpA_2);
        ushort8 tmpA_3 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA+384));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest+384), tmpA_3);
        '''
    elif  blockA == 16:
        src = r'''
        ushort8 tmpA_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest), tmpA_0);
        ushort8 tmpA_1 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA+128));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest+128), tmpA_1);
        '''
    elif blockA == 8:
        src = r'''
        ushort8 tmpA_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrA));
        intel_sub_group_block_write_us8((const __local ushort*)(lA_ptr_dest), tmpA_0);
        '''
    elif blockA == 4:
        src = r'''
        ushort4 tmpA_0 = intel_sub_group_block_read_us4((const __global ushort*)(ptrA));
        intel_sub_group_block_write_us4((const __local ushort*)(lA_ptr_dest), tmpA_0);
        '''
    else:
        print(f'error: not support blockA{blockA}')
    return src

def gen_SLMB(blockB):
    src = ''
    if  blockB == 24:
        src = r'''
        ushort8 tmpB_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest), tmpB_0);
        ushort8 tmpB_1 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB+128));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest+128), tmpB_1);
        ushort8 tmpB_2 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB+256));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest+256), tmpB_2);
        '''

    elif blockB == 32:
        src = r'''
        ushort8 tmpB_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest), tmpB_0);
        ushort8 tmpB_1 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB+128));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest+128), tmpB_1);
        ushort8 tmpB_2 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB+256));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest+256), tmpB_2);
        ushort8 tmpB_3 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB+384));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest+384), tmpB_3);
        '''
    elif  blockB == 16:
        src = r'''
        ushort8 tmpB_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest), tmpB_0);
        ushort8 tmpB_1 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB+128));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest+128), tmpB_1);
        '''
    elif blockB == 8:
        src = r'''
        ushort8 tmpB_0 = intel_sub_group_block_read_us8((const __global ushort*)(ptrB));
        intel_sub_group_block_write_us8((const __local ushort*)(lB_ptr_dest), tmpB_0);
        '''
    elif blockB == 4:
        src = r'''
        ushort4 tmpB_0 = intel_sub_group_block_read_us4((const __global ushort*)(ptrB));
        intel_sub_group_block_write_us4((const __local ushort*)(lB_ptr_dest), tmpB_0);
        '''
    else:
        print(f'error: not support blockB{blockB}')
    return src

def test_SLM_FMA(M, N, K,regM, regN, sgM, sgN, BK, blkA, blkB):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
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

        //__global half *ptrA = A + m_idx * K;
        //__global half *ptrB = B + n_idx;
        __global half *ptrA = A + get_group_id(0)*K*BM+sgid*blkA*SG_SZ;
        __global half *ptrB = B + get_group_id(1)*K*BN+ sgid*blkB*SG_SZ;
        __global half *ptrC = C + m_idx * N + n_idx;

        __local half lA[BM*BK];
        __local half lB[BK*BN];

        __local half* lA_ptr_dest = lA + sgid * blkA*SG_SZ;
        __local half* lB_ptr_dest = lB + sgid * blkB*SG_SZ;

        __local half* lA_ptr_src = lA + sgid_M * regM *BK;
        __local half* lB_ptr_src = lB + sgid_N * regN * SG_SZ;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int j = 0; j < K; j += BK) {
#if 1
            //copy A to SLM.
''' +    gen_SLMA(blkA) + r'''
''' +    gen_SLMB(blkB) + r'''

            barrier(CLK_LOCAL_MEM_FENCE);
#endif
            __local half* lA_ptr = lA_ptr_src;
            __local half* lB_ptr = lB_ptr_src;

            // FMA Matmul([BM, BK], [BK, BN]) = [BM, BN]
            for(int i = 0; i < BK; i+=SG_SZ) {

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
#if 1
            barrier(CLK_LOCAL_MEM_FENCE);
#endif
            ptrA += BM*BK;
            ptrB += BN*BK;
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

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 80
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)

    alpha = np.random.rand(N).astype(np.float16)


    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor(C) for _ in range(REPEAT)]

    kernel_slm = kernel_cache(gen_slm_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN} -DblkA={blkA} -DblkB={blkB}")
    #kernel_slm = kernel_cache(src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN} -DblkA={blkA} -DblkB={blkB}")

    GWS = [M//regM , N//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        # kernel_slm.enqueue("gemm", GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
        kernel_slm.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    rd_bytes = (N*K+M*K)*2
    for time in ns:
        print(f'TPUT: [SLM]:{flops/time:.1f} GFLOPS, {rd_bytes/time:.1f} GBS, ratio:{flops/rd_bytes:.1f} rd:{rd_bytes/(1024**2):.1f}MB us:{time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK} sgM:{sgM} sgN:{sgN} BLK_A:{blkA} BLK_B:{blkB}')
    print(f'SLB:{(BM*BK*2 + BN*BK*2)/1024}')

    print("----------------------------------------------------")
    if 1:
        tC_ref = cl.tensor(C)
        tA_repack = cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [M // BM, K // BK, BM, BK]).astype(np.float16))
        tB_repack = cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [N // BN, K // BK, BK, BN]).astype(np.float16))
        kernel_ref = kernel_cache(cl_kernel_sources_ref, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN}")
        kernel_ref.enqueue("gemm", [M, N], [8 , 8], tA_list[0], tB_list[0],tC_ref, M, N, K)
        kernel_ref.enqueue("repackA", [M, K], [16, 16], tA_list[0], tA_repack, M, K)
        kernel_ref.enqueue("repackB", [K, N], [16, 16], tB_list[0], tB_repack, K, N)
        kernel_slm.enqueue(func, GWS, LWS, tA_repack, tB_repack,tC_slm_list[0], M, N, K)
        cl.finish()
        compare(tC_ref.numpy(), tC_slm_list[0].numpy())
        if 0:
            Arepack = tA_repack.numpy()
            Brepack = tB_repack.numpy()
            for mb in range(0, M // BM):
                for kb in range(0, K // BK):
                    for subm in range (0, BM):
                        for subk in range(0, BK):
                            midx = mb * BM + subm
                            kidx = kb * BK + subk
                            assert Arepack[mb, kb, subm , subk] ==  A[midx , kidx]
            for nb in range(0, N // BN):
                for kb in range(0, K // BK):
                    for subn in range (0, BN):
                        for subk in range(0, BK):
                            nidx = nb * BN + subn
                            kidx = kb * BK + subk
                            assert Brepack[nb, kb, subk, subn] ==  B[kidx, nidx]


def test_FMA(M, N, K,regM, regN, sgM, sgN, BK):
    func = f'gemm_rM{regM}_rN{regN}_sM{sgM}_sN{sgN}_M{M}_N{N}_K{K}'
    gen_opt_src =  r'''
    __attribute__((intel_reqd_sub_group_size(SG_SZ)))
    __kernel void
    ''' + f'{func}' + r'''(__global half * A, __global half *B,  __global half *C,  int M, int N, int K) {
        int sgid = get_sub_group_id();
        int sgid_N = sgid % sgN;
        int sgid_M = sgid / sgN;
        ''' + f'int regM = {regM};\nint regN = {regN};' + r'''

        int m_idx = get_group_id(0) * BM  + sgid_M * regM;
        int n_idx = get_group_id(1) * BN  + sgid_N * SG_SZ * regN;

        //__global half *ptrA = A + m_idx * K;
        //__global half *ptrB = B + n_idx;
        __global half *ptrA = A + get_group_id(0)*K*BM+sgid_M * regM *BK;
        __global half *ptrB = B + get_group_id(1)*K*BN+ sgid_N * regN * SG_SZ;
        __global half *ptrC = C + m_idx * N + n_idx;

        '''  + "\n\t ".join([f"half sum{m}_{n} = 0;" for m in range(regM) for n in range(regN)]) + r''';

        for(int j = 0; j < K; j += BK) {

            // FMA Matmul([BM, BK], [BK, BN]) = [BM, BN]
            for(int i = 0; i < BK; i+=SG_SZ) {

                '''  + "\n\t\t ".join([f"ushort input{m} = intel_sub_group_block_read_us((const __global ushort*)(ptrA + {m} * BK));" for m in range(regM)]) + r'''

                __attribute__((opencl_unroll_hint))
                for (int kk = 0; kk < SG_SZ; kk++) {

                    '''  + "\n\t\t\t ".join([f"half bb{n} = as_half(intel_sub_group_block_read_us((const __global ushort*)(ptrB + {n} * SG_SZ)));" for n in range(regN)]) + r'''
                    '''  + "\n\t\t\t ".join([f"half aa{m} = as_half(intel_sub_group_broadcast(input{m}, kk));" for m in range(regM)]) + r'''
                    ''' + "\n\t\t\t".join([f"sum{m}_{n} = fma(aa{m}, bb{n}, sum{m}_{n});" for m in range(regM) for n in range(regN)]) + r'''
                    ptrB += BN;
                }
                ptrA +=SG_SZ;
            }
            ptrA -= BK;
            ptrA += BM*BK;
            //ptrA += BM*BK;
            //ptrB += BN*BK;
        }
        ''' +  gen_store_C(regM, regN) + r'''
    }
    '''
    # print(gen_opt_src)
    cl.profiling(True)

    SG_SZ = 16
    SLM_SZ = 64*1024
    BM = regM*sgM
    BN = regN*sgN*SG_SZ

    np.random.seed(0)
    vRANGE = 1
    REPEAT = 80
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C = np.random.randint(-vRANGE, vRANGE+1, [M, N]).astype(np.float16)

    alpha = np.random.rand(N).astype(np.float16)


    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_slm_list = [cl.tensor(C) for _ in range(REPEAT)]

    kernel_opt= kernel_cache(gen_opt_src, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN} -DsgM={sgM} -DsgN={sgN}")

    GWS = [M//regM , N//(regN)]
    LWS = [sgM, sgN * SG_SZ]
    assert sgM *sgN * SG_SZ <= 1024, f" LWS:{LWS} exceed 1024 limitation"
    cl.finish()

    for i in range(0, REPEAT):
        # kernel_slm.enqueue("gemm", GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
        kernel_opt.enqueue(func, GWS, LWS, tA_list[i], tB_list[i],tC_slm_list[i], M, N, K)
    ns = cl.finish()
    flops = M * N * K * 2
    rd_bytes = (N*K+M*K)*2
    for time in ns:
        print(f'TPUT: [SLM]:{flops/time:.1f} GFLOPS, {rd_bytes/time:.1f} GBS, ratio:{flops/rd_bytes:.1f} rd:{rd_bytes/(1024**2):.1f}MB us:{time*1e-3:.1f}')
    print("----------------------------------------------------")
    print(f'GWS:{GWS}, LWS:{LWS}, M:{M}/{BM}, N:{N}/{BN}, K:{K}/{BK} sgM:{sgM} sgN:{sgN}')
    print(f'SLB:{(BM*BK*2 + BN*BK*2)/1024}')

    print("----------------------------------------------------")
    if 1:
        tC_ref = cl.tensor(C)
        tA_repack = cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [M // BM, K // BK, BM, BK]).astype(np.float16))
        tB_repack = cl.tensor(np.random.randint(-vRANGE, vRANGE+1, [N // BN, K // BK, BK, BN]).astype(np.float16))
        kernel_ref = kernel_cache(cl_kernel_sources_ref, options=f"-DSG_SZ={SG_SZ} -DBK={BK} -DBM={BM} -DBN={BN}")
        kernel_ref.enqueue("gemm", [M, N], [8 , 8], tA_list[0], tB_list[0],tC_ref, M, N, K)
        kernel_ref.enqueue("repackA", [M, K], [16, 16], tA_list[0], tA_repack, M, K)
        kernel_ref.enqueue("repackB", [K, N], [16, 16], tB_list[0], tB_repack, K, N)
        kernel_opt.enqueue(func, GWS, LWS, tA_repack, tB_repack,tC_slm_list[0], M, N, K)
        cl.finish()
        compare(tC_ref.numpy(), tC_slm_list[0].numpy())
        if 0:
            Arepack = tA_repack.numpy()
            Brepack = tB_repack.numpy()
            for mb in range(0, M // BM):
                for kb in range(0, K // BK):
                    for subm in range (0, BM):
                        for subk in range(0, BK):
                            midx = mb * BM + subm
                            kidx = kb * BK + subk
                            assert Arepack[mb, kb, subm , subk] ==  A[midx , kidx]
            for nb in range(0, N // BN):
                for kb in range(0, K // BK):
                    for subn in range (0, BN):
                        for subk in range(0, BK):
                            nidx = nb * BN + subn
                            kidx = kb * BK + subk
                            assert Brepack[nb, kb, subk, subn] ==  B[kidx, nidx]


if __name__ == '__main__':

    SG_SZ =16
    regM=16
    regN=2
    # Each subgroup will copy A_blk*SG_SZ into SLM A and B_blk*SG_SZ into SLM B
    A_blk = 16
    B_blk= 16
    bk=64

    WGS_M = 16
    WGS_N = 64
    # A_blk = bm*bk%(sgM*sgN*SG_SZ) = sgM*regM*bk%(sgM*sgN*SG_SZ) = regM*bk % (sgN*SG_SZ), bk is the multiple of SG_SZ. Just ensure regM % sgN == 0
    # bn*bk%(sgM*sgN*SG_SZ) = sgN*regN*SG_SZ*bk%(sgM*sgN*SG_SZ)=regN*bk%sgM, bk is 64. ususually sgM is the divisor of 64.
    sgN = regM*bk//(SG_SZ*A_blk)
    sgM = regN*bk // (B_blk)

    assert regM*bk % (sgN*SG_SZ) == 0 and regN*bk%sgM == 0
    assert sgN * sgM <=64

    bm = sgM * regM
    bn = sgN * regN * SG_SZ

    assert bm*bk%(sgM*sgN*SG_SZ) == 0 and bn*bk%(sgM*sgN) == 0 and bk % SG_SZ == 0
    assert bm*bk//(sgM*sgN*SG_SZ) == A_blk and bn*bk//(sgM*sgN*SG_SZ) == B_blk

    K = bk*20
    M = regM * sgM * WGS_M
    N = regN * sgN * SG_SZ * WGS_N
    print(f'----------SLB:{(bm*bk*2 + bn*bk*2)/1024}')

    assert K%bk == 0
    # A770 FMA tput: 512 XVE *16 lanes *2 FMA_OPS *2.4G = 39.3 TFLOPS
    test_SLM_FMA(M, N, K, regM, regN, sgM=sgM, sgN=sgN,BK=bk, blkA=A_blk, blkB=B_blk)
    test_FMA(M, N, K,regM, regN, sgM, sgN, BK=bk)

# TPUT: [SLM]:21200.6 GFLOPS, us: 1215.5
