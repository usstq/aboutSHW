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

def test_FMA_basic_fp16():
    src =  r'''
    #include <cm/cm.h>
    #include <cm/cmtl.h>
    template <typename T, unsigned EXEC_SIZE>
    inline _GENX_ vector<T, EXEC_SIZE> madd(vector<T, EXEC_SIZE> s1,
                                        vector<T, EXEC_SIZE> s2,
                                        vector<T, EXEC_SIZE> s3) {
  vector<T, EXEC_SIZE> dst;

  asm("mad (M1, %4) %0 %1 %2 %3"
      : "=r"(dst)
      : "r"(s1), "r"(s2), "r"(s3), "n"(EXEC_SIZE));

  return dst;
}

    extern "C" _GENX_MAIN_ void gemm(svmptr_t A [[type("svmptr_t")]],
                                  svmptr_t B [[type("svmptr_t")]],
                                  svmptr_t C [[type("svmptr_t")]],
                                  int K, int N) {
        matrix<half, regM, 16> matA;
        matrix<half, regN/16, 16> matB;
        matrix<half, regM*regN/16, 16> sum = 0;
        auto nthrM = cm_local_size(0);
        auto nthrN = cm_local_size(1);
        auto ithrM = cm_local_id(0);
        auto ithrN = cm_local_id(1);

#define DUMMY_READ 0
#if DUMMY_READ
        cm_slm_init(1024*64);
        uint slmA0 = cm_slm_alloc(nthrM*BK*regM * sizeof(half));
        uint slmB0 = cm_slm_alloc(nthrN*BK*regN * sizeof(half));

        slmA0 += ithrM * (BK*regM) * sizeof(half);
        slmB0 += ithrN *regN * sizeof(half);
#else
        B += cm_global_id(1) * (regN * sizeof(half));
        A += cm_global_id(0) * regM * sizeof(half) * K;
#endif

        C += cm_global_id(0) * (regM * sizeof(half) * N);
        C += cm_global_id(1) * regN * sizeof(half);
        for (int kkk = 0; kkk < K; kkk += BK) {
            for (int kk = 0; kk < BK; kk+=16){
                #pragma unroll
                for (int idx = 0; idx < regM; idx++) {
#if DUMMY_READ
                    cm_svm_block_read_unaligned(slmA0 + idx*BK*sizeof(half), matA.row(idx));
#else
                    cm_svm_block_read(A + idx * K * sizeof(half), matA.row(idx));
#endif
                }
                #pragma unroll
                for (int j = 0; j < 16; j++) {
#if DUMMY_READ
                    cm_svm_block_read_unaligned(slmB0,matB.format<half>());
#else
                    cm_svm_block_read(B,matB.format<half>());
#endif
                    #pragma unroll
                    for (int idx_m = 0; idx_m < regM; idx_m++) {
                        vector<half,16> bastA = matA.row(idx_m).replicate<16, 1>(j);
                        #pragma unroll
                        for (int idx_n = 0; idx_n < regN/16; idx_n++) {
                            sum.row(idx_m*regN/16 + idx_n) = madd(matB.row(idx_n), bastA, sum.row(idx_m*regN/16 + idx_n));
                        }
                    }
 #if DUMMY_READ
                    slmB0 += nthrN*regN*sizeof(half);
#else
                    B += N*sizeof(half);
#endif
                }
#if DUMMY_READ
                slmA0 += 16*sizeof(half);
#else
                A += 16*sizeof(half);
#endif
            }
#if DUMMY_READ
        slmB0 -= BK*nthrN*regN*sizeof(half);
        slmA0 -= BK*sizeof(half);
#endif
        }

        matrix_ref<half, regM, regN> sum_ref = sum.format<half, regM, regN>();
        #pragma unroll
        for (int idx_m = 0; idx_m < regM; idx_m++) {
            cm_svm_block_write<half, regN>(C + idx_m*N*sizeof(half), sum_ref.row(idx_m));
        }
    }
    '''
    REPEAT = 20
    regM = 24
    regN = 32
    nthrM = 8
    nthrN = 8
    WGS_M = 8
    WGS_N = 8
    global_nthrM = nthrM*WGS_M
    global_nthrN = nthrN*WGS_N
    BK = 64
    # M = regM*global_nthrM
    # N = regN*global_nthrN
    M = regM*global_nthrM
    N = regN*global_nthrN
    K = BK*16
    # np.random.seed(0)
    vRANGE = 2
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [K, N]).astype(np.float16)
    C_ref = np.matmul(A, B)
    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float16)) for _ in range(REPEAT)]

    SG_SZ = 16
    kernel = kernel_cache(src, options=f"-cmc -march=DG2 -Qxcm_opt_report -mdump_asm -g2 -DregM={regM} -DregN={regN} -DBK={BK} -DSG_SZ=16")

    for i in range(0, REPEAT):
        kernel.enqueue("gemm", [global_nthrM, global_nthrN],[nthrM, nthrN], tA_list[i], tB_list[i],tC_list[i], K, N)
    ns = cl.finish()
    flops = M * N * K * 2
    print("----------------------------------------------------")
    print(f'\M:{M}, N:{N}, K:{K}')
    print("----------------------------------------------------")
    for time_opt in ns:
        print(f'TPUT: [W/O SLM]:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}')

    compare(C_ref, tC_list[0].numpy())

cl.profiling(True)
test_FMA_basic_fp16()
print("-------------------------")