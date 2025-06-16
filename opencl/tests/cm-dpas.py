#!/usr/bin/python3
from clops import cl
import numpy as np
import time
from clops import compare
import torch
from clops import to_cl
from clops.utils import *

def test_INT8():
    src =  r'''
template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8d,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template<int M, int N>
void show<half, int M, int N>(const matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}



#ifdef CM_HAS_LSC_UNTYPED_2D
#pragma message (">>>>>> CM_HAS_LSC_UNTYPED_2D OK")
#else
#error "----------------------------"
#endif

#define SystolicDepth 8
#define OpsPerChannel 4
#define regN 16
#define regK (SystolicDepth*OpsPerChannel)

    extern "C" _GENX_MAIN_ void gemm(int8_t* A [[type("svmptr_t")]],
                                  int8_t* B [[type("svmptr_t")]],
                                  int32_t* C [[type("svmptr_t")]],
                                  unsigned int M, unsigned int K, unsigned int N) {
        int m_idx = cm_global_id(0) * regM * tileM;
        int n_idx = cm_global_id(1) * regN * 1;
        matrix<int8_t, tileM, regM * regK> matA;
        matrix<int8_t, 1, regK*regN> matB;
        matrix<int32_t, tileM*1, regM*regN> matC = 0;
        matrix<int8_t, tileM*1, regM*regN> matC8 = 0;

        lsc::block_2d_desc<int8_t, 1, regM, regK> descA{(int8_t*)A, M-1, K-1, K-1, 0, m_idx};
        lsc::block_2d_desc<int8_t, 1, regK, regN> descB{(int8_t*)B, K-1, N-1, N-1, n_idx, 0};
        lsc::block_2d_desc<int32_t, 1, regM, regN> descC{(int32_t*)C, M-1, 4*N-1, 4*N-1, n_idx, m_idx};

#if 1
        for (int kb = 0; kb < K; kb+= BK) {
            #pragma unroll
            for (int ki = 0; ki < BK ; ki+=regK) {
                descB.set_block_y(kb+ki);
                cm_load<lsc::LoadOp::VNNI>(matB.format<int8_t>(), descB);
                descA.set_block_x(kb+ki);
                #pragma unroll
                for (int mt_idx = 0; mt_idx <  tileM; mt_idx ++) {
                    descA.set_block_y(m_idx + mt_idx* regM);
                    cm_load<lsc::LoadOp::Normal>(matA[mt_idx].format<int8_t>(), descA);
                }
                int tileC_idx = 0;
                #pragma unroll
                for (int mt_idx = 0; mt_idx <  tileM; mt_idx ++) {
                    matC.row(tileC_idx).format<int32_t>() = cm_dpas<CM_PRECISION_S8, CM_PRECISION_S8, SystolicDepth, regM>(
                            matC.row(tileC_idx).format<int32_t>(),
                            matB[0].format<int32_t>(),
                            matA[mt_idx].format<int32_t>());
                    tileC_idx++;
                }
            }
        }
        show(matC[0].format<int32_t,regM,regN>());
        #pragma unroll
        for (int mt_idx = 0; mt_idx <  tileM; mt_idx ++) {
            descC.set_block_y(m_idx + mt_idx*regM);
            descC.set_block_x(n_idx);
            cm_store(descC, matC[mt_idx*1].format<int32_t>());
        }
#endif
    }
    '''
    REPEAT = 1
    regM = 8
    regN = 16
    BK = 64
    tileM = 2
    tileN = 1

    nthrM = 1
    nthrN = 1

    BM = nthrM* regM * tileM
    BN = nthrN* regN * tileN

    WGS_M = 1
    WGS_N = 1
    global_nthrM = nthrM*WGS_M
    global_nthrN = nthrN*WGS_N

    M = regM*tileM*global_nthrM
    N = regN*tileN*global_nthrN
    K = BK*2
    # np.random.seed(0)
    vRANGE = 2
    A = torch.randint(-vRANGE, vRANGE+1, [M, K], dtype=torch.int32)
    B = torch.randint(-vRANGE, vRANGE+1, [K, N], dtype=torch.int32)
    C = torch.zeros(M, N, dtype=torch.int32)

    C_ref = A @ B
    tA_list = [to_cl(A.to(torch.int8)) for _ in range(REPEAT)]
    tB_list = [to_cl(B.to(torch.int8)) for _ in range(REPEAT)]
    tC_list = [to_cl(C.to(torch.int32)) for _ in range(REPEAT)]

    SG_SZ = 16
    kernel =  cl.kernels(src, options=f"-cmc -mdump_asm -g2 -Qxcm_register_file_size=256 -DregM={regM} -DregN={regN} -DtileM={tileM} -DtileN={tileN} -DBK={BK} -BM={BM} -BN={BN}")

    for i in range(0, REPEAT):
        kernel.enqueue("gemm", [global_nthrM, global_nthrN],[nthrM, nthrN], tA_list[i], tB_list[i],tC_list[i], M, K, N)
    ns = cl.finish()
    flops = M * N * K * 2
    print("----------------------------------------------------")
    print(f'M:{M}, N:{N}, K:{K}')
    for time_opt in ns:
        print(f'TPUT: [W/O SLM]:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}')
    # print(C_ref.detach().numpy())
    # print(tC_list[0].numpy())
    compare(C_ref.detach().numpy(), tC_list[0].numpy())


def test_FP16():
    src =  r'''
template<typename T, int M, int N>
void show_int(const matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8d,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}


template<typename T, int M, int N>
void show_float(const matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template <typename T1, typename T2>
CM_INLINE void Transpose_16x4(matrix_ref<T1, 4, 16> in,
                               matrix_ref<T2, 16, 4> out) {
    out.row(0) = in.column(0);
    out.row(1) = in.column(1);
    out.row(2) = in.column(2);
    out.row(3) = in.column(3);
    out.row(4) = in.column(4);
    out.row(5) = in.column(5);
    out.row(6) = in.column(6);
    out.row(7) = in.column(7);
    out.row(8) = in.column(8);
    out.row(9) = in.column(9);
    out.row(10) = in.column(10);
    out.row(11) = in.column(11);
    out.row(12) = in.column(12);
    out.row(13) = in.column(13);
    out.row(14) = in.column(14);
    out.row(15) = in.column(15);
}

template <typename T1, typename T2>
CM_INLINE void VNNI_32x16(matrix_ref<T1, 32, 16> in,
                               matrix_ref<T2, 32, 16> out) {
    matrix_ref<T1, 8, 64> in_tmp = in.format<T1, 8, 64>();
    matrix_ref<T1, 8, 64> out_tmp = out.format<T2, 8, 64>();
    #pragma unroll
    for (int idx = 0; idx < 8; idx++) {
        Transpose_16x4(in_tmp[idx].format<T1, 4, 16>(), out_tmp[idx].format<T2, 16, 4>());
    }
}

template <typename T1, typename T2>
CM_INLINE void Transpose_16x8(matrix_ref<T1, 16, 8> in,
                               matrix_ref<T2, 8, 16> out) {
  out.row(0) =  in.column(0);
  out.row(1) =  in.column(1);
  out.row(2) =  in.column(2);
  out.row(3) =  in.column(3);
  out.row(4) =  in.column(4);
  out.row(5) =  in.column(5);
  out.row(6) =  in.column(6);
  out.row(7) =  in.column(7);

}


#ifdef CM_HAS_LSC_UNTYPED_2D
#pragma message (">>>>>> CM_HAS_LSC_UNTYPED_2D OK")
#else
#error "----------------------------"
#endif
#define SystolicDepth 8
#define regN 16
#define regK 32

    extern "C" _GENX_MAIN_ _GENX_FLOAT_CONTROL_(CM_RTE) void gemm(svmptr_t A [[type("svmptr_t")]],
                                  svmptr_t B [[type("svmptr_t")]],
                                  svmptr_t C [[type("svmptr_t")]],
                                  unsigned int M, unsigned int K, unsigned int N) {
        int m_idx = cm_global_id(0) * regM*2;
        int n_idx = cm_global_id(1) * regN;
        matrix<half, 2, regM*regK> matA;
        matrix<uint64_t, regN, regK/4> tempB;

        matrix<uint64_t, regK/4, regN> matB;
        matrix<float, 2, regM*regN> matC = 0;
        //#qA and qB reuse register with A, B
        matrix_ref<int8_t, regM, regK> qmatA0 = matA[0].format<int8_t, 2, regM * regK>().row(0).format<int8_t,regM,regK>();
        matrix_ref<int8_t, regM, regK> qmatA1 = matA[1].format<int8_t, 2, regM * regK>().row(0).format<int8_t,regM,regK>();
        matrix_ref<half, regK/4, regN*4> matB_half = matB.format<half, regK/4, regN*4>().format<half, regK/4, regN*4>();

        //matrix<int8_t,regK, regN> qmatB_vnni;
        matrix_ref<int8_t,regK, regN> qmatB_vnni = matB.format<int8_t, 2, regK*regN>().row(0).format<int8_t,regK, regN>();

        matrix<int32_t, 2, regM*regN> qmatC = 0;

        lsc::block_2d_desc<half, 1, regM, regK> descA{(half*)A, M-1, K*2-1, K*2-1, 0, m_idx};
        lsc::block_2d_desc<uint64_t, 1, regN, regK/4> descB{(uint64_t*)B, N-1, 2*K-1, 2*K-1, 0, n_idx};
        lsc::block_2d_desc<float, 1, regM, regN> descC{(float*)C, M-1, 4*N-1, 4*N-1, n_idx, m_idx};

        for (int kb = 0; kb < K; kb+= BK) {
            #pragma unroll
            for (int ki = 0; ki < BK ; ki+=regK) {
                qmatC = 0;
                //#load B
                descB.set_block_x((kb+ki)/4);
                //cm_load<lsc::LoadOp::Transpose >(matB.format<uint64_t>(), matB);
                cm_load<lsc::LoadOp::Normal>(tempB.format<uint64_t>(), descB);
                Transpose_16x8(tempB, matB);
                //#load A, 16x32 float, each tile 8*32 float
                descA.set_block_x(kb+ki);
                descA.set_block_y(m_idx);
                cm_load<lsc::LoadOp::Normal>(matA[0].format<half>(), descA);
                descA.set_block_y(m_idx+regM);
                cm_load<lsc::LoadOp::Normal>(matA[1].format<half>(), descA);
                //#quantize A per sybolic block,  fp16 scale
                half scaleA[2];
                scaleA[0] = half(127.0) / cm_reduced_max<half>(cm_abs<half>(matA[0]).format<half>());
                scaleA[1] = half(127.0) / cm_reduced_max<half>(cm_abs<half>(matA[1]).format<half>());
                //#quantize B per sybolic block, fp16 scale
                half scaleB = half(127.0) / cm_reduced_max<half>(cm_abs<half>(matB_half).format<half>());

                qmatA0 = cm_mul<int8_t>(matA[0], scaleA[0]);
                qmatA1 = cm_mul<int8_t>(matA[1], scaleA[1]);

                qmatB_vnni.format<int8_t>() = cm_mul<int8_t>(matB_half.format<half>(), scaleB);
                //show_int(qmatB[0].format<int8_t, 32, 16>());
                //#make B to be format of VNNI
                //show_int(qmatB_vnni.format<int8_t, 8, 64>());
                qmatC.row(0).format<int32_t>() = cm_dpas<CM_PRECISION_S8, CM_PRECISION_S8, SystolicDepth, regM>(
                        qmatC.row(0).format<int32_t>(),
                        qmatB_vnni.format<int32_t>(),
                       qmatA0.format<int32_t>());
                qmatC.row(1).format<int32_t>() = cm_dpas<CM_PRECISION_S8, CM_PRECISION_S8, SystolicDepth, regM>(
                        qmatC.row(1).format<int32_t>(),
                        qmatB_vnni.format<int32_t>(),
                        qmatA1.format<int32_t>());
                //show_int(qmatC[0].format<int32_t, regM, regN>());
                float dq_scale[2];

                dq_scale[0] = (float)(1.0f/(scaleA[0]*scaleB));
                dq_scale[1] = (float)(1.0f/(scaleA[1]*scaleB));

                matC.row(0) += cm_mul<float>(qmatC.row(0), dq_scale[0]);
                matC.row(1) += cm_mul<float>(qmatC.row(1), dq_scale[1]);

                //show_float(matC[0].format<half, regM, regN>());

            }
        }
        descC.set_block_y(m_idx);
        cm_store(descC, matC[0].format<float>());
        descC.set_block_y(m_idx+regM);
        cm_store(descC, matC[1].format<float>());
    }
    '''
    REPEAT = 1
    regM = 8
    regN = 16
    BK = 64
    tileM = 2
    tileN = 1

    nthrM = 1
    nthrN = 1

    BM = nthrM* regM * tileM
    BN = nthrN* regN * tileN

    WGS_M = 1
    WGS_N = 1
    global_nthrM = nthrM*WGS_M
    global_nthrN = nthrN*WGS_N

    M = regM*tileM*global_nthrM
    N = regN*tileN*global_nthrN
    K = BK*1
    # np.random.seed(0)
    vRANGE = 2
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
    B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
    C_ref = np.matmul(A, np.transpose(B, (1, 0))).astype(np.float32)
    tA_list = [cl.tensor(A) for _ in range(REPEAT)]
    tB_list = [cl.tensor(B) for _ in range(REPEAT)]
    tC_list = [cl.tensor([M, N], np.dtype(np.float32)) for _ in range(REPEAT)]

    SG_SZ = 16
    kernel =  cl.kernels(src, options=f"-cmc -mdump_asm -g2 -DregM={regM} -DregN={regN} -DBK={BK} -BM={BM} -BN={BN}")

    for i in range(0, REPEAT):
        kernel.enqueue("gemm", [global_nthrM, global_nthrN],[nthrM, nthrN], tA_list[i], tB_list[i],tC_list[i], M, K, N)
    ns = cl.finish()
    flops = M * N * K * 2
    print("----------------------------------------------------")
    print(f'M:{M}, N:{N}, K:{K}')
    print("----------------------------------------------------")
    for time_opt in ns:
        print(f'TPUT: [W/O SLM]:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}')

    compare(C_ref, tC_list[0].numpy(), atol=0.3, rtol=0.02)

cl.profiling(True)


# test tput.
# test_INT8()
test_FP16()