import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

# prototyping CM kernels
from clops import cl
import numpy as np
import time

# ugemm perform gemm in 1 GPU HW thread
# load submatrix from A & B
# 

N_REGM = 2
N_REGN = 2
BLK_M = 8*N_REGM
BLK_N = 16*N_REGN

src=r'''
#pyeval f"#define N_REGM {N_REGM}"
#pyeval f"#define N_REGN {N_REGN}"

#ifndef CM_HAS_LSC_UNTYPED_2D 
#error "need CM_HAS_LSC_UNTYPED_2D has not defined"
#endif

template<typename T, int M, int N>
void show(const char * name, const matrix<T, M, N> mat) {
    //printf("%s [\n", name);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%6.0f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

#define SystolicDepth 8
#define RepeatCount 8
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount
#define REG_N 16

//# ugemm's K must be integer multiple of REG_K(16)
//# ugemm's M must be integer multiple of REG_M(8)
//# ugemm's N must be integer multiple of REG_N(16)

template<int num_REGM, int num_REGN>
void ugemm(matrix_ref<float, num_REGM * num_REGN, REG_M*REG_N> res,
           lsc::block_2d_desc<half, 1, REG_M, REG_K>& b2dA,
           lsc::block_2d_desc<half, 1, REG_K, REG_N>& b2dB,
           int m0, int n0, int total_K) {
    matrix<half, num_REGM, REG_M*REG_K> Amat;
    matrix<half, num_REGN, REG_K*REG_N> Bmat;

    for(int k = 0; k < total_K; k += REG_K) {
        b2dA.set_block_x(k);
        b2dB.set_block_y(k);
        
        #pragma unroll (num_REGM)
        for(int m = 0; m < num_REGM; m++) {
            cm_load<lsc::Normal>(Amat[m], b2dA.set_block_y(m0 + m*REG_M));
        }

        #pragma unroll (num_REGN)
        for(int n = 0; n < num_REGN; n++) {
            cm_load<lsc::VNNI>(Bmat[n], b2dB.set_block_x(n0 + n*REG_N));
        }

        #pragma unroll (num_REGM)
        for(int m = 0; m < num_REGM; m++) {
            #pragma unroll (num_REGN)
            for(int n = 0; n < num_REGN; n++) {
                auto cid = m*num_REGN + n;
                res[cid] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            res[cid].format<float>(),
                            Bmat[n].format<int32_t>(),
                            Amat[m].format<int32_t>());
            }
        }
    }
}

extern "C" _GENX_MAIN_ void cm_test(
    const half* A [[type("svmptr_t")]], int widthA, int heightA,
    const half* B [[type("svmptr_t")]], int widthB, int heightB,
    float* C [[type("svmptr_t")]], int widthC, int heightC,
    int m0,
    int n0,
    int total_K
    ) {
    //# https://github.com/intel/cm-compiler/blob/cmc_monorepo_110/clang/lib/Headers/cm/include/cm/lsc/block2d.h
    lsc::block_2d_desc<half, 1, REG_M, REG_K> b2dA((half*)A, heightA - 1, widthA*sizeof(half) - 1, widthA*sizeof(half) - 1, 0, 0);
    lsc::block_2d_desc<half, 1, REG_K, REG_N> b2dB((half*)B, heightB - 1, widthB*sizeof(half) - 1, widthB*sizeof(half) - 1, 0, 0);
    
    matrix<float, N_REGM * N_REGN, REG_M*REG_N> Cmat = 0;
    ugemm<N_REGM, N_REGN>(Cmat, b2dA, b2dB, m0, n0, total_K);

    lsc::block_2d_desc<float, 1, REG_M, REG_N> b2dC((float*)C, heightC - 1, widthC*sizeof(float) - 1, widthC*sizeof(float) - 1, 0, 0);

    for(int m = 0; m < N_REGM; m++) {
        for(int n = 0; n < N_REGN; n++) {
            auto cid = m*N_REGN + n;
            cm_store(b2dC.set_block_y(m0 + m*REG_M).set_block_x(n0 + n*REG_N), Cmat[cid].format<float>());
            //show("Cmat=", Cmat[cid].format<float, REG_M, REG_N>());
        }
    }
}

'''

GWS=[1]
LWS=[1]
low = -4
high = 5
total_K = 64
total_N = 1024
total_M = 256
A = np.random.randint(low,high,[total_M, total_K]).astype(np.float16)
B = np.random.randint(low,high,[total_K, total_N]).astype(np.float16)

#A[:, 64:] = 0
#B[64:,:] = 0

C = A.astype(np.float32) @ B.astype(np.float32)

t_A = cl.tensor(A)
t_B = cl.tensor(B)
t_C = cl.tensor([total_M, total_N], np.dtype(np.float32))

def pyeval(src):
    result_src = ""
    for line in src.splitlines():
        if line.startswith("#pyeval"):
            new_line = eval(line[8:])
            result_src += new_line + "\n"
            # print(f"[pyeval] {new_line}")
        else:
            result_src += line + "\n"
    return result_src

m0 = 7
n0 = 8

subA = A[m0:m0+BLK_M, :]
subB = B[:, n0:n0+BLK_N]
subC = subA @ subB
print("================ subA")
print(subA)
print("================ subB")
print(subB)
print("================ subC")
print(subC)

cm_kernels = cl.kernels(pyeval(src), f"-cmc -mdump_asm -g2 ")
cm_kernels.enqueue("cm_test", GWS, LWS,
                    t_A, total_K, total_M,
                    t_B, total_N, total_K,
                    t_C, total_N, total_M, 
                    m0, n0, total_K)
cl.finish()
C1 = t_C.numpy()
subC1 = C1[m0:m0+BLK_M, n0:n0+BLK_N]
print(subC1)
assert np.allclose(subC, subC1)
#cm_kernels.enqueue("cm_test", GWS, LWS, q_len, kv_len, t_q, t_k, t_v, t_out, t_mask)
