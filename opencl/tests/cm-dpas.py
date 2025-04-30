#!/usr/bin/python3
from clops import cl
import numpy as np


# the decorator `cl.source` turns testcm() into a compiled opencl kernel object
# by invoking clbuild inside.
#
# thanks to `-mdump_asm -g2`, we can find following dumps after running:
#    vISA asm file:   cm_check.visaasm
#    GEN asm file:    cm_check.asm (with interleaved src-code & line-number)
src = r'''
#include <cm/cm.h>
#include <cm/cmtl.h>

#ifndef CM_HAS_LSC_UNTYPED_2D
#error "CM_HAS_LSC_UNTYPED_2D is required!!!!"
#endif

template<int RegM, int RegN>
void dpas_regs(svmptr_t pdst, svmptr_t psrc1, svmptr_t psrc2) {
    matrix<float, RegM*RegN, XXX_AccSize> acc;// The matrix C
    matrix<int32_t, RegN, XXX_Src1Size> src1; // The matrix B
    matrix<int32_t, RegM, XXX_Src2Size> src2; // The matrix A

    lsc::block_2d_desc<short, RegN, 16, 16*RegN> Desc((__global short*)(psrc1), 16, RegN*16, RegN*16*sizeof(half)-1, 0, 0);

    cm_svm_block_read(pdst, acc);
    cm_svm_block_read(psrc1, src1);
    cm_svm_block_read(psrc2, src2);

    //# loop over K dimension
    //# register-blocking MatMul

    #define SystolicDepth 8
    #pragma unroll (RegM)
    for(int m = 0; m < RegM; m++) {
        #pragma unroll (RegN)
        for(int n = 0; n < RegN; n++) {
            acc.row(m*RegN + n) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, XXX_RepeatCount>(
                                            acc.row(m*RegN + n),
                                            src1.row(n),
                                            src2.row(m));
        }
    }
    cm_svm_block_write(pdst, acc);
}

extern "C" _GENX_MAIN_ void cm_dpas_check(
    svmptr_t pdst [[type("svmptr_t")]],
    svmptr_t psrc1 [[type("svmptr_t")]],
    svmptr_t psrc2 [[type("svmptr_t")]]
    ) {
    dpas_regs<1,1>(pdst, psrc1, psrc2);
    printf("Hello,CM\n");
}
'''

'''
register width: 512 bit
so SIMD-16 is required for DPAS in Xe2

The systolic depth must be equal to 8 for all the XMX functions.
The K dimension is calculated as the product of the systolic depth
and the number of operations per channel, so:

hf x hf : K = 8*2
'''

regM = 1
regN = 1

RepeatCount = 8
M = RepeatCount * regM  # RepeatCount 1~8
K = 16                  # 16 x half  = 256 bits
N = 16 * regN           # 16 x float = 512 bits

AccSize = M*N
Src1Size = K*N//2
Src2Size = M*K//2

print(f" {M=} {K=} {N=} {Src1Size=} {Src2Size=} ")
B = np.random.randint(-4,5,[K, N]).astype(np.float16)
A = np.random.randint(-4,5,[M, K]).astype(np.float16)
C = A @ B

def repack_B(B):
    K, N = B.shape
    newB = np.zeros([K//2, N*2], np.float16)
    for k in range(K//2):
        for n in range(N):
            newB[k, 2*n+0] = B[2*k+0, n]
            newB[k, 2*n+1] = B[2*k+1, n]
    return newB

tA = cl.tensor(A)
tB = cl.tensor(repack_B(B))
tC = cl.tensor(np.zeros([M, N], np.float32))

kernels = cl.kernels(src, f"-cmc -mdump_asm -g2 -D XXX_Src1Size={Src1Size} -D XXX_Src2Size={Src2Size} -D XXX_AccSize={AccSize} -D XXX_RepeatCount={RepeatCount}")
kernels.enqueue("cm_dpas_check", [1,1],[1,1], tC, tB, tA)

cl.finish()

C1 = tC.numpy()
print(C1.shape, C.shape)

if not np.allclose(C, C1):
    print(C)
    print(C1)
else:
    print("passed.")
