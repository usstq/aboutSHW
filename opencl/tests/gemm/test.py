#!/usr/bin/python3
from clops import cl
import numpy as np
import time
from clops import compare
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
files = ['kernel.hpp']
src = [open(f'{base_dir}/{file}').read() for file in files]
src.append(
# from kernel.cpp
r'''
#define ATTR [[type("svmptr_t")]]

extern "C" _GENX_MAIN_ void gemm(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    const uint BLOCK_WG_M = _BLOCK_SG_M * _SG_M;
    const uint BLOCK_WG_N = _BLOCK_SG_N * _SG_N;
    const uint size_slm_a = BLOCK_WG_M * _BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = _BLOCK_WG_K * BLOCK_WG_N * sizeof(half);
    cm_slm_init(size_slm_a + size_slm_b);
    auto slm = cm_slm_alloc(size_slm_a + size_slm_b);

    gemm_xmx<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        _SG_SIZE, _BLOCK_SG_M, _BLOCK_SG_N, _SG_M, _SG_N, _BLOCK_WG_K>(slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}

_GENX_MAIN_ void repack_f16(int K, int N, half* src ATTR, half* dst ATTR) {
    repack_f16<_SG_SIZE, _BLOCK_SG_M, _BLOCK_SG_N, _SG_M, _SG_N, _BLOCK_WG_K>(K, N, src, dst);
}
''')
src = '\n'.join(src)

SG_SIZE = 8
BLOCK_SG_M = 16
BLOCK_SG_N = 16
SG_M = 8
SG_N = 8
BLOCK_WG_K = 64
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N

kernels = cl.kernels(src, f'''-cmc -mdump_asm -g2 -D_SG_SIZE={SG_SIZE} -D_BLOCK_SG_M={BLOCK_SG_M} -D_BLOCK_SG_N={BLOCK_SG_N}
                     -D_SG_M={SG_M} -D_SG_N={SG_N} -D_BLOCK_WG_K={BLOCK_WG_K}''')
cl.profiling(True)

M = 128*2
N = 128*2
K = 64*2
B = np.random.randint(-2,2,[K, N]).astype(np.float16)
A = np.random.randint(-2,2,[M, K]).astype(np.float16)
C_ref = np.matmul(A, B)

tA = cl.tensor(A)
tB = cl.tensor(B)
tB_pack = cl.tensor(np.zeros([K, N], np.float16))
tC = cl.tensor(np.zeros([M, N], np.float16))
kernels.enqueue("repack_f16", [N // BLOCK_WG_N, K // BLOCK_WG_K], [1, 1], K, N, tB, tB_pack)
cl.finish()
kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
cl.finish()

compare(C_ref, tC.numpy())
print("----------------------------------------------------")
print(f'M:{M}, N:{N}, K:{K}')
print("----------------------------------------------------")
for i in range(0, 10):
    kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
    ns = cl.finish()
    flops = M * N * K * 2
    for time_opt in ns:
        print(f'TPUT:{flops/time_opt:.1f} GFLOPS, us: {time_opt*1e-3:.1f}')
