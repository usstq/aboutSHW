#!/usr/bin/python3
from clops import cl
import numpy as np
import time
from clops import compare
import os
from clops.utils import Colors
#input(f'{os.getpid()} xxx')
''' config for this file
run:
CM_FE_DIR=/mnt/luocheng/aboutSHW/opencl/cm python tests/gemm/test.py
onednn:
export OV_BUILD_PATH=/mnt/luocheng/2025.1/openvino/build
with cliloader:
export PATH=$PATH:/mnt/luocheng/opencl-intercept-layer/install/bin/
CM_FE_DIR=`pwd`/cm cliloader -dv -cdt  --dump-dir ./dump/ python tests/gemm/test.py
'''
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

_GENX_MAIN_ void repack_f16(half* src ATTR, half* dst ATTR, int K, int N) {
    repack_f16<_SG_SIZE, _BLOCK_SG_M, _BLOCK_SG_N, _SG_M, _SG_N, _BLOCK_WG_K>(src, dst, K, N);
}

_GENX_MAIN_ void gemm_nocopy(SurfaceIndex src_a ATTR_BUF, SurfaceIndex src_b ATTR_BUF, SurfaceIndex dst ATTR_BUF, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
    const int SG_SIZE = 8;
    const int BLOCK_REG_N = SG_SIZE;
    // sg blocking
    const int BLOCK_SG_M = 64;
    const int BLOCK_SG_N = 16;
    const int SG_M = 2;
    const int SG_N = 16;
    const int BLOCK_WG_K = 64;	// same in sg

    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    const uint size_slm_a = BLOCK_WG_M * _BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = 0;
    cm_slm_init(size_slm_a + size_slm_b);
    auto slm = cm_slm_alloc(size_slm_a + size_slm_b);

    gemm_64x16_no_slmb<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        SG_SIZE, BLOCK_SG_M, BLOCK_SG_N, SG_M, SG_N>(slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}
''')
src = '\n'.join(src)

SG_SIZE = 8
use_v1 = False
if use_v1:
    BLOCK_SG_M = 32 #16
    BLOCK_SG_N = 16 #64
    SG_M = 4
    SG_N = 8
    BLOCK_WG_K = 128 #32
    BLOCK_WG_M = BLOCK_SG_M * SG_M
    BLOCK_WG_N = BLOCK_SG_N * SG_N
    SLM_SIZE_A=BLOCK_WG_M * BLOCK_WG_K * 2 // 1024
    SLM_SIZE_B=BLOCK_WG_N * BLOCK_WG_K * 2 // 1024
else:
    BLOCK_SG_M = 64 #16
    BLOCK_SG_N = 16 #64
    SG_M = 2
    SG_N = 16
    BLOCK_WG_K = 64 #32
    BLOCK_WG_M = BLOCK_SG_M * SG_M
    BLOCK_WG_N = BLOCK_SG_N * SG_N
    SLM_SIZE_A=BLOCK_WG_M * BLOCK_WG_K * 2 // 1024
    SLM_SIZE_B=0

kernels = cl.kernels(src, f'''-cmc -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2 -D_SG_SIZE={SG_SIZE} -D_BLOCK_SG_M={BLOCK_SG_M} -D_BLOCK_SG_N={BLOCK_SG_N}
                     -D_SG_M={SG_M} -D_SG_N={SG_N} -D_BLOCK_WG_K={BLOCK_WG_K}''')
cl.profiling(True)

M = 4096*2
N = 2048
K = 128*15
B = np.random.randint(-1,2,[K, N]).astype(np.float16)
A = np.random.randint(-1,2,[M, K]).astype(np.float16)
print(f'SLM used A: {SLM_SIZE_A}KB + B: {SLM_SIZE_B}KB = {SLM_SIZE_A+SLM_SIZE_B}KB')
tA = cl.tensor(A)
tB = cl.tensor(B)
tB_pack = cl.tensor(np.zeros([K, N], np.float16))
tB_trans = cl.tensor(B.transpose().copy())
tC = cl.tensor(np.zeros([M, N], np.float16))
if use_v1:
    kernels.enqueue("repack_f16", [N // BLOCK_WG_N, K // BLOCK_WG_K], [1, 1], tB, tB_pack, K, N)
    cl.finish()
    kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
    cl.finish()
else:
    kernels.enqueue("gemm_nocopy", [N // BLOCK_SG_N, M // BLOCK_SG_M], [SG_N, SG_M], tA, tB_trans, tC, M, N, K, K, K, N)
    cl.finish()

if K < 1024:
    C_ref = np.matmul(A, B)
    compare(C_ref, tC.numpy())
    print(f'{Colors.GREEN}passed{Colors.END}')
print("-----------------------------------------------------------------------------")
A_SIZE = M*K*2//1024//1024
B_SIZE = K*N*2//1024//1024
C_SIZE = M*N*2//1024//1024
ALL_SZIE = A_SIZE + B_SIZE + C_SIZE
print(f'{M=} {N=} {K=} {A_SIZE=}MB {B_SIZE=}MB {C_SIZE=}MB {ALL_SZIE=}MB')
print("-----------------------------------------------------------------------------")
flops = M * N * K * 2
repeat_size = (N/BLOCK_WG_N*M*K+K*N+M*N)*2
if 1:
    beg = time.time()
    for i in range(0, 100):
        if use_v1:
            kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
        else:
            kernels.enqueue("gemm_nocopy", [N // BLOCK_SG_N, M // BLOCK_SG_M], [SG_N, SG_M], tA, tB_trans, tC, M, N, K, K, K, N)
    cl.finish()
    end = time.time()
    time_opt = (end - beg) * 1e9 / 100
    print(f'CUR    TPUT:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')
else:
    for i in range(0, 50):
        kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
        ns = cl.finish()
        #repeat_size1 = (M*K+M/BLOCK_WG_M*K*N+M*N)*2
        for time_opt in ns:
            print(f'TPUT:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')
#print(f'loop M first {repeat_size//1024//1024/1024:.2f} GB loop N first {repeat_size1//1024//1024/1024:.2f} GB')

def test_onednn(M, K, N, K_group_size, w_dtype):
    tP1 = cl.tensor()

    if w_dtype == cl.onednn_dtype.f16:
        #B = np.random.randint(-1,2,[K, N]).astype(np.float16)
        #C = A @ B
        #print("ref is calculated!")

        tB = cl.tensor(B.transpose().copy())
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, -1, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.f16, tB, cl.tensor(), cl.tensor())
        linear.forward(tA, tC, tP1)

        cl.finish()
        #C_ref = np.matmul(A, B)
        #compare(C_ref, tC.numpy())
        beg = time.time()
        for i in range(0, 100):
            linear.forward(tA, tC, tP1)
        ns = cl.finish()
        end = time.time()
        time_opt = (end - beg) * 1e9 / 100
        print(f'ONEDNN TPUT:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

if os.environ['OV_BUILD_PATH']:
    test_onednn(M, K, N, 0, cl.onednn_dtype.f16)
