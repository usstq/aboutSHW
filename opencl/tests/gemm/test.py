#!/usr/bin/python3
import math
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

#define ABS(x) (x) < 0 ? -(x) : (x)

CM_INLINE void get_mn(uint& id_wg_m, uint& id_wg_n, uint M, uint N, int slice_no, int slice, const int BLOCK_WG_M, const int BLOCK_WG_N) {
    uint id_wg_mn = cm_group_id(0);
    if (slice_no == 0) {
        if (slice == 0) {
            // loop M first, N is shared, total = N/256*M+N
            uint WG_MN = M / BLOCK_WG_M;
            id_wg_m = id_wg_mn % WG_MN;
            id_wg_n = id_wg_mn / WG_MN;
        } else {
            // loop N first, M is shared, total = M/128*N+M
            uint WG_MN = N / BLOCK_WG_N;
            id_wg_n = id_wg_mn % WG_MN;
            id_wg_m = id_wg_mn / WG_MN;
        }
    } else {
        uint wg_x = slice > 0 ? N / BLOCK_WG_N : M / BLOCK_WG_M;
        uint slice_no_abs = ABS(slice_no);
        uint slice_abs = ABS(slice);
        int id_wg_mn_in_reminder = (int)id_wg_mn - (int)(slice_no_abs * slice_abs * wg_x);
        uint slice_idx;
        // in [slice_no x slice]
        if (id_wg_mn_in_reminder < 0) {
            slice_idx = id_wg_mn / (slice_abs * wg_x);
            uint rem_in_slice = id_wg_mn % (slice_abs * wg_x);
            uint x = rem_in_slice % slice_abs;
            uint y = rem_in_slice / slice_abs;
            id_wg_m = slice > 0 ? x + slice_idx * slice_abs : y;
            id_wg_n = slice < 0 ? x + slice_idx * slice_abs : y;
        } else {
            uint slice_rem = slice_abs + (slice_no > 0 ? 1 : -1);
            slice_idx = id_wg_mn_in_reminder / (slice_rem * wg_x);
            uint rem_in_slice = id_wg_mn_in_reminder % (slice_rem * wg_x);
            uint x = rem_in_slice % slice_rem;
            uint y = rem_in_slice / slice_rem;
            id_wg_m = slice > 0 ? x + slice_idx * slice_rem + slice_no_abs * slice_abs : y;
            id_wg_n = slice < 0 ? x + slice_idx * slice_rem + slice_no_abs * slice_abs : y;
        }
    }
}

#if (CM_GENX >= 1270 && CM_GENX < 1280)

_GENX_MAIN_ void gemm(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc) {
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

// if slice_no == 0, slice is linear type, 0 for loop M first and 1 for loop N first
// if slice_no != 0, use wg blocking schema:
//    slice > 0 means the slice is in M dimension else is in N dimension
//    slice_no > 0 means the slice size of reminder will be slice+1 else be slice-1
_GENX_MAIN_ void gemm_nocopy_xe1(SurfaceIndex src_a ATTR_BUF, SurfaceIndex src_b ATTR_BUF, SurfaceIndex dst ATTR_BUF, uint M, uint N, uint K, uint lda, uint ldb, uint ldc,
    int slice_no, int slice) {
    const int SG_SIZE = _SG_SIZE;
    const int BLOCK_REG_N = SG_SIZE;
    // sg blocking
    const int BLOCK_SG_M = _BLOCK_SG_M;
    const int BLOCK_SG_N = _BLOCK_SG_N;
    const int SG_M = _SG_M;
    const int SG_N = _SG_N;
    const int BLOCK_WG_K = _BLOCK_WG_K;	// same in sg

    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    const uint size_slm_a = BLOCK_WG_M * _BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = 0;
    cm_slm_init(size_slm_a + size_slm_b);
    auto slm = cm_slm_alloc(size_slm_a + size_slm_b);

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    gemm_xe1_f16<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        SG_SIZE, BLOCK_SG_M, BLOCK_SG_N, SG_M, SG_N>(id_wg_m, id_wg_n, slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}
#endif

#if CM_GENX >= 1280

_GENX_MAIN_ void gemm_nocopy_xe2(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc,
    int slice_no, int slice) {
    const int SG_SIZE = _SG_SIZE;
    const int BLOCK_REG_N = SG_SIZE;
    // sg blocking
    const int BLOCK_SG_M = _BLOCK_SG_M;
    const int BLOCK_SG_N = _BLOCK_SG_N;
    const int SG_M = _SG_M;
    const int SG_N = _SG_N;
    const int BLOCK_WG_K = _BLOCK_WG_K;	// same in sg

    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    const uint size_slm_a = BLOCK_WG_M * _BLOCK_WG_K * sizeof(half);
    const uint size_slm_b = 0;
    auto slm = 0;

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    gemm_xe2_f16<half, float, half, CmPrecisionType::CM_Precision_FP16, CmPrecisionType::CM_Precision_FP16,
        SG_SIZE, BLOCK_SG_M, BLOCK_SG_N, SG_M, SG_N>(id_wg_m, id_wg_n, slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
}
#endif

''')
src = '\n'.join(src)

devinfo = cl.dev_info()
if 'cl_intel_subgroup_2d_block_io' in devinfo['CL_DEVICE_EXTENSIONS']:
    SG_SIZE = 16
else:
    SG_SIZE = 8

M = 4096*2
N = 2048
#M, N = N, M
K = 128*15
K_GROUP_SIZE = 128
K_GROUPS = K // K_GROUP_SIZE
A = np.random.randint(-1,2,[M, K]).astype(np.float16)
if 1:
    B = np.random.randint(-1,2,[K, N]).astype(np.float16)
else:
    B_q = np.random.randint(0,3,[K, N]).astype(np.int8)
    B_scale = np.random.randint(-4,5,[K_GROUPS, N]).astype(np.float16)
    B_zp = np.random.randint(0,3,[K_GROUPS, N]).astype(np.int8)
    
    B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_GROUP_SIZE, axis=0)) * B_scale.repeat(K_GROUP_SIZE, axis=0)

tA = cl.tensor(A)
tB = cl.tensor(B)
tC = cl.tensor(np.zeros([M, N], np.float16))
tB_trans = cl.tensor(B.transpose().copy())
print("-----------------------------------------------------------------------------")
A_SIZE = M*K*2//1024//1024
B_SIZE = K*N*2//1024//1024
C_SIZE = M*N*2//1024//1024
ALL_SZIE = A_SIZE + B_SIZE + C_SIZE
print(f'{M=} {N=} {K=} {A_SIZE=}MB {B_SIZE=}MB {C_SIZE=}MB {ALL_SZIE=}MB')
print("-----------------------------------------------------------------------------")
flops = M * N * K * 2
cl.profiling(True)

def test_org():
    BLOCK_SG_M = 32 #16
    BLOCK_SG_N = 16 #64
    SG_M = 4
    SG_N = 8
    BLOCK_WG_K = 128 #32
    BLOCK_WG_M = BLOCK_SG_M * SG_M
    BLOCK_WG_N = BLOCK_SG_N * SG_N
    SLM_SIZE_A=BLOCK_WG_M * BLOCK_WG_K * 2 // 1024
    SLM_SIZE_B=BLOCK_WG_N * BLOCK_WG_K * 2 // 1024

    kernels = cl.kernels(src, f'''-cmc -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2 -D_SG_SIZE={SG_SIZE} -D_BLOCK_SG_M={BLOCK_SG_M} -D_BLOCK_SG_N={BLOCK_SG_N}
                        -D_SG_M={SG_M} -D_SG_N={SG_N} -D_BLOCK_WG_K={BLOCK_WG_K}''')
    cl.profiling(True)

    print(f'SLM used A: {SLM_SIZE_A}KB + B: {SLM_SIZE_B}KB = {SLM_SIZE_A+SLM_SIZE_B}KB')
    tB_pack = cl.tensor(np.zeros([K, N], np.float16))
    kernels.enqueue("repack_f16", [N // BLOCK_WG_N, K // BLOCK_WG_K], [1, 1], tB, tB_pack, K, N)
    cl.finish()
    kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
    cl.finish()

    if K < 1024:
        C_ref = np.matmul(A, B)
        compare(C_ref, tC.numpy())
        print(f'{Colors.GREEN}passed{Colors.END}')
    repeat_size = (N/BLOCK_WG_N*M*K+K*N+M*N)*2
    if 1:
        beg = time.time()
        for i in range(0, 100):
            kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
        cl.finish()
        end = time.time()
        time_opt = (end - beg) * 1e9 / 100
        print(f'ORG    TPUT(f16):{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')
    else:
        for i in range(0, 50):
            kernels.enqueue("gemm", [M // BLOCK_SG_M, N // BLOCK_SG_N], [SG_M, SG_N], tA, tB_pack, tC, M, N, K, K, N, N)
            ns = cl.finish()
            #repeat_size1 = (M*K+M/BLOCK_WG_M*K*N+M*N)*2
            for time_opt in ns:
                print(f'TPUT:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

def pack_i8_to_i4(B_q):
    assert B_q.ndim == 2
    K = B_q.shape[0]
    N = B_q.shape[1]
    B_q4 = np.zeros([K, N//2], dtype=np.int8)
    for k in range(K):
        for n in range(N//2):
            even = (B_q[k,2*n]) & 0xF
            odd = (B_q[k,2*n+1]) & 0xF
            B_q4[k,n] = even | (odd << 4)
    return B_q4

def test_v2():
    BLOCK_SG_M = 64 #16
    BLOCK_SG_N = 16 #64
    SG_M = 2
    SG_N = 16
    BLOCK_WG_K = 64 #32
    BLOCK_WG_M = BLOCK_SG_M * SG_M
    BLOCK_WG_N = BLOCK_SG_N * SG_N
    SLM_SIZE_A=BLOCK_WG_M * BLOCK_WG_K * 2 // 1024
    SLM_SIZE_B=0
    kernel_name = 'gemm_nocopy_xe1'
    if SG_SIZE == 16:
        kernel_name = 'gemm_nocopy_xe2'
        BLOCK_SG_N = 32
        SG_N = 8
        SG_M = 4
        BLOCK_WG_M = BLOCK_SG_M * SG_M
        BLOCK_WG_N = BLOCK_SG_N * SG_N
        SLM_SIZE_A=0

    # -dumpVisaOptionsAll -noDPASMacro  -noschedule -dumpSchedule -nocompaction"
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2 -D_SG_SIZE={SG_SIZE} -D_BLOCK_SG_M={BLOCK_SG_M} -D_BLOCK_SG_N={BLOCK_SG_N}
                        -D_SG_M={SG_M} -D_SG_N={SG_N} -D_BLOCK_WG_K={BLOCK_WG_K}''')

    print(f'SLM used A: {SLM_SIZE_A}KB + B: {SLM_SIZE_B}KB = {SLM_SIZE_A+SLM_SIZE_B}KB')
    # loop N first:[0, 1], loop M first:[0, 0]; block M first[slice_no, slice(>0)], block N first[slice_no, slice(<0)]
    #default linear
    slice_no = 0
    slice = N < M
    if 1:
        block_m = M // BLOCK_WG_M
        block_n = N // BLOCK_WG_N
        bias = BLOCK_WG_M / BLOCK_WG_N
        devinfo = cl.dev_info()
        eu_xecore = 16 if SG_SIZE == 8 else 8
        xecores = devinfo["CL_DEVICE_MAX_COMPUTE_UNITS"] // eu_xecore
        sm = math.sqrt(xecores * 2 / bias)
        sn = math.sqrt(xecores * 2 * bias)
        if N < M:
            s0 = math.ceil(sn)
            if block_n % s0 == 0:
                slice = -s0
                slice_no = block_n // slice
            else:
                if s0 * 1.5 < block_n:
                    rem = block_n % s0
                    expect_slice_no = (block_n + s0 - 1) // s0 - (s0 - rem)
                    if expect_slice_no > 0:
                        assert expect_slice_no >= 0, f'{block_n=} {expect_slice_no=} {s0=}'
                        slice = -s0
                        slice_no = -expect_slice_no
        else:
            # TODO: big weight matrix?
            pass
    print(f'{xecores=} {block_m=} {block_n=} {slice=} {slice_no=}')
    for i in range(50):
        kernels.enqueue(kernel_name, [M // BLOCK_WG_M * N // BLOCK_WG_N * SG_N, SG_M], [SG_N, SG_M], tA, tB_trans, tC, M, N, K, K, K, N, slice_no, slice)
    cl.finish()

    if K < 1024:
        C_ref = np.matmul(A, B)
        compare(C_ref, tC.numpy())
        print(f'{Colors.GREEN}passed{Colors.END}')
    repeat_size = (N/BLOCK_WG_N*M*K+K*N+M*N)*2
    if 0:
        beg = time.time()
        for i in range(0, 100):
            kernels.enqueue(kernel_name, [M // BLOCK_WG_M * N // BLOCK_WG_N * SG_N, SG_M], [SG_N, SG_M], tA, tB_trans, tC, M, N, K, K, K, N, slice_no, slice)
        cl.finish()
        end = time.time()
        time_opt = (end - beg) * 1e9 / 100
        print(f'NOCOPY TPUT(f16):{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')
    else:
        for i in range(0, 100):
            kernels.enqueue(kernel_name, [M // BLOCK_WG_M * N // BLOCK_WG_N * SG_N, SG_M], [SG_N, SG_M], tA, tB_trans, tC, M, N, K, K, K, N, slice_no, slice)
            ns = cl.finish()
            #repeat_size1 = (M*K+M/BLOCK_WG_M*K*N+M*N)*2
            for time_opt in ns:
                print(f'(CUR)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f}:{repeat_size/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

def test_onednn(M, K, N, K_group_size, w_dtype):
    tP1 = cl.tensor()

    if w_dtype == cl.onednn_dtype.f16:
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, cl.onednn_dtype.undef, M, K, N, -1, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.f16, tB_trans, cl.tensor(), cl.tensor(), cl.tensor())
        linear.forward(tA, tC, cl.tensor())

        cl.finish()
        # C = A @ B
        # assert(np.allclose(C, tC.numpy()))
        # print('done')

    if w_dtype == cl.onednn_dtype.u4:

        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        C = A @ B
        print("ref is calculated!")

        # pack weight into 4bit
        B_q4 = pack_i8_to_i4(B_q.transpose().copy())
        B_zp4 = pack_i8_to_i4(B_zp)

        tBq4 = cl.tensor(B_q4)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp4)
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, cl.onednn_dtype.undef, M, K, N, K_group_size, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.u4, tBq4, tBs, tBz, cl.tensor())
        linear.forward(tA, tC, cl.tensor())
        cl.finish()
        assert(np.allclose(C, tC.numpy()))
        print('done')
        
    name = 'f16' if w_dtype == cl.onednn_dtype.f16 else 'u4 '
    beg = time.time()
    for i in range(0, 100):
        linear.forward(tA, tC, tP1)
    ns = linear.finish()
    end = time.time()
    for i, time_opt in enumerate(ns):
        print(f'(DNN)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

    # time_opt = (end - beg) * 1e9 / 100
    # print(f'ONEDNN TPUT({name}):{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

if 0:
    test_org()

test_v2()
print('============================================')
if os.environ['OV_BUILD_PATH']:
    test_onednn(M, K, N, 0, cl.onednn_dtype.f16)
    #test_onednn(M, K, N, K_GROUP_SIZE, cl.onednn_dtype.u4)
