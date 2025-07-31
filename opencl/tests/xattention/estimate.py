#from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F

import numpy as np

from clops import cl
import clops
import time
from clops import compare
import os
from clops.utils import Colors

cl.profiling(True)

def get_ref(Q: torch.Tensor, K: torch.Tensor, block_size, S, threshold=0.9, causal=True, wg_n=256):
    # Q, K, V shape = (B, H, L, d)
    B, num_kv_head, k_len, d = K.shape
    B, num_q_head, q_len, d = Q.shape
    assert num_q_head % num_kv_head == 0
    H = num_q_head
    K = K.repeat_interleave(num_q_head // num_kv_head, 1)
    
    Q_block = (q_len + block_size -1) // block_size
    K_block = (k_len + block_size -1) // block_size

    # Antidiagonal Reshaping (Stride-S)
    Q_resh = torch.cat([Q[:, :, S-1-i::S, :] for i in range(S)], dim=-1)
    K_resh = torch.cat([K[:, :, i::S, :] for i in range(S)], dim=-1)

    # Attention Estimation
    A = Q_resh @ K_resh.transpose(-1, -2)
    #print(f'{A.shape=} {Q_resh.shape=} {K_resh.shape=}')
    A = A / math.sqrt(d) / S

    ################# TODO: fuse gemm+softmax begin
    # [1, 32, 256, 8192]
    B, _, q_len_strided, k_len_strided = A.shape
    # [1, 32, 256, 1]
    A_max = torch.max(A, dim=-1, keepdim=True)[0]                               # real max needed for compensation [b, num_q_head, q_len_strided, 1]
    # [1, 32, 256, 32, 256]
    A_5d = A.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_n, wg_n)
    # [1, 32, 256, 32, 1]
    A_5d_max = torch.max(A_5d, dim=-1, keepdim=True)[0]                         # local wg max needed for compensation [b, num_q_head, q_len_strided, k_len_strided // wg_n, 1]
    # [1, 32, 256, 32, 256]
    A_exp_partial = torch.exp(A_5d - A_5d_max)
    # [1, 32, 256, 32, 32, 8]
    A_exp_partial = A_exp_partial.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_n, wg_n // (block_size // S), block_size // S)
    # [1, 32, 256, 32, 32]
    A_exp_partial_sum = A_exp_partial.sum(dim=-1)                               # local wg sum

    # stage 2:
    # [1, 32, 256, 32, 32] * [1, 32, 256, 32, 1]
    A_exp_horz_sum = A_exp_partial_sum * torch.exp(A_5d_max - A_max.unsqueeze(dim=-1))
    # [1, 32, 256, 32 * 32]
    A_exp_horz_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided, -1)
    A_exp_horz_sum = A_exp_horz_sum / A_exp_horz_sum.sum(dim=-1, keepdim=True)
    A_exp_vert_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided // (block_size // S), block_size // S, -1)
    A_exp_vert_sum = A_exp_vert_sum.sum(dim=-2)
    A_sum = F.softmax(A, dim=-1).reshape(B, num_q_head, q_len_strided // (block_size // S), (block_size // S), k_len_strided // (block_size // S), (block_size // S))
    A_sum = A_sum.sum(dim=-1).sum(dim=-2)
    if not torch.allclose(A_sum, A_exp_vert_sum, atol=0.01, rtol=0.01):
        print(f'{A_sum=}\n{A_exp_vert_sum=}')
        assert 0
    ###################### fuse gemm+softmax end

    #A = F.softmax(A, dim=-1, dtype=torch.float32).to(Q.dtype)
    return A


# q: [B, Hq, L_q, S]
# k: [B, Hk, L_k, S]
def test(q:torch.Tensor, k:torch.Tensor, block_size=128, threshold=0.9, stride=16, causal=True):
    B, Hq, Lq, S = q.shape
    _, Hk, Lk, _ = k.shape
    M = Lq // stride
    N = Lk // stride
    K = stride * S

    tQ = cl.tensor(q.detach().numpy())
    tK = cl.tensor(k.detach().numpy())
    tC = cl.tensor(np.zeros([B, Hq, M, N], np.float16))

    # kernel
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = ['estimate.hpp']
    src = [open(f'{base_dir}/{file}').read() for file in files]
    src.append(
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

    _GENX_MAIN_ void gemm_qk(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc, int slice_no, int slice) {
        const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
        const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
        const uint size_slm_b = 0;
        auto slm = 0;
        uint b_hq = cm_group_id(2);
        uint b = b_hq / HQ;
        uint hq = b_hq % HQ;
        uint hk = hq / (HQ / HK);

        static_assert(HQ % HK == 0, "HQ must be multiple of HK");

        uint id_wg_m, id_wg_n;
        get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

        src_a += (b * HQ + hq) * M * lda * (uint)sizeof(half);
        src_b += (b * HK + hk) * N * ldb * (uint)sizeof(half);
        dst += (b * HQ + hq) * M * ldc * (uint)sizeof(half);
#if (BLOCK_SG_M == 64 && BLOCK_SG_N == 32)
            gemm_qk_8x2_xe2(id_wg_m, id_wg_n, slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
#elif (BLOCK_SG_M == 32 && BLOCK_SG_N == 32)
            gemm_qk_4x2_xe2(id_wg_m, id_wg_n, slm, src_a, src_b, dst, M, N, K, lda, ldb, ldc);
#else
            static_assert(false, "BLOCK_SG_M and BLOCK_SG_N not support");
#endif
    }

#if 0
    _GENX_MAIN_ void gemm_qk_gqa(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t dst ATTR, uint M, uint N, uint K, uint lda, uint ldb, uint ldc, int slice_no, int slice) {
        const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
        const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
        const uint size_slm_b = 0;
        auto slm = 0;
        uint b_hk = cm_group_id(2);
        uint b = b_hk / HK;
        uint hk = b_hk % HK;
        uint hq = hk * (HQ / HK);

        static_assert(HQ % HK == 0, "HQ must be multiple of HK");

        uint id_wg_m, id_wg_n;
        get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

        src_b += (b * HK + hk) * N * ldb * (uint)sizeof(half);
        for (uint i = 0; i < HQ / HK; i++) {
            svmptr_t cur_a = src_a + (b * HQ + hq + i) * M * lda * (uint)sizeof(half);
            svmptr_t cur_dst = dst + (b * HQ + hq + i) * M * ldc * (uint)sizeof(half);
            gemm_qk_8x2_xe2(id_wg_m, id_wg_n, slm, cur_a, src_b, cur_dst, M, N, K, lda, ldb, ldc);
        }
    }
#endif
    ''')
    src = '\n'.join(src)

    kernel_name = 'gemm_qk'
    BLOCK_SG_M = 64 #32
    BLOCK_SG_N = 32
    SG_M = 2
    SG_N = 4
    BLOCK_WG_M = BLOCK_SG_M * SG_M
    BLOCK_WG_N = BLOCK_SG_N * SG_N

    ################# TODO: fuse gemm+softmax begin
    #assert BLOCK_WG_N % block_size == 0, "a block must be in a workgroup"

    jit_option = '-abortonspill -noschedule '
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2
                       -DSTRIDE={stride} -DHQ={Hq} -DHK={Hk} -DHEAD_SIZE={S} -DSG_M={SG_M} -DSG_N={SG_N} -DBLOCK_SG_N={BLOCK_SG_N} -DBLOCK_SG_M={BLOCK_SG_M} -DINV_S={1 / math.sqrt(S) / stride}''')
    # loop N first:[0, 1], loop M first:[0, 0]; block M first[slice_no, slice(>0)], block N first[slice_no, slice(<0)]
    #default linear
    slice_no = 0
    slice = N < M
    if 1:
        block_m = M // BLOCK_WG_M
        block_n = N // BLOCK_WG_N
        bias = BLOCK_WG_M / BLOCK_WG_N
        devinfo = cl.dev_info()
        eu_xecore = 8
        xecores = devinfo["CL_DEVICE_MAX_COMPUTE_UNITS"] // eu_xecore
        sm = math.sqrt(xecores * 2 / bias)
        sn = math.sqrt(xecores * 2 * bias)
        if 1 or N < M:
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
    slice_no = 0
    slice = 0
    print(f'{xecores=} {block_m=} {block_n=} {slice=} {slice_no=}')

    for i in range(10):
        kernels.enqueue(kernel_name, [M // BLOCK_WG_M * N // BLOCK_WG_N * SG_N, SG_M, B * Hq], [SG_N, SG_M, 1], tQ, tK, tC, M, N, K, K, K, N, slice_no, slice)
    cl.finish()

    C_ref = get_ref(q, k, block_size=block_size, S=stride, threshold=threshold, causal=causal, wg_n=BLOCK_WG_N)
    compare(C_ref.detach().numpy(), tC.numpy())
    print(f'{Colors.GREEN}passed{Colors.END}')
    flops = B * Hq * M * N * K * 2
    for i in range(0, 100):
        kernels.enqueue(kernel_name, [M // BLOCK_WG_M * N // BLOCK_WG_N * SG_N, SG_M, B * Hq], [SG_N, SG_M, 1], tQ, tK, tC, M, N, K, K, K, N, slice_no, slice)
    ns = cl.finish()
    for i, time_opt in enumerate(ns):
        print(f'(CUR)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

    #mask = torch.zeros([1, 1], dtype=torch.bool)
    #return mask

def main():
    # 106 T/s:
    # bsz = 1
    # q_head = 1
    # k_head = 1
    # q_len = 1024*16*2
    # k_len = 1024*128
    # dim = 128
    # block_size = 128
    # stride = 16

    # 84 T/s:
    bsz = 1
    q_head = 32
    k_head = 4
    q_len = 1024*4*1  # if q_len=1024*4*2 ==> 95 T/s
    k_len = 1024*128
    dim = 128
    block_size = 128
    stride = 16

    assert (q_len // stride) >= 256 and (k_len // stride) >= 256, "minimum block size should be 256*256"

    if True:
        q = torch.randn((bsz,q_head,q_len,dim),dtype=torch.float32)
        k = torch.randn((bsz,k_head,k_len,dim),dtype=torch.float32)

        np.save("q.npy", q.numpy())
        np.save("k.npy", k.numpy())
    else:
        q = torch.from_numpy(np.load("q.npy"))
        k = torch.from_numpy(np.load("k.npy"))

    q = q.to(torch.float16)
    k = k.to(torch.float16)

    test(q, k, block_size=block_size, threshold=0.9, stride=stride, causal=False)


if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    # requirements:
    # num_q_head == num_kv_head
    # chunk size alignment
    # causal_mask
    main()