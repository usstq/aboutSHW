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

def get_gemm_ref(Q: torch.Tensor, K: torch.Tensor, block_size, S, threshold=0.9, causal=True, wg_x=256):
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
    # [1, 32, 256, 64, 128]
    A_5d = A.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_x, wg_x)
    # [1, 32, 256, 64, 1]
    A_5d_max = torch.max(A_5d, dim=-1, keepdim=True)[0]                         # local wg max needed for compensation [b, num_q_head, q_len_strided, k_len_strided // wg_n, 1]
    # [1, 32, 64, 256]
    A_5d_max_ret = torch.transpose(A_5d_max, -2, -3).squeeze(dim=-1)
    # [1, 32, 256, 64, 128]
    A_exp_partial = torch.exp(A_5d - A_5d_max)
    # [1, 32, 256, 64, 16, 8]
    A_exp_partial = A_exp_partial.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_x, wg_x // (block_size // S), block_size // S)
    # [1, 32, 256, 64, 16]
    A_exp_partial_sum = A_exp_partial.sum(dim=-1)                               # local wg sum
    # [1, 32, 256, 64 * 16]
    A_exp_partial_sum_ret = torch.reshape(A_exp_partial_sum, (B, num_q_head, q_len_strided, -1))

    # stage 2:
    # [1, 32, 256, 64, 16] * [1, 32, 256, 64, 1]
    A_exp_horz_sum = A_exp_partial_sum * torch.exp(A_5d_max - A_max.unsqueeze(dim=-1))
    # [1, 32, 256, 64 * 16]
    A_exp_horz_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided, -1)
    A_exp_horz_sum = A_exp_horz_sum / A_exp_horz_sum.sum(dim=-1, keepdim=True)
    A_exp_vert_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided // (block_size // S), block_size // S, -1)
    A_exp_vert_sum = A_exp_vert_sum.sum(dim=-2)
    A_sum = F.softmax(A, dim=-1).reshape(B, num_q_head, q_len_strided // (block_size // S), (block_size // S), k_len_strided // (block_size // S), (block_size // S))
    A_sum = A_sum.sum(dim=-1).sum(dim=-2)
    # [1, 32, 32, 64 * 16]
    if not torch.allclose(A_sum, A_exp_vert_sum, atol=0.01, rtol=0.01):
        print(f'{A_sum=}\n{A_exp_vert_sum=}')
        assert 0
    ###################### fuse gemm+softmax end

    #A = F.softmax(A, dim=-1, dtype=torch.float32).to(Q.dtype)
    return torch.transpose(A, -1, -2), A_max.squeeze(-1), A_5d_max_ret, A_exp_partial_sum_ret, A_sum

def get_softmax_ref(A: torch.Tensor, block_size, S, causal=True):
    # [1, 32, 256, 8192]
    B, num_q_head, q_len_strided, k_len_strided = A.shape
    A_sum = F.softmax(A, dim=-1).reshape(B, num_q_head, q_len_strided // (block_size // S), (block_size // S), k_len_strided // (block_size // S), (block_size // S))
    A_sum = A_sum.sum(dim=-1).sum(dim=-2)
    return A_sum

# kernel
base_dir = os.path.dirname(os.path.abspath(__file__))
files = ['estimate.hpp', 'find_block.hpp']
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

_GENX_MAIN_ void gemm_qk(svmptr_t src_a ATTR, svmptr_t src_b ATTR, svmptr_t c_max ATTR, svmptr_t c_max_wg ATTR, svmptr_t c_exp_partial_sum ATTR, 
    uint M, uint N, uint K, uint lda, uint ldb, uint ldc, int slice_no, int slice) {
    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    const uint size_slm_b = 0;
    uint b_hq = cm_group_id(2);
    uint b = b_hq / HQ;
    uint hq = b_hq % HQ;
    uint hk = hq / (HQ / HK);
    const uint slm_size = SG_M * BLOCK_WG_N * sizeof(half);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    static_assert(HQ % HK == 0, "HQ must be multiple of HK");

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    src_a += (b * HK + hk) * M * lda * (uint)sizeof(half);
    src_b += (b * HQ + hq) * N * ldb * (uint)sizeof(half);

    // c_max: [b, hq, n_pad]
    // c_max_wg: [b, hq, m_groups, n_pad]
    // c_exp_partial_sum: [b, hq, n_pad, m_groups*BLOCK_WG_M/(BLOCK_SIZE/STRIDE)]
    uint n_pad = (N + BLOCK_WG_N - 1) / BLOCK_WG_N * BLOCK_WG_N;
    c_max += (b * HQ + hq) * n_pad * (uint)sizeof(half);
    uint m_groups = (M + BLOCK_WG_M - 1) / BLOCK_WG_M;
    c_max_wg += (b * HQ + hq) * m_groups * n_pad * (uint)sizeof(half);

    const uint sum_per_n_token_in_block = BLOCK_SIZE / STRIDE;
    const uint m_after_sum_in_group = BLOCK_WG_M / sum_per_n_token_in_block;
    const uint m_after_sum_pad = m_after_sum_in_group * m_groups;
    c_exp_partial_sum += (b * HQ + hq) * m_after_sum_pad * n_pad * (uint)sizeof(half);

    gemm_kq_8x2_xe2(id_wg_m, id_wg_n, slm, src_a, src_b, c_max, c_max_wg, c_exp_partial_sum, M, N, K, lda, ldb, ldc);
}

_GENX_MAIN_ void find_block(svmptr_t c_max ATTR, svmptr_t c_max_wg ATTR, svmptr_t c_exp_partial_sum ATTR, svmptr_t c_sum ATTR, uint m_pad, uint n_after_sum_pad) {
    // c_max:             [b, hq, m_pad]
    // c_max_wg:          [b, hq, m_groups, m_pad]
    // c_exp_partial_sum: [b, hq, m_pad, n_after_sum_pad]
    // c_sum:             [b, hq, m_pad/TOKEN_IN_BLOCK, n_after_sum_pad]
    // [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
    // global:            [m_pad/TOKEN_IN_BLOCK, hq, b]
    const int TOKEN_IN_BLOCK = BLOCK_SIZE / STRIDE;
    const int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;
    uint m = cm_group_id(0);
    uint hq = cm_group_id(1);
    uint b = cm_group_id(2);
    c_max += (b * HQ + hq) * m_pad * (uint)sizeof(half);
    c_max_wg += (b * HQ + hq) * (n_after_sum_pad / TOKEN_SHARE_MAX) * m_pad * (uint)sizeof(half);
    c_exp_partial_sum += (b * HQ + hq) * m_pad * n_after_sum_pad * (uint)sizeof(half);
    c_sum += (b * HQ + hq) * m_pad / TOKEN_IN_BLOCK * n_after_sum_pad * (uint)sizeof(half);
    find(m, c_max, c_max_wg, c_exp_partial_sum, c_sum, m_pad, n_after_sum_pad);
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
SG_M = 4
SG_N = 4
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N
KV_BLOCK_SIZE = 256

HQ = 32
HK = 4
HEAD_SIZE = 128
STRIDE = 16
BLOCK_SIZE = 128
Q_LEN = 1024*4*1  # if q_len=1024*4*2 ==> 95 T/s
K_LEN = 1024*128

################# TODO: fuse gemm+softmax begin
#assert BLOCK_WG_N % block_size == 0, "a block must be in a workgroup"

jit_option = '-abortonspill -noschedule '
kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2
                    -DSTRIDE={STRIDE} -DHQ={HQ} -DHK={HK} -DHEAD_SIZE={HEAD_SIZE} -DSG_M={SG_M} -DSG_N={SG_N} -DBLOCK_SG_N={BLOCK_SG_N} -DBLOCK_SG_M={BLOCK_SG_M}
                    -DBLOCK_SIZE={BLOCK_SIZE} -DINV_S={1 / math.sqrt(HEAD_SIZE) / STRIDE} -DKV_BLOCK_SIZE={KV_BLOCK_SIZE} -DBLOCK_SHARE_MAX={BLOCK_WG_M} -DUSE_KQ=1''')

# q: [B, Hq, L_q, S]
# k: [B, Hk, L_k, S]
def test_gemm(q:torch.Tensor, k:torch.Tensor, block_size=128, threshold=0.9, stride=16, causal=True):
    B, Hq, Lq, S = q.shape
    _, Hk, Lk, _ = k.shape
    M = Lq // stride
    N = Lk // stride
    K = stride * S
    M_kq = N
    N_kq = M

    tQ = cl.tensor(q.detach().numpy())
    tK = cl.tensor(k.detach().numpy())
    tC = cl.tensor(np.zeros([B, Hq, M_kq, N_kq], np.float16))

    # [1, 32, 256]
    N_kq_out_pad = (N_kq + BLOCK_WG_N - 1) // BLOCK_WG_N * BLOCK_WG_N
    max_init = np.ones([B, Hq, N_kq_out_pad], np.float16) * -60000.0
    tC_max = cl.tensor(max_init)  # [b, hq, M], init must be -inf
    # [1, 32, 64, 256]
    M_kq_groups = (M_kq + BLOCK_WG_M - 1) // BLOCK_WG_M
    tC_max_wg = cl.tensor(np.zeros([B, Hq, M_kq_groups, N_kq_out_pad], np.float16))
    # [1, 32, 256, 64 * 16]
    sum_per_token_in_block = block_size // stride
    M_kq_after_sum_in_group = BLOCK_WG_M // sum_per_token_in_block
    M_kq_after_sum_pad = M_kq_after_sum_in_group * M_kq_groups
    tC_exp_partial_sum = cl.tensor(np.zeros([B, Hq, N_kq_out_pad, M_kq_after_sum_pad], np.float16))

    # loop N first:[0, 1], loop M first:[0, 0]; block M first[slice_no, slice(>0)], block N first[slice_no, slice(<0)]
    #default linear
    slice_no = 0
    slice = N_kq < M_kq
    if 1:
        block_m = M_kq // BLOCK_WG_M
        block_n = N_kq // BLOCK_WG_N
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
    slice = 1
    print(f'{xecores=} {block_m=} {block_n=} {slice=} {slice_no=}')

    # gemm
    for i in range(10):
        kernels.enqueue(kernel_name, [M_kq // BLOCK_WG_M * N_kq // BLOCK_WG_N * SG_N, SG_M, B * Hq], [SG_N, SG_M, 1], tK, tQ, tC_max, tC_max_wg, tC_exp_partial_sum, M_kq, N_kq, K, K, K, N_kq, slice_no, slice)
    cl.finish()

    # , [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
    C_ref, A_max, A_5d_max_ret, A_exp_partial_sum_ret, A_sum = get_gemm_ref(q, k, block_size=block_size, S=stride, threshold=threshold, causal=causal, wg_x=BLOCK_WG_M)
    compare(A_max.detach().numpy(), tC_max.numpy())
    print(f'{Colors.GREEN}gemm:max passed{Colors.END}')
    compare(A_5d_max_ret.detach().numpy(), tC_max_wg.numpy())
    print(f'{Colors.GREEN}gemm:max_wg passed{Colors.END}')
    compare(A_exp_partial_sum_ret.detach().numpy(), tC_exp_partial_sum.numpy())
    print(f'{Colors.GREEN}gemm:exp_partial passed{Colors.END}')
    if 1:
        flops = B * Hq * M * N * K * 2
        for i in range(0, 100):
            kernels.enqueue(kernel_name, [M_kq // BLOCK_WG_M * N_kq // BLOCK_WG_N * SG_N, SG_M, B * Hq], [SG_N, SG_M, 1], tK, tQ, tC_max, tC_max_wg, tC_exp_partial_sum, M_kq, N_kq, K, K, K, N_kq, slice_no, slice)
            ns = cl.finish()
            for i, time_opt in enumerate(ns):
                print(f'(GEMM)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')


    return tC_max, tC_max_wg, tC_exp_partial_sum, A_sum


def test_find(tC_max, tC_max_wg, tC_exp_partial_sum, A_sum):
    # [1, 32, 256, 64 * 16]
    B, _, N_kq_out_pad, M_kq_after_sum_pad = tC_exp_partial_sum.shape
    N_after_sum_pad = M_kq_after_sum_pad  # 1024=8192/8
    M_pad = N_kq_out_pad                        # multiple of 128
    sum_per_n_token_in_block = BLOCK_SIZE // STRIDE

    # [1, 32, 32, 64 * 16]
    tC_sum = cl.tensor(np.zeros([B, HQ, M_pad // sum_per_n_token_in_block, N_after_sum_pad], np.float16))
    # find
    for i in range(1):
        kernels.enqueue("find_block", [M_pad // sum_per_n_token_in_block, HQ, B], [1, 1, 1], tC_max, tC_max_wg, tC_exp_partial_sum, tC_sum, M_pad, N_after_sum_pad)
    cl.finish()
    compare(A_sum.detach().numpy(), tC_sum.numpy())
    print(f'{Colors.GREEN}find:sum passed{Colors.END}')
    if 1:
        for i in range(0, 100):
            kernels.enqueue("find_block", [M_pad // sum_per_n_token_in_block, HQ, B], [1, 1, 1], tC_max, tC_max_wg, tC_exp_partial_sum, tC_sum, M_pad, N_after_sum_pad)
        ns = cl.finish()
        for i, time_opt in enumerate(ns):
            print(f'(FIND)TPUT_{i}: {time_opt*1e-3:,.0f} us')

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
    q_head = HQ
    k_head = HK
    q_len = Q_LEN
    k_len = K_LEN
    dim = HEAD_SIZE
    block_size = BLOCK_SIZE
    stride = STRIDE

    assert (q_len // stride) >= 128 and (k_len // stride) >= 128, "minimum block size should be 128*128"

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

    tC_max, tC_max_wg, tC_exp_partial_sum, A_sum = test_gemm(q, k, block_size=block_size, threshold=0.9, stride=stride, causal=False)
    test_find(tC_max, tC_max_wg, tC_exp_partial_sum, A_sum)


if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    # requirements:
    # num_q_head == num_kv_head
    # chunk size alignment
    # causal_mask
    main()