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

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

def get_partial_softmax_ref(A: torch.Tensor, block_size, S, wg_k=256, wg_q=128, valid_q=0):
    # [1, 32, 256, 8192]
    B, num_q_head, q_len_strided, k_len_strided = A.shape
    A_pad = torch.ones([B, num_q_head, rnd_up(q_len_strided, wg_q), rnd_up(k_len_strided, wg_k)], dtype=A.dtype) * -60000.0
    A_pad[:,:,:q_len_strided,:k_len_strided] = A
    q_len_strided, k_len_strided = rnd_up(q_len_strided, wg_q), rnd_up(k_len_strided, wg_k)

    # [1, 32, 256, 1]
    A_max = torch.max(A_pad, dim=-1, keepdim=True)[0]                           # real max needed for compensation [b, num_q_head, q_len_strided, 1]
    # [1, 32, 256, 64, 128]
    A_5d = A_pad.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_k, wg_k)
    # [1, 32, 256, 64, 1]
    A_5d_max = torch.max(A_5d, dim=-1, keepdim=True)[0]                         # local wg max needed for compensation [b, num_q_head, q_len_strided, k_len_strided // wg_n, 1]
    # [1, 32, 64, 256]
    A_5d_max_ret = torch.transpose(A_5d_max, -2, -3).squeeze(dim=-1)
    # [1, 32, 256, 64, 128]
    A_exp_partial = torch.exp(A_5d - A_5d_max) * (A_5d_max > -60000.0)
    # [1, 32, 256, 64, 16, 8]
    A_exp_partial = A_exp_partial.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_k, wg_k // (block_size // S), block_size // S)
    # [1, 32, 256, 64, 16]
    A_exp_partial_sum = A_exp_partial.sum(dim=-1)                               # local wg sum
    # [1, 32, 256, 64 * 16]
    A_exp_partial_sum_ret = torch.reshape(A_exp_partial_sum, (B, num_q_head, q_len_strided, -1))

    # set invalid q to 0
    if valid_q:
        A_max[:,:,valid_q:,:] = 0
        A_5d_max_ret[:,:,:,valid_q:] = 0
        A_exp_partial_sum_ret[:,:,valid_q:,:] = 0

    # stage 2:
    # [1, 32, 256, 64, 16] * [1, 32, 256, 64, 1]
    A_exp_horz_sum = A_exp_partial_sum * torch.exp(A_5d_max - A_max.unsqueeze(dim=-1))
    # [1, 32, 256, 64 * 16]
    A_exp_horz_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided, -1)
    A_exp_horz_sum = A_exp_horz_sum / (A_exp_horz_sum.sum(dim=-1, keepdim=True)+1e-5)
    A_exp_vert_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided // (block_size // S), block_size // S, -1)
    A_exp_vert_sum = A_exp_vert_sum.sum(dim=-2)
    A_sum = F.softmax(A_pad, dim=-1) * (A_pad > -60000.0)
    if valid_q:
        A_sum[:,:,valid_q:,:] = 0
    A_sum = A_sum.reshape(B, num_q_head, q_len_strided // (block_size // S), (block_size // S), k_len_strided // (block_size // S), (block_size // S))

    A_sum = A_sum.sum(dim=-1).sum(dim=-2)
    # [1, 32, 32, 64 * 16]
    if not torch.allclose(A_sum, A_exp_vert_sum, atol=0.01, rtol=0.01):
        print(f'{A_sum=}\n{A_exp_vert_sum=}')
        assert 0
    ###################### fuse gemm+softmax end
    return A_max.squeeze(-1).contiguous(), A_5d_max_ret.contiguous(), A_exp_partial_sum_ret.contiguous(), A_sum.contiguous()

def get_gemm_ref(Q_org: torch.Tensor, K: torch.Tensor, q_start_strided, block_size, S, threshold=0.9, causal=True, wg_k=256, wg_q=128):
    # Q, K, V shape = (B, H, L, d)
    B, num_kv_head, k_len, d = K.shape
    # drop tails which are less than stride
    Q = Q_org[:,:,:Q_org.shape[2]//S*S,:]

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
    if causal:
        causal_mask = torch.zeros(
            (
                B,
                num_q_head,
                q_len//S,
                k_len//S,
            ),
        )
        #causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
        chunk_start = q_start_strided
        chunk_end = chunk_start + q_len//S
        causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
            torch.ones(
                1,
                num_q_head,
                q_len//S,
                q_len//S,
            )
            * float("-60000.0"),
            diagonal=1,
        )
        A += causal_mask

    #A = F.softmax(A, dim=-1, dtype=torch.float32).to(Q.dtype)
    return get_partial_softmax_ref(A, block_size=block_size, S=S, wg_k=wg_k, wg_q=wg_q)

def find_blocks_ref(input_tensor_org, threshold, q_block, k_block, causal, current_index):
    input_tensor = input_tensor_org[...,:q_block,:k_block]
    B, HQ, Q_block, K_block = input_tensor.shape
    #input_tensor = input_tensor.to(float)
    input_tensor = input_tensor.to(torch.float32)
    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    required_sum = total_sum * threshold
    #print(f'{threshold=}, {required_sum=}, {total_sum=}')
    if causal:
        chunk_num = Q_block
        block_num = K_block
        mask = torch.zeros_like(input_tensor, dtype=torch.bool)
        mask[:, :, :, current_index : current_index + chunk_num] = (
            torch.eye(chunk_num, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, HQ, chunk_num, chunk_num)
        )
        mask[:, :, :, 0] = 1
        other_values = input_tensor.masked_fill(
            mask, 0
        )
        sorted_values, _ = torch.sort(
            other_values, dim=-1, descending=True, stable=True
        )
        sorted_values = sorted_values.to(input_tensor.device)

        sorted_values = torch.cat(
            [
                torch.zeros(
                    (B, HQ, chunk_num, 1), device=input_tensor.device
                ),
                torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                sorted_values[:, :, :, :-2],
            ],
            dim=-1,
        )

        _, org_index = torch.sort(
            torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
            dim=-1,
            descending=True, stable=True
        )
        cumulative_sum_without_self = torch.cat(
            [
                torch.zeros(
                    (B, HQ, chunk_num, 1), device=input_tensor.device
                ),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)

        index_mask = cumulative_sum_without_self < required_sum
        index = torch.where(index_mask,org_index,0)
        mask = mask.view(B,HQ*chunk_num,block_num)
        index = index.view(B,HQ*chunk_num,block_num)
        mask[:,torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),index] = True
        mask = mask.view(B,HQ,chunk_num,block_num)
    else:
        mask = torch.zeros_like(input_tensor, dtype=torch.int8)
        sorted_values, org_index = torch.sort(
            input_tensor, dim=-1, descending=True, stable=True
        )
        #print(f'{sorted_values=} {org_index=}')
        sorted_values = sorted_values.to(input_tensor.device)
        cumulative_sum_without_self = torch.cat(
            [
                torch.zeros(
                    (B, HQ, Q_block, 1), device=input_tensor.device
                ),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)
        #print(f'{cumulative_sum_without_self=}')
        index_mask = cumulative_sum_without_self < required_sum
        #print(f'{index_mask=}')
        index = torch.where(index_mask, org_index, 0)
        mask = mask.view(B, HQ * Q_block, K_block)
        index = index.view(B, HQ * Q_block, K_block)
        mask[
            :,
            torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
            index,
        ] = 1
        mask = mask.view(B, HQ, Q_block, K_block)
    #print(f'ref_mask={mask}')
    return mask, sorted_values, org_index, cumulative_sum_without_self, required_sum

# kernel
base_dir = os.path.dirname(os.path.abspath(__file__))
files = ['sort.hpp', 'estimate.hpp', 'find_block.hpp']
src = [open(f'{base_dir}/{file}').read() for file in files]
src.append(
r'''
#define ABS(x) (x) < 0 ? -(x) : (x)

CM_INLINE void get_mn(uint& id_wg_m, uint& id_wg_n, uint M, uint N, int slice_no, int slice, const int BLOCK_WG_M, const int BLOCK_WG_N) {
    uint id_wg_mn = cm_group_id(0);
    if (slice_no == 0) {
        if (slice == 0) {
            // loop M first, N is shared, total = N/256*M+N
            uint WG_MN = (M + BLOCK_WG_M - 1) / BLOCK_WG_M;
            id_wg_m = id_wg_mn % WG_MN;
            id_wg_n = id_wg_mn / WG_MN;
        } else {
            // loop N first, M is shared, total = M/128*N+M
            uint WG_MN = (N + BLOCK_WG_N - 1) / BLOCK_WG_N;
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

_GENX_MAIN_ void gemm_qk(svmptr_t key_cache ATTR, svmptr_t query ATTR, svmptr_t block_indices ATTR, svmptr_t block_indices_begins ATTR, svmptr_t kq_max ATTR, svmptr_t kq_max_wg ATTR, svmptr_t kq_exp_partial_sum ATTR, 
    uint M, uint N, uint K, uint query_stride, int slice_no, int slice, uint q_start_strided) {
    const uint BLOCK_WG_M = BLOCK_SG_M * SG_M;
    const uint BLOCK_WG_N = BLOCK_SG_N * SG_N;
    const uint size_slm_b = 0;
    uint hq = cm_group_id(2);
    uint hk = hq / (HQ / HK);
    const uint slm_size = SG_M * BLOCK_WG_N * sizeof(half);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    static_assert(HQ % HK == 0, "HQ must be multiple of HK");

    uint id_wg_m, id_wg_n;
    get_mn(id_wg_m, id_wg_n, M, N, slice_no, slice, BLOCK_WG_M, BLOCK_WG_N);

    // key cache: [block, HQ, KV_BLOCK_SIZE, HEAD_SIZE_KEY]
#if USE_INT8
    key_cache += hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
#else
    key_cache += hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half));
#endif
    // query: [l_q, HQ * HEAD_SIZE]
    query += hq * HEAD_SIZE * (uint)sizeof(half);

    // kq_max: [hq, n_pad]
    // kq_max_wg: [hq, m_groups, n_pad]
    // kq_exp_partial_sum: [hq, n_pad, m_groups*BLOCK_WG_M/(BLOCK_SIZE/STRIDE)]
    uint n_pad = (N + BLOCK_WG_N - 1) / BLOCK_WG_N * BLOCK_WG_N;
    kq_max += hq * n_pad * (uint)sizeof(half);
    uint m_groups = (M + BLOCK_WG_M - 1) / BLOCK_WG_M;
    kq_max_wg += hq * m_groups * n_pad * (uint)sizeof(half);

    const uint sum_per_n_token_in_block = BLOCK_SIZE / STRIDE;
    const uint m_after_sum_in_group = BLOCK_WG_M / sum_per_n_token_in_block;
    const uint m_after_sum_pad = m_after_sum_in_group * m_groups;
    kq_exp_partial_sum += hq * m_after_sum_pad * n_pad * (uint)sizeof(half);

#define CONCAT_IMPL(a, b) gemm_kq_ ##a ##x ##b ##_xe2
#define CONCAT(x, y) CONCAT_IMPL(x, y)
#define FUNC CONCAT(BLOCK_SG_M, BLOCK_SG_N)
    FUNC(id_wg_m, id_wg_n, hq, slm, key_cache, query, block_indices, block_indices_begins, kq_max, kq_max_wg, kq_exp_partial_sum, M, N, K, query_stride, q_start_strided);
}

_GENX_MAIN_ void find_block(svmptr_t kq_max ATTR, svmptr_t kq_max_wg ATTR, svmptr_t kq_exp_partial_sum ATTR, svmptr_t kq_sum ATTR, svmptr_t block_mask ATTR, uint q_stride, uint q_stride_pad, uint k_block_pad,
    float thresh, uint causal_start_index) {
    // kq_max:             [b, hq, q_stride_pad]
    // kq_max_wg:          [b, hq, m_groups, q_stride_pad]
    // kq_exp_partial_sum: [b, hq, q_stride_pad, k_block_pad]
    // kq_sum:             [b, hq, q_stride_pad/TOKEN_IN_BLOCK, k_block_pad]
    // block_mask:         [b, hq, q_stride_pad/TOKEN_IN_BLOCK, k_block_pad]
    // [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
    // global:            [q_stride_pad/TOKEN_IN_BLOCK, hq, b]
    const int TOKEN_IN_BLOCK = BLOCK_SIZE / STRIDE;
    const int TOKEN_SHARE_MAX = BLOCK_SHARE_MAX / TOKEN_IN_BLOCK;
    uint m = cm_group_id(0);
    uint hq = cm_group_id(1);
    uint b = cm_group_id(2);
    kq_max += (b * HQ + hq) * q_stride_pad * (uint)sizeof(half);
    kq_max_wg += (b * HQ + hq) * (k_block_pad / TOKEN_SHARE_MAX) * q_stride_pad * (uint)sizeof(half);
    kq_exp_partial_sum += (b * HQ + hq) * q_stride_pad * k_block_pad * (uint)sizeof(half);
    kq_sum += (b * HQ + hq) * q_stride_pad / TOKEN_IN_BLOCK * k_block_pad * (uint)sizeof(half);
    block_mask += (b * HQ + hq) * q_stride_pad / TOKEN_IN_BLOCK * k_block_pad;

    const uint slm_size = 32 * 16 * sizeof(ushort);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    find(slm, m, kq_max, kq_max_wg, kq_exp_partial_sum, kq_sum, block_mask, q_stride, q_stride_pad, k_block_pad, thresh, causal_start_index);
}

''')
src = '\n'.join(src)

kernel_name = 'gemm_qk'
BLOCK_SG_M = 32 #32
BLOCK_SG_N = 32
SG_M = 4
SG_N = 4
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N
KV_BLOCK_SIZE = 256

HQ = 32
HK = 8
HEAD_SIZE = 128
STRIDE = 16
BLOCK_SIZE = 128
THRESH = 0.9
IS_CAUSAL = 1
USE_INT8 = 1
if USE_INT8:
    HEAD_SIZE_KEY = HEAD_SIZE + 2 * 2
else:
    HEAD_SIZE_KEY = HEAD_SIZE

FIND_DEBUG_ACC = 0 # only acc test needed

jit_option = '-abortonspill -noschedule '
kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2
                    -DSTRIDE={STRIDE} -DHQ={HQ} -DHK={HK} -DHEAD_SIZE={HEAD_SIZE} -DSG_M={SG_M} -DSG_N={SG_N} -DBLOCK_SG_N={BLOCK_SG_N} -DBLOCK_SG_M={BLOCK_SG_M}
                    -DBLOCK_SIZE={BLOCK_SIZE} -DINV_S={1 / math.sqrt(HEAD_SIZE) / STRIDE} -DKV_BLOCK_SIZE={KV_BLOCK_SIZE} -DBLOCK_SHARE_MAX={BLOCK_WG_M} -DUSE_KQ=1
                    -DDEBUG_ACC={FIND_DEBUG_ACC} -DIS_CAUSAL={IS_CAUSAL} -DUSE_INT8={USE_INT8} -DHEAD_SIZE_KEY={HEAD_SIZE_KEY}''')

def quant_i8(k:torch.Tensor):
    B, Hk, Lk, S = k.shape
    k_pad = torch.zeros([B, Hk, rnd_up(Lk, KV_BLOCK_SIZE), HEAD_SIZE], dtype=k.dtype)
    k_pad[:, :, :Lk,:] = k

    B, Hk, Lk, S = k_pad.shape
    k_i8 = torch.zeros([B, Hk, Lk, HEAD_SIZE_KEY], dtype=torch.uint8)
    k_i8_4d = k_i8.reshape([B, Hk, -1, KV_BLOCK_SIZE * HEAD_SIZE_KEY])
    max = torch.max(k_pad, dim=-1, keepdim=True)[0].to(torch.float32)
    min = torch.min(k_pad, dim=-1, keepdim=True)[0].to(torch.float32)
    diff_value = torch.masked_fill(max - min, max == min, 0.001)
    scale = 255/ diff_value
    zp = -min * scale
    quanted = k_pad * scale + zp
    quanted = quanted.clamp(0, 255)
    # weights
    k_i8_4d[:, :, :, :KV_BLOCK_SIZE * HEAD_SIZE] = torch.reshape(quanted, [B, Hk, Lk // KV_BLOCK_SIZE, -1])
    # scale
    k_i8_4d[:, :, :, KV_BLOCK_SIZE * HEAD_SIZE : (KV_BLOCK_SIZE * (HEAD_SIZE + 2))] = (1.0 / scale).to(torch.float16).reshape([B, Hk, Lk // KV_BLOCK_SIZE, -1]).view(dtype=torch.uint8)
    # zp
    k_i8_4d[:, :, :, (KV_BLOCK_SIZE * (HEAD_SIZE + 2)) : ] = zp.to(torch.float16).reshape([B, Hk, Lk // KV_BLOCK_SIZE, -1]).view(dtype=torch.int8)

    return k_i8

# q: [B, Hq, L_q, S]
# k: [B, Hk, L_k, S]
def test_gemm(q:torch.Tensor, k:torch.Tensor, block_size=128, q_start_strided=0, threshold=0.9, stride=16, causal=True, perf=True):
    B, Hq, Lq, S = q.shape
    _, Hk, Lk, _ = k.shape
    Lk = Lk // stride * stride
    M = Lq // stride                  # will slient drop the tails which is less than `stride`
    N = Lk // stride
    K = stride * S
    M_kq = N
    N_kq = M

    k = k[:,:,:Lk,:]                  # will slient drop the tails which is less than `stride`

    # k -> key_cache blocked layout
    block_indices_begins = torch.zeros(B + 1).to(torch.int32)
    block_indices = torch.arange((Lk + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE)
    # simulate random block allocation
    perm_idx = torch.randperm(block_indices.shape[0])
    inv_per_idx = torch.argsort(perm_idx)
    block_indices = block_indices[inv_per_idx]

    t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
    # t_past_lens = cl.tensor(past_lens.to(torch.int32).detach().numpy())                   # M_kq has already been calculated
    t_block_indices_begins = cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
    # t_subsequence_begins = cl.tensor(subsequence_begins.to(torch.int32).detach().numpy()) # N_kq has already been calculated

    if USE_INT8:
        k_pad = quant_i8(k)
        key_cache = k_pad.reshape([B, Hk, -1, KV_BLOCK_SIZE, HEAD_SIZE_KEY]).permute(0, 2, 1, 3, 4)
    else:
        k_pad = torch.zeros([B, Hk, (Lk + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE * KV_BLOCK_SIZE, HEAD_SIZE], dtype=k.dtype)
        k_pad[:, :, :Lk,:] = k
        key_cache = k_pad.reshape([B, Hk, -1, KV_BLOCK_SIZE, HEAD_SIZE]).permute(0, 2, 1, 3, 4)
    key_cache = key_cache[:,perm_idx,:,:,:].contiguous()
    t_key_cache = cl.tensor(key_cache.detach().numpy())

    # [B, Hq, L, S] -> [B, L, Hq * S]
    q_3d = q.permute(0, 2, 1, 3).reshape([B, Lq, -1])
    # simulate layout of q,k,v is [B, L, Hq * S..Hkv * S]
    q_3d_with_padding = torch.zeros([B, Lq, Hq * S * 2], dtype=q_3d.dtype)
    q_3d_with_padding[:, :, : Hq * S] = q_3d
    t_query = cl.tensor(q_3d_with_padding.detach().numpy())

    # [1, 32, 256]
    q_stride_pad = (N_kq + BLOCK_WG_N - 1) // BLOCK_WG_N * BLOCK_WG_N
    max_init = np.ones([B, Hq, q_stride_pad], np.float16) * -60000.0
    t_kq_max = cl.tensor(max_init)  # [b, hq, M], init must be -inf
    # [1, 32, 64, 256]
    M_kq_groups = (M_kq + BLOCK_WG_M - 1) // BLOCK_WG_M
    t_kq_max_wg = cl.tensor(np.zeros([B, Hq, M_kq_groups, q_stride_pad], np.float16))
    # [1, 32, 256, 64 * 16]
    sum_per_token_in_block = block_size // stride
    M_kq_after_sum_in_group = BLOCK_WG_M // sum_per_token_in_block
    k_block_pad = M_kq_after_sum_in_group * M_kq_groups
    t_kq_exp_partial_sum = cl.tensor(np.zeros([B, Hq, q_stride_pad, k_block_pad], np.float16))

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
    for i in range(1):
        kernels.enqueue(kernel_name, [M_kq_groups * (q_stride_pad // BLOCK_WG_N) * SG_N, SG_M, Hq], [SG_N, SG_M, 1], t_key_cache, t_query, t_block_indices, t_block_indices_begins, t_kq_max, t_kq_max_wg, t_kq_exp_partial_sum, M_kq, N_kq, K, K * HQ * 2, slice_no, slice, q_start_strided)
    cl.finish()

    # [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
    kq_max_ref, kq_5d_max_ret_ref, kq_exp_partial_sum_ret_ref, _ = get_gemm_ref(q, k, block_size=block_size, q_start_strided=q_start_strided, S=stride, threshold=threshold, causal=causal, wg_k=BLOCK_WG_M, wg_q=BLOCK_WG_N)
    kq_max_ref_np = kq_max_ref.detach().numpy()[..., :N_kq]
    t_kq_max_np = t_kq_max.numpy()[..., :N_kq]
    compare(kq_max_ref_np, t_kq_max_np)
    print(f'{Colors.GREEN}gemm:max passed{Colors.END}')
    kq_5d_max_ret_ref_np = kq_5d_max_ret_ref.detach().numpy()[..., :N_kq]
    t_kq_max_wg_np = t_kq_max_wg.numpy()[..., :N_kq]
    compare(kq_5d_max_ret_ref_np, t_kq_max_wg_np)
    print(f'{Colors.GREEN}gemm:max_wg passed{Colors.END}')
    kq_exp_partial_sum_ret_ref_np = kq_exp_partial_sum_ret_ref.detach().numpy()[:,:,:N_kq,:]
    t_kq_exp_partial_sum_np = t_kq_exp_partial_sum.numpy()[:,:,:N_kq,:]
    compare(kq_exp_partial_sum_ret_ref_np, t_kq_exp_partial_sum_np)
    print(f'{Colors.GREEN}gemm:exp_partial passed{Colors.END}')
    if perf:
        flops = B * Hq * M * N * K * 2
        for i in range(0, 100):
            kernels.enqueue(kernel_name, [M_kq // BLOCK_WG_M * q_stride_pad // BLOCK_WG_N * SG_N, SG_M, Hq], [SG_N, SG_M, 1], t_key_cache, t_query, t_block_indices, t_block_indices_begins, t_kq_max, t_kq_max_wg, t_kq_exp_partial_sum, M_kq, N_kq, K, K * HQ * 2, slice_no, slice, q_start_strided)
            ns = cl.finish()
            for i, time_opt in enumerate(ns):
                print(f'(GEMM)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

    return t_kq_max, t_kq_max_wg, t_kq_exp_partial_sum


# q_stride = q_len // stride
# k_stride = k_len // stride
def test_find(q_stride, k_stride, block_size, S, wg_k, wg_q, perf, causal):
    qk = torch.randint(-2000, 3000, size=[1, HQ, q_stride, k_stride], dtype=torch.int16).to(dtype=torch.float16)
    assert wg_k % block_size == 0, "wg_k should be multiple of block_size then there is no tails from block_size"
    assert wg_q % block_size == 0, "wg_q should be multiple of block_size then there is no tails from block_size"
    qk_max, kq_5d_max, qk_exp_partial_sum, qk_sum = get_partial_softmax_ref(qk, block_size, S, wg_k, wg_q, valid_q=q_stride)
    t_kq_max = cl.tensor(qk_max.detach().numpy())
    t_kq_max_wg = cl.tensor(kq_5d_max.detach().numpy())
    t_kq_exp_partial_sum = cl.tensor(qk_exp_partial_sum.detach().numpy())

    sum_per_n_token_in_block = BLOCK_SIZE // STRIDE
    q_block = div_up(q_stride, sum_per_n_token_in_block)
    k_block = div_up(k_stride, sum_per_n_token_in_block)
    assert k_block >= q_block, "k block should be larger than q_block"
    mask_ref, sorted_value_ref, sorted_index_ref, cumulative_sum_without_self, required_sum = find_blocks_ref(qk_sum, THRESH, q_block, k_block, causal=causal, current_index=k_block-q_block)

    # [1, 32, 256, 64 * 16]
    B, _, q_stride_pad, k_block_pad = t_kq_exp_partial_sum.shape
    assert k_block_pad == (k_stride + wg_k - 1) // wg_k * wg_k // sum_per_n_token_in_block, "k_block padded to wg_k / block_size"
    assert q_stride_pad == (q_stride + wg_q - 1) // wg_q * wg_q, "q_stride_pad padded to wg_q / stride"

    # [1, 32, 32, 64 * 16]
    t_kq_sum = cl.tensor(np.zeros([B, HQ, q_stride_pad // sum_per_n_token_in_block, k_block_pad], np.float16))
    t_mask = cl.tensor(np.zeros([B, HQ, q_stride_pad // sum_per_n_token_in_block, k_block_pad], np.int8))
    # find
    for i in range(1):
        kernels.enqueue("find_block", [q_stride_pad // sum_per_n_token_in_block, HQ, B], [1, 1, 1], t_kq_max, t_kq_max_wg, t_kq_exp_partial_sum, t_kq_sum, t_mask, q_stride, q_stride_pad, k_block_pad, THRESH,
                        k_block-q_block)
    cl.finish()
    if FIND_DEBUG_ACC == 1:
        compare(qk_sum.detach().numpy(), t_kq_sum.numpy())
        print(f'{Colors.GREEN}find:sum passed{Colors.END}')

    kq_exp_partial_sum_np = t_kq_exp_partial_sum.numpy()
    cur_sorted_value = kq_exp_partial_sum_np[:, :, 1::sum_per_n_token_in_block, :]
    cur_sorted_index = kq_exp_partial_sum_np[:, :, 3::sum_per_n_token_in_block, :].view(dtype=np.ushort)
    cur_accum_value  = kq_exp_partial_sum_np[:, :, 6::sum_per_n_token_in_block, :]
    sorted_value_ref = sorted_value_ref.detach().numpy()
    cur_sorted_value = cur_sorted_value[...,:q_block,:k_block]
    compare(sorted_value_ref, cur_sorted_value)
    print(f'{Colors.GREEN}find:sort_value passed{Colors.END}')
    sorted_index_ref = sorted_index_ref.to(torch.uint16).detach().numpy()

    # minor error may get different index
    if causal:
        # index of causal mask for line 0 is also 0, reference will only skip 1 point instead of 2 points(causal+leftmost)
        if q_block == k_block:
            sorted_index_ref[...,0, 2:] = sorted_index_ref[...,0, 1:-1]
        # index 0,1 are for diag+leftmost
        sorted_index_ref[...,:2] = 0
        cur_sorted_index[...,:2] = 0
    cur_sorted_index = cur_sorted_index[...,:q_block,:k_block]
    error_idx = np.where(sorted_index_ref != cur_sorted_index)
    if error_idx[0].shape[0]:
        # print(f'{error_idx=}\nidx_ref={sorte_index_ref[error_idx]}\nidx_cur={cur_sorted_index[error_idx]}')
        error_idx_ref = (error_idx[0], error_idx[1], error_idx[2], sorted_index_ref[error_idx])
        error_idx_cur = (error_idx[0], error_idx[1], error_idx[2], cur_sorted_index[error_idx])
        if not np.allclose(qk_sum[error_idx_ref], qk_sum[error_idx_cur], atol=0.01, rtol=0.01):
            pos = np.where(np.abs(qk_sum[error_idx_ref] - qk_sum[error_idx_cur]) > 0.01)
            print(f'ref={qk_sum[error_idx_ref][pos]}\ncur={qk_sum[error_idx_cur][pos]}')
            raise('error')

    #compare(sorte_index_ref, cur_sorted_index)
    print(f'{Colors.GREEN}find:sort_index passed{Colors.END}')

    if FIND_DEBUG_ACC == 1:
        cur_accum_value = cur_accum_value[...,:q_block,:k_block-2]
        cumulative_sum_without_self_np = cumulative_sum_without_self.detach().numpy()
        compare(cumulative_sum_without_self_np[...,:-2], cur_accum_value)
        print(f'{Colors.GREEN}find:accum passed{Colors.END}')

    cur_mask = t_mask.numpy()
    cur_mask = cur_mask[...,:q_block,:k_block]
    if (mask_ref & cur_mask).sum() / mask_ref.sum() < 0.95:
        raise Exception(f'mask intersection less than 95%')
    # minor error may get different mask index
    # mask_ref_np =  mask_ref.detach().numpy()
    # cur_mask = tMask.numpy()
    # error_idx = np.where(mask_ref_np != cur_mask)
    # if error_idx[0].shape[0]:
    #     if not np.allclose(A_sum[error_idx], tC_sum.numpy()[error_idx], atol=0.01, rtol=0.01):
    #         print(f'{error_idx=}\nidx_ref={mask_ref_np[error_idx]}\nidx_cur={cur_mask[error_idx]}')
    #         raise Exception(f'the diff value should be smaller than 0.01, ref={A_sum[error_idx]} cur={tC_sum.numpy()[error_idx]}')
    # print(f'{Colors.GREEN}find:mask passed{Colors.END}')

    if perf:
        for i in range(0, 100):
            kernels.enqueue("find_block", [q_stride_pad // sum_per_n_token_in_block, HQ, B], [1, 1, 1], t_kq_max, t_kq_max_wg, t_kq_exp_partial_sum, t_kq_sum, t_mask, q_stride, q_stride_pad, k_block_pad, THRESH,
                            k_block-q_block)
        ns = cl.finish()
        for i, time_opt in enumerate(ns):
            print(f'(FIND)TPUT_{i}: {time_opt*1e-3:,.0f} us')

def test_func():
    q_head = HQ
    k_head = HK
    dim = HEAD_SIZE
    block_size = BLOCK_SIZE
    stride = STRIDE
    sizes = [
        # normal case
        (256 * STRIDE, 256 * STRIDE, 'normal+causal start == 0'),           # normal case:causal start == 0
        (4 * 1024, 128 * 1024, 'normal'),                                   # normal case:4k * 128k
        (128 * STRIDE, 256 * STRIDE, 'normal+smallest block'),              # smallest block just suitable for one workgroup
        # query tails: tails are less than STRIDE
        (128 * STRIDE + 7, 256 * STRIDE, 'M tail=7 of query'),
        (4 * 1024 + 5, 128 * 1024, 'M tail=5 of query'),
        # query tails: tails are multiple of STRIDE
        (STRIDE, 256 * STRIDE, 'M tail=16 of query'),                       # query has tail which is not enough for a workgroup
        ((128+1) * STRIDE, 256 * STRIDE, 'M tail of smallest query'),       # query has tail which is not full for its last block in M dimension
        (4 * 1024 + STRIDE*3, 128 * 1024, 'M tail of bigger query'),        # query has tail which is not full for its last block in M dimension
        (4 * 1024 + STRIDE*7+2, 128 * 1024, 'M tail=x*STRIDE+y of query'),  # query has tail which is not full for its last block in M dimension
        # key tails: tails are less than STRIDE
        (128 * STRIDE, 256 * STRIDE + 7, 'N tail=7 of key'),
        (4 * 1024 + 5, 128 * 1024 + 3, 'M tail=5&N tail=3 of key'),
        # key tails: tails are multiple of STRIDE
        (128 * STRIDE, STRIDE * (128 + 1), 'N tail=16 of key'),                # key has tail which is not enough for a workgroup
        (4 * 1024, 128*1024+STRIDE*2, 'N tail=32 of key'),                     # key has tail which is not enough for a workgroup
        (4 * 1024 + 3*STRIDE+5, 128 * 1024 + 7*STRIDE+ 3, 'M tail=3.5&N tail=7.3 of key'),
    ]
    for q_len, k_len, prompt in sizes:
        assert q_len // STRIDE >= 1, "there should be at least 1 row for gemm"
        print(f'{Colors.BLUE}test gemm("{prompt}") query: [{q_len}, {dim}*{stride}] key:[{k_len}, {dim}*{stride}]...{Colors.END}')

        q = torch.randint(-2, 3, size=[1, q_head, q_len, dim], dtype=torch.int16).to(dtype=torch.float16)
        k = torch.randint(-2, 3, size=[1, k_head, k_len, dim], dtype=torch.int16).to(dtype=torch.float16)

        q_start_strided = k_len // stride - q_len // stride
        assert q_start_strided >= 0, "length of key cache must be greater or equal than query"
        test_gemm(q, k, block_size=block_size, q_start_strided=q_start_strided, threshold=THRESH, stride=stride, causal=True, perf=False)
        test_find(q_len // stride, k_len // stride, block_size, stride, wg_k=BLOCK_WG_M, wg_q=BLOCK_WG_N, perf=False, causal=True)

def test_perf():
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
    bsz = 1 # must be 1(gemm_qk does not support batch which M,N may be vary for each batch)
    q_head = HQ
    k_head = HK
    dim = HEAD_SIZE
    block_size = BLOCK_SIZE
    stride = STRIDE

    Q_LEN = 1024*4*2  # if q_len=1024*4*2 ==> 95 T/s
    K_LEN = 1024*128
    q_len = Q_LEN
    k_len = K_LEN

    assert (q_len // stride) >= 128 and (k_len // stride) >= 128, "minimum block size should be 128*128"

    q = torch.randint(-2, 3, size=[bsz, q_head, q_len, dim], dtype=torch.int16).to(dtype=torch.float16)
    k = torch.randint(-2, 3, size=[bsz, k_head, k_len, dim], dtype=torch.int16).to(dtype=torch.float16)

    test_gemm(q, k, block_size=block_size, q_start_strided=k_len // stride - q_len // stride, threshold=THRESH, stride=stride, causal=True)
    test_find(q_len // stride, k_len // stride, block_size, stride, wg_k=BLOCK_WG_M, wg_q=BLOCK_WG_N, perf=True, causal=True)


def main():
    if 1:
        test_func()
    if 1:
        test_perf()

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    # requirements:
    # num_q_head == num_kv_head
    # chunk size alignment
    # causal_mask
    main()