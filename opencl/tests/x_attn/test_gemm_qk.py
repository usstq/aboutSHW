import os

import torch
import numpy as np

from clops import cl
from clops import compare
from clops.utils import Colors

from references import get_gemm_ref

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

#  determine Xe1/Xe2
def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

# CM_GRF_WIDTH = get_cm_grf_width()
# print(f'{CM_GRF_WIDTH=}')

# gemmQK

kernel_name = 'gemm_qk'
BLOCK_SG_M = 64 #32
BLOCK_SG_N = 32
SG_M = 4
SG_N = 8
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N
KV_BLOCK_SIZE = 256   # should BLOCK_WG_N be dividable by KV_BLOCK_SIZE?

HQ = 32
HK = 8
HEAD_SIZE = 128
STRIDE = 16
BLOCK_SIZE = 128
THRESH = 1.0
IS_CAUSAL = 0
USE_INT8 = 0
if USE_INT8:
    HEAD_SIZE_KEY = HEAD_SIZE + 2 * 2
else:
    HEAD_SIZE_KEY = HEAD_SIZE

# loop order walks HQ first and the step is WALK_HQ, 1 means not walk HQ, 2 means walks 2 heads first. Valid value: 1, 2, 4...
WALK_HQ = 2 if HQ != HK else 1
SOFTMAX_TYPE = 'float' # 'half'

FIND_DEBUG_ACC = 0 # only acc test needed

INV_S = 1 / torch.sqrt(torch.tensor([HEAD_SIZE], dtype=torch.float32)) / STRIDE
INV_S = int(INV_S.view(dtype=torch.uint32))

kernels = None
def create_kernels(force_create=False):
    global kernels
    if kernels and not force_create: return

    src = r'''#include "xattn_gemm_qk.hpp"'''
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"compiling {cwd} ...")

    # jit_option = '-abortonspill -noschedule '
    jit_option = ' '
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size=256 -I{cwd}
                        -DSTRIDE={STRIDE} -DHQ={HQ} -DHK={HK} -DHEAD_SIZE={HEAD_SIZE} -DSG_M={SG_M} -DSG_N={SG_N} -DBLOCK_SG_N={BLOCK_SG_N} -DBLOCK_SG_M={BLOCK_SG_M}
                        -DBLOCK_SIZE={BLOCK_SIZE} -DINV_S={INV_S} -DKV_BLOCK_SIZE={KV_BLOCK_SIZE} -DBLOCK_SHARE_MAX={BLOCK_WG_N} -DWALK_HQ={WALK_HQ}
                        -DIS_CAUSAL={IS_CAUSAL} -DUSE_INT8={USE_INT8} -DHEAD_SIZE_KEY={HEAD_SIZE_KEY} -DSOFTMAX_TYPE={SOFTMAX_TYPE}
                        -DDEBUG_ACC={FIND_DEBUG_ACC}
                        ''')

def quant_i8(k:torch.Tensor):
    B, Hk, Lk, S = k.shape
    k_pad = torch.zeros([B, Hk, rnd_up(Lk, KV_BLOCK_SIZE), HEAD_SIZE], dtype=k.dtype)
    k_pad[:, :, :Lk,:] = k

    B, Hk, Lk, S = k_pad.shape
    k_i8 = torch.zeros([B, Hk, Lk, HEAD_SIZE_KEY], dtype=torch.uint8)
    k_i8_4d = k_i8.reshape([B, Hk, -1, KV_BLOCK_SIZE * HEAD_SIZE_KEY])
    if 0:
        max = torch.max(k_pad, dim=-1, keepdim=True)[0].to(torch.float32)
        min = torch.min(k_pad, dim=-1, keepdim=True)[0].to(torch.float32)
        diff_value = torch.masked_fill(max - min, max == min, 0.001)
        scale = 255/ diff_value
        zp = -min * scale
        quanted = k_pad * scale + zp
        quanted = quanted.clamp(0, 255)
        scale = 1.0 / scale
    else:
        scale = torch.randint(-1, 3, [B, Hk, Lk, 1], dtype=torch.float16)
        zp = torch.randint(0, 3, [B, Hk, Lk, 1], dtype=torch.float16)
        quanted = torch.randint(0, 3, [B, Hk, Lk, S], dtype=torch.float16)
        k_pad = (quanted - zp) * scale
        k = k_pad[:, :, :k.shape[2], :]
        k_pad[:,:,k.shape[2]:,:] = 0
    # weights
    k_i8_4d[:, :, :, :KV_BLOCK_SIZE * HEAD_SIZE] = torch.reshape(quanted, [B, Hk, Lk // KV_BLOCK_SIZE, -1])
    # scale
    k_i8_4d[:, :, :, KV_BLOCK_SIZE * HEAD_SIZE : (KV_BLOCK_SIZE * (HEAD_SIZE + 2))] = (scale).to(torch.float16).reshape([B, Hk, Lk // KV_BLOCK_SIZE, -1]).view(dtype=torch.uint8)
    # zp
    k_i8_4d[:, :, :, (KV_BLOCK_SIZE * (HEAD_SIZE + 2)) : ] = zp.to(torch.float16).reshape([B, Hk, Lk // KV_BLOCK_SIZE, -1]).view(dtype=torch.int8)

    return k_i8, k

# q: [B, Hq, L_q, S]
# k: [B, Hk, L_k, S]
def test_gemm(q:torch.Tensor, k:torch.Tensor, block_size=128, q_start_strided=0, threshold=0.9, stride=16, causal=True, perf=True):
    create_kernels()

    B, Hq, Lq, S = q.shape
    _, Hk, Lk, _ = k.shape
    Lk = Lk // stride * stride
    M = Lq // stride                  # will slient drop the tails which is less than `stride`
    N = Lk // stride
    K = stride * S

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
        k_pad, k = quant_i8(k)
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

    q_stride_pad = rnd_up(M, BLOCK_WG_M)
    # [1, 32, 64, 256]
    softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
    N_kq_groups = div_up(N, BLOCK_WG_N)
    t_kq_max_wg = cl.tensor(np.ones([B, Hq, N_kq_groups, q_stride_pad], softmax_type))
    # [1, 32, 256, 64 * 16]
    sum_per_token_in_block = block_size // stride
    k_block_in_group = BLOCK_WG_N // sum_per_token_in_block
    k_block_pad = k_block_in_group * N_kq_groups
    t_kq_exp_partial_sum = cl.tensor(np.ones([B, Hq, q_stride_pad, k_block_pad], softmax_type))

    # loop N first:[0, 1], loop M first:[0, 0]; block M first[slice_no, slice(>0)], block N first[slice_no, slice(<0)]
    #default linear
    slice_no = 0
    slice = 0
    print(f'{slice=} {slice_no=}')
    query_stride = K * HQ * 2   # i.e. (stride * S) * HQ * 2
    M_kq_groups = q_stride_pad // BLOCK_WG_M

    global_size = [N_kq_groups * M_kq_groups * SG_N * WALK_HQ, SG_M, Hq // WALK_HQ]

    # gemm warmup
    for i in range(1):
        kernels.enqueue(kernel_name, global_size, [SG_N, SG_M, 1],
                        t_key_cache, t_query, t_block_indices, t_block_indices_begins,
                        t_kq_max_wg, t_kq_exp_partial_sum,
                        M, N, K, query_stride, slice_no, slice, q_start_strided)
    cl.finish()

    if not perf:
        # [1, 32, 256], [1, 32, 64, 256], [1, 32, 256, 64 * 16], A_sum:[1, 32, 32, 64 * 16]
        kq_max_ref, kq_5d_max_ret_ref, kq_exp_partial_sum_ret_ref, _ = get_gemm_ref(q, k, block_size=block_size, q_start_strided=q_start_strided, S=stride, threshold=threshold, causal=causal, wg_k=BLOCK_WG_N, wg_q=BLOCK_WG_M)
        kq_5d_max_ret_ref_np = kq_5d_max_ret_ref.detach().numpy()[..., :M]
        t_kq_max_wg_np = t_kq_max_wg.numpy()[..., :M]
        compare(kq_5d_max_ret_ref_np, t_kq_max_wg_np)
        print(f'{Colors.GREEN}gemm:max_wg passed{Colors.END}')
        kq_exp_partial_sum_ret_ref_np = kq_exp_partial_sum_ret_ref.detach().numpy()[:,:,:M,:]
        t_kq_exp_partial_sum_np = t_kq_exp_partial_sum.numpy()[:,:,:M,:]
        compare(kq_exp_partial_sum_ret_ref_np, t_kq_exp_partial_sum_np)
        print(f'{Colors.GREEN}gemm:exp_partial passed{Colors.END}')
    else:
        flops = B * Hq * M * N * K * 2
        for i in range(0, 100):
            kernels.enqueue(kernel_name, global_size, [SG_N, SG_M, 1],
                            t_key_cache, t_query, t_block_indices, t_block_indices_begins,
                            t_kq_max_wg, t_kq_exp_partial_sum,
                            M, N, K, query_stride, slice_no, slice, q_start_strided)
            ns = cl.finish()
            for i, time_opt in enumerate(ns):
                print(f'(GEMM)TPUT_{i}:{flops/time_opt:,.0f} GFLOPS, BW:{(M*K+K*N+M*N)*2/time_opt:,.0f} GB/s {time_opt*1e-3:,.0f} us')

    return t_kq_max_wg, t_kq_exp_partial_sum

def test_func():
    global FIND_DEBUG_ACC
    FIND_DEBUG_ACC = 1
    create_kernels(True)

    q_head = HQ
    k_head = HK
    dim = HEAD_SIZE
    block_size = BLOCK_SIZE
    stride = STRIDE
    sizes = [
        (256 * STRIDE, 256 * STRIDE, '4k'),
        # normal case
        (512 * STRIDE, 512 * STRIDE, '8k'),
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

        q = torch.randint(-2, 4, size=[1, q_head, q_len, dim], dtype=torch.int16).to(dtype=torch.float16)
        # q = torch.arange(1*q_head*q_len*dim).reshape(1, q_head, q_len, dim).to(dtype=torch.float16)
        k = torch.randint(-2, 4, size=[1, k_head, k_len, dim], dtype=torch.int16).to(dtype=torch.float16)
        # k = torch.arange(1*k_head*k_len*dim).reshape(1, k_head, k_len, dim).to(dtype=torch.float16)

        if IS_CAUSAL:
            q_start_strided = k_len // stride - q_len // stride
            assert q_start_strided >= 0, "length of key cache must be greater or equal than query"
        else:
            q_start_strided = 0
        test_gemm(q, k, block_size=block_size, q_start_strided=q_start_strided, threshold=THRESH, stride=stride, causal=IS_CAUSAL, perf=False)

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
    global FIND_DEBUG_ACC
    FIND_DEBUG_ACC = 0
    create_kernels(True)

    # 84 T/s:
    bsz = 1 # must be 1(gemm_qk does not support batch which M,N may be vary for each batch)
    q_head = HQ
    k_head = HK
    dim = HEAD_SIZE
    block_size = BLOCK_SIZE
    stride = STRIDE

    Q_LEN = 1024*4*1  # if q_len=1024*4*2 ==> 95 T/s
    K_LEN = 1024*128
    q_len = Q_LEN
    k_len = K_LEN

    assert (q_len // stride) >= 128 and (k_len // stride) >= 128, "minimum block size should be 128*128"

    q = torch.randint(-2, 4, size=[bsz, q_head, q_len, dim], dtype=torch.int16).to(dtype=torch.float16)
    k = torch.randint(-2, 4, size=[bsz, k_head, k_len, dim], dtype=torch.int16).to(dtype=torch.float16)

    test_gemm(q, k, block_size=block_size, q_start_strided=k_len // stride - q_len // stride, threshold=THRESH, stride=stride, causal=IS_CAUSAL)

def main():
    test_func()
    test_perf()

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)

    cl.profiling(True)
    
    # requirements:
    # num_q_head == num_kv_head
    # chunk size alignment
    # causal_mask
    main()