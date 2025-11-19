import os

import torch
import numpy as np

from clops import cl
from clops import compare
from clops.utils import Colors

from references import get_partial_softmax_ref, find_blocks_ref

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

BLOCK_SG_M = 64 #32
BLOCK_SG_N = 32
SG_M = 4
SG_N = 8
BLOCK_WG_M = BLOCK_SG_M * SG_M
BLOCK_WG_N = BLOCK_SG_N * SG_N

HQ = 32
HK = 8
HEAD_SIZE = 128
STRIDE = 16
BLOCK_SIZE = 128
THRESH = 0.9
IS_CAUSAL = 1
SOFTMAX_TYPE = 'float' # 'half'

FIND_DEBUG_ACC = 0 # only acc test needed

kernels = None
def create_kernels(force_create=False):
    global kernels
    if kernels and not force_create: return

    src = r'''#include "xattn_find_block.hpp"'''
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"compiling {cwd} ...")

    jit_option = '-abortonspill -noschedule '
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size=256 -I{cwd}
                        -DSTRIDE={STRIDE} -DHQ={HQ} -DHK={HK} -DHEAD_SIZE={HEAD_SIZE} 
                        -DBLOCK_SIZE={BLOCK_SIZE} -DBLOCK_SHARE_MAX={BLOCK_WG_N} 
                        -DDEBUG_ACC={FIND_DEBUG_ACC} -DIS_CAUSAL={IS_CAUSAL} -DSOFTMAX_TYPE={SOFTMAX_TYPE}
                        ''')

def cmp_mask(ref_mask_np, cur_mask_np, ref_sum, cur_exp_partial_sum_np):
    sum_per_token_in_block = BLOCK_SIZE // STRIDE
    diff_idx = np.where(ref_mask_np != cur_mask_np)
    if diff_idx[0].shape[0]:
        # case 1: if 2+ values are very close, anyone selected is valid although its index is different
        diff_idx_t = torch.tensor(np.array(diff_idx)).transpose(0, 1)
        unique_rows, counts_rows = torch.unique(diff_idx_t[:,:-1], dim=0, return_counts=True)
        idxes = []
        repeated_rows = unique_rows[counts_rows > 1]
        for repeated_row in repeated_rows:
            vals = []
            full_pos = []
            for idx in range(diff_idx_t.shape[0]):
                if torch.all(diff_idx_t[idx][:-1] == repeated_row):
                    pos = diff_idx_t[idx].tolist()
                    vals.append(ref_sum[*pos])
                    full_pos.append(pos)
                    idxes.append(idx)
            if torch.allclose(torch.tensor(vals, dtype=vals[0].dtype), vals[0], atol=0.01):
                print(f'{Colors.YELLOW}similar float detected:{Colors.END} idx={full_pos}, vals={vals}')
            else:
                print(f'{Colors.RED}not close float detected:{Colors.END} idx={full_pos}, vals={vals}')
                raise Exception('failed')
        if len(idxes):
            indices_to_remove = torch.tensor(idxes)
            mask_one = np.ones_like(diff_idx[0], dtype=np.bool)
            mask_one[indices_to_remove] = False
            diff_idx = (diff_idx[0][mask_one], diff_idx[1][mask_one], diff_idx[2][mask_one], diff_idx[3][mask_one])

        ref_err_rows = torch.from_numpy(ref_sum.detach().numpy()[diff_idx[:3]])
        #cur_err_rows = torch.from_numpy(cur_sum_np[diff_idx[:3]])
        #cur_thresh = cur_err_rows.sum(dim=-1) * THRESH
        cur_thresh = ref_thresh = ref_err_rows.sum(dim=-1) * THRESH

        kq_exp_partial_sum_np = cur_exp_partial_sum_np
        cur_sorted_value = kq_exp_partial_sum_np[:, :, 1::sum_per_token_in_block, :].view(dtype=np.float16)
        cur_sorted_index = kq_exp_partial_sum_np[:, :, 3::sum_per_token_in_block, :].view(dtype=np.ushort)
        cur_accum_value  = kq_exp_partial_sum_np[:, :, 6::sum_per_token_in_block, :].view(dtype=np.float16)
        if not FIND_DEBUG_ACC:
            print(f'{Colors.RED}please set {Colors.BLUE}FIND_DEBUG_ACC=1{Colors.END} to get `cur_accum_value` for checking if there is an accuracy problem{Colors.END}')
            raise Exception('please set FIND_DEBUG_ACC=1')
        if SOFTMAX_TYPE == 'float':
            # results only uses 2 bytes
            cur_sorted_index = cur_sorted_index[...,:cur_sorted_index.shape[-1] // 2].copy()
            cur_accum_value = cur_accum_value[...,:cur_accum_value.shape[-1] // 2].copy()
        # first 2 elements are reserved for first and diag position
        cur_sorted_index[...,:2] = 65535
        # index(X) of different k_blocks 
        error_last_pos_np = np.array(diff_idx[3]).reshape([-1, 1])
        # the index(Y) of X in sorted list
        error_last_idx_np = np.where(cur_sorted_index[diff_idx[:3]] == error_last_pos_np)
        error_idx_np = (*diff_idx[:3], error_last_idx_np[-1])
        # lookup accum value using index Y in cumsum list
        error_accum = cur_accum_value[error_idx_np]
        # case 2: if accum is very close to thresh, it should be a minor float error
        if np.allclose(error_accum, cur_thresh, rtol=0.01, atol=0.01):
            print(f'{Colors.YELLOW}minor float error detected:{Colors.END} idx={diff_idx}, accum={error_accum}, thresh={cur_thresh}')
        else:
            print(f'{Colors.RED}mask is not same:{Colors.END} idx={diff_idx}, accum={error_accum}, thresh={cur_thresh}')
            raise Exception('mask is not same')


# q_stride = q_len // stride
# k_stride = k_len // stride
def test_find(q_len, q_stride, k_stride, block_size, S, wg_k, wg_q, perf, causal):
    create_kernels()

    qk = torch.randint(-2000, 3000, size=[1, HQ, q_stride, k_stride], dtype=torch.int16).to(dtype=torch.float32)
    assert wg_k % block_size == 0, "wg_k should be multiple of block_size then there is no tails from block_size"
    assert wg_q % block_size == 0, "wg_q should be multiple of block_size then there is no tails from block_size"
    qk_max, kq_5d_max, qk_exp_partial_sum, qk_sum = get_partial_softmax_ref(qk, block_size, S, wg_k, wg_q, valid_q=q_stride)
    softmax_type = np.float16 if SOFTMAX_TYPE == 'half' else np.float32
    t_kq_max_wg = cl.tensor(kq_5d_max.detach().numpy().astype(softmax_type))
    t_kq_exp_partial_sum = cl.tensor(qk_exp_partial_sum.detach().numpy().astype(softmax_type))

    sum_per_n_token_in_block = BLOCK_SIZE // STRIDE
    q_block = div_up(q_stride, sum_per_n_token_in_block)
    k_block = div_up(k_stride, sum_per_n_token_in_block)
    assert k_block >= q_block, "k block should be larger than q_block"
    mask_ref, sorted_value_ref, sorted_index_ref, cumulative_sum_without_self, required_sum = find_blocks_ref(qk_sum, THRESH, q_block, k_block, causal=causal, current_index=k_block-q_block)

    # [1, 32, 256, 64 * 16]
    B, _, q_stride_pad, k_block_pad = t_kq_exp_partial_sum.shape
    assert k_block_pad == (k_stride + wg_k - 1) // wg_k * wg_k // sum_per_n_token_in_block, "k_block padded to wg_k / block_size"
    assert q_stride_pad == (q_stride + wg_q - 1) // wg_q * wg_q, "q_stride_pad padded to wg_q / stride"
    q_block_input = q_stride_pad // sum_per_n_token_in_block
    q_block_pad = div_up(q_len, BLOCK_SIZE)

    # [1, 32, 32, 64 * 16]
    t_kq_sum = cl.tensor(np.zeros([B, HQ, q_block_input, k_block_pad], dtype=np.float16))
    # mask shape: [rnd_up(q_stride, WG_M) / 8, rnd_up(k_stride, WG_N) / 8]
    t_mask = cl.tensor(np.ones([B, HQ, q_block_pad, k_block_pad], np.int8) * 100)
    params = [t_kq_max_wg, t_kq_exp_partial_sum, t_mask, q_len, q_stride, q_stride_pad, q_block_pad, k_block_pad, k_block-q_block, THRESH]
    if FIND_DEBUG_ACC:
        params += [t_kq_sum]
    # find
    for i in range(1):
        kernels.enqueue("find_block", [q_block_pad, HQ, B], [1, 1, 1], *params)
    cl.finish()
    if not perf:
        if FIND_DEBUG_ACC == 1:
            compare(qk_sum.detach().numpy(), t_kq_sum.numpy())
            print(f'{Colors.GREEN}find:sum passed{Colors.END}')

        kq_exp_partial_sum_np = t_kq_exp_partial_sum.numpy()
        cur_sorted_value = kq_exp_partial_sum_np[:, :, 1::sum_per_n_token_in_block, :].view(dtype=np.float16)
        cur_sorted_index = kq_exp_partial_sum_np[:, :, 3::sum_per_n_token_in_block, :].view(dtype=np.ushort)
        cur_accum_value  = kq_exp_partial_sum_np[:, :, 6::sum_per_n_token_in_block, :].view(dtype=np.float16)
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
        if q_block != q_block_pad:
            assert np.all(cur_mask[:,:,-(q_block_pad - q_block):,:] == 1), f"new pad value must be one"
        cur_mask = cur_mask[...,:q_block,:k_block]
        if not np.all(cur_mask == mask_ref.to(torch.int8).detach().numpy()):
            cmp_mask(mask_ref.to(torch.int8).detach().numpy(), cur_mask, qk_sum, kq_exp_partial_sum_np)
        # minor error may get different mask index
        # mask_ref_np =  mask_ref.detach().numpy()
        # cur_mask = tMask.numpy()
        # error_idx = np.where(mask_ref_np != cur_mask)
        # if error_idx[0].shape[0]:
        #     if not np.allclose(A_sum[error_idx], tC_sum.numpy()[error_idx], atol=0.01, rtol=0.01):
        #         print(f'{error_idx=}\nidx_ref={mask_ref_np[error_idx]}\nidx_cur={cur_mask[error_idx]}')
        #         raise Exception(f'the diff value should be smaller than 0.01, ref={A_sum[error_idx]} cur={tC_sum.numpy()[error_idx]}')
        # print(f'{Colors.GREEN}find:mask passed{Colors.END}')

    else:
        for i in range(0, 100):
            kernels.enqueue("find_block", [q_block_pad, HQ, B], [1, 1, 1], *params)
        ns = cl.finish()
        for i, time_opt in enumerate(ns):
            print(f'(FIND)TPUT_{i}: {time_opt*1e-3:,.0f} us')

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

        if IS_CAUSAL:
            q_start_strided = k_len // stride - q_len // stride
            assert q_start_strided >= 0, "length of key cache must be greater or equal than query"
        else:
            q_start_strided = 0
        test_find(q_len, q_len // stride, k_len // stride, block_size, stride, wg_k=BLOCK_WG_M, wg_q=BLOCK_WG_N, perf=False, causal=IS_CAUSAL)

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
    test_find(q_len, q_len // stride, k_len // stride, block_size, stride, wg_k=BLOCK_WG_M, wg_q=BLOCK_WG_N, perf=True, causal=IS_CAUSAL)

def main():
    test_func()
    test_perf()

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)

    cl.profiling(True)

    main()