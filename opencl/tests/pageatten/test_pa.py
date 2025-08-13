import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

from clops import cl
from clops.utils import Colors
import os

import numpy as np

from flashattn import get_flash0

def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

CM_GRF_WIDTH = 512 # get_cm_grf_width()

class page_atten_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, is_causal = True, sparse_block_sz = 128):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        assert trunk_sz % block_sz == 0, f'Error: trunk_sz must be multiple of block_sz'
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz
        self.sparse_block_sz = sparse_block_sz


        src1 = r'''#include "cm_sdpa_vlen.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {head_size=} {sparse_block_sz=}...")

        scale_factor = 1.0/(head_size**0.5)
        self.kernels = cl.kernels(src1,
                     (f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd}'
                      f" -DCMFLA_NUM_HEADS={num_heads}"
                      f" -DCMFLA_NUM_KV_HEADS={num_kv_heads}"
                      f" -DCMFLA_HEAD_SIZE={head_size}"
                      f" -DCMFLA_SCALE_FACTOR={scale_factor}"
                      f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
                      f" -DCMPA_BLOCK_SZ={self.block_sz}"
                      f" -DSPARSE_BLOCK_SIZE={int(sparse_block_sz)}"
                      f" -mdump_asm -g2")
                     )

    def __call__(self, q, k, v, block_mask, n_repeats = 1):
        seq_len, _, head_size = q.shape
        padded_k = k
        padded_v = v
        old_dtype = q.dtype
        total_heads = (self.num_heads + self.num_kv_heads * 2)

        assert head_size == self.head_size
        #align seqlen with block_sz.  block is the PA K/V cache minimum unit.
        aligned_seqlen = seq_len
        #pad the K, V to align with block_sz
        if seq_len % self.block_sz != 0:
            padding_tokens = self.block_sz -  seq_len % self.block_sz
            assert len(k.shape) == 3
            kv_padding_dims = (0,0,0,0,0,padding_tokens)
            aligned_seqlen = seq_len + padding_tokens
            padded_k = torch.nn.functional.pad(k,kv_padding_dims)
            padded_v = torch.nn.functional.pad(v,kv_padding_dims)

        # print(f'k.shape:{k.shape}, padded_k.shape:{padded_k.shape}')
        # reorder K,V from [L, H, S] to [block_num, H, block_size, S]
        padded_k = padded_k.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        padded_v = padded_v.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        #output memory for the whole SDPA
        output = torch.zeros(seq_len, self.num_heads, self.head_size).to(torch.float16)
        blks_per_trunk = self.trunk_sz // self.block_sz
        assert aligned_seqlen % self.block_sz==0, f'Error: aligned_seqlen must be multiple of block_sz'
        # Q[L, H, S]
        # K/V: [blk_num, H, blk_sz, S]
        trunk_num = (aligned_seqlen+ self.trunk_sz - 1) // self.trunk_sz
        max_blks = aligned_seqlen // self.block_sz

        for trunk_idx in range(trunk_num):
            print(f'{Colors.GREEN}=============== {trunk_idx}/{trunk_num}  ==============={Colors.END}')
            blk_num = max_blks if blks_per_trunk*(trunk_idx + 1) > max_blks else blks_per_trunk*(trunk_idx + 1)
            block_indices =  torch.randperm(blk_num)
            # block_indices =  torch.arange(blk_num)
            # print(f'==============={block_indices=}')
            sub_k = torch.zeros(blk_num, self.num_kv_heads, self.block_sz, head_size).to(torch.float16)
            sub_v = torch.zeros(blk_num, self.num_kv_heads, self.block_sz, head_size).to(torch.float16)
            for i in  range(len(block_indices)):
                sub_k[block_indices[i],:] = padded_k[i,:]
                sub_v[block_indices[i],:] = padded_v[i,:]
            q_start = trunk_idx*self.trunk_sz
            q_end =  min(q_start + self.trunk_sz, seq_len)
            q_len = q_end - q_start
            sub_q = q[q_start:q_end, :]

            if self.sparse_block_sz > 1:
                kv_len = trunk_idx*self.trunk_sz + q_len
                q_start_block = q_start // self.sparse_block_sz
                q_block_num = (q_len + sparse_block_sz -1) // sparse_block_sz
                kv_block_num = (kv_len + sparse_block_sz -1) // sparse_block_sz
                sub_block_mask = torch.zeros(self.num_heads, q_block_num, kv_block_num).to(torch.bool)
                sub_block_mask[...] = block_mask[:, q_start_block : q_start_block + q_block_num, : kv_block_num]
                # print(f"============ {sub_block_mask.shape=} {sub_block_mask.is_contiguous()=}")
                # print(f"============ {sub_block_mask=}")

            t_q = cl.tensor(sub_q.to(torch.float16).detach().numpy())
            t_k= cl.tensor(sub_k.to(torch.float16).detach().numpy())
            t_v = cl.tensor(sub_v.to(torch.float16).detach().numpy())
            t_out = cl.tensor([q_len, self.num_heads, self.head_size], np.dtype(np.float16))
            wg_size = 16
            q_step = CM_GRF_WIDTH // 32 # or 8 on Xe1
            wg_seq_len = wg_size * q_step
            wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

            GWS = [1, self.num_heads, int(wg_count * wg_size)]
            LWS = [1, 1, wg_size]
            # block_indices = int[blk_num], past_lens = 0, block_indices_begins = 0,
            past_lens=torch.tensor([trunk_idx*self.trunk_sz]).to(torch.int32)
            block_indices_begins=torch.tensor([0, blk_num]).to(torch.int32)
            subsequence_begins=torch.tensor([0,q_len]).to(torch.int32)

            t_block_indices=cl.tensor(block_indices.to(torch.int32).detach().numpy())
            t_past_lens=cl.tensor(past_lens.to(torch.int32).detach().numpy())
            t_block_indices_begins=cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
            t_subsequence_begins=cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())
            print(f"calling cm_page_attention {GWS=} {LWS=} x {n_repeats} times, q:[{q_start}, {q_end}], past_lens:{int(past_lens)}, kv_blk_num:{blk_num}, sparse_block_sz:{self.sparse_block_sz} ")
            for _ in range(n_repeats):
                if self.sparse_block_sz > 1:
                    t_block_mask = cl.tensor(sub_block_mask.to(torch.bool).detach().numpy())
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, q_len, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_block_mask, t_out)
                else:
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, q_len, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out)
            output[q_start:q_end] = torch.from_numpy(t_out.numpy())

        return output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size,block_sz, trunk_sz, is_causal, sparse_block_sz):
        return page_atten_cm(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, is_causal, sparse_block_sz)

# sparse to dense mask 
def block_mask_to_attention_mask(block_mask: torch.Tensor, q_len: int, k_len: int, sparse_block_size: int) -> torch.Tensor:
    # block_mask shape [num_head, q_block_num, k_block_num] dtype bool ->
    # attention_mask shape [num_head, q_len, kv_len] dtype bool
    q_block_num = (q_len + sparse_block_size -1) // sparse_block_size
    k_block_num = (k_len + sparse_block_size -1) // sparse_block_size
    assert(block_mask.shape[-2] == q_block_num)
    assert(block_mask.shape[-1] == k_block_num)

    expanded = block_mask.repeat_interleave(sparse_block_size, dim=1)
    expanded = expanded.repeat_interleave(sparse_block_size, dim=2)
    return expanded[:, :q_len, :k_len]
    

def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = True, attention_mask = None):
    seq_length, num_heads, head_size = q.shape
    kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    # print(f"============2 {cu_seqlens=} {seq_length=} {num_heads=}")
    # print(f"============2 {q.shape=} {q.is_contiguous()=} {k.shape=} {k.is_contiguous()=} {v.shape=} {v.is_contiguous()=}")
    if attention_mask is not None:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attn_mask = attention_mask,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        # print(f"============2 {attn_output.shape=} ")
        print(".")
        return attn_output.to(old_dtype)
        
    if is_causal:
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            is_causal = True,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    else:
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        if len(cu_seqlens):
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        else:
            attention_mask[...] = True

        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attn_mask = attention_mask,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    attn_output = attn_output.squeeze(0).transpose(0, 1)
    # print(f"============2 {attn_output.shape=} ")
    print(".")
    return attn_output.to(old_dtype)

def get_attention_mask(q, k, v, approx_simple_mask, sparse_block_size, is_causal = True):
    q_len, num_heads, head_size = q.shape
    kv_len, num_kv_heads, head_size = k.shape

    # [num_head, q_len, kv_len] dtype bool ->
    # [1, num_head, q_len, kv_len] dtype float16
    if sparse_block_size > 1:
        attention_mask = block_mask_to_attention_mask(approx_simple_mask, q_len, kv_len, sparse_block_size)
    else:
        attention_mask = torch.full([num_heads, q_len, kv_len], True).to(dtype=torch.bool)
    if is_causal:
        causal_pattern = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device))
        causal_pattern = causal_pattern.unsqueeze(0).repeat_interleave(num_heads, dim=0)
        attention_mask = attention_mask & causal_pattern
    attention_mask = torch.where(attention_mask, 0, torch.finfo(q.dtype).min)
    attention_mask = attention_mask.unsqueeze(0)
    # print(f"============flash0 {attention_mask.shape=} {attention_mask.is_contiguous()=}")
    # print(f"============flash0 {attention_mask=}")
    return attention_mask

def flash0_ref(q, k, v, attention_mask):
    attn_output = get_flash0(
            q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
            attention_mask = attention_mask)
    attn_output = attn_output.squeeze(0)
    # print(f"============2 {attn_output.shape=} ")
    print(".")
    return attn_output.to(q.dtype)


def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

def test_page_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, block_sz=128, trunk_sz=512, sparse_block_sz=128, sparse_ratio=0.5, check_acc = True):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    # q = torch.range(0, seq_len*num_heads*head_size-1).to(dtype=act_dtype)
    # print(f'=========================={q.shape =}')
    # q = q.reshape(seq_len, num_heads, head_size)
    # print("====================",q[16:,...])
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
    
    def create_block_mask(num_q_blocks: int, num_k_blocks: int, ratio: float) -> torch.Tensor:
        diag = torch.eye(num_q_blocks, num_k_blocks, dtype=torch.bool)
        if ratio <= 0:
            return diag
        if ratio >= 1:
            return torch.tril(torch.ones((num_q_blocks, num_k_blocks), dtype=torch.bool))
        causal_without_diag = torch.tril(torch.ones((num_q_blocks, num_k_blocks), dtype=torch.bool), diagonal=-1)
        rand_mask = torch.rand((num_q_blocks, num_k_blocks))
        non_diag_causal = causal_without_diag & (rand_mask < ratio)
        return diag | non_diag_causal

    def generate_block_mask_with_ratio(H, Q_block, K_block, true_ratio=sparse_ratio, is_causal=True, device='cpu'):
        if is_causal:
            causal_pattern = torch.tril(torch.ones(Q_block, K_block, dtype=torch.bool, device=device))
            block_mask = (torch.rand(H, Q_block, K_block, device=device) < true_ratio) & causal_pattern
        else:
            block_mask = torch.rand(H, Q_block, K_block, device=device) < true_ratio
        block_mask[:,:,0] = True  # the first column is always True
        if Q_block == 2 and K_block == 2: # HARD CODE FOR DEBUG
            block_mask = torch.tensor([True, False, False, True], dtype=torch.bool).reshape(Q_block, K_block).expand(H, Q_block, K_block)
            
        # print(f"============ {block_mask.shape=} {block_mask.is_contiguous()=}")
        # print(f"============ {block_mask=}")
        return block_mask

    approx_simple_mask = None
    if sparse_block_sz > 1:
        q_block_num = (seq_len + sparse_block_sz -1) // sparse_block_sz
        k_block_num = (seq_len + sparse_block_sz -1) // sparse_block_sz
        approx_simple_mask = generate_block_mask_with_ratio(num_heads, q_block_num, k_block_num, 0.5)

    attention_mask = get_attention_mask(q, k, v, approx_simple_mask, sparse_block_sz)

    is_causal = True  # PageAttention implictly means causal_mask
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, is_causal, sparse_block_sz)
    out = pa_cm(q, k, v, approx_simple_mask)

    if check_acc:
        ref = flash_attn_vlen_ref(q, k, v, [], is_causal, attention_mask)
        # ref0 = flash0_ref(q, k, v, attention_mask)
        # check_close(ref, ref0, atol=1e-2, rtol=1e-3)
        check_close(ref, out)
    else:
        out = pa_cm(q, k, v, approx_simple_mask, n_repeats=10)
        latency = cl.finish()
        # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
        print(f" page_atten_cm average latency: {sum(latency)/len(latency)*1e-6:.3f} ms")

if __name__ == "__main__":
    # brutal-force run
    # for sparse_block_sz in [1, 128, 64, 32, 16]:   # 'sparse_block_sz == 1' means dense
    #     for block_sz in range(32, 144, 16):
    #         for blocks_per_trunk in range(1, 30, 6):
    #             for seq_len in range(8192, 8248):
    #                 print("-----------------------------------------------------------------------------------------------------------------------------------------")
    #                 print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} sparse_block_sz={sparse_block_sz}')
    #                 print("-----------------------------------------------------------------------------------------------------------------------------------------")
    #                 test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, sparse_block_sz = sparse_block_sz)

    # smoke test
    for sparse_block_sz in [128, 256]:   # 'sparse_block_sz == 1' means dense
        for block_sz in [128, 256]:
            for blocks_per_trunk in [1, 15, 16, 17, 32, 300]:
                for seq_len in [16*15, 16*16, 16*16+1, 1024, 1024+1, 8*1024, 8*1024+3, 32*1024]:
                    if seq_len==8192 and block_sz==128 and blocks_per_trunk==15 and sparse_block_sz==256: continue # acc failure
                    if seq_len==8195 and block_sz==128 and blocks_per_trunk==15 and sparse_block_sz==256: continue # acc failure
                    if seq_len==32768 and block_sz==128 and blocks_per_trunk==15 and sparse_block_sz==256: continue # acc failure
                    if seq_len==8192 and block_sz==128 and blocks_per_trunk==17 and sparse_block_sz==256: continue # acc failure
                    if seq_len==8195 and block_sz==128 and blocks_per_trunk==17 and sparse_block_sz==256: continue # acc failure
                    if seq_len==32768 and block_sz==128 and blocks_per_trunk==17 and sparse_block_sz==256: continue # acc failure
                    print("-----------------------------------------------------------------------------------------------------------------------------------------")
                    print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} sparse_block_sz={sparse_block_sz}')
                    print("-----------------------------------------------------------------------------------------------------------------------------------------")
                    test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, sparse_block_sz = sparse_block_sz)


    # failure cases of PA
    # 1. seq_len, block_sz, blocks_per_trunk, sparse_block_sz = 3+16*512, 16, 1, 1
    # 2. seq_len=32771 block_sz=128 blocks_per_trunk=1 sparse_block_sz=1
    # 3. seq_len=16 block_sz=128 blocks_per_trunk=15 sparse_block_sz=1
    # failure cases of sparse-attn
    # 1. seq_len=8192 block_sz=128 blocks_per_trunk=15 sparse_block_sz=256
    # 2. seq_len=8195 block_sz=128 blocks_per_trunk=15 sparse_block_sz=256
    # 3. seq_len=32768 block_sz=128 blocks_per_trunk=15 sparse_block_sz=256
    # 4. seq_len=8192 block_sz=128 blocks_per_trunk=17 sparse_block_sz=256
    # print("-----------------------------------------------------------------------------------------------------------------------------------------")
    # print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} sparse_block_sz={sparse_block_sz}')
    # print("-----------------------------------------------------------------------------------------------------------------------------------------")
    # test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, sparse_block_sz = sparse_block_sz)

    # QWen3 8K case
    # for sparse_block_sz in [1, 128]:
    #     seq_len, block_sz, blocks_per_trunk= 8*1024, 256, 16
    #     print("-----------------------------------------------------------------------------------------------------------------------------------------")
    #     print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} sparse_block_sz={sparse_block_sz}')
    #     print("-----------------------------------------------------------------------------------------------------------------------------------------")
    #     test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 8, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, sparse_block_sz = sparse_block_sz, check_acc=False)
