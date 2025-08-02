import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

from clops import cl
import os

import numpy as np

def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

CM_GRF_WIDTH = get_cm_grf_width()

class flash_attn_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, is_causal = False):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        self.block_sz = 64
        self.trunk_sz = 512

        src1 = r'''#include "cm_sdpa_vlen.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {head_size=} ...")

        scale_factor = 1.0/(head_size**0.5)
        self.kernels = cl.kernels(src1,
                     (f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd}'
                      f" -DCMFLA_NUM_HEADS={num_heads}"
                      f" -DCMFLA_NUM_KV_HEADS={num_kv_heads}"
                      f" -DCMFLA_HEAD_SIZE={head_size}"
                      f" -DCMFLA_SCALE_FACTOR={scale_factor}"
                      f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
                      f" -DCMPA_BLOCK_SZ={self.block_sz}"
                      f" -mdump_asm -g2")
                     )

    def qkv_fused(self, q, k, v, n_repeats = 1):
        seq_len, _, head_size = q.shape
        trunk_sz = self.trunk_sz
        trunk_num = seq_len // trunk_sz
        assert trunk_sz%self.block_sz == 0 and  seq_len%trunk_sz==0, f'Error: trunk_sz must be multiple of trunk_sz/block_sz'

        old_dtype = q.dtype
        total_heads = (self.num_heads + self.num_kv_heads * 2)
        assert head_size == self.head_size

        k = k.reshape(seq_len//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        v = v.reshape(seq_len//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        output = torch.zeros(seq_len, self.num_heads, self.head_size).to(torch.float16)
        blks_per_trunk = trunk_sz // self.block_sz

        # Q[L, H, S]
        # K/V: [blk_num, H, blk_sz, S]
        for trunk_idx in range(trunk_num):
            blk_num = blks_per_trunk * (trunk_idx + 1)
            # block_indices =  torch.randperm(blk_num)
            block_indices =  torch.arange(blk_num)
             # print(block_indices)
            sub_k = torch.zeros(blk_num, self.num_kv_heads, self.block_sz, head_size).to(torch.float16)
            sub_v = torch.zeros(blk_num, self.num_kv_heads, self.block_sz, head_size).to(torch.float16)
            for i in  range(len(block_indices)):
                sub_k[block_indices[i],:] = k[i,:]
                sub_v[block_indices[i],:] = v[i,:]
            q_start = trunk_idx*trunk_sz
            q_end =  q_start + trunk_sz
            sub_q = q[q_start:q_end, :]

            t_q = cl.tensor(sub_q.to(torch.float16).detach().numpy())
            t_k= cl.tensor(sub_k.to(torch.float16).detach().numpy())
            t_v = cl.tensor(sub_v.to(torch.float16).detach().numpy())
            t_out = cl.tensor([trunk_sz, self.num_heads, self.head_size], np.dtype(np.float16))
            wg_size = 16
            q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
            wg_seq_len = wg_size * q_step
            wg_count = (trunk_sz + wg_seq_len - 1) // wg_seq_len

            GWS = [1, self.num_heads, int(wg_count * wg_size)]
            LWS = [1, 1, wg_size]
            # block_indices = int[blk_num], past_lens = 0, block_indices_begins = 0,
            past_lens=torch.tensor([trunk_idx*trunk_sz]).to(torch.int32)
            block_indices_begins=torch.tensor([0, blk_num]).to(torch.int32)
            subsequence_begins=torch.tensor([0,seq_len]).to(torch.int32)

            t_block_indices=cl.tensor(block_indices.to(torch.int32).detach().numpy())
            t_past_lens=cl.tensor(past_lens.to(torch.int32).detach().numpy())
            t_block_indices_begins=cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
            t_subsequence_begins=cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())
            q_len = trunk_sz
            print(f"calling qkv_fused {GWS=} {LWS=} x {n_repeats} times")
            for _ in range(n_repeats):
                self.kernels.enqueue("cm_sdpa_qkv_fused", GWS, LWS, q_len, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out)
            output[q_start:q_end] = torch.from_numpy(t_out.numpy())

        return output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, is_causal):
        return flash_attn_cm(num_heads, num_kv_heads, head_size, is_causal)

def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = False):
    seq_length, num_heads, head_size = q.shape
    kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    # print(f"============2 {cu_seqlens=} {seq_length=} {num_heads=}")
    # print(f"============2 {q.shape=} {q.is_contiguous()=} {k.shape=} {k.is_contiguous()=} {v.shape=} {v.is_contiguous()=}")
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


def flash_attn(q, k, v, cu_seqlens):
    _, num_heads, head_size = q.shape
    func = flash_attn_cm.create_instance(num_heads, head_size)
    # return flash_attn_vlen_ref(q, k, v, cu_seqlens)
    return func(q, k, v, cu_seqlens)


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

def test_flash_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    import numpy as np

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    is_causal = True
    ref = flash_attn_vlen_ref(q, k, v, [], is_causal=is_causal)

    func = flash_attn_cm.create_instance(num_heads, num_kv_heads, head_size, is_causal)


    out = func.qkv_fused(q, k, v)
    # out = func.qkv_fused(q, k, v, n_repeats=20)
    # latency = cl.finish()
    # # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    # print(f" qkv_fused_causal {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    # print(ref)
    # print(out)
    check_close(ref, out)
    #assert 0

if __name__ == "__main__":
    test_flash_attn_causal_batch1(seq_len=8192, num_heads = 28, num_kv_heads = 4, head_size = 128)
