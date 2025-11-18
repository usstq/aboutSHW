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

CM_GRF_WIDTH = get_cm_grf_width()
print(f"{CM_GRF_WIDTH=}")
if CM_GRF_WIDTH == 256:
    xe_arch = 1
else:
    xe_arch = 2

def quan_per_token(kv):
    blk_num, kv_heads, blksz, *_ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim = True)
    kv_min = kv.amin(dim=-1, keepdim = True)
    qrange = kv_max - kv_min

    INTMAX = 255.0
    INTMIN = 0.0
    INTRAGNE = INTMAX - INTMIN
    kv_scale = ((INTRAGNE)/qrange).to(dtype=torch.half)
    kv_zp = ((0.0-kv_min)*kv_scale+INTMIN).to(dtype=torch.half)
    #round half to even
    kv_INT8 = torch.round((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    # print("################################################################################")
    # print(f'KV\n:{kv.reshape(64, 16)}')
    # print(f'kv_INT8\n:{kv_INT8.reshape(64, 16)}')
    # print(f'kv_scale\n:{kv_scale.reshape( 1, 64)}')
    # print(f'kv_zp\n:{kv_zp.reshape( 1, 64)}')
    # print("################################################################################")

    dq_scale = (1.0/kv_scale).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    return torch.concat((kv_INT8, dq_scale, kv_zp), dim=-1)

def dequant_per_token(kv, head_size, blk_size):
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:,:,:head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:,:,head_size * blk_size:head_size * blk_size + blk_size * 2].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)
    kv_zp = kv[:,:,head_size * blk_size + blk_size * 2:head_size * blk_size + blk_size * 4].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)

    # print("dequant_kv_u8 = ", kv_u8)
    # print("dequant_kv_scale = ", kv_scale.reshape(blk_num, kv_head_num, blk_size))
    # print("dequant_kv_zp    = ", kv_zp.reshape(blk_num, kv_head_num, blk_size))

    kv_dequant = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)

    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(blk_size):
                kv_dequant[m,n,i,:] = (kv_u8[m,n,i,:].to(dtype=torch.float16) - kv_zp[m,n,i,0].to(dtype=torch.float16)) * kv_scale[m,n,i,0].to(dtype=torch.float16)

    return kv_dequant

def ALIGN_UP(x, y):
    return (x + y -1) // y * y

def DIV_UP(x, y):
    return (x + y -1) // y
class page_atten_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal = True, sparse_block_sz = 128):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        assert trunk_sz % block_sz == 0, f'Error: trunk_sz must be multiple of block_sz'
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz
        self.sparse_block_sz = sparse_block_sz
        self.compressed_kvcache = compressed_kvcache

        src1 = r'''#include "cm_pa_kernel.hpp"'''
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
                      f" -DCMPA_KVCACHE_U8={int(compressed_kvcache)}"
                      f" -DCMPA_XE_ARCH={int(xe_arch)}"
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
            padded_k = torch.nn.functional.pad(k,kv_padding_dims, "constant", 1)
            padded_v = torch.nn.functional.pad(v,kv_padding_dims, "constant", 1)
            #padding all to NAN to simulate the NAN case  when fp16
            if self.compressed_kvcache == False:
                padded_k.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00
                padded_v.view(torch.uint16)[seq_len:aligned_seqlen] = 0xfe00

        # print(f'k.shape:{k.shape}, padded_k.shape:{padded_k.shape}')
        # reorder K,V from [L, H, S] to [block_num, H, block_size, S]
        k_cache = padded_k.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        v_cache = padded_v.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        if self.compressed_kvcache:
            k_cache = quan_per_token(k_cache)
            v_cache = quan_per_token(v_cache)
        else:
            k_cache = k_cache.reshape(aligned_seqlen//self.block_sz, self.num_kv_heads, -1)
            v_cache = v_cache.reshape(aligned_seqlen//self.block_sz, self.num_kv_heads, -1)
        #output memory for the whole SDPA
        output = torch.zeros(seq_len, self.num_heads, self.head_size).to(torch.float16)
        blks_per_trunk = self.trunk_sz // self.block_sz
        assert aligned_seqlen % self.block_sz==0, f'Error: aligned_seqlen must be multiple of block_sz'
        # Q[L, H, S]
        # K/V: [blk_num, H, blk_sz, S]
        trunk_num = (aligned_seqlen+ self.trunk_sz - 1) // self.trunk_sz
        max_blks = aligned_seqlen // self.block_sz

        kv_dtype = torch.uint8 if self.compressed_kvcache else torch.half
        #extra half zp and half scale per token. totally 4 bytes.
        token_sz = (head_size+4) if self.compressed_kvcache else (head_size)

        if self.sparse_block_sz > 1:
            block_mask_list = []
            block_mask_in_wg_list = []
            for trunk_idx in range(trunk_num):
                q_start = trunk_idx*self.trunk_sz
                q_end =  min(q_start + self.trunk_sz, seq_len)
                q_len = q_end - q_start

                wg_size = 16
                q_step = CM_GRF_WIDTH // 32 # or 8 on Xe1
                wg_seq_len = wg_size * q_step
                wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

                kv_len = trunk_idx*self.trunk_sz + q_len
                q_start_block = 0
                q_block_num = (q_len + self.sparse_block_sz -1) // self.sparse_block_sz
                kv_block_num = (kv_len + self.sparse_block_sz -1) // self.sparse_block_sz
                sub_block_mask = torch.zeros(self.num_heads, q_block_num, kv_block_num).to(torch.bool)
                sub_block_mask[...] = block_mask[trunk_idx, :, q_start_block : q_start_block + q_block_num, : kv_block_num]

                sub_block_mask_in_wg = torch.zeros(self.num_heads, wg_count, kv_block_num).to(torch.bool)

                for head_idx in range(0, self.num_heads):
                    for wg_id in range(0, wg_count):
                        # default skip all the blocks.
                        sub_block_mask_in_wg[head_idx, wg_id, :] = False
                        qblk_start_wg = wg_id*wg_seq_len // self.sparse_block_sz
                        qblk_end_wg = DIV_UP(min(wg_id*wg_seq_len + wg_seq_len, q_len), self.sparse_block_sz)
                        # print(f'############qblk_start_wg = {qblk_start_wg}, qblk_end_wg = {qblk_end_wg}, submask shape = {sub_block_mask.shape}')
                        for kv_blk_idx in range(0, kv_block_num):
                            for qblk_idx in range(qblk_start_wg, qblk_end_wg):
                                if sub_block_mask[head_idx, qblk_idx, kv_blk_idx] == True:
                                    sub_block_mask_in_wg[head_idx, wg_id, kv_blk_idx] = True
                # print(f'{sub_block_mask=}')
                # print(f'{sub_block_mask_in_wg=}')
                block_mask_list.append(sub_block_mask)
                block_mask_in_wg_list.append(sub_block_mask_in_wg)

        cl.finish()

        for _ in range(n_repeats):
            for trunk_idx in range(trunk_num):
                print(f'{Colors.GREEN}=============== {trunk_idx}/{trunk_num}  ==============={Colors.END}')
                blk_num = max_blks if blks_per_trunk*(trunk_idx + 1) > max_blks else blks_per_trunk*(trunk_idx + 1)
                block_indices =  torch.randperm(blk_num)
                # block_indices =  torch.arange(blk_num)
                # print(f'==============={block_indices=}')
                sub_k = torch.zeros(blk_num, self.num_kv_heads, self.block_sz*token_sz).to(kv_dtype)
                sub_v = torch.zeros(blk_num, self.num_kv_heads, self.block_sz*token_sz).to(kv_dtype)
                for i in  range(len(block_indices)):
                    sub_k[block_indices[i],:] = k_cache[i,:]
                    sub_v[block_indices[i],:] = v_cache[i,:]
                q_start = trunk_idx*self.trunk_sz
                q_end =  min(q_start + self.trunk_sz, seq_len)
                q_len = q_end - q_start
                sub_q = q[q_start:q_end, :]

                wg_size = 16
                q_step = CM_GRF_WIDTH // 32 # or 8 on Xe1
                wg_seq_len = wg_size * q_step
                wg_count = (q_len + wg_seq_len - 1) // wg_seq_len

                # if self.sparse_block_sz > 1:
                #     kv_len = trunk_idx*self.trunk_sz + q_len
                #     q_start_block = 0
                #     q_block_num = (q_len + self.sparse_block_sz -1) // self.sparse_block_sz
                #     kv_block_num = (kv_len + self.sparse_block_sz -1) // self.sparse_block_sz
                #     sub_block_mask = torch.zeros(self.num_heads, q_block_num, kv_block_num).to(torch.bool)
                #     sub_block_mask[...] = block_mask[trunk_idx, :, q_start_block : q_start_block + q_block_num, : kv_block_num]

                #     sub_block_mask_in_wg = torch.zeros(self.num_heads, wg_count, kv_block_num).to(torch.bool)
                #     for head_idx in range(0, self.num_heads):
                #         for wg_id in range(0, wg_count):
                #             # default skip all the blocks.
                #             sub_block_mask_in_wg[head_idx, wg_id, :] = False
                #             qblk_start_wg = DIV_UP(wg_id*wg_seq_len, self.sparse_block_sz)
                #             qblk_end_wg = DIV_UP(min(wg_id*wg_seq_len + wg_seq_len, q_len), self.sparse_block_sz)
                #             # print(f'############qblk_start_wg = {qblk_start_wg}, qblk_end_wg = {qblk_end_wg}, submask shape = {sub_block_mask.shape}')
                #             for kv_blk_idx in range(0, kv_block_num):
                #                 for qblk_idx in range(qblk_start_wg, qblk_end_wg):
                #                     if sub_block_mask[head_idx, qblk_idx, kv_blk_idx] == True:
                #                         sub_block_mask_in_wg[head_idx, wg_id, kv_blk_idx] = True

                    # print(f"============ {sub_block_mask.shape=} {sub_block_mask.is_contiguous()=}")
                    # print(f"============ {sub_block_mask=}")

                t_q = cl.tensor(sub_q.to(torch.float16).detach().numpy())
                t_k= cl.tensor(sub_k.to(kv_dtype).detach().numpy())
                t_v = cl.tensor(sub_v.to(kv_dtype).detach().numpy())
                t_out = cl.tensor([q_len, self.num_heads, self.head_size], np.dtype(np.float16))

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

                kv_precision = "U8" if self.compressed_kvcache else "F16"
                print(f"calling cm_page_attention {GWS=} {LWS=} x {n_repeats} times, q:[{q_start}, {q_end}], past_lens:{int(past_lens)}, kv_blk_num:{blk_num}, sparse_block_sz:{self.sparse_block_sz} kv_cache:{kv_precision}")
                if self.sparse_block_sz > 1:
                    t_block_mask = cl.tensor(block_mask_list[trunk_idx].to(torch.bool).detach().numpy())
                    t_block_mask_in_wg  = cl.tensor(block_mask_in_wg_list[trunk_idx].to(torch.bool).detach().numpy())
                    validate = True
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, t_block_mask, t_block_mask_in_wg, q_len, t_block_mask.shape[1], t_block_mask.shape[2], validate)
                else:
                    self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, q_len)
                output[q_start:q_end] = torch.from_numpy(t_out.numpy())

        return output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size,block_sz, trunk_sz, compressed_kvcache, is_causal, sparse_block_sz):
        return page_atten_cm(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal, sparse_block_sz)

# sparse to dense mask
def block_mask_to_attention_mask(block_mask: torch.Tensor, q_len: int, kv_len: int, sparse_block_size: int, trunk_sz: int) -> torch.Tensor:
    # block_mask shape [num_trunks, num_head, q_block_num, k_block_num] dtype bool ->
    # attention_mask shape [num_head, q_len, kv_len] dtype bool
    trunk_num, num_head, q_block_num, kv_block_num = block_mask.shape
    assert((q_len + trunk_sz -1) // trunk_sz == trunk_num)
    assert((trunk_sz + sparse_block_size -1) // sparse_block_size == q_block_num)
    assert((kv_len + sparse_block_size -1) // sparse_block_size == kv_block_num)

    attention_mask = torch.zeros([num_head, q_len, kv_len], device=block_mask.device, dtype=torch.bool)
    for trunk_idx in range(trunk_num):
        expanded = block_mask[trunk_idx].repeat_interleave(sparse_block_size, dim=1)
        expanded = expanded.repeat_interleave(sparse_block_size, dim=2)
        q_start = trunk_idx * trunk_sz
        q_end = min(q_start + trunk_sz, q_len)
        remaining = q_end - q_start
        attention_mask[ : , q_start : q_end, : ] = expanded[:, :remaining, :kv_len]

    return attention_mask


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

def get_attention_mask(q, k, v, approx_simple_mask, sparse_block_size, trunk_sz, is_causal = True):
    q_len, num_heads, head_size = q.shape
    kv_len, num_kv_heads, head_size = k.shape

    # [num_head, q_len, kv_len] dtype bool ->
    # [1, num_head, q_len, kv_len] dtype float16
    if sparse_block_size > 1:
        attention_mask = block_mask_to_attention_mask(approx_simple_mask, q_len, kv_len, sparse_block_size, trunk_sz)
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


def check_close(input, other, atol=1e-2, rtol=1e-2):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        print(input)
        print(other)
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

def count_false_percentage(mask):
    B, H, NQ, NL = mask.shape
    tril_mask = torch.tril(torch.ones((NQ, NL), dtype=torch.bool, device=mask.device))
    expanded_tril = tril_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    # Count elements in the tril region
    tril_elements = torch.sum(expanded_tril).item()
    # Count False elements in the tril region
    false_in_tril = torch.sum(~mask & expanded_tril).item()
    # Calculate percentage
    if tril_elements > 0:
        false_percentage = (false_in_tril / tril_elements) * 100
    else:
        false_percentage = 0.0
    return false_percentage

def test_page_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, block_sz=128, trunk_sz=512, compressed_kvcache=False, sparse_block_sz=128, sparse_ratio=0.5, check_acc = True):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    density = 1.0

    if compressed_kvcache:
        k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / 4.0
        k[0:seq_len:3, :, :] = (k[0:seq_len:3, :, :] + 0.25)/ 2.0
        v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        v[0:seq_len:3, :, :] = (v[0:seq_len:3, :, :] + 0.25)/ 2.0
    else:
        k = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)
        v = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)/high

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

    def generate_block_mask_with_ratio(num_heads, seq_len, trunk_sz, true_ratio=sparse_ratio, is_causal=True, device='cpu'):
        trunk_num = (seq_len + trunk_sz -1) // trunk_sz
        q_block_num = (trunk_sz + sparse_block_sz -1) // sparse_block_sz
        k_block_num = (seq_len + sparse_block_sz -1) // sparse_block_sz

        if true_ratio <= 0:
            block_mask = torch.full([trunk_num, num_heads, q_block_num, k_block_num], False).to(dtype=torch.bool)
        elif true_ratio >= 1:
            block_mask = torch.full([trunk_num, num_heads, q_block_num, k_block_num], True).to(dtype=torch.bool)
        else:
            block_mask = torch.rand(trunk_num, num_heads, q_block_num, k_block_num, device=device) < true_ratio

        if is_causal:
            causal_pattern = torch.tril(torch.ones(q_block_num, k_block_num, dtype=torch.bool, device=device))
            block_mask = block_mask & causal_pattern
        else:
            block_mask = block_mask
        block_mask[:,:,:,0] = True  # the first column is always True
        # if q_block_num == 2 and k_block_num == 2: # HARD CODE FOR DEBUG
        #     block_mask = torch.tensor([True, False, False, True], dtype=torch.bool).reshape(q_block_num, k_block_num).expand(trunk_num, num_heads, q_block_num, k_block_num)

        return block_mask

    approx_simple_mask = None
    if sparse_block_sz > 1:
        approx_simple_mask = generate_block_mask_with_ratio(num_heads, seq_len, trunk_sz, 1.0 - sparse_ratio)
        percentage = count_false_percentage(approx_simple_mask)
        # print(f"Percentage of False elements: {percentage:.2f}%")
        # print(f"============ block_mask.shape={approx_simple_mask.shape}")
        # print(f"============ block_mask={approx_simple_mask}")
        density = 1.0 - percentage / 100.0

    is_causal = True  # PageAttention implictly means causal_mask
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal, sparse_block_sz)
    out = pa_cm(q, k, v, approx_simple_mask)
    latency = cl.finish()

    if check_acc:
        attention_mask = get_attention_mask(q, k, v, approx_simple_mask, sparse_block_sz, trunk_sz)
        ref = flash_attn_vlen_ref(q, k, v, [], is_causal, attention_mask)
        # ref0 = flash0_ref(q, k, v, attention_mask)
        # check_close(ref, ref0, atol=1e-2, rtol=1e-3)
        # print(ref)
        # print(out)
        check_close(ref, out)
    else:
        rep = 15
        out = pa_cm(q, k, v, approx_simple_mask, n_repeats=rep)
        latency = cl.finish()
        trunks = len(latency) // rep
        trunk_lat = []
        # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
        if rep >= 15:
            print(f'====================================================================================')
            flops = 1 * num_heads * seq_len * seq_len * head_size * 2 * density
            for trunk_idx in range(trunks):
                cur_density = 1.0 - count_false_percentage(approx_simple_mask[trunk_idx:trunk_idx+1]) / 100.0  if sparse_block_sz > 1 else 1.0
                cur_flops = 1 * num_heads * trunk_sz * (trunk_sz*(trunk_idx+1)) * head_size * 2 * cur_density
                lat = latency[trunk_idx:-1:trunks]
                avg = sum(lat[10:])/len(lat[10:])*1e-6
                trunk_lat.append(avg)
                print(f'[trunk {trunk_idx}] density {cur_density:.2f}, MFU {cur_flops/(avg*1e6):,.0f} GFLOPS, average latency: {avg:.3f} ms')
            total_lat = sum(trunk_lat)
            print(f"[total]: PA_causal {seq_len=} , {trunks=}, density {density:.2f}, compressKVCache {compressed_kvcache}, MFU {flops/(total_lat*1e6):,.0f} GFLOPS, latency: {total_lat:.3f} ms")
            print(f'====================================================================================')

def test_ov():
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)
    def get_tensor(name, dtype=np.float16):
        with open(name, 'rb') as f:
            data = f.read()
            np_data = np.frombuffer(data, dtype=dtype).copy()
            return torch.from_numpy(np_data)

    def check_tril(block_mask: torch.Tensor):
        """
        Checks if each [H, W] slice in a [B, H, W] boolean tensor is lower triangular.
        Returns a list of coordinates where the mask is True above the diagonal.
        """
        B, H, W = block_mask.shape
        assert H == W, "Only square matrices can be checked for tril structure."

        # Create a mask for upper triangular part (excluding diagonal)
        upper_tri_mask = torch.triu(torch.ones(H, W, dtype=torch.bool), diagonal=1)

        violations = []
        for b in range(B):
            # Get the upper triangle violations for this batch
            violation_coords = torch.nonzero(block_mask[b] & upper_tri_mask, as_tuple=False)
            for coord in violation_coords:
                h, w = coord.tolist()
                violations.append((b, h, w))

        if violations:
            print("Not a tril. Violations found at coordinates:")
            for v in violations:
                print(f"Batch {v[0]}, Row {v[1]}, Col {v[2]}")
        else:
            print("All slices are lower triangular.")

        return violations

    def check_tril_all(tensor_dir, file_suffix, block_mask_shape):
        files_checked = 0
        is_tril = True
        for filename in os.listdir(tensor_dir):
            if filename.endswith(file_suffix):
                file_path = os.path.join(tensor_dir, filename)
                block_mask  = get_tensor(file_path, dtype=np.int8).reshape(block_mask_shape)
                print(f'{block_mask.shape=}')
                B, H, W = block_mask.shape
                violations = check_tril(block_mask[:,:,:H])
                is_tril &= (len(violations) == 0)
                files_checked += 1
        print(f'checked {files_checked} files')
        return is_tril & files_checked > 0

    compressed_kvcache = False
    xattn_thresh = 100
    sparse_block_sz, kv_block_size, trunk_sz = 128, 256, 4096 # trunk_sz no use
    num_heads, num_kv_heads, head_size = 32, 8, 128
    base = '/home/ceciliapeng/openvino/tensors_bin_pr/'

    query = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_src0__f16__1024_4096_1_1__bfyx.bin').reshape([1024, num_heads*head_size])
    key_cache   = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_updated_src_3__f16__5_8_256_128__bfyx.bin', np.int8 if compressed_kvcache else np.float16).reshape([-1, num_kv_heads, kv_block_size, head_size])
    value_cache   = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_updated_src_4__f16__5_8_256_128__bfyx.bin', np.int8 if compressed_kvcache else np.float16).reshape([-1, num_kv_heads, kv_block_size, head_size])
    # low = -1
    # high = 2
    # act_dtype = torch.float16
    # query = torch.ones(*query.shape).to(dtype=act_dtype)
    # key_cache = torch.ones(*key_cache.shape).to(dtype=act_dtype)
    # value_cache = torch.ones(*value_cache.shape).to(dtype=act_dtype)/high

    q_len = query.shape[0]
    valid_num_blks = key_cache.shape[0] - 1 # genai usually generates one more blocks than required
    q_block_pad = (q_len + sparse_block_sz - 1) // sparse_block_sz

    block_mask  = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_intermediates_4__boolean__8192_1_1_1__bfyx.bin', dtype=np.int8).reshape([num_heads, q_block_pad, -1])
    block_mask_in_wg  = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_intermediates_5__boolean__4096_1_1_1__bfyx.bin', dtype=np.int8).reshape([num_heads, -1, block_mask.shape[-1]])
    for i in range(num_heads):
        print(f'{i}:{block_mask[i,:]=}')
    # block_mask = torch.ones(block_mask.shape, dtype=torch.bool)
    # block_mask_in_wg = torch.ones(block_mask_in_wg.shape, dtype=torch.bool)

    past_lens = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_src5__i32__1_1_1_1__bfyx.bin', dtype=np.int32).reshape([1])
    subsequence_begins = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_src6__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])
    block_indices = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_src7__i32__4_1_1_1__bfyx.bin', dtype=np.int32).reshape([valid_num_blks])
    block_indices_begins = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_src8__i32__2_1_1_1__bfyx.bin', dtype=np.int32).reshape([2])

    full_dense = False
    if xattn_thresh >= 1:
        full_dense = check_tril_all(base, "_intermediates_4__boolean__8192_1_1_1__bfyx.bin", block_mask.shape)
        assert(full_dense, "SHOULD be full dense if XAttn thresh larger than 1.0")

    ov_out = get_tensor(base + 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_26732_dst0__f16__1024_4096_1_1__bfyx.bin').reshape([q_len, num_heads*head_size])

    # q [q_len, num_heads*head_size], k/v cache [num_blks, num_kv_heads, kv_block_size, head_size]
    print(f'{query.shape = }, {key_cache.shape = }, {value_cache.shape = },  {ov_out.shape = }')
    # print(f'{block_indices=}')

    is_causal = True
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, kv_block_size, trunk_sz, compressed_kvcache, is_causal, sparse_block_sz)

    t_query = cl.tensor(query.detach().numpy())
    t_key_cache = cl.tensor(key_cache.detach().numpy())
    t_value_cache = cl.tensor(value_cache.detach().numpy())
    t_block_indices = cl.tensor(block_indices.to(torch.int32).detach().numpy())
    t_block_indices_begins = cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
    t_past_lens = cl.tensor(past_lens.to(torch.int32).detach().numpy())
    t_subsequence_begins = cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())

    output = torch.zeros(q_len, num_heads*head_size).to(torch.float16)
    t_out = cl.tensor(output.detach().numpy())

    wg_size = 16
    q_step = CM_GRF_WIDTH // 32 # or 8 on Xe1
    wg_seq_len = wg_size * q_step
    wg_count = (q_len + wg_seq_len - 1) // wg_seq_len
    GWS = [1, pa_cm.num_heads, int(wg_count * wg_size)]
    LWS = [1, 1, wg_size]

    kv_precision = "U8" if pa_cm.compressed_kvcache else "F16"
    print(f"calling cm_page_attention {GWS=} {LWS=} sparse_block_sz:{pa_cm.sparse_block_sz} kv_cache:{kv_precision}")
    if pa_cm.sparse_block_sz > 1:
        t_block_mask = cl.tensor(block_mask.to(torch.bool).detach().numpy())
        t_block_mask_in_wg  = cl.tensor(block_mask_in_wg.to(torch.bool).detach().numpy())
        pa_cm.kernels.enqueue("cm_page_attention", GWS, LWS, t_query, t_key_cache, t_value_cache, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, t_block_mask, t_block_mask_in_wg, q_len, t_block_mask.shape[1], t_block_mask.shape[2])
    else:
        pa_cm.kernels.enqueue("cm_page_attention", GWS, LWS, t_query, t_key_cache, t_value_cache, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out, q_len)
    latency = cl.finish()

    ut_out = torch.from_numpy(t_out.numpy().reshape(-1, num_heads, head_size))
    ov_out = ov_out.reshape(-1, num_heads, head_size)
    check_close(ut_out, ov_out)

    enable_dequant_check = True if compressed_kvcache else False
    if enable_dequant_check: # TODO: there is bug in this check?
        k_dequan = dequant_per_token(key_cache.reshape(-1, num_kv_heads, kv_block_size*(head_size+2*2)), head_size, kv_block_size)
        v_dequan = dequant_per_token(value_cache.reshape(-1, num_kv_heads, kv_block_size*(head_size+2*2)), head_size, kv_block_size)
        # print(f'{k_dequan.shape = }, {v_dequan.shape = }')

        # => q [q_len, num_heads, head_size], k/v [kv_len, num_kv_heads, head_size]
        q_3d = query.reshape(q_len, num_heads, head_size).contiguous()
        k_3d = k_dequan[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()
        v_3d = v_dequan[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()
    else:
        # => q [q_len, num_heads, head_size], k/v [kv_len, num_kv_heads, head_size]
        q_3d = query.reshape(q_len, num_heads, head_size).contiguous()
        k_3d = key_cache[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()
        v_3d = value_cache[:valid_num_blks, :].transpose(1,2).reshape(-1, num_kv_heads, head_size).contiguous()

    print(f'{q_3d.shape = }, {k_3d.shape = }, {v_3d.shape = }')

    attention_mask = None if full_dense else get_attention_mask(q_3d, k_3d, v_3d, block_mask.unsqueeze(0), sparse_block_sz, trunk_sz)
    ref = flash_attn_vlen_ref(q_3d, k_3d, v_3d, [], is_causal, attention_mask)
    # print(f'{ref.shape=}')

    # check_close(ref, ov_out)
    check_close(ref, ut_out)

    print(f'{Colors.GREEN}test_ov done.{Colors.END}')

if __name__ == "__main__":

    # test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=True, sparse_block_sz = sparse_block_sz, sparse_ratio=sparse_ratio, check_acc=True)
    # test_page_attn_causal_batch1(32768, num_heads = 32, num_kv_heads = 8, head_size = 128, block_sz=256, trunk_sz=4096, compressed_kvcache=False, sparse_block_sz = 1, check_acc=False)
    #ACC test PA base
    if 1:
        for block_sz in range(32, 144, 16):
            for blocks_per_trunk in range(1, 30, 6):
                for seq_len in range(8192, 8248, 3):
                    for compressed_kv in [True,False]:
                        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                        print(f'[PA_BASE_ACC_TETS]: seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache=={"U8" if compressed_kv else "F16"} sparse_block_sz=1')
                        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                        test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kv, sparse_block_sz = 1, check_acc=True)
                        test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kv, sparse_block_sz = 1, check_acc=False)

        for block_sz in range(128, 257, 32):
                for seq_len in range(32768, 32810):
                    for trunk_num in range(1, 21):
                        for compressed_kvcache in [True,False,]:
                            seq_in_blks = (seq_len + block_sz -1 ) // block_sz
                            blocks_per_trunk = seq_in_blks // trunk_num if seq_in_blks % trunk_num == 0 else seq_in_blks // (trunk_num - 1)
                            print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                            print(f'[PA_BASE_ACC_TETS]:seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache={"U8" if compressed_kvcache else "F16"}')
                            print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                            test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache, sparse_block_sz=1, check_acc=True)
                            test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache, sparse_block_sz=1, check_acc=True)
    #ACC test sparse X Attention:
    if 0:
        for sparse_block_sz in [128, 256, 64,]:
            for block_sz in range (64, 256, 16):
                for sparse_ratio in [0.5, 0.75]:
                    for blocks_per_trunk in [1, 15, 16, 17, 32, 300]:
                        for seq_len in [16*15, 16*16, 16*16+1, 1024, 1024+1, 8*1024, 8*1024+3, 16*1024]:
                            for compressed_kvcache in [True,False,]:
                                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                                print(f'[XATTENION_ACC_TETS]:seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache={"U8" if compressed_kvcache else "F16"} {sparse_block_sz=} {sparse_ratio=}')
                                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                                test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 32, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache, sparse_block_sz = sparse_block_sz, sparse_ratio=sparse_ratio, check_acc=True)

    # perf for sparse X attention.
    if 0:
        seq_len = 32*1024
        block_sz = 256
        trunk_sz=seq_len
        sparse_block_sz = 128

        test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 4, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=False, sparse_block_sz = sparse_block_sz, sparse_ratio=0.5, check_acc=False)
        # test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 4, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=True, sparse_block_sz = sparse_block_sz, sparse_ratio=0.5, check_acc=False)

        # test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 4, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=False, sparse_block_sz = sparse_block_sz, sparse_ratio=0.8, check_acc=False)
        # test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 4, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=True, sparse_block_sz = sparse_block_sz, sparse_ratio=0.8, check_acc=False)

    # perf for sparse X attention, with QWen3 8K case
    if 0:
        for sparse_block_sz in [128]:
            for density in [100.0, 0.8, 0.5, 0.25, 0.1]:
                # seq_len, block_sz, blocks_per_trunk= 8*1024, 256, 16*2
                seq_len, block_sz, blocks_per_trunk= 32*1024, 256, 128
                trunk_sz = blocks_per_trunk*block_sz
                print("-----------------------------------------------------------------------------------------------------------------------------------------")
                print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} sparse_block_sz={sparse_block_sz}')
                print("-----------------------------------------------------------------------------------------------------------------------------------------")
                test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 8, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=False, sparse_block_sz = sparse_block_sz, sparse_ratio=1.0-density, check_acc=False)
                test_page_attn_causal_batch1(seq_len, num_heads = 32, num_kv_heads = 8, head_size = 128, block_sz=block_sz, trunk_sz=trunk_sz,  compressed_kvcache=True, sparse_block_sz = sparse_block_sz, sparse_ratio=1.0-density, check_acc=False)

    # test_ov()
