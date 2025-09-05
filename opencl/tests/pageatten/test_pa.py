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
# print(f'CM_GRF_WIDTH is {CM_GRF_WIDTH}')

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

class page_atten_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal = True):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        assert trunk_sz % block_sz == 0, f'Error: trunk_sz{trunk_sz} must be multiple of block_sz{trunk_sz}'
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz
        self.compressed_kvcache = compressed_kvcache



        src1 = r'''#include "cm_pa_kernel.hpp"'''
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
                    f" -DCMPA_KVCACHE_U8={int(compressed_kvcache)}"
                    f" -mdump_asm -g2")
                    )


    def __call__(self, q, k, v, n_repeats = 1):
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

        # reorder K,V from [L, H, S] to [block_num, H, block_size, S]
        k_cache = padded_k.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        v_cache = padded_v.reshape(aligned_seqlen//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        if self.compressed_kvcache:
            k_cache = quan_per_token(k_cache)
            v_cache = quan_per_token(v_cache)
        else:
            k_cache = k_cache.reshape(aligned_seqlen//self.block_sz, self.num_kv_heads, -1)
            v_cache = v_cache.reshape(aligned_seqlen//self.block_sz, self.num_kv_heads, -1)
        # print(f'-----------------------------------------------------------------')
        # print(f'[KINT8]={k_quan}')
        # print(f'[VINT8]={v_quan}')
        # print(f'[K]={k}')
        # print(f'[V]={v}')
        # print(f'-----------------------------------------------------------------')
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

        cl.finish()
        for _ in range(n_repeats):
            for trunk_idx in range(trunk_num):
                blk_num = max_blks if blks_per_trunk*(trunk_idx + 1) > max_blks else blks_per_trunk*(trunk_idx + 1)
                block_indices =  torch.randperm(blk_num).to(torch.uint32)
                # block_indices =  torch.arange(blk_num)
                sub_k = torch.zeros(blk_num, self.num_kv_heads, self.block_sz*token_sz).to(kv_dtype)
                sub_v = torch.zeros(blk_num, self.num_kv_heads, self.block_sz*token_sz).to(kv_dtype)
                for i in  range(len(block_indices)):
                    sub_k[block_indices[i],:] = k_cache[i,:]
                    sub_v[block_indices[i],:] = v_cache[i,:]
                q_start = trunk_idx*self.trunk_sz
                q_end =  min(q_start + self.trunk_sz, seq_len)
                q_len = q_end - q_start
                sub_q = q[q_start:q_end, :]

                t_q = cl.tensor(sub_q.to(torch.float16).detach().numpy())
                t_k= cl.tensor(sub_k.to(kv_dtype).detach().numpy())
                t_v = cl.tensor(sub_v.to(kv_dtype).detach().numpy())
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
                print(f"calling cm_page_attention {GWS=} {LWS=} x {n_repeats} times, q:[{q_start}, {q_end}], past_lens:{int(past_lens)}, kv_blk_num:{blk_num}")
                self.kernels.enqueue("cm_page_attention", GWS, LWS, q_len, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_out)
                output[q_start:q_end] = torch.from_numpy(t_out.numpy())

        return output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size,block_sz, trunk_sz, compressed_kvcache, is_causal):
        return page_atten_cm(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal)

def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = True):
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


def check_close(input, other, atol=1e-2, rtol=1e-2):
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

def test_page_attn_causal_batch1(seq_len, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache = True, rep = 20):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024, threshold=10_000)

    import numpy as np

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)

    if compressed_kvcache:
        k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / 4.0
        k[0:seq_len:3, :, :] = (k[0:seq_len:3, :, :] + 0.25)/ 2.0
        v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
        v[0:seq_len:3, :, :] = (v[0:seq_len:3, :, :] + 0.25)/ 2.0
    else:
        k = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)
        v = torch.rand(seq_len, num_kv_heads, head_size).to(dtype=act_dtype)/high

    is_causal = True
    ref = flash_attn_vlen_ref(q, k, v, [], is_causal=is_causal)
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, compressed_kvcache, is_causal)

    out = pa_cm(q, k, v, rep)
    # out = func(q, k, v, n_repeats=20)
    latency = cl.finish()
    trunks = len(latency) // rep
    trunk_lat = []
    if rep >= 15:
        print(f'====================================================================================')
        for trunk_idx in range(trunks):
            lat = latency[trunk_idx:-1:trunks]
            avg = sum(lat[10:])/len(lat[10:])*1e-6
            trunk_lat.append(avg)
            print(f'[trunk{trunk_idx}] average latency: {avg:.3f} ms')
        print(f"[total]: PA_causal {seq_len=} , {trunks=} latency: {sum(trunk_lat):.3f} ms")
        print(f'====================================================================================')

    check_close(ref, out)

if __name__ == "__main__":


    seq_len =  8192
    test_page_attn_causal_batch1(seq_len, num_heads = 28, num_kv_heads = 4, head_size = 128, block_sz=64, trunk_sz=8192, compressed_kvcache=True, rep=20)
    test_page_attn_causal_batch1(seq_len, num_heads = 28, num_kv_heads = 4, head_size = 128, block_sz=64, trunk_sz=8192, compressed_kvcache=False, rep=20)

    if 0:
        for block_sz in range(16, 32, 16):
            for blocks_per_trunk in [1,]:
                for seq_len in range(1024*32, 1024*32+4):
                    for compressed_kvcache in [True,False,]:
                        print("\n##################################################################################################################################################################")
                        print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache={"U8" if compressed_kvcache else "F16"}')
                        print("##################################################################################################################################################################")
                        test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 64, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache)
    if 1:
        test_page_attn_causal_batch1(32771, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=256, trunk_sz=128*256, compressed_kvcache=True)
        test_page_attn_causal_batch1(32771, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=256, trunk_sz=128*256, compressed_kvcache=False)
        for block_sz in range(128, 257, 32):
            for seq_len in range(32768, 32810):
                for trunk_num in range(1, 21):
                    for compressed_kvcache in [True,False,]:

                        seq_in_blks = (seq_len + block_sz -1 ) // block_sz
                        blocks_per_trunk = seq_in_blks // trunk_num if seq_in_blks % trunk_num == 0 else seq_in_blks // (trunk_num - 1)
                        print("\n##################################################################################################################################################################")
                        print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk} kv_cache={"U8" if compressed_kvcache else "F16"}')
                        print("##################################################################################################################################################################")
                        test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz, compressed_kvcache=compressed_kvcache)



