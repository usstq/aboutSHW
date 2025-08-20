import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

from clops import cl
import os

import numpy as np

debug = False
def get_cm_grf_width():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_get_grf_width(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_get_grf_width", [1], [1], t_info)
    return t_info.numpy()[0]

CM_GRF_WIDTH = get_cm_grf_width()

def align_up(a, b):
   return (a+b-1) // b * b

def div_up(a, b):
   return (a+b-1) // b

class page_atten_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, block_sz, trunk_sz, is_causal = False):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        assert trunk_sz % block_sz == 0, f'Error: trunk_sz must be multiple of block_sz'
        self.block_sz = block_sz
        self.trunk_sz = trunk_sz


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
            padding_tokens = (self.block_sz -  seq_len % self.block_sz) % self.block_sz
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
            blk_num = max_blks if blks_per_trunk*(trunk_idx + 1) > max_blks else blks_per_trunk*(trunk_idx + 1)
            #block_indices =  torch.randperm(blk_num)
            block_indices =  torch.arange(blk_num)
             # print(block_indices)
            sub_k = torch.zeros(blk_num, self.num_kv_heads, self.block_sz, head_size).to(torch.float16)
            sub_v = torch.zeros(blk_num, self.num_kv_heads, self.block_sz, head_size).to(torch.float16)
            for i in  range(len(block_indices)):
                sub_k[block_indices[i],:] = padded_k[i,:]
                sub_v[block_indices[i],:] = padded_v[i,:]
            q_start = trunk_idx*self.trunk_sz
            q_end =  min(q_start + self.trunk_sz, seq_len)
            q_len = q_end - q_start
            sub_q = q[q_start:q_end, :]

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
            wg_begins=torch.tensor([0,wg_count]).to(torch.int32)


            t_block_indices=cl.tensor(block_indices.to(torch.int32).detach().numpy())
            t_past_lens=cl.tensor(past_lens.to(torch.int32).detach().numpy())
            t_block_indices_begins=cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
            t_subsequence_begins=cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())
            t_wg_begins=cl.tensor(wg_begins.to(torch.int32).detach().numpy())
            print(f"calling cm_page_attention {GWS=} {LWS=} x {n_repeats} times, q:[{q_start}, {q_end}], past_lens:{int(past_lens)}, kv_blk_num:{blk_num}")
            for _ in range(n_repeats):
                self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_wg_begins,t_out)
            output[q_start:q_end] = torch.from_numpy(t_out.numpy())

        return output



    def dynamic_batch(self, q, k, v, q_start_list, q_len_list):
        seq_len, _, head_size = q.shape
        assert len(q_start_list) == len(q_len_list), f'same batch number in start_blk_list and blk_num_list'
        assert head_size == self.head_size
        q_end_list = d=[x + y for x, y in zip(q_start_list, q_len_list)]

        wg_size = 16
        q_step = CM_GRF_WIDTH // 32 # or 8 on Xe1
        wg_seq_len = wg_size * q_step

        total_kv_blks = 0
        total_q_len = 0

        batch_size_in_sequences = len(q_start_list)
        past_lens=torch.zeros(batch_size_in_sequences).to(torch.int32)
        subsequence_begins=torch.zeros(batch_size_in_sequences+1).to(torch.int32)
        block_indices_begins=torch.zeros(batch_size_in_sequences+1).to(torch.int32)
        wg_begins=torch.zeros(batch_size_in_sequences+1).to(torch.int32)

        wg_count = 0
        wg_begins[0] = 0
        subsequence_begins[0]=0
        block_indices_begins[0]=0
        for i in range(batch_size_in_sequences):
            past_lens[i]=q_start_list[i]
            subsequence_begins[i+1]=subsequence_begins[i] + q_len_list[i]
            #todo: align for blocks
            aligned_kv_cache_blocks = div_up(q_end_list[i], self.block_sz)
            block_indices_begins[i+1] = block_indices_begins[i] + aligned_kv_cache_blocks
            total_kv_blks += aligned_kv_cache_blocks
            total_q_len += q_len_list[i]

            subseq_wgs = div_up(q_len_list[i], wg_seq_len)
            wg_begins[i+1] = wg_begins[i]+subseq_wgs
            wg_count += subseq_wgs

        assert total_kv_blks==block_indices_begins[-1]

        block_indices =  torch.arange(total_kv_blks)
        if debug:
            print(f'PA:{q_start_list=}')
            print(f'PA:{q_len_list=}')
            print(f'PA:{past_lens=}')
            print(f'PA:{subsequence_begins=}')
            print(f'PA:{block_indices_begins=}')
            print(f'PA:{block_indices=}')
            print(f'PA:{wg_begins=}')


        if debug:
            print("\nqkv:")
            print("======================================================================================================================")
            print(f'Q:{q.shape}\n{q}')
            print(f'K:{k.shape}\n{k}')
            print(f'V:{v.shape}\n{v}')
            print("======================================================================================================================")
        # dummy empty k,v,q
        q_batched = q[0:0, :]
        k_batched = k[0:0, :]
        v_batched = v[0:0, :]

        # Q[L, H, S]
        # K/V: [blk_num, H, blk_sz, S]

        for subseq_idx in range(batch_size_in_sequences):
            q_start=q_start_list[subseq_idx]
            q_end=q_end_list[subseq_idx]
            kv_cached_padding = (self.block_sz - q_end % self.block_sz) % self.block_sz

            subq = q[q_start:q_end, :]
            subk = k[0:q_end, :]
            subv = v[0:q_end, :]

            #K/V is [L,H,S]
            assert len(k.shape) == 3  and len(v.shape) == 3
            kv_padding_dims = (0,0,0,0,0,kv_cached_padding)
            subk = torch.nn.functional.pad(subk,kv_padding_dims)
            subv = torch.nn.functional.pad(subv,kv_padding_dims)

            q_batched=torch.cat((q_batched, subq), 0)
            k_batched=torch.cat((k_batched, subk), 0)
            v_batched=torch.cat((v_batched, subv), 0)
        # if debug:
        #     print("\nqkv_batched before trans:")
        #     print("======================================================================================================================")
        #     print(f'Q:{q_batched.shape}\n{q_batched}')
        #     print(f'K:{k_batched.shape}\n{k_batched}')
        #     print(f'V:{v_batched.shape}\n{v_batched}')
        #     print("======================================================================================================================")

        assert k_batched.shape[0]%self.block_sz == 0 and  k_batched.shape[0] ==  v_batched.shape[0]
        #K, V cached to [block_num, head, block_sz, head_sz]
        k_batched = k_batched.reshape(k_batched.shape[0]//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        v_batched = v_batched.reshape(v_batched.shape[0]//self.block_sz, self.block_sz, self.num_kv_heads, self.head_size).transpose(1,2).contiguous()
        if debug:
            print("\nqkv_batched after trans:")
            print("======================================================================================================================")
            print(f'Q:{q_batched.shape}\n{q_batched}')
            print(f'K:{k_batched.shape}\n{k_batched}')
            print(f'V:{v_batched.shape}\n{v_batched}')
            print("======================================================================================================================")
        t_q = cl.tensor(q_batched.to(torch.float16).detach().numpy())
        t_k= cl.tensor(k_batched.to(torch.float16).detach().numpy())
        t_v = cl.tensor(v_batched.to(torch.float16).detach().numpy())
        t_out = cl.tensor([total_q_len, self.num_heads, self.head_size], np.dtype(np.float16))


        GWS = [1, self.num_heads, int(wg_count * wg_size)]
        LWS = [1, 1, wg_size]

        t_block_indices=cl.tensor(block_indices.to(torch.int32).detach().numpy())
        t_past_lens=cl.tensor(past_lens.to(torch.int32).detach().numpy())
        t_block_indices_begins=cl.tensor(block_indices_begins.to(torch.int32).detach().numpy())
        t_subsequence_begins=cl.tensor(subsequence_begins.to(torch.int32).detach().numpy())
        t_wg_begins = cl.tensor(wg_begins.to(torch.int32).detach().numpy())
        print(f'calling cm_page_attention {GWS=} {LWS=}')

        self.kernels.enqueue("cm_page_attention", GWS, LWS, t_q, t_k, t_v, t_past_lens, t_block_indices, t_block_indices_begins, t_subsequence_begins, t_wg_begins, t_out)
        return torch.from_numpy(t_out.numpy())

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size,block_sz, trunk_sz, is_causal):
        return page_atten_cm(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, is_causal)

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

def test_page_attn_causal_batch1(seq_len, num_heads, num_kv_heads, head_size, block_sz, trunk_sz):
    cl.profiling(True)
    torch.manual_seed(1)
    torch.set_printoptions(linewidth=1024)
    import numpy as np

    low = -1
    high = 3
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    is_causal = True
    ref = flash_attn_vlen_ref(q, k, v, [], is_causal=is_causal)
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, trunk_sz, is_causal)
    out = pa_cm(q, k, v)

    # out = func(q, k, v, n_repeats=20)
    # latency = cl.finish()
    # # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    # print(f" qkv_fused_causal {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    # print(ref)
    # print(out)
    check_close(ref, out)
    #assert 0


def test_dynamic_batch(qstart_list, qlen_list,num_heads, num_kv_heads, head_size, block_sz):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    q_end_list = d=[x + y for x, y in zip(qstart_list, qlen_list)]

    seq_len = max(q_end_list)

    import numpy as np

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    is_causal = True
    ref = flash_attn_vlen_ref(q, k, v, [], is_causal=is_causal)
    if debug:
        print(f'reftesnsor:\n{ref}')
    pa_cm = page_atten_cm.create_instance(num_heads, num_kv_heads, head_size, block_sz, block_sz, is_causal=True)
    out_ref = ref[0:0, :]

    for subseq_idx in range(len(qstart_list)):
        sub_out = ref[qstart_list[subseq_idx]:q_end_list[subseq_idx], :]
        out_ref=torch.cat((out_ref, sub_out), 0)
    if debug:
        print(f'outref_tesnror:\n{out_ref}')

    out = pa_cm.dynamic_batch(q, k, v,qstart_list, qlen_list)
    if debug:
        print(f'cm_tensor:\n{out}')
    # latency = cl.finish()
    # # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    # print(f" qkv_fused_causal {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    # print(ref)
    # print(out)
    check_close(out_ref, out)
    #assert 0

if __name__ == "__main__":
    if 0:

        for block_sz in range(16, 32, 16):
            for blocks_per_trunk in [1,]:
                for seq_len in range(1024*32, 1024*32+4):
                    print("-----------------------------------------------------------------------------------------------------------------------------------------")
                    print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk}')
                    print("-----------------------------------------------------------------------------------------------------------------------------------------")
                    test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 16, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz)

        # for block_sz in range(32, 144, 16):
        for block_sz in range(17, 32, 2):
            for blocks_per_trunk in range(1, 30, 6):
                for seq_len in range(1, 5):
                    print("-----------------------------------------------------------------------------------------------------------------------------------------")
                    print(f'seq_len={seq_len} block_sz={block_sz} blocks_per_trunk={blocks_per_trunk}')
                    print("-----------------------------------------------------------------------------------------------------------------------------------------")
                    test_page_attn_causal_batch1(seq_len, num_heads = 1, num_kv_heads = 1, head_size = 128, block_sz=block_sz, trunk_sz=blocks_per_trunk*block_sz)
        # test_page_attn_causal_batch1(1025, num_heads = 28, num_kv_heads = 4, head_size = 128, block_sz=32, trunk_sz=64)
    else:
        # qstart_list = [32]
        # qstart_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        # qlen_list = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

        qstart_list = [127, 1023, 1, 255, 54]
        qlen_list = [32,11, 2, 1,7]
        test_dynamic_batch(qstart_list, qlen_list, num_heads = 1, num_kv_heads = 1, head_size = 16, block_sz=16)

