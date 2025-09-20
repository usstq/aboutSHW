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
    def __init__(self, batch, num_heads, num_kv_heads, head_size, is_causal, fuse_v):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_causal = is_causal
        self.batch = batch
        self.fusev = fuse_v
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
                      f" -DCMFLA_V_FUSED={int(fuse_v)}"
                      f" -DCMFLA_IS_CAUSAL={int(is_causal)}"
                      f" -mdump_asm -g2")
                     )

    def sdpa(self, q, k, v, n_repeats = 1):
        batch, seq_len, head_num, head_size = q.shape

        old_dtype = q.dtype
        assert head_size == self.head_size
        t_q = cl.tensor(q.to(torch.float16).detach().numpy())
        t_k = cl.tensor(k.to(torch.float16).detach().numpy())
        t_v = cl.tensor(v.to(torch.float16).detach().numpy())

        t_out = cl.tensor([batch, seq_len, self.num_heads, self.head_size], np.dtype(np.float16))
        wg_size = 16
        q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        wg_seq_len = wg_size * q_step
        wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
        GWS = [batch, self.num_heads, wg_count * wg_size]
        LWS = [1, 1, wg_size]
        print(f"calling sdpa {GWS=} {LWS=} x {n_repeats} times")
        for _ in range(n_repeats):
            self.kernels.enqueue("cm_sdpa", GWS, LWS, seq_len, t_q, t_k, t_v, t_out)
        attn_output = torch.from_numpy(t_out.numpy()).to(old_dtype)
        return attn_output
    def __call__(self, q, k, v, cu_seqlens, n_repeats = 1):
        q_len = q.shape[0]
        kv_len = k.shape[0]
        old_dtype = q.dtype
        assert q_len == kv_len
        t_q = cl.tensor(q.to(torch.float16).detach().numpy())
        t_k = cl.tensor(k.to(torch.float16).detach().numpy())
        t_v = cl.tensor(v.to(torch.float16).detach().numpy())
        t_cu_seqlens = cl.tensor(np.array(cu_seqlens, dtype=np.int32))
        t_out = cl.tensor([q.shape[0], self.num_heads, self.head_size], np.dtype(np.float16))

        max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        wg_size =  (max_seq_len + q_step - 1)//q_step
        need_wg_mapping = 0
        if wg_size > 16:
            # seq_len is too big to fit into a single work-group
            # will use fixed work-group size 16, process 16*16 (or 16*8 on xe1)
            # part of sequence, in this case, kernel needs to figure-out which part
            # it needs to handle
            need_wg_mapping = 1
            wg_size = 16

        if need_wg_mapping:
            wg_count = 0
            wg_seq_len = wg_size * q_step
            for i in range(len(cu_seqlens) - 1):
                wg_count += (cu_seqlens[i+1] - cu_seqlens[i] + wg_seq_len - 1) // wg_seq_len
        else:
            wg_count = (len(cu_seqlens) - 1)
        GWS = [self.num_heads, wg_count * wg_size]
        LWS = [1, wg_size]
        print(f"calling {need_wg_mapping=} {q_step=} {max_seq_len=} {wg_count=} ...")
        print(f"{GWS=} {LWS=}")
        for _ in range(n_repeats):
            self.kernels.enqueue("cm_sdpa_vlen", GWS, LWS, t_cu_seqlens, need_wg_mapping, t_q, t_k, t_v, t_out)
        attn_output = torch.from_numpy(t_out.numpy()).to(old_dtype)
        return attn_output

    @staticmethod
    @functools.cache
    def create_instance(batch, num_heads, num_kv_heads, head_size, is_causal, fuse_v):
        return flash_attn_cm(batch, num_heads, num_kv_heads, head_size, is_causal, fuse_v)

def flash_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = False):
    batch, seq_length, num_heads, head_size = q.shape
    batch, kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    # print(f"============2 {cu_seqlens=} {seq_length=} {num_heads=}")
    if is_causal:
        attn_output = F.scaled_dot_product_attention(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
            is_causal = True,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )
    else:
        attention_mask = torch.zeros([batch, 1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        if len(cu_seqlens):
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        else:
            attention_mask[...] = True
        print(f"============2 {q.shape=} {q.is_contiguous()=} {k.shape=} {k.is_contiguous()=} {v.shape=} {v.is_contiguous()=}")
        print(f"============2 {attention_mask.shape=} ")

        attn_output = F.scaled_dot_product_attention(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
            attn_mask = attention_mask,
            dropout_p=0.0,
            enable_gqa = (num_kv_heads != num_heads)
        )

    attn_output = attn_output.transpose(1, 2)
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

def test_flash_attn_cm(seq_len, sub_seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    import numpy as np
    q_len = kv_len = seq_len
    cu_seqlens = torch.tensor([i for i in range(0, seq_len, sub_seq_len)] + [seq_len], dtype=torch.int32)
    print(f'{cu_seqlens=}')

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [q_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    ref = flash_attn_vlen_ref(q, k, v, cu_seqlens)

    func = flash_attn_cm.create_instance(num_heads, num_kv_heads, head_size, False)
    out = func(q, k, v, cu_seqlens)
    check_close(ref, out)

    out = func(q, k, v, cu_seqlens, 100)
    latency = cl.finish()
    # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    print(f" {seq_len=} {sub_seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")


def test_flash_attn_causal_batch1(batch, seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, fuse_v = False):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    import numpy as np

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [batch, seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [batch, seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [batch, seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

    is_causal = False
    ref = flash_attn_vlen_ref(q, k, v, [], is_causal=is_causal)

    func = flash_attn_cm.create_instance(batch, num_heads, num_kv_heads, head_size, is_causal, fuse_v)

    qkv = torch.cat((q,k,v), 2)
    v = qkv if fuse_v else v
    out = func.sdpa(q, k, v)
    out = func.sdpa(q, k, v, n_repeats=20)
    latency = cl.finish()
    # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    print(f" sdpa_causal {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    check_close(ref, out)
    #assert 0

if __name__ == "__main__":
    test_flash_attn_causal_batch1(batch=16, seq_len=128, num_heads = 1, num_kv_heads = 1, head_size = 80)
    test_flash_attn_causal_batch1(batch=16, seq_len=128, num_heads = 1, num_kv_heads = 1, head_size = 80, fuse_v=True)

