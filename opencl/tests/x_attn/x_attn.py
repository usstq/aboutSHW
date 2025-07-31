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
        src1 = r'''#include "pa_sdpa_prefill_prefetch.hpp"'''
        cwd = os.path.dirname(os.path.realpath(__file__))
        print(f"compiling {cwd} {num_heads=} {num_kv_heads=} {head_size=} ...")

        scale_factor = 1.0/(head_size**0.5)
        self.kernels = cl.kernels(src1,
                     (f'-cmc -Qxcm_jit_option="-abortonspill" -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd}'
                      f" -DCMFLA_NUM_HEADS={num_heads}"
                      f" -DCMFLA_NUM_KV_HEADS={num_kv_heads}"
                      f" -DCMFLA_HEAD_SIZE={head_size}"
                      f" -DCMFLA_SCALE_FACTOR={scale_factor}"
                      f" -DCAUSAL_MASK={int(is_causal)}"
                      f" -mdump_asm -g2")
                     )

    def __call__(self, q, k, v, block_mask, n_repeats = 1):
        batch, seq_len, q_heads, head_size = q.shape
        batch, seq_len_kv, kv_heads, head_size = k.shape 
        old_dtype = q.dtype
        assert q_heads == self.num_heads
        assert kv_heads == self.num_kv_heads
        assert head_size == self.head_size

        t_q = cl.tensor(q.to(torch.float16).detach().numpy())
        t_k = cl.tensor(k.to(torch.float16).detach().numpy())
        t_v = cl.tensor(v.to(torch.float16).detach().numpy())
        t_out = cl.tensor([batch, seq_len, self.num_heads, self.head_size], np.dtype(np.float16))
        v_before_padding = (q_heads + kv_heads) * head_size

        wg_size = 16
        q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        wg_seq_len = wg_size * q_step
        wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
      
        GWS = [batch, self.num_heads, wg_count * wg_size]
        LWS = [1, 1, wg_size]

        print(f"calling cm_pa_sdpa_prefill {GWS=} {LWS=} x {n_repeats} times")
        for _ in range(n_repeats):
            self.kernels.enqueue("cm_pa_sdpa_prefill", GWS, LWS, t_q, t_k, t_v, t_out, seq_len, seq_len_kv, v_before_padding)
        attn_output = torch.from_numpy(t_out.numpy()).to(old_dtype)
        return attn_output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, is_causal):
        return flash_attn_cm(num_heads, num_kv_heads, head_size, is_causal)

def flash_attn_ref(q, k, v, approx_simple_mask, block_size):
    batch, seq_length, num_heads, head_size = q.shape
    batch, kv_seq_length, num_kv_heads, head_size = k.shape
    old_dtype = q.dtype
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    print(f"============2 {batch=} {seq_length=} {kv_seq_length=} {num_heads=}")
    print(f"============2 {q.shape=} {q.is_contiguous()=} {k.shape=} {k.is_contiguous()=} {v.shape=} {v.is_contiguous()=}")
    print(f"============2 {approx_simple_mask.shape=} {approx_simple_mask.is_contiguous()=} {block_size=}")
    
    q_block_num = (seq_length + block_size -1) // block_size
    k_block_num = (kv_seq_length + block_size -1) // block_size
    assert(approx_simple_mask.shape[-2] == q_block_num)
    assert(approx_simple_mask.shape[-1] == k_block_num)
   
    # sparse to dense mask 
    def block_mask_to_attention_mask(block_mask: torch.Tensor, q_len: int, k_len: int, block_size: int) -> torch.Tensor:
        expanded = block_mask.repeat_interleave(block_size, dim=2)
        expanded = expanded.repeat_interleave(block_size, dim=3)
        return expanded[:, :, :q_len, :k_len]

    attention_mask = block_mask_to_attention_mask(approx_simple_mask, seq_length, kv_seq_length, block_size)

    attn_output = F.scaled_dot_product_attention(
        q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
        attn_mask = attention_mask,
        dropout_p=0.0,
        enable_gqa = (num_kv_heads != num_heads)
    )
    attn_output = attn_output.squeeze(0).transpose(0, 1)
    print(f"============2 {attn_output.shape=} ")    
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

def test_flash_attn(batch, seq_len, seq_len_kv, num_heads = 16, num_kv_heads = 16, head_size = 80, block_size = 64, is_causal = False):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)
    
    import numpy as np

    low = -1
    high = 2
    act_dtype = torch.float16
    q = torch.randint(low, high, [batch, seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [batch, seq_len_kv, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [batch, seq_len_kv, num_kv_heads, head_size]).to(dtype=act_dtype)/high
    
    def generate_block_mask_with_ratio(B, H, Q_block, K_block, true_ratio=0.5, is_causal=False, device='cpu'):
        if is_causal:
            causal_pattern = torch.tril(torch.ones(Q_block, K_block, dtype=torch.bool, device=device))
            rand = torch.rand(B, H, Q_block, K_block, device=device)
            block_mask = (rand < true_ratio) & causal_pattern
        else:
            block_mask = torch.rand(B, H, Q_block, K_block, device=device) < true_ratio
            
        return block_mask

    q_block_num = (seq_len + block_size -1) // block_size
    k_block_num = (seq_len_kv + block_size -1) // block_size
    approx_simple_mask = generate_block_mask_with_ratio(batch, num_heads, q_block_num, k_block_num, 0.5, is_causal)

    ref = flash_attn_ref(q, k, v, approx_simple_mask, block_size)

    func = flash_attn_cm.create_instance(num_heads, num_kv_heads, head_size, is_causal)
    
    out = func(q, k, v, approx_simple_mask)
    out = func(q, k, v, approx_simple_mask, n_repeats=20)
    latency = cl.finish()
    # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    print(f" test_flash_attn {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    check_close(ref, out)
    #assert 0

if __name__ == "__main__":
    test_flash_attn(batch = 1, seq_len=1024, seq_len_kv=1024, num_heads = 28, num_kv_heads = 4, head_size = 128, is_causal = False)
