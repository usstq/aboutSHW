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
print(f'======================CM_GRF_WIDTH is {CM_GRF_WIDTH}')

class sage_attn_cm:
    def __init__(self, num_heads, num_kv_heads, head_size, qkfused = False, vfused = False, is_causal = False, unroll_cnt=32):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.qkfused = qkfused
        self.vfused = vfused
        self.is_causal = is_causal

        self.unroll_cnt = unroll_cnt
        self.state_blk_sz = 32
        self.local_sz = 32 if CM_GRF_WIDTH == 512 else 64
        src1 = r'''#include "sage_sdpa_vlen.hpp"'''
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
                      f" -DCMKMEAN_UNROLL_NUM={int(self.unroll_cnt)}"
                      f" -DCMKMEAN_STATE_BLK={int(self.state_blk_sz)}"
                      f" -DCMKMEAN_LOCAL_SZ={int(self.local_sz)}"
                      f" -DCMFLA_QK_FUSED={int(self.qkfused)}"
                      f" -DCMFLA_V_FUSED={int(self.vfused)}"
                      f" -mdump_asm -g2")
                     )

    def __call__(self, q, k, v, n_repeats = 1):
        seq_len, _, head_size = q.shape
        old_dtype = q.dtype
        total_heads = (self.num_heads + self.num_kv_heads * 2)
        assert head_size == self.head_size
        t_Q = [cl.tensor(q.to(torch.float16).detach().numpy()) for _ in range(n_repeats)]
        t_K = [cl.tensor(k.to(torch.float16).detach().numpy()) for _ in range(n_repeats)]
        t_V = [cl.tensor(v.to(torch.float16).detach().numpy()) for _ in range(n_repeats)]
        qkv = None
        t_QKV = []
        if self.qkfused or self.vfused:
            qkv = torch.cat((q,k,v), 1)
            t_QKV = [cl.tensor(qkv.to(torch.float16).detach().numpy()) for _ in range(n_repeats)]


        t_out = cl.tensor([seq_len, self.num_heads, self.head_size], np.dtype(np.float16))
        t_dqscale_q = [cl.tensor(torch.zeros(self.num_heads, seq_len).to(torch.float32).detach().numpy()) for _ in range(n_repeats)]
        t_dqscale_k = [cl.tensor(torch.zeros(self.num_kv_heads, seq_len).to(torch.float32).detach().numpy()) for _ in range(n_repeats)]
        t_mean_k = [cl.tensor(torch.zeros(1, self.num_kv_heads, self.head_size).to(torch.float16).detach().numpy()) for _ in range(n_repeats)]

        local_sz = self.local_sz

        kmean_seq_blk = (seq_len + local_sz - 1) // local_sz
        kmean_seq_blk = (kmean_seq_blk + self.unroll_cnt - 1) // self.unroll_cnt * self.unroll_cnt
        kmean_lws = [1, 1, local_sz]
        kmean_gws = [self.num_kv_heads, self.head_size//self.state_blk_sz, local_sz]

        quan_lws = [local_sz]
        quan_gws=[(self.num_kv_heads*seq_len+local_sz-1)//local_sz*local_sz]
        print(f'QUAN GWS:{quan_gws}, QUAN LWS:{quan_lws}')
        print(f'KMEAN GWS:{kmean_gws}, KMEAN LWS:{kmean_lws}')

        cl.finish()
        for i in range(n_repeats):
            t_q= t_QKV[i] if self.qkfused else t_Q[i]
            t_k= t_QKV[i] if self.qkfused else t_K[i]
            self.kernels.enqueue("cm_kmean", kmean_gws, kmean_lws, seq_len, kmean_seq_blk, t_k, t_mean_k[i])
            self.kernels.enqueue("cm_quantize_qk", quan_gws, quan_lws, seq_len, t_q, t_k, t_dqscale_q[i], t_dqscale_k[i], t_mean_k[i])
        lat=cl.finish()
        assert len(lat) == n_repeats * 2
        lat_kmean=lat[0:n_repeats*2:2]
        lat_quan=lat[1:n_repeats*2:2]
        kmean_rdbytes = seq_len*self.num_kv_heads*self.head_size*2
        quan_rdbytes=(seq_len*self.num_heads*self.head_size+seq_len*self.num_kv_heads*self.head_size)*2
        if n_repeats > 10:
            quan_ns=sum(lat_quan[5:])/len(lat_quan[5:])
            kmean_ns=sum(lat_kmean[5:])/len(lat_kmean[5:])
            print(f'[K_MEAN]: latency: {kmean_ns*1e-3:.2f} us, read:{kmean_rdbytes/kmean_ns:.2f} GB/S')
            print(f'[QUAN_KV]: latency: {quan_ns*1e-3:.2f} us, read:{quan_rdbytes/quan_ns:.2f} GB/S, write:{(quan_rdbytes/2+(self.num_heads+self.num_kv_heads)*seq_len*4)/quan_ns:.2f} GB/S')

        wg_size = 16
        q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        wg_seq_len = wg_size * q_step
        wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
        GWS = [1, self.num_heads, wg_count * wg_size]
        LWS = [1, 1, wg_size]
        print(f"calling qkv_fused {GWS=} {LWS=} x {n_repeats} times")
        for i in range(n_repeats):
            t_q= t_QKV[i] if self.qkfused else t_Q[i]
            t_k= t_QKV[i] if self.qkfused else t_K[i]
            t_v= t_QKV[i] if self.vfused else t_V[i]
            self.kernels.enqueue("cm_sage_sdpa", GWS, LWS, seq_len, t_q, t_k, t_v, t_dqscale_q[i], t_dqscale_k[i], t_out)
        attn_output = torch.from_numpy(t_out.numpy()).to(old_dtype)
        return attn_output

    @staticmethod
    @functools.cache
    def create_instance(num_heads, num_kv_heads, head_size, qkfused, vfused, is_causal):
        return sage_attn_cm(num_heads, num_kv_heads, head_size, qkfused, vfused,is_causal)

def sage_attn_vlen_ref(q, k, v, cu_seqlens, is_causal = False):
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


def sage_attn(q, k, v, cu_seqlens):
    _, num_heads, head_size = q.shape
    func = sage_attn_cm.create_instance(num_heads, head_size)
    # return sage_attn_vlen_ref(q, k, v, cu_seqlens)
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

def test_sage_attn_causal_batch1(seq_len, num_heads = 16, num_kv_heads = 16, head_size = 80, qkfused = False, vfused = False):
    cl.profiling(True)
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    import numpy as np

    low = -1
    high = 3
    act_dtype = torch.float16
    is_causal = True
    #manually produce different scales for different token.
    q_factor = torch.randint(high, high+3, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k_factor = torch.randint(high, high+3, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    q_factor[:,:,0:head_size:3] = torch.randint(high+3,high+6,[1]).to(dtype=act_dtype)
    k_factor[:,:,0:head_size:3] = torch.randint(high+3,high+6,[1]).to(dtype=act_dtype)

    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype) / q_factor
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype) / k_factor
    # add bias to k to simulate outlier
    bias_low=3
    bias_high=7
    step=3
    k[:,:,0:head_size:step] += torch.randint(bias_low,bias_high,[1]).to(dtype=act_dtype)/bias_high


    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
    ref = sage_attn_vlen_ref(q, k, v, [], is_causal)

    func = sage_attn_cm.create_instance(num_heads, num_kv_heads, head_size, qkfused, vfused, is_causal)

    out = func(q, k, v)
    out = func(q, k, v, n_repeats=20)
    latency = cl.finish()
    # for i,ns in enumerate(latency): print(f"[{i}]  {ns*1e-6:.3f} ms")
    print(f" qkv_fused_causal {seq_len=} average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")
    check_close(ref, out)

if __name__ == "__main__":
    # for seqlen in range(1025, 1055, 1):
    #     test_sage_attn_causal_batch1(seqlen, num_heads = 28, num_kv_heads = 4, head_size = 128)

    for seqlen in range(9216, 9217, 1024):
        # todo: add other head_size support,cmload limitation.
        for head_size in range(160, 192, 32):
            print(f'-----------------------------------------------')
            print(f'seq={seqlen}, head_size={head_size}')
            print(f'-----------------------------------------------')
            test_sage_attn_causal_batch1(seqlen, num_heads = 28, num_kv_heads = 4, head_size = head_size, qkfused=False, vfused=False)
            test_sage_attn_causal_batch1(seqlen, num_heads = 28, num_kv_heads = 4, head_size = head_size, qkfused=True, vfused=True)
            test_sage_attn_causal_batch1(seqlen, num_heads = 28, num_kv_heads = 4, head_size = head_size, qkfused=False, vfused=True)

    # test_sage_attn_causal_batch1(1025, num_heads = 28, num_kv_heads = 4, head_size = 96)

