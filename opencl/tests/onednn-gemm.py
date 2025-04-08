#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import torch


def pack_i8_to_i4(B_q):
    assert B_q.ndim == 2
    K = B_q.shape[0]
    N = B_q.shape[1]
    B_q4 = np.zeros([K, N//2], dtype=np.int8)
    for k in range(K):
        for n in range(N//2):
            even = (B_q[k,2*n]) & 0xF
            odd = (B_q[k,2*n+1]) & 0xF
            B_q4[k,n] = even | (odd << 4)
    return B_q4


def test_mm(M, K, N, K_group_size, w_dtype):
    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    tA = cl.tensor(A)
    tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
    tP1 = cl.tensor()

    if w_dtype == cl.onednn_dtype.f16:
        B = np.random.randint(-1,2,[K, N]).astype(np.float16)
        C = A @ B
        print("ref is calculated!")

        tB = cl.tensor(B)
        with_wc_scales = False
        with_wc_zp = False
        with_silu = False
        with_binmul = False
        mm = cl.onednn_matmul(cl.onednn_dtype.f16, w_dtype, K, N, 0,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu,
                            with_binmul)
        linear = mm.get_linear(cl.onednn_dtype.f16, tB, cl.tensor(), cl.tensor())


    if w_dtype == cl.onednn_dtype.s8:
        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        B_q = np.random.randint(-1,2,[K, N]).astype(np.int8)
        B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float32)
        B_zp = np.random.randint(-1,2,[K_groups, N]).astype(np.int8)

        B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)

        C = A @ B
        print("ref is calculated!")

        tBq = cl.tensor(B_q)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp)

        with_wc_scales = True
        with_wc_zp = True
        with_silu = False
        with_binmul = False
        mm = cl.onednn_matmul(cl.onednn_dtype.f16, w_dtype, K, N, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu,
                            with_binmul)
        linear = mm.get_linear(cl.onednn_dtype.s8, tBq, tBs, tBz)


    if w_dtype == cl.onednn_dtype.u4:
        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        B_q = np.random.randint(0,3,[K, N]).astype(np.int8)
        B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float16)
        B_zp = np.random.randint(0,3,[K_groups, N]).astype(np.int8)

        BinarySrc = np.random.randint(-2,3,[M, N]).astype(np.float16)
        
        B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)

        C = A @ B
        C = C * BinarySrc
        print("ref is calculated!")

        # pack weight into 4bit
        B_q4 = pack_i8_to_i4(B_q)
        B_zp4 = pack_i8_to_i4(B_zp)

        tBq4 = cl.tensor(B_q4)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp4)

        with_wc_scales = True
        with_wc_zp = True
        with_silu = False
        with_binmul = True
        mm = cl.onednn_matmul(cl.onednn_dtype.f16, w_dtype, K, N, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu,
                            with_binmul)
        linear = mm.get_linear(cl.onednn_dtype.u4, tBq4, tBs, tBz)
        
        tP1 = cl.tensor(BinarySrc)

    linear.forward(tA, tC, tP1)

    cl.finish()

    C1 = tC.numpy()

    if not np.allclose(C, C1):
        print(C)
        print(C1)


from torch import nn
class config:
    hidden_size = 2048
    intermediate_size = 8192
    num_experts = 4
    num_experts_per_tok = 2
    moe_intermediate_size = 768
    num_hidden_layers = 1
    sliding_window = 4096
    num_attention_heads = 32
    num_key_value_heads = 32
    head_dim = 128
    attention_bias = False
    rms_norm_eps = 1e-6
    attention_dropout = 0

class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj




def test_mlp(K_group_size, w_dtype):
    mlp = Qwen3MoeMLP(config, intermediate_size = config.moe_intermediate_size)
    mlp = mlp.eval()
    is_quantized = w_dtype != cl.onednn_dtype.f16
    with_wc_scales = is_quantized
    with_wc_zp = is_quantized
    with_silu = False
    with_binmul = False

    mlp_up = cl.onednn_matmul(cl.onednn_dtype.f16, w_dtype, config.hidden_size, config.moe_intermediate_size, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu,
                            True)
    mlp_gate = cl.onednn_matmul(cl.onednn_dtype.f16, w_dtype, config.hidden_size, config.moe_intermediate_size, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            True,
                            with_binmul)

    mlp_down = cl.onednn_matmul(cl.onednn_dtype.f16, w_dtype, config.moe_intermediate_size, config.hidden_size, K_group_size,
                            with_wc_scales,
                            with_wc_zp,
                            with_silu,
                            with_binmul)

    linear_up = mlp_up.get_linear(cl.onednn_dtype.f32, cl.tensor(mlp.up_proj.weight.detach().numpy()), cl.tensor(), cl.tensor())
    linear_gate = mlp_gate.get_linear(cl.onednn_dtype.f32, cl.tensor(mlp.gate_proj.weight.detach().numpy()), cl.tensor(), cl.tensor())
    linear_down = mlp_down.get_linear(cl.onednn_dtype.f32, cl.tensor(mlp.down_proj.weight.detach().numpy()), cl.tensor(), cl.tensor())

    M = 120

    input = torch.randint(-1, 2, size=[M, config.hidden_size], dtype=torch.float16)
    
    x = cl.tensor(input.numpy())
    temp_gate = cl.tensor(np.zeros([M, config.moe_intermediate_size], dtype=np.float16))
    temp_up = cl.tensor(np.zeros([M, config.moe_intermediate_size], dtype=np.float16))
    dst = cl.tensor(np.zeros([M, config.hidden_size], dtype=np.float16))

    linear_gate.forward(x, temp_gate, cl.tensor())
    linear_up.forward(x, temp_up, temp_gate)
    linear_down.forward(temp_up, dst, cl.tensor())
    cl.finish()

    print(x.numpy())
    print(temp_gate.numpy())
    print(temp_up.numpy())
    
    dst_ref = mlp.forward(input.to(dtype=torch.float32))
    
    ref = dst_ref.detach().numpy()
    act = dst.numpy()
    if not np.allclose(ref, act):
        print("============")
        print(ref)
        print("============")
        print(act)

np.set_printoptions(linewidth=1024)
#test_mm(M = 1, K = 768, N = 2048, K_group_size = 0, w_dtype = cl.onednn_dtype.f16)
#test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.s8)
#test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.u4)
test_mlp(K_group_size = 128, w_dtype = cl.onednn_dtype.f16)
