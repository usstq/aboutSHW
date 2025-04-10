#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import torch
import time, sys


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

        tB = cl.tensor(B.transpose().copy())
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, -1, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.f16, tB, cl.tensor(), cl.tensor())

    if w_dtype == cl.onednn_dtype.s8:
        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        B_q = np.random.randint(-1,2,[K, N]).astype(np.int8)
        B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float16)
        B_zp = np.random.randint(-1,2,[K_groups, N]).astype(np.int8)

        B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)

        C = A @ B
        print("ref is calculated!")

        tBq = cl.tensor(B_q.transpose().copy())
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp)

        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, K_group_size, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.s8, tBq, tBs, tBz)


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
        B_q4 = pack_i8_to_i4(B_q.transpose().copy())
        B_zp4 = pack_i8_to_i4(B_zp)

        tBq4 = cl.tensor(B_q4)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp4)
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, K_group_size, cl.onednn_matmul_type.with_bin_mul,
                                  cl.onednn_dtype.u4, tBq4, tBs, tBz)

        tP1 = cl.tensor(BinarySrc)

    linear.forward(tA, tC, tP1)

    cl.finish()

    C1 = tC.numpy()

    if not np.allclose(C, C1):
        print(C)
        print(C1)
    else:
        print("================ PASSED ==================" , M, K, N, w_dtype)


from torch import nn
class config:
    hidden_size = 2048
    intermediate_size = 8192
    num_experts = 128
    num_experts_per_tok = 8
    moe_intermediate_size = 768
    num_hidden_layers = 1
    sliding_window = 4096
    num_attention_heads = 32
    num_key_value_heads = 32
    head_dim = 128
    attention_bias = False
    rms_norm_eps = 1e-6
    attention_dropout = 0
    debug = False

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

    # top_x： token index
    def forward_expert(self, hidden_states, expert_mask, routing_weights, final_hidden_states):
        # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
        # expert_mask[j,k] == 1 means apply current expert to token k as the top-j
        
        # using CPU to get token_list & routing_weights index from expert_mask
        top_x = []
        idx = []
        for j in range(expert_mask.shape[0]):
            for k in range(expert_mask.shape[1]):
                if (expert_mask[j,k] != 0):
                    top_x.append(k)
                    idx.append(j)
        if len(top_x) == 0:
            if config.debug: print("skip")
            return

        if config.debug:
            print("========== expert_mask")
            print(expert_mask)
            print("========== top_x", top_x)
            print("========== idx", idx)

        x = hidden_states[top_x, :] # slicing2d
        if config.debug:
            print("========== x")
            print(x.detach().numpy())
        # expert_layer
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if config.debug:
            print("========== y")
            print(y.detach().numpy())

        # routing_weights[i, j] i'th token's top-j's routing weight
        # y[i, :]  i'th token's all hidden-states
        if config.debug:
            print("========== routing_weights[top_x, idx, None]")
            print(routing_weights[top_x, idx, None])

        y = y * routing_weights[top_x, idx, None]

        if config.debug:
            print("========== y * routing_weights")
            print(y.detach().numpy())

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, torch.tensor(top_x, dtype=torch.int32), y)


ocl_sources=r'''
// [n_tokens, count] [1, 1]
__kernel void gather_2d_ref(
    const __global half* src_tok,
    __global half* dst_tok,
    const __global half* src_rweight,
    __global half* dst_rweight,
    __global int * tok_index,
    __global int * top_index,
    int tok_size,
    int rweight_size) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += tok_idx * tok_size;
    dst_tok += k * tok_size;

    dst_tok[off] = src_tok[off];

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[tok_idx * rweight_size + top_idx];
    }
}

__kernel void index_add_(
    const __global half* src_tok,
    __global half* dst_tok,
    __global int * tok_index,
    int tok_size) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += k * tok_size;
    dst_tok += tok_idx * tok_size;

    dst_tok[off] += src_tok[off];
}

'''

class scratchBuff:
    def __init__(self):
        self.x = None
        self.rweights = None
        self.n_tokens = -1
        self.act_gate = None
        self.act_up = None

class onednnMLP:
    def __init__(self, mlp, expert_id):
        self.w_dtype = cl.onednn_dtype.f16
        self.config = mlp.config
        self.M = -1
        self.expert_id = expert_id
        self.K_group_size = -1
        self.weight_up = cl.tensor(mlp.up_proj.weight.detach().to(dtype=torch.float16).numpy())
        self.weight_gate = cl.tensor(mlp.gate_proj.weight.detach().to(dtype=torch.float16).numpy())
        self.weight_down = cl.tensor(mlp.down_proj.weight.detach().to(dtype=torch.float16).numpy())
        self.ocl_kernels = cl.kernels(ocl_sources, "-D FMACNT=4 -D UNROLL=4")

    def update_batch(self, batch):
        if self.M != batch:
            empty_cl_tensor = cl.tensor()
            if config.debug: print("======== create linear_up =========")
            self.linear_up = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.config.hidden_size, self.config.moe_intermediate_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        cl.onednn_dtype.f16, 
                                        self.weight_up, empty_cl_tensor, empty_cl_tensor)
            if config.debug: print("======== create linear_gate =========")
            self.linear_gate = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.config.hidden_size, self.config.moe_intermediate_size,
                                        self.K_group_size, cl.onednn_matmul_type.with_silu_bin_mul,
                                        cl.onednn_dtype.f16, 
                                        self.weight_gate, empty_cl_tensor, empty_cl_tensor)
            if config.debug: print("======== create linear_down =========")
            self.linear_down = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        cl.onednn_dtype.f16, 
                                        self.weight_down, empty_cl_tensor, empty_cl_tensor)

            self.linear_down2 = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_mul_per_row,
                                        cl.onednn_dtype.f16, 
                                        self.weight_down, empty_cl_tensor, empty_cl_tensor)
            if (batch == 1):
                self.linear_down3 = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_mul_per_row_sum,
                                        cl.onednn_dtype.f16, 
                                        self.weight_down, empty_cl_tensor, empty_cl_tensor)
            self.M = batch

    def forward(self, input):
        M = input.shape[0]
        self.update_batch(M)
        temp_gate = cl.tensor(np.zeros([M, config.moe_intermediate_size], dtype=np.float16))
        temp_up = cl.tensor(np.zeros([M, config.moe_intermediate_size], dtype=np.float16))
        dst = cl.tensor(np.zeros([M, config.hidden_size], dtype=np.float16))

        self.linear_up.forward(input, temp_up, cl.tensor())
        self.linear_gate.forward(input, temp_gate, temp_up)
        self.linear_down.forward(temp_gate, dst, cl.tensor())
        return dst


    # top_x： token index
    def forward_expert(self, hidden_states, expert_mask, routing_weights, final_hidden_states, scratch):
        # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
        # expert_mask[j,k] == 1 means apply current expert to token k as the top-j
        
        # x = hidden_states[top_x, :] # slicing2d
        if hidden_states.shape[0] == 1:
            # fast path - check 
            # using CPU to get token_list & routing_weights index from expert_mask
            idx, top_x = torch.where(expert_mask)
            if top_x.numel() == 0:
                if config.debug: print("skip")
                return
            
            x = hidden_states
            if config.debug:
                print("========== x")
                print(x.numpy())

            # expert_layer
            self.update_batch(1)
            if scratch.n_tokens != 1:
                scratch.n_tokens = 1
                scratch.act_gate = cl.tensor(np.zeros([1, config.moe_intermediate_size], dtype=np.float16))
                scratch.act_up = cl.tensor(np.zeros([1, config.moe_intermediate_size], dtype=np.float16))

            # extract single rweights is time-consuming, just build a offseted tensor with just 1 element
            routing_weights.offset = idx[0]*2
            self.linear_up.forward(x, scratch.act_up, cl.tensor())
            self.linear_gate.forward(x, scratch.act_gate, scratch.act_up)
            self.linear_down3.forward(scratch.act_gate, final_hidden_states, routing_weights)   # with routing weights applied
        else:
            # slow path
            t0 = time.time()
            idx, top_x = torch.where(expert_mask)
            if top_x.numel() == 0:
                if config.debug: print("skip")
                return

            n_tokens = len(top_x)
            # print("n_tokens=", n_tokens)
            t1 = time.time()
            # print(f">>>>>>>> expert_mask:{expert_mask.shape}  {(t1-t0)*1e3:.3f} ms")
            
            if config.debug:
                print("========== expert_mask")
                print(expert_mask)
                print("========== top_x", top_x)
                print("========== idx", idx)

            self.update_batch(n_tokens)
            if scratch.n_tokens != n_tokens:
                scratch.n_tokens = n_tokens
                scratch.act_gate = cl.tensor(np.zeros([n_tokens, config.moe_intermediate_size], dtype=np.float16))
                scratch.act_up = cl.tensor(np.zeros([n_tokens, config.moe_intermediate_size], dtype=np.float16))
                scratch.y = cl.tensor(np.zeros([n_tokens, config.hidden_size], dtype=np.float16))
                scratch.x = cl.tensor(np.zeros([n_tokens, self.config.hidden_size], dtype=np.float16))
                scratch.rweights = cl.tensor(np.zeros([n_tokens], dtype=np.float16))

            tok_index = cl.tensor(np.array(top_x, dtype=np.int32))
            top_index = cl.tensor(np.array(idx, dtype=np.int32))
            

            self.ocl_kernels.enqueue("gather_2d_ref", [n_tokens, self.config.hidden_size],[1, 1],
                                    hidden_states, scratch.x,
                                    routing_weights, scratch.rweights,
                                    tok_index, top_index,
                                    hidden_states.shape[1], routing_weights.shape[1])

            if config.debug:
                print("========== x")
                print(scratch.x.numpy())

            # expert_layer
            self.linear_up.forward(scratch.x, scratch.act_up, cl.tensor())
            self.linear_gate.forward(scratch.x, scratch.act_gate, scratch.act_up)
            self.linear_down2.forward(scratch.act_gate, scratch.y, scratch.rweights)   # with routing weights applied

            if config.debug:
                print("========== y")
                print(scratch.y.numpy())

                # routing_weights[i, j] i'th token's top-j's routing weight
                # y[i, :]  i'th token's all hidden-states
                print("========== routing_weights[top_x, idx, None]", routing_weights.shape, routing_weights.shape[1])
                print(scratch.rweights.numpy())

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            # final_hidden_states.index_add_(0, torch.tensor(top_x), y)
            
            self.ocl_kernels.enqueue("index_add_", [n_tokens, self.config.hidden_size],[1, 1],
                                    scratch.y, final_hidden_states,
                                    tok_index, 
                                    hidden_states.shape[1])





def test_mlp(K_group_size, w_dtype):
    torch.manual_seed(0)

    mlp1 = Qwen3MoeMLP(config, intermediate_size = config.moe_intermediate_size)
    mlp1 = mlp1.eval()
    mlp2 = onednnMLP(mlp1)

    n_tokens = 8
    input = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)

    dst_ref = mlp1.forward(input.to(dtype=torch.float32))
    input_cl = cl.tensor(input.to(dtype=torch.float16).detach().numpy())
    dst_cur = mlp2.forward(input_cl)
    dst_cur = mlp2.forward(input_cl)
    dst_cur = mlp2.forward(input_cl)

    dst_ref = dst_ref.detach().numpy()
    dst_cur = dst_cur.numpy()
    if not np.allclose(dst_ref, dst_cur):
        print("============ dst_ref")
        print(dst_ref)
        print("============ dst_cur")
        print(dst_cur)

    print("\n"*6,f":::::::::: TEST EXPERT:::::::::: n_tokens={n_tokens} num_experts={config.num_experts} num_experts_per_tok={config.num_experts_per_tok} ")

    hidden_states = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)

    # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
    selected_experts = torch.randint(0, config.num_experts, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16).type(torch.LongTensor)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=config.num_experts).permute(2, 1, 0)
    # routing_weights[i, j] i'th token's top-j's routing weight
    routing_weights = torch.randint(0, 10, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16)

    # reference
    final_hidden_states1 = torch.zeros(n_tokens, config.hidden_size, dtype=torch.float32)
    for expert_id in range(config.num_experts):
        mlp1.forward_expert(hidden_states.to(dtype=torch.float32),
                            expert_mask[expert_id,...],
                            routing_weights.to(dtype=torch.float32),
                            final_hidden_states1)

    # impl
    final_hidden_states2 = cl.tensor(np.zeros([n_tokens, config.hidden_size], dtype=np.float16))
    cl_hidden_states = cl.tensor(hidden_states.detach().numpy())
    cl_routing_weights = cl.tensor(routing_weights.detach().numpy())
    for expert_id in range(config.num_experts):
        mlp2.forward_expert(cl_hidden_states,
                            expert_mask[expert_id,...],
                            cl_routing_weights,
                            final_hidden_states2)

    dst_ref = final_hidden_states1.detach().numpy()
    dst_cur = final_hidden_states2.numpy()
    
    if not np.allclose(dst_ref, dst_cur):
        print("============ dst_ref")
        print(dst_ref)
        print("============ dst_cur")
        print(dst_cur)


def test_moe(n_tokens = 120, running_experts = config.num_experts):
    torch.manual_seed(0)

    moe1 = [Qwen3MoeMLP(config, intermediate_size = config.moe_intermediate_size).eval() for _ in range(config.num_experts)]
    moe2 = [onednnMLP(mlp, i) for i,mlp in enumerate(moe1)]

    hidden_states = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)

    # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
    selected_experts = torch.randint(0, config.num_experts, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16).type(torch.LongTensor)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=config.num_experts).permute(2, 1, 0)
    # routing_weights[i, j] i'th token's top-j's routing weight
    routing_weights = torch.randint(0, 10, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16)

    # reference
    if config.debug: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> torch >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    final_hidden_states1 = torch.zeros(n_tokens, config.hidden_size, dtype=torch.float32)
    for expert_id in range(running_experts):
        moe1[expert_id].forward_expert(hidden_states.to(dtype=torch.float32),
                                       expert_mask[expert_id,...],
                                       routing_weights.to(dtype=torch.float32),
                                       final_hidden_states1)

    if config.debug: print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> onednn >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # impl
    scratch = scratchBuff()
    final_hidden_states2 = cl.tensor(np.zeros([n_tokens, config.hidden_size], dtype=np.float16))
    cl_hidden_states = cl.tensor(hidden_states.detach().numpy())
    cl_routing_weights = cl.tensor(routing_weights.detach().numpy())    
    for expert_id in range(running_experts):
        moe2[expert_id].forward_expert(cl_hidden_states,
                                       expert_mask[expert_id,...],
                                       cl_routing_weights,
                                       final_hidden_states2, scratch)

    dst_ref = final_hidden_states1.detach().numpy()
    dst_cur = final_hidden_states2.numpy()
    
    if not np.allclose(dst_ref, dst_cur):
        print("============ dst_ref")
        print(dst_ref)
        print("============ dst_cur")
        print(dst_cur)
    

    for r in range(10):
        cl.finish()
        t0 = time.time()
        for expert_id in range(running_experts):
            moe2[expert_id].forward_expert(cl_hidden_states,
                                           expert_mask[expert_id,...],
                                           cl_routing_weights,
                                           final_hidden_states2, scratch)
        latency_ns = cl.finish()
        t1 = time.time()
        for t in latency_ns:
            print(f"\t gpu-kernel: {t*1e-3:.3f} us")

        print(f" [{r}] :  {(t1 - t0)*1e3 : .3f} ms")

np.set_printoptions(linewidth=1024)

# cl.profiling(True)

test_moe(1)
test_moe(4096, running_experts = 1)

sys.exit()

if 0:
    test_mm(M = 1, K = 768, N = 2048, K_group_size = 0, w_dtype = cl.onednn_dtype.f16)
    test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.s8)
    test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.u4)

test_mlp(K_group_size = 128, w_dtype = cl.onednn_dtype.f16)
