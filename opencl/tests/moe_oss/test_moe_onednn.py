#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import torch
import time, sys

oss_alpha=1.702
oss_limit=7.0

from torch import nn
class config:
    hidden_size = 2880
    intermediate_size = 2880
    num_experts = 32
    num_experts_per_tok = 4
    moe_intermediate_size = 2880
    weight_group_size = 32
    debug = False

SZ_LAYOUT = 0
class WeightQ4A:
    def __init__(self, K, N, K_group_size):
        self.K = K
        self.N = N
        self.K_group_size = K_group_size
        assert K % K_group_size == 0
        self.K_groups = K // K_group_size
        self.raw_weight_q = np.random.randint(7,9,[K, N]).astype(np.int8)
        self.raw_weight_s = np.random.randint(-4,5,[self.K_groups, N]).astype(np.float16)/32.0
        self.raw_weight_z = np.random.randint(8,9,[self.K_groups, N]).astype(np.int8)
        self.raw_bias = np.random.randint(-1,3,[1, N]).astype(np.float16)/32.0
        self.weight = (self.raw_weight_q.astype(np.float32) - self.raw_weight_z.astype(np.float32).repeat(K_group_size, axis=0)) * self.raw_weight_s.repeat(K_group_size, axis=0)
        # [N, K]
        self.weight_q4 = self.pack_i8_to_i4(self.raw_weight_q.transpose().copy())
        self.weight_scale = self.relayout_sz(self.raw_weight_s)
        self.weight_zp4 = self.pack_i8_to_i4(WeightQ4A.relayout_sz(self.raw_weight_z))
        self.weight_bias = self.raw_bias.copy()
        self.alpha = oss_alpha
        self.nbytes = self.weight_q4.nbytes + self.weight_scale.nbytes + self.weight_zp4.nbytes + self.weight_bias.nbytes

    @staticmethod
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

    @staticmethod
    def relayout_sz(sz):
        if SZ_LAYOUT == 0:
            return sz
        k_groups, N = sz.shape
        assert N % 2 == 0
        new_sz = np.zeros([N//2, k_groups*2], sz.dtype)
        for n in range(N//2):
            for k in range(k_groups):
                new_sz[n, 2*k + 0] = sz[k, 2*n+0]
                new_sz[n, 2*k + 1] = sz[k, 2*n+1]
        return new_sz

    def print(self):
        print(f"WeightQ4A: K={self.K}, N={self.N}, K_group_size={self.K_group_size}, K_groups={self.K_groups}")
        print(f"raw_weight_q.shape={self.raw_weight_q.shape},raw_weight_q=\n{self.raw_weight_q[:16,:16]}")
        print(f"raw_weight_s.shape={self.raw_weight_s.shape},raw_weight_s=\n{self.raw_weight_s[:16,:16]}")
        print(f"raw_weight_z.shape={self.raw_weight_z.shape},raw_weight_z=\n{self.raw_weight_z[:16,:16]}")
        print(f"weight.shape={self.weight.shape},weight=\n{self.weight[:16,:16]}")
        print(f"weight_q4.shape={self.weight_q4.shape}, weight_q4=\n{self.weight_q4[:16,:16]}")
        print(f"weight_scale.shape={self.weight_scale.shape}, weight_scale=\n{self.weight_scale[:16,:16]}")
        print(f"weight_zp4.shape={self.weight_zp4.shape}, weight_zp4=\n{self.weight_zp4[:16,:16]}")


class GPT_OSS_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.group_size = config.weight_group_size
        self.gate = WeightQ4A(self.hidden_size, self.intermediate_size, self.group_size)
        self.up = WeightQ4A(self.hidden_size, self.intermediate_size, self.group_size)
        self.down = WeightQ4A(self.intermediate_size, self.hidden_size, self.group_size)

    def post_proc(self, gate_proj, up_proj):
        up_proj = up_proj.clamp(min=-oss_limit, max=oss_limit).numpy()
        gate_proj = gate_proj.clamp(min=None, max=oss_limit).numpy()
        def glu(x, alpha):
            return x * (1 / (1 + np.exp(-alpha * x)))
        gate_proj = glu(gate_proj, oss_alpha) * (up_proj + 1)
        return torch.from_numpy(gate_proj)


    def forward(self, x):
        input = x
        gate_proj = (input @ torch.from_numpy(self.gate.weight)) + self.gate.raw_bias

        #print("reference: gate_proj = ", gate_proj.detach().numpy())

        up_proj = (input @ torch.from_numpy(self.up.weight)) + self.up.raw_bias
        down_proj = self.post_proc(gate_proj, up_proj) @ torch.from_numpy(self.down.weight) + self.down.raw_bias
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

        x = hidden_states[top_x, :] # slicing 2d
        if config.debug:
            print("========== x")
            print(x.detach().numpy())
        # expert_layer
        y = self.forward(x)
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
// [n_tokens, count] [1, 16]
__attribute__((intel_reqd_sub_group_size(16)))
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

    // dst_tok[off] = src_tok[off];
    ushort value = intel_sub_group_block_read_us((const __global ushort *)(src_tok + off));
    intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), value);

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[tok_idx * rweight_size + top_idx];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
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

    // dst_tok[off] += src_tok[off];
    half src_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(src_tok + off)));
    half dst_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(dst_tok + off)));
    half value = dst_value + src_value;
    intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_ushort(value));
}

// gws[batch, intermediate_size]
// lws[1, 16]
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gate_up_post_proc(
    const __global half* gate,
    const __global half* up,
    __global half* output) {

    int m = get_global_id(0);
    int n = get_global_id(1);

    int offset = m * INTERMEDIATE_SIZE + n;

    const half oss_alpha = -1.702;
    const half oss_limit = 7.0;
    const half oss_neg_limit = -7.0;
    const half oss_min_none = -3e38;

    half src_gate = as_half(intel_sub_group_block_read_us((const __global ushort *)(gate + offset)));
    half src_up = as_half(intel_sub_group_block_read_us((const __global ushort *)(up + offset)));

    //if(src_up < oss_neg_limit) {
    //    src_up = oss_neg_limit  ;
    //} else if(src_up > oss_limit) {
    //    src_up = oss_limit;
    //}

    //if(src_gate > oss_limit) {
    //    src_gate = oss_limit;
    //}

    src_up = clamp(src_up, oss_neg_limit, oss_limit);
    src_gate = clamp(src_gate, oss_min_none, oss_limit);

    half value = src_gate * ( 1.0 / (1.0 + native_exp(oss_alpha * src_gate))) * (src_up + 1.0h);
    intel_sub_group_block_write_us((__global ushort *)(output + offset), as_ushort(value));
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
        self.w_dtype = cl.onednn_dtype.u4
        self.b_dtype = cl.onednn_dtype.f16
        self.config = mlp.config
        self.M = -1
        self.expert_id = expert_id
        self.K_group_size = config.weight_group_size
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.total_latency = 0.0

        self.up_weight = cl.tensor(mlp.up.weight_q4)
        self.up_scale = cl.tensor(mlp.up.weight_scale)
        self.up_zp4 = cl.tensor(mlp.up.weight_zp4)
        self.up_bias = cl.tensor(mlp.up.weight_bias)

        self.gate_weight = cl.tensor(mlp.gate.weight_q4)
        self.gate_scale = cl.tensor(mlp.gate.weight_scale)
        self.gate_zp4 = cl.tensor(mlp.gate.weight_zp4)
        self.gate_bias = cl.tensor(mlp.gate.weight_bias)

        self.down_weight = cl.tensor(mlp.down.weight_q4)
        self.down_scale = cl.tensor(mlp.down.weight_scale)
        self.down_zp4 = cl.tensor(mlp.down.weight_zp4)
        self.down_bias = cl.tensor(mlp.down.weight_bias)

        self.ocl_kernels = cl.kernels(ocl_sources, f"-D FMACNT=4 -D UNROLL=4 -D INTERMEDIATE_SIZE={config.intermediate_size}")

    def update_batch(self, batch):
        if self.M != batch:
            empty_cl_tensor = cl.tensor()

            self.linear_up = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.hidden_size, self.config.moe_intermediate_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        self.w_dtype, 
                                        self.up_weight, self.up_scale, self.up_zp4, self.up_bias)

            self.linear_gate = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.hidden_size, self.config.moe_intermediate_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        self.w_dtype, 
                                        self.gate_weight, self.gate_scale, self.gate_zp4, self.gate_bias)

            self.linear_down = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        self.w_dtype, 
                                        self.down_weight, self.down_scale, self.down_zp4, self.down_bias)
            
            
            self.linear_down2 = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_mul_per_row,
                                        self.w_dtype, 
                                        self.down_weight, self.down_scale, self.down_zp4, self.down_bias)
            self.M = batch

    def forward(self, input):
        M = input.shape[0]
        self.update_batch(M)
        temp_gate = cl.tensor(np.zeros([M, config.intermediate_size], dtype=np.float16))
        temp_up = cl.tensor(np.zeros([M, config.intermediate_size], dtype=np.float16))
        dst = cl.tensor(np.zeros([M, config.hidden_size], dtype=np.float16))

        self.linear_gate.forward(input, temp_gate, cl.tensor())
        self.linear_up.forward(input, temp_up, cl.tensor())
        self.ocl_kernels.enqueue("gate_up_post_proc", [M, config.intermediate_size],[1, 16],
                                    temp_gate, temp_up,
                                    temp_gate)
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

            # expert_layer
            self.update_batch(1)
            if scratch.n_tokens != 1:
                scratch.n_tokens = 1
                scratch.act_gate = cl.tensor(np.zeros([1, config.moe_intermediate_size], dtype=np.float16))
                scratch.act_up = cl.tensor(np.zeros([1, config.moe_intermediate_size], dtype=np.float16))

            # extract single rweights is time-consuming, just build a offseted tensor with just 1 element
            routing_weights.offset = idx[0]*2
            self.linear_up.forward(x, scratch.act_up, cl.tensor())
            self.linear_gate.forward(x, scratch.act_gate, cl.tensor())
            self.ocl_kernels.enqueue("gate_up_post_proc", [1, self.intermediate_size],[1, 16],
                                    scratch.act_gate, scratch.act_up,
                                    scratch.act_gate)
            self.linear_down2.forward(scratch.act_gate, final_hidden_states, routing_weights)   # with routing weights applied
        else:
            # slow path
            idx, top_x = torch.where(expert_mask)
            if top_x.numel() == 0:
                if config.debug: print("skip")
                return

            n_tokens = len(top_x)
            # print("n_tokens=", n_tokens)

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
            
            t0 = time.time()
            self.ocl_kernels.enqueue("gather_2d_ref", [n_tokens, self.config.hidden_size],[1, 16],
                                    hidden_states, scratch.x,
                                    routing_weights, scratch.rweights,
                                    tok_index, top_index,
                                    hidden_states.shape[1], routing_weights.shape[1])

            # expert_layer
            self.linear_up.forward(scratch.x, scratch.act_up, cl.tensor())
            self.linear_gate.forward(scratch.x, scratch.act_gate, cl.tensor())
            self.ocl_kernels.enqueue("gate_up_post_proc", [n_tokens, self.intermediate_size],[1, 16],
                                    scratch.act_gate, scratch.act_up,
                                    scratch.act_gate)
            self.linear_down2.forward(scratch.act_gate, scratch.y, scratch.rweights)   # with routing weights applied

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            # final_hidden_states.index_add_(0, torch.tensor(top_x), y)
            self.ocl_kernels.enqueue("index_add_", [n_tokens, self.config.hidden_size],[1, 16],
                                    scratch.y, final_hidden_states,
                                    tok_index, 
                                    hidden_states.shape[1])
            cl.finish()
            t1 = time.time()
            print(f">>>>>>>> expert_mask:{expert_mask.shape}, token_num = {n_tokens}, {(t1-t0)*1e3:.3f} ms")


def test_mlp(K_group_size, n_tokens=8):
    torch.manual_seed(0)

    mlp1 = GPT_OSS_MLP(config)
    mlp1 = mlp1.eval()
    mlp2 = onednnMLP(mlp1, 0)

    input = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)

    dst_ref = mlp1.forward(input.to(dtype=torch.float32))
    input_cl = cl.tensor(input.to(dtype=torch.float16).detach().numpy())
    dst_cur = mlp2.forward(input_cl)
    # dst_cur = mlp2.forward(input_cl)
    # dst_cur = mlp2.forward(input_cl)

    #dst_ref = dst_ref.detach().numpy()
    dst_cur = dst_cur.numpy()
    if not np.allclose(dst_ref, dst_cur, rtol=1.e-2, atol=1.e-1):
        print("============ dst_ref")
        print(dst_ref)
        print("============ dst_cur")
        print(dst_cur)
    else:
        print("mlp test forward pass!")

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

    scratch = scratchBuff()
    # impl
    final_hidden_states2 = cl.tensor(np.zeros([n_tokens, config.hidden_size], dtype=np.float16))
    cl_hidden_states = cl.tensor(hidden_states.detach().numpy())
    cl_routing_weights = cl.tensor(routing_weights.detach().numpy())
    for expert_id in range(config.num_experts):
        mlp2.forward_expert(cl_hidden_states,
                            expert_mask[expert_id,...],
                            cl_routing_weights,
                            final_hidden_states2, scratch)

    dst_ref = final_hidden_states1.detach().numpy()
    dst_cur = final_hidden_states2.numpy()
    
    if not np.allclose(dst_ref, dst_cur, rtol=1.e-2, atol=1.e-1):
        print("============ dst_ref")
        print(dst_ref)
        print("============ dst_cur")
        print(dst_cur)
    else:
        print("mlp test forward_expert pass!")


def test_moe(n_tokens = 120, running_experts = config.num_experts):
    torch.manual_seed(0)

    moe1 = [GPT_OSS_MLP(config).eval() for _ in range(config.num_experts)]
    moe2 = [onednnMLP(mlp, i) for i,mlp in enumerate(moe1)]

    hidden_states = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)

    # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
    selected_experts = torch.randint(0, config.num_experts, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16).type(torch.LongTensor)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=config.num_experts).permute(2, 1, 0)
    # routing_weights[i, j] i'th token's top-j's routing weight
    routing_weights = torch.randint(0, 10, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16)

    # reference
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> torch >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    final_hidden_states1 = torch.zeros(n_tokens, config.hidden_size, dtype=torch.float32)
    for expert_id in range(running_experts):
        moe1[expert_id].forward_expert(hidden_states.to(dtype=torch.float32),
                                       expert_mask[expert_id,...],
                                       routing_weights.to(dtype=torch.float32),
                                       final_hidden_states1)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> onednn >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
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
        # for t in latency_ns:
        #     print(f"\t gpu-kernel: {t*1e-3:.3f} us")

        print(f" [{r}] :  {(t1 - t0)*1e3 : .3f} ms")

np.set_printoptions(linewidth=1024)

cl.profiling(True)

test_mlp(K_group_size = 32, n_tokens=12)
test_moe(1024)


sys.exit()


