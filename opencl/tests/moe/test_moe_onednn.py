#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import torch
import time, sys
import torch.nn.functional as F

from torch import nn
class config:
    hidden_size = 768
    intermediate_size = 2048
    num_experts = 128
    num_experts_per_tok = 8
    moe_intermediate_size = 2048
    weight_group_size = 128
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
        self.weight = (self.raw_weight_q.astype(np.float32) - self.raw_weight_z.astype(np.float32).repeat(K_group_size, axis=0)) * self.raw_weight_s.repeat(K_group_size, axis=0)
        # [N, K]
        self.weight_q4 = self.pack_i8_to_i4(self.raw_weight_q.transpose().copy())
        self.weight_scale = self.relayout_sz(self.raw_weight_s)
        self.weight_zp4 = self.pack_i8_to_i4(WeightQ4A.relayout_sz(self.raw_weight_z))
        self.nbytes = self.weight_q4.nbytes + self.weight_scale.nbytes + self.weight_zp4.nbytes
        
        self.weight_bias = np.random.randint(-1,3,[1, N]).astype(np.float16)/32.0

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


class QWEN_MLP(nn.Module):
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
        up_proj = up_proj.numpy()
        gate_proj = gate_proj.numpy()
        def glu(x):
            return x * (1 / (1 + np.exp(-x)))
        gate_proj = glu(gate_proj) * up_proj
        return torch.from_numpy(gate_proj)

    def forward(self, x):
        input = x
        gate_proj = input @ torch.from_numpy(self.gate.weight)

        #print("reference: gate_proj = ", gate_proj.detach().numpy())

        up_proj = (input @ torch.from_numpy(self.up.weight))
        down_proj = self.post_proc(gate_proj, up_proj) @ torch.from_numpy(self.down.weight)
        return down_proj

    # top_x： token index
    def forward_expert(self, hidden_states, expert_mask, routing_weights, final_hidden_states):
        # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
        # expert_mask[?,j,k] == 1 means apply current expert to token k as the top-j
        
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
        x = hidden_states[top_x, :] # slicing 2d

        # expert_layer
        y = self.forward(x)

        # routing_weights[i, j] i'th token's top-j's routing weight
        # y[i, :]  i'th token's all hidden-states
        y = y * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, torch.tensor(top_x, dtype=torch.int32), y)


ocl_sources=r'''

#define TYPE half
__kernel void softmax_topk(
    const __global TYPE* input, // [input_batch, sort_in_num]
    __global uint* output_index, // [input_batch, TOP_K]
    __global TYPE* output // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;

    uint sort_position = 0;

    __local TYPE local_input[VALUE_NUM];
    __local TYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

    TYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    local_input[sort_index] = in_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        TYPE value = local_input[i];
        if(value >= in_value) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        TYPE value = local_input[i];
        if(value > in_value) {
            sort_position++;
        }
    }
    if (sort_position < TOP_K) {
        local_output[sort_position] = in_value;
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sort_position == 0) {
        float softmax_total = 1.0;
        TYPE max_v = local_output[0];
        local_output[0] = 1;
        for(uint i = 1; i < TOP_K; i++) {
            local_output[i] = native_exp(local_output[i] - max_v);
            softmax_total += local_output[i];
        }
        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < TOP_K; i++) {
            output[i] = local_output[i]/softmax_total;
            output_index[i] = local_index[i];
        }
    }
}

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

'''

class scratchBuff:
    def __init__(self):
        self.x = None
        self.rweights = None
        self.n_tokens = -1
        self.act_gate = None
        self.act_up = None

def get_ocl_kernel():
    return cl.kernels(ocl_sources, f"-D FMACNT=4 -D UNROLL=4 -D INTERMEDIATE_SIZE={config.intermediate_size} -D TOP_K={config.num_experts_per_tok} -D VALUE_NUM={config.num_experts}")


class onednnMLP:
    def __init__(self, mlp, expert_id):
        self.w_dtype = cl.onednn_dtype.u4
        self.b_dtype = cl.onednn_dtype.undef
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

        self.gate_weight = cl.tensor(mlp.gate.weight_q4)
        self.gate_scale = cl.tensor(mlp.gate.weight_scale)
        self.gate_zp4 = cl.tensor(mlp.gate.weight_zp4)

        self.down_weight = cl.tensor(mlp.down.weight_q4)
        self.down_scale = cl.tensor(mlp.down.weight_scale)
        self.down_zp4 = cl.tensor(mlp.down.weight_zp4)


        self.up_bias = cl.tensor(mlp.up.weight_bias)
        self.gate_bias = cl.tensor(mlp.gate.weight_bias)
        self.down_bias = cl.tensor(mlp.down.weight_bias)

        self.ocl_kernels = get_ocl_kernel()
        #cl.kernels(ocl_sources, f"-D FMACNT=4 -D UNROLL=4 -D INTERMEDIATE_SIZE={config.intermediate_size} -D TOP_K={config.num_experts_per_tok} -D VALUE_NUM={config.num_experts}")

    def update_batch(self, batch):
        if self.M != batch:
            empty_cl_tensor = cl.tensor()
            
            self.linear_up = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.hidden_size, self.config.moe_intermediate_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        self.w_dtype, 
                                        self.up_weight, self.up_scale, self.up_zp4, empty_cl_tensor)

            self.linear_gate = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.hidden_size, self.config.moe_intermediate_size,
                                        self.K_group_size, cl.onednn_matmul_type.with_silu_bin_mul,
                                        self.w_dtype, 
                                        self.gate_weight, self.gate_scale, self.gate_zp4, empty_cl_tensor)

            self.linear_down = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.none,
                                        self.w_dtype, 
                                        self.down_weight, self.down_scale, self.down_zp4, empty_cl_tensor)

            self.linear_down2 = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, self.b_dtype, batch, self.config.moe_intermediate_size, self.config.hidden_size,
                                        self.K_group_size, cl.onednn_matmul_type.with_bin_mul_per_row,
                                        self.w_dtype, 
                                        self.down_weight, self.down_scale, self.down_zp4, empty_cl_tensor)
            self.M = batch

    def forward(self, input):
        M = input.shape[0]
        self.update_batch(M)
        temp_gate = cl.tensor(np.zeros([M, config.intermediate_size], dtype=np.float16))
        temp_up = cl.tensor(np.zeros([M, config.intermediate_size], dtype=np.float16))
        dst = cl.tensor(np.zeros([M, config.hidden_size], dtype=np.float16))

        self.linear_up.forward(input, temp_up, cl.tensor())
        self.linear_gate.forward(input, temp_gate, temp_up)
        self.linear_down.forward(temp_gate, dst, cl.tensor())
        return dst


    # top_x： token index
    # def forward_expert(self, hidden_states, idx, top_x, routing_weights, final_hidden_states, scratch):
    def forward_expert(self, hidden_states, expert_mask, routing_weights, final_hidden_states, scratch):
        # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
        # expert_mask[?,j,k] == 1 means apply current expert to token k as the top-j
        
        # x = hidden_states[top_x, :] # slicing2d

        idx, top_x = torch.where(expert_mask)
        if top_x.numel() == 0:
            if config.debug: print("skip")
            return

        n_tokens = len(top_x)
        #print("n_tokens=", n_tokens)

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
        self.linear_gate.forward(scratch.x, scratch.act_gate, scratch.act_up)
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
        #print(f">>>>>>>> expert_mask:{expert_mask.shape}, token_num = {n_tokens}, {(t1-t0)*1e3:.3f} ms")


def test_mlp(K_group_size, n_tokens=8):
    torch.manual_seed(0)

    mlp1 = QWEN_MLP(config)
    mlp1 = mlp1.eval()
    mlp2 = onednnMLP(mlp1, 0)

    input = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)

    dst_ref = mlp1.forward(input.to(dtype=torch.float32))
    input_cl = cl.tensor(input.to(dtype=torch.float16).detach().numpy())
    dst_cur = mlp2.forward(input_cl)

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

    low, high = -1.0, 1.0
    hidden_states = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)
    router_logits = (np.random.rand(n_tokens, config.num_experts) * (high - low) + low).astype(np.float16)
    
    routing_weights = F.softmax(torch.from_numpy(router_logits), dim=1, dtype=torch.float)
    #print(f"routing_weights_softmax={routing_weights}")
    routing_weights, selected_experts = torch.topk(routing_weights, k=config.num_experts_per_tok, dim=1)
    #print(f"routing_weights_topk={routing_weights}")
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float16).reshape(n_tokens, config.num_experts_per_tok)
    selected_experts = selected_experts.reshape(n_tokens, config.num_experts_per_tok)

    print(f"selected_experts=\n{selected_experts}")
    print(f"routing_weights=\n{routing_weights}")

    # # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
    # selected_experts = torch.randint(0, config.num_experts, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16).type(torch.LongTensor)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=config.num_experts).permute(2, 1, 0)
    # # routing_weights[i, j] i'th token's top-j's routing weight
    # routing_weights = torch.randint(0, 10, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16)

    # reference
    final_hidden_states1 = torch.zeros(n_tokens, config.hidden_size, dtype=torch.float32)
    for expert_id in range(config.num_experts):
        mlp1.forward_expert(hidden_states.to(dtype=torch.float32),
                            expert_mask[expert_id,...],
                            routing_weights.to(dtype=torch.float32),
                            final_hidden_states1)

    # impl
    # self.ocl_kernels.enqueue("softmax_topk",[n_tokens, config.num_experts],
    #                         [1, config.num_experts],
    #                         t_router_logits,
    #                         t_topk_idx, t_topk_weights)
    scratch = scratchBuff()

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

    moe1 = [QWEN_MLP(config).eval() for _ in range(config.num_experts)]
    moe2 = [onednnMLP(mlp, i) for i,mlp in enumerate(moe1)]

    low, high = -1.0, 1.0
    hidden_states = torch.randint(-1, 2, size=[n_tokens, config.hidden_size], dtype=torch.float16)
    router_logits = (np.random.rand(n_tokens, config.num_experts) * (high - low) + low).astype(np.float16)
    
    routing_weights = F.softmax(torch.from_numpy(router_logits), dim=1, dtype=torch.float)
    #print(f"routing_weights_softmax={routing_weights}")
    routing_weights, selected_experts = torch.topk(routing_weights, k=config.num_experts_per_tok, dim=1)
    #print(f"routing_weights_topk={routing_weights}")
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float16).reshape(n_tokens, config.num_experts_per_tok)
    selected_experts = selected_experts.reshape(n_tokens, config.num_experts_per_tok)

    # expert_mask[i,j,k] == 1 means expert i is selected by token k as the top-j
    # selected_experts = torch.randint(0, config.num_experts, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16).type(torch.LongTensor)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=config.num_experts).permute(2, 1, 0)
    # routing_weights[i, j] i'th token's top-j's routing weight
    # routing_weights = torch.randint(0, 10, size=[n_tokens, config.num_experts_per_tok], dtype=torch.float16)

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


