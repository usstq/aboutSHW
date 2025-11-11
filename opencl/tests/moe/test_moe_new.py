#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys

import torch
import torch.nn.functional as F
from torch import nn

dump_input_output = 0

#GROUP_SIZE only support 128 or -1, don't modify it
GROUP_SIZE = -1

K = INTERMEDIATE_SIZE = 768
N = HIDDEN_SIZE = 2048

MAX_TOPK = 8
N_EXPERTS = 128
# xe2:32, xe1:16
SUBGROUP_SIZE = 16
SUBGROUP_NUM = 8
N_BLOCK = 4
SZ_LAYOUT = 0

M = 1

if GROUP_SIZE == -1:
    GATE_UP_GROUP_SIZE = HIDDEN_SIZE
    DOWN_GROUP_SIZE = INTERMEDIATE_SIZE
else:
    GATE_UP_GROUP_SIZE = GROUP_SIZE
    DOWN_GROUP_SIZE = GROUP_SIZE
    assert (INTERMEDIATE_SIZE % GROUP_SIZE) == 0
    assert (HIDDEN_SIZE % GROUP_SIZE) == 0
#K_groups = K // GROUP_SIZE

GATE_IDX=0
UP_IDX=1
DOWN_IDX=2

torch.set_printoptions(threshold=float('inf'))
np.set_printoptions(threshold=np.inf)

class WeightQ4A:
    def __init__(self, K, N, K_group_size):
        self.K = K
        self.N = N
        self.K_group_size = K_group_size
        assert K % K_group_size == 0
        self.K_groups = K // K_group_size
        self.raw_weight_q = np.random.randint(0,3,[K, N]).astype(np.int8)
        self.raw_weight_s = np.random.randint(-4,5,[self.K_groups, N]).astype(np.float16)/32.0
        self.raw_weight_z = np.random.randint(0,3,[self.K_groups, N]).astype(np.int8)
        self.weight = (self.raw_weight_q.astype(np.float32) - self.raw_weight_z.astype(np.float32).repeat(K_group_size, axis=0)) * self.raw_weight_s.repeat(K_group_size, axis=0)
        # [N, K]
        self.weight_q4 = self.pack_i8_to_i4(self.raw_weight_q.transpose().copy())
        self.weight_scale = self.relayout_sz(self.raw_weight_s)
        self.weight_zp4 = self.pack_i8_to_i4(WeightQ4A.relayout_sz(self.raw_weight_z))
        self.nbytes = self.weight_q4.nbytes + self.weight_scale.nbytes + self.weight_zp4.nbytes

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

class MoE:
    def __init__(self, N_experts, hidden_size, intermediate_size, gate_up_group_size, down_group_size):
        self.N_experts = N_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_group_size = gate_up_group_size
        self.down_group_size = down_group_size

        # gate/up/down - [expert_num, 3]
        self.experts = [[WeightQ4A(hidden_size, intermediate_size, gate_up_group_size), WeightQ4A(hidden_size, intermediate_size, gate_up_group_size), WeightQ4A(intermediate_size, hidden_size, down_group_size)] for _ in range(N_experts)]
        self.gate_fused_weight, self.gate_fused_weight_offset = MoE.fuse_to_single_weight(N_experts, self.experts, GATE_IDX)
        self.gate_fused_scale, self.gate_fused_scale_offset = MoE.fuse_to_single_scale(N_experts, self.experts, GATE_IDX)
        self.gate_fused_zp4, self.gate_fused_zp4_offset = MoE.fuse_to_single_zp(N_experts, self.experts, GATE_IDX)

        self.up_fused_weight, self.up_fused_weight_offset = MoE.fuse_to_single_weight(N_experts, self.experts, UP_IDX)
        self.up_fused_scale, self.up_fused_scale_offset = MoE.fuse_to_single_scale(N_experts, self.experts, UP_IDX)
        self.up_fused_zp4, self.up_fused_zp4_offset = MoE.fuse_to_single_zp(N_experts, self.experts, UP_IDX)

        self.down_fused_weight, self.down_fused_weight_offset = MoE.fuse_to_single_weight(N_experts, self.experts, DOWN_IDX)
        self.down_fused_scale, self.down_fused_scale_offset = MoE.fuse_to_single_scale(N_experts, self.experts, DOWN_IDX)
        self.down_fused_zp4, self.down_fused_zp4_offset = MoE.fuse_to_single_zp(N_experts, self.experts, DOWN_IDX)

    @staticmethod
    def fuse_to_single_weight(expert_num, experts, idx):
        weight_size = expert_num * experts[0][idx].weight_q4.size
        fused_weight = np.zeros([weight_size], dtype=np.uint8)
        # offset = wei/scale/zp, 1 per expert
        fused_weight_offset = np.zeros([expert_num], dtype=np.uint32)
        offset = 0
        for i in range(expert_num):
            w = experts[i][idx]
            fused_weight[offset:offset + w.weight_q4.nbytes] = w.weight_q4.view(np.uint8).reshape(-1)
            fused_weight_offset[i] = offset
            offset += w.weight_q4.nbytes
        
        # print(f"fused_weight_offset={fused_weight_offset}")
        # print(f"fused_weight.size={fused_weight.size}, offset={offset}")
        # experts[0][1].print()
        # print(f"fused_weight={fused_weight[672:720]}")
        
        gate_wei_size = HIDDEN_SIZE * INTERMEDIATE_SIZE // 2
        up_wei_size = gate_wei_size
        down_wei_size = INTERMEDIATE_SIZE * HIDDEN_SIZE // 2
        all_wei_size = [gate_wei_size, up_wei_size, down_wei_size]

        expected_size = expert_num * all_wei_size[idx]
        print(f"idx = {idx}, expected_size={expected_size}, gate_wei_size={gate_wei_size}, up_wei_size={up_wei_size}, down_wei_size={down_wei_size}")

        assert fused_weight.size == expected_size, f"fused_weight.size={fused_weight.size}, expected_size={expected_size}"
        return fused_weight, fused_weight_offset

    @staticmethod
    def fuse_to_single_scale(expert_num, experts, idx):
        scale_size = expert_num * experts[0][idx].weight_scale.size * 2
        fused_scale = np.zeros([scale_size], dtype=np.uint8)
        # offset = wei/scale/zp, 1 per expert
        fused_scale_offset = np.zeros([expert_num], dtype=np.uint32)
        offset = 0
        for i in range(expert_num):
            w = experts[i][idx]
            fused_scale[offset:offset + w.weight_scale.nbytes] = w.weight_scale.view(np.uint8).reshape(-1)
            fused_scale_offset[i] = offset
            offset += w.weight_scale.nbytes

        gate_scale_size = (HIDDEN_SIZE // GATE_UP_GROUP_SIZE) * INTERMEDIATE_SIZE * 2
        up_scale_size = gate_scale_size
        down_scale_size = (INTERMEDIATE_SIZE // DOWN_GROUP_SIZE) * HIDDEN_SIZE  * 2
        all_scale_size = [gate_scale_size, up_scale_size, down_scale_size]

        expected_size = expert_num * all_scale_size[idx]
        print(f"idx = {idx}, expected_size={expected_size}, gate_scale_size={gate_scale_size}, up_scale_size={up_scale_size}, down_scale_size={down_scale_size}")

        assert fused_scale.size == expected_size, f"fused_scale.size={fused_scale.size}, expected_size={expected_size}"
        return fused_scale, fused_scale_offset

    @staticmethod
    def fuse_to_single_zp(expert_num, experts, idx):
        zp_size = expert_num * experts[0][idx].weight_zp4.size
        fused_zp = np.zeros([zp_size], dtype=np.uint8)
        # offset = wei/scale/zp, 1 per expert
        fused_zp_offset = np.zeros([expert_num], dtype=np.uint32)
        offset = 0
        for i in range(expert_num):
            w = experts[i][idx]
            fused_zp[offset:offset + w.weight_zp4.nbytes] = w.weight_zp4.view(np.uint8).reshape(-1)
            fused_zp_offset[i] = offset
            offset += w.weight_zp4.nbytes

        gate_zp_size = (HIDDEN_SIZE // GATE_UP_GROUP_SIZE) * INTERMEDIATE_SIZE // 2
        up_zp_size = gate_zp_size
        down_zp_size = (INTERMEDIATE_SIZE // DOWN_GROUP_SIZE) * HIDDEN_SIZE // 2
        all_zp_size = [gate_zp_size, up_zp_size, down_zp_size]

        expected_size = expert_num * all_zp_size[idx]
        print(f"idx = {idx}, expected_size={expected_size}, gate_zp_size={gate_zp_size}, up_zp_size={up_zp_size}, down_zp_size={down_zp_size}")

        assert fused_zp.size == expected_size, f"fused_zp.size={fused_zp.size}, expected_size={expected_size}"
        return fused_zp, fused_zp_offset

moe_op = MoE(N_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, GATE_UP_GROUP_SIZE, DOWN_GROUP_SIZE)

low, high = -1.0, 1.0
hidden_states = (np.random.rand(M, HIDDEN_SIZE) * (high - low) + low).astype(np.float16)
router_logits = (np.random.rand(M, N_EXPERTS) * (high - low) + low).astype(np.float16)


def get_reference(hidden_states, router_logits, moe_op):
    routing_weights = F.softmax(torch.from_numpy(router_logits), dim=1, dtype=torch.float)
    #print(f"routing_weights_softmax={routing_weights}")
    routing_weights, selected_experts = torch.topk(routing_weights, k=MAX_TOPK, dim=1)
    #print(f"routing_weights_topk={routing_weights}")
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(torch.float16).reshape(M, MAX_TOPK)
    selected_experts = selected_experts.reshape(M, MAX_TOPK)

    print(f"selected_experts={selected_experts}")
    print(f"routing_weights={routing_weights}")

    final_hidden_states = torch.zeros((M, HIDDEN_SIZE), dtype=torch.float16).numpy()

    #expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=N_EXPERTS).permute(0, 2, 1)  # [M, N_EXPERTS, MAX_TOPK]
    #expert_hit = torch.greater(expert_mask.sum(dim=(-1,-2)), 0).nonzero()
    #print("expert_mask=", expert_mask)
    #print(f"expert_hit={expert_hit}")
    #for expert_idx in expert_hit:
    expert_top_idx = 0
    for expert_idx in selected_experts[0]:
        # print(f"expert_idx={expert_idx}")
        expert_layer = moe_op.experts[expert_idx]

        up_proj = (hidden_states @ expert_layer[1].weight).astype(np.float16)
        # print(f"current_state={hidden_states}")
        # print(f"expert_layer[1].weight={expert_layer[1].weight[:16,:16]}")
        # print(f"up_proj[{expert_idx}]={up_proj}")

        gate_proj = (hidden_states @ expert_layer[0].weight).astype(np.float16)
        def silu(x):
            return x * (1 / (1 + np.exp(-x)))
        gate_proj = silu(gate_proj) * up_proj

        # print(f"gate_up_proj[{expert_idx}]={gate_proj}")

        down_proj = (gate_proj @ expert_layer[2].weight).astype(np.float16)
        current_hidden_states = down_proj * routing_weights[0][expert_top_idx].numpy()
        expert_top_idx += 1

        # print(f"down_proj[{expert_idx}]={current_hidden_states}")

        final_hidden_states += current_hidden_states
    final_hidden_states = final_hidden_states.reshape(M, HIDDEN_SIZE)
    return final_hidden_states


reference= get_reference(hidden_states, router_logits, moe_op)
print()
print()


t_hidden_states = cl.tensor(hidden_states)

t_router_logits = cl.tensor(router_logits)
t_topk_idx = cl.tensor(np.zeros([M, MAX_TOPK], dtype=np.int32))
t_topk_weights = cl.tensor(np.zeros([M, MAX_TOPK], dtype=np.float16))

t_gate_fused_weights = cl.tensor(moe_op.gate_fused_weight)
t_gate_fused_scale = cl.tensor(moe_op.gate_fused_scale)
t_gate_fused_zp4 = cl.tensor(moe_op.gate_fused_zp4)

t_up_fused_weights = cl.tensor(moe_op.up_fused_weight)
t_up_fused_scale = cl.tensor(moe_op.up_fused_scale)
t_up_fused_zp4 = cl.tensor(moe_op.up_fused_zp4)

t_down_fused_weights = cl.tensor(moe_op.down_fused_weight)
t_down_fused_scale = cl.tensor(moe_op.down_fused_scale)
t_down_fused_zp4 = cl.tensor(moe_op.down_fused_zp4)

t_gate_up_output = cl.tensor(np.zeros([M*MAX_TOPK, INTERMEDIATE_SIZE], dtype=np.float16))
t_down_output = cl.tensor(np.zeros([M*MAX_TOPK, HIDDEN_SIZE], dtype=np.float16))
t_final_hidden_state = cl.tensor(np.zeros([M, HIDDEN_SIZE], dtype=np.float16))


moe_mlp_cl_source_file = "./moe_mlp_new.cl"
with open(moe_mlp_cl_source_file, "r") as file:
    # Read the entire file content into a string
    moe_mlp_src = file.read()

moe_opt_cl_source_file = "./moe_opt.cl"
with open(moe_opt_cl_source_file, "r") as file:
    # Read the entire file content into a string
    moe_opt_src = file.read()

cl.profiling(True)
TYPE="half"
TYPE_SIZE=2
moe_opt_kernels = cl.kernels(moe_opt_src,
                            f"-D TYPE={TYPE} \
                            -D VALUE_NUM={N_EXPERTS} \
                            -D TOP_K={MAX_TOPK} \
                            -D TYPE_SIZE={TYPE_SIZE} \
                            -D HIDDEN_SIZE={HIDDEN_SIZE} \
                            -D SZ_LAYOUT={SZ_LAYOUT}", "./moe_opt_dump")

moe_mlp_kernels = cl.kernels(moe_mlp_src,
                            f"-D N_BLOCK={N_BLOCK} \
                            -D TYPE={TYPE} \
                            -D TYPE_SIZE={TYPE_SIZE} \
                            -D SUBGROUP_SIZE={SUBGROUP_SIZE}\
                            -D GATE_UP_GROUP_SIZE={GATE_UP_GROUP_SIZE} \
                            -D DOWN_GROUP_SIZE={DOWN_GROUP_SIZE} \
                            -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} \
                            -D HIDDEN_SIZE={HIDDEN_SIZE} \
                            -D MAX_TOPK={MAX_TOPK} \
                            -D SZ_LAYOUT={SZ_LAYOUT}", "./moe_mlp_dump")


moe_opt_kernels.enqueue("softmax_topk",[M, N_EXPERTS],
                                        [1, N_EXPERTS],
                                        t_router_logits,
                                        t_topk_idx, t_topk_weights)

cl.finish()
o_topk_idx = t_topk_idx.numpy()
o_topk_weights = t_topk_weights.numpy()
print(f"o_topk_idx={o_topk_idx}")
print(f"o_topk_weights={o_topk_weights}")
print()

print(f"GWS = [{MAX_TOPK}, {SUBGROUP_SIZE}, {INTERMEDIATE_SIZE//N_BLOCK}], LWS = [1, {SUBGROUP_SIZE}, {SUBGROUP_NUM}]")
moe_mlp_kernels.enqueue("mlp_gate_up",[MAX_TOPK, SUBGROUP_SIZE, INTERMEDIATE_SIZE//N_BLOCK], 
                                    [1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                    t_topk_idx, t_gate_fused_weights, t_gate_fused_scale, t_gate_fused_zp4, t_up_fused_weights, t_up_fused_scale, t_up_fused_zp4, t_hidden_states,
                                    t_gate_up_output)


# cl.finish()
# o_gate_up_output = t_gate_up_output.numpy()
# print(f"o_gate_up_output={o_gate_up_output}")
# print()

moe_mlp_kernels.enqueue("mlp_down",[MAX_TOPK, SUBGROUP_SIZE, HIDDEN_SIZE//N_BLOCK],
                                [1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                t_topk_idx, t_down_fused_weights, t_down_fused_scale, t_down_fused_zp4, t_gate_up_output, t_topk_weights,
                                t_down_output)

max_work_group_size = cl.dev_info()["CL_DEVICE_MAX_WORK_GROUP_SIZE"]
print("max_work_group_size=", max_work_group_size)
moe_mlp_kernels.enqueue("mlp_reduce",[1, HIDDEN_SIZE],
                                    [1, 1024],
                                    t_down_output, t_final_hidden_state)

cl.finish()

np.set_printoptions(suppress=True)
result = t_final_hidden_state.numpy()
if not np.allclose(reference, result, rtol=1e-02, atol=1e1):
    print(f'reference = {reference[:, :32]}')
    print(f'result = {result[:, :32]}')
    print(f'{reference.shape=} {result.shape=}')
    # print(f' diff = {reference - result}')
    print(f"diff.max = {(reference - result).max()}")
    # print(f' reference = {reference}')
    print(f"rel_diff.max = {(abs(reference - result)/(abs(reference) + 1e-4)).max()}")
    exit(0)
else:
    print(f'reference = {reference[:, :32]}')
    print(f'result = {result[:, :32]}')
    print("================ PASSED ==================" , M, K, N)
print(f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} HIDDEN_SIZE={HIDDEN_SIZE}")


def dump_buffer(tensor, name, is_fp16 = 0, n_experts = N_EXPERTS):
    
    if is_fp16:
        src_buf = tensor.numpy().view(np.float16)
    else:
        src_buf = tensor.numpy()
    shape = src_buf.shape
    #print(f"{name}.orig_shape = {shape}")

    # wei/scale/zp is 1D tensor [experts * per_expert]
    if len(shape) == 1 and shape[0] % n_experts == 0:
        per_expert = shape[0] // n_experts
        src_buf = src_buf.reshape(n_experts, per_expert)
        shape = src_buf.shape
        #print(f"{name}.reshaped = {shape}")

    if is_fp16:
        print(f"float16 {name}[{shape[0]}][{shape[1]}]","={")
    else:
        print(f"uint8_t {name}[{shape[0]}][{shape[1]}]","={")
    for j in range(shape[0]):
        for i in range(shape[1]):
            print(f"{src_buf[j][i]}, ",end="")
        print("")
    print("};")


if dump_input_output:
    dump_buffer(t_hidden_states, "input", 1)
    dump_buffer(t_router_logits, "router_logits", 1)
    dump_buffer(t_gate_fused_weights, "gate_fused_weights")
    dump_buffer(t_gate_fused_scale, "gate_fused_scale", 1)
    dump_buffer(t_gate_fused_zp4, "gate_fused_zp4")
    dump_buffer(t_up_fused_weights, "up_fused_weights")
    dump_buffer(t_up_fused_scale, "up_fused_scale", 1)
    dump_buffer(t_up_fused_zp4, "up_fused_zp4")
    dump_buffer(t_down_fused_weights, "down_fused_weights")
    dump_buffer(t_down_fused_scale, "down_fused_scale",1)
    dump_buffer(t_down_fused_zp4, "down_fused_zp4")
    dump_buffer(t_final_hidden_state, "output", 1)
    exit()

print("Performance test...\n")
loop_cnt = 100
all_layers = []

actual_mem = hidden_states.nbytes + router_logits.nbytes + (moe_op.gate_fused_weight.nbytes + moe_op.gate_fused_weight_offset.nbytes) / N_EXPERTS * MAX_TOPK
actual_mem += (moe_op.up_fused_weight.nbytes + moe_op.up_fused_weight_offset.nbytes) / N_EXPERTS * MAX_TOPK
actual_mem += (moe_op.down_fused_weight.nbytes + moe_op.down_fused_weight_offset.nbytes) / N_EXPERTS * MAX_TOPK
actual_mem += MAX_TOPK * INTERMEDIATE_SIZE * 2 * M / N_EXPERTS * MAX_TOPK + MAX_TOPK * HIDDEN_SIZE * 2 * M  / N_EXPERTS * MAX_TOPK + HIDDEN_SIZE * 2 * M  / N_EXPERTS * MAX_TOPK

total_mem_size = 0
while len(all_layers) < loop_cnt and total_mem_size < 8e9:
    all_layers.append([
        cl.tensor(hidden_states),  #t_hidden_states   0
        cl.tensor(router_logits),  #t_router_logits    1
        cl.tensor(np.zeros([M, MAX_TOPK], dtype=np.int32)), #t_topk_idx  2
        cl.tensor(np.zeros([M, MAX_TOPK], dtype=np.float16)),#t_topk_weights 3
        cl.tensor(moe_op.gate_fused_weight), #t_gate_fused_weights 4
        cl.tensor(moe_op.gate_fused_scale), #t_gate_fused_scale  5
        cl.tensor(moe_op.gate_fused_zp4), #t_gate_fused_zp4  6
        cl.tensor(moe_op.up_fused_weight), #t_up_fused_weights 7
        cl.tensor(moe_op.up_fused_scale), #t_up_fused_scale 8
        cl.tensor(moe_op.up_fused_zp4), #t_up_fused_zp4  9
        cl.tensor(moe_op.down_fused_weight), #t_down_fused_weights 10
        cl.tensor(moe_op.down_fused_scale), #t_down_fused_scale 11
        cl.tensor(moe_op.down_fused_zp4), #t_down_fused_zp4  12
        cl.tensor(np.zeros([M*MAX_TOPK, INTERMEDIATE_SIZE], dtype=np.float16)), #t_gate_up_output 13
        cl.tensor(np.zeros([M*MAX_TOPK, HIDDEN_SIZE], dtype=np.float16)),#t_down_output 14
        cl.tensor(np.zeros([M, HIDDEN_SIZE], dtype=np.float16))#t_final_hidden_state 15
    ])
    total_mem_size += hidden_states.nbytes + router_logits.nbytes + moe_op.gate_fused_weight.nbytes + moe_op.gate_fused_scale.nbytes + moe_op.gate_fused_zp4.nbytes
    total_mem_size += moe_op.up_fused_weight.nbytes + moe_op.up_fused_scale.nbytes + moe_op.up_fused_zp4.nbytes
    total_mem_size += moe_op.down_fused_weight.nbytes + moe_op.down_fused_scale.nbytes + moe_op.down_fused_zp4.nbytes
    total_mem_size += MAX_TOPK * INTERMEDIATE_SIZE * 2 * M + MAX_TOPK * HIDDEN_SIZE * 2 * M + HIDDEN_SIZE * 2 * M

print(f"nlayers={len(all_layers)} total_mem_size={total_mem_size*1e-6:.3f} MB")

for i in range(loop_cnt):
    j  = i % len(all_layers)
    moe_opt_kernels.enqueue("softmax_topk",[M, N_EXPERTS],
                                        [1, N_EXPERTS],
                                        all_layers[j][1],
                                        all_layers[j][2], all_layers[j][3])
    moe_mlp_kernels.enqueue("mlp_gate_up",[MAX_TOPK, SUBGROUP_SIZE, INTERMEDIATE_SIZE//N_BLOCK],
                                    [1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                    all_layers[j][2], all_layers[j][4], all_layers[j][5], all_layers[j][6],
                                    all_layers[j][7], all_layers[j][8], all_layers[j][9], all_layers[j][0],
                                    all_layers[j][13])
    moe_mlp_kernels.enqueue("mlp_down",[MAX_TOPK, SUBGROUP_SIZE, HIDDEN_SIZE//N_BLOCK],
                                [1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                all_layers[j][2], all_layers[j][10], all_layers[j][11], all_layers[j][12], all_layers[j][13], all_layers[j][3],
                                all_layers[j][14])
    moe_mlp_kernels.enqueue("mlp_reduce",[1, HIDDEN_SIZE],
                                    [1, 1024],
                                    all_layers[j][14], all_layers[j][15])

latency = cl.finish()
i = 0
for ns in range(0, len(latency), 4):
    print(f"{i}: {latency[4*i+0]*1e-6:.3f} ms, {latency[4*i+1]*1e-6:.3f} ms, {latency[4*i+2]*1e-6:.3f} ms, {latency[4*i+3]*1e-6:.3f} ms")
    i += 1

total_latency = sum(latency)
print(f"actual_mem={actual_mem*1e-6:.3f} MB")
print(f"Total latency of {loop_cnt} runs: {total_latency*1e-6:.3f} ms, avg: {total_latency*1e-6/loop_cnt:.3f} ms")
print(f"Throughput: {actual_mem*loop_cnt*1e-6/ (total_latency*1e-6):.3f} GB/s")