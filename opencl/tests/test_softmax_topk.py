#!/usr/bin/python3
from clops import cl
import numpy as np
import os
import torch
import time, sys

print(dir(cl))

cl.profiling(True)

src = r'''
    ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
#define TYPE half

__kernel void softmax_topk_opt_3(
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
    //if (sort_position >= TOP_K) return;
    
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

'''


# M = 1024
# N = 128
# TOP_K = 4

def test_softmax_topk(M,N,TOP_K):
    kernel = cl.kernels(src, f"-D VALUE_NUM={N} -D TOP_K={TOP_K}")

    np.random.seed(0)
    A0 = np.random.uniform(-2.0, 2.0, [M, N]).astype(np.float16) #input
    B0 = np.zeros([M, TOP_K]).astype(np.float16) #output
    C0 = np.zeros([M, TOP_K]).astype(np.float16) #output

    #print("A = ", A0)

    A = cl.tensor(A0)
    B = cl.tensor(B0)
    C_index = cl.tensor(np.zeros([M, TOP_K]).astype(np.int32))
    C_value = cl.tensor(C0)


    #kernel.enqueue("softmax_topk_ref", [M, N], [1, N], A, B)
    for i in range(100):
        kernel.enqueue("softmax_topk_opt_3", [M, N], [1, N], A, C_index, C_value)
    ret = B.numpy()
    opt_index = C_index.numpy()
    opt_value = C_value.numpy()

    ref_softmax_result = torch.nn.functional.softmax(torch.tensor(A0), dim=1)
    top_k_values, top_k_indices = torch.topk(ref_softmax_result, k=TOP_K, dim=1)
    top_k_values = top_k_values / top_k_values.sum(dim=1, keepdim=True)
    print(top_k_values.sum(0))

    if not np.allclose(opt_index, top_k_indices):
        print("softmax_topk_ref top_k_indices misMatch!!!")
        loc = np.where(opt_index != top_k_indices.numpy())
        
        print(f"{opt_value=}")
        print(f"{top_k_values=}")
        print(loc)
        for i in range(len(loc[0])):
            y , x = loc[0][i], loc[1][i]
            print(f"{y}, {x}   {opt_value[y, x]}@{opt_index[y,x]} vs {top_k_values[y, x]}@{top_k_indices[y,x]}   {A0[y, opt_index[y,x]]}  vs {A0[y, top_k_indices[y,x]]}")
        #print("opt_index=\n", opt_index)
        #print("top_k_indices=\n",top_k_indices.numpy())
    else:
        print("softmax_topk_ref top_k_indices sucessfully!!!")

    if not np.allclose(opt_value, top_k_values):
        print("softmax_topk_ref top_k_values misMatch!!!")
        print("opt_value=\n", opt_value)
        print("top_k_values=\n",top_k_values.numpy())
        print("opt_index=\n", opt_index)
        print("top_k_indices=\n",top_k_indices.numpy())        
        vals = A0[0, opt_index]
        vals = np.exp(vals)
        vals = vals / vals.sum()
        print("A0[0, top_k_indices]=\n", vals)
    else:
        print("softmax_topk_ref top_k_values  sucessfully!!!")

    profiling = cl.finish()
   
    for duration in profiling:
        print(f" cost: {duration/1000} us ")


test_softmax_topk(1, 128, 8)
#test_softmax_topk(1024, 128, 8)
