#!/usr/bin/python3
from clops import cl
import numpy as np
import os
import torch
import time, sys

print(dir(cl))

cl.profiling(True)

src = '''
    ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

#define TOP_K 8
#define TYPE half

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void softmax_topk_ref(
    const __global TYPE* input, // [input_batch, sort_in_num]
    __global TYPE* output, // [input_batch, TOP_K]
    __global uint* output_index // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt;
    output += batch * TOP_K;
    output_index += batch * TOP_K;
    float softmax_total = 0.0;
    float softmax_current = 0.0;
    TYPE in_value = input[sort_index];
    uint sort_position = 0;

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_cnt; i++) {
        TYPE value = input[i];
        softmax_total += native_exp(value);
        if(i==sort_index) {
            softmax_current = native_exp(value);
            continue;
        }
        if(value > in_value || (value == in_value && i < sort_index)) {
            sort_position++;
        }
        if(sort_position >= TOP_K)
            return;
    }

    output[sort_position] = softmax_current / softmax_total;
    output_index[sort_position] = sort_index;
}


__attribute__((intel_reqd_sub_group_size(32)))
__kernel void softmax_topk_opt(
    const __global TYPE* input, // [input_batch, sort_in_num]
    __global TYPE* output, // [input_batch, TOP_K]
    __global uint* output_index // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;
    output += batch * TOP_K;
    output_index += batch * TOP_K;
    float softmax_total = 0.0;
    float softmax_current = 0.0;
    
    uint sort_position = 0;
    __local TYPE local_input[VALUE_NUM];
    local_input[sort_index] = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    TYPE in_value = local_input[sort_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = 0; i < sort_index; i++) {
        TYPE value = local_input[i];
        softmax_total += native_exp(value);
        if(value > in_value || (value == in_value && i < sort_index)) {
            sort_position++;
        }
    }
    if(sort_position >= TOP_K)
        return;
    softmax_current = native_exp(in_value);
    softmax_total += softmax_current;
    for(uint i = sort_index + 1; i < sort_cnt; i++) {
        TYPE value = local_input[i];
        softmax_total += native_exp(value);
        if(value > in_value || (value == in_value && i < sort_index)) {
            sort_position++;
        }
    }
    if(sort_position >= TOP_K)
        return;

    output[sort_position] = softmax_current / softmax_total;
    output_index[sort_position] = sort_index;
}

__attribute__((intel_reqd_sub_group_size(32)))
__kernel void softmax_topk_opt_2(
    const __global TYPE* input, // [input_batch, sort_in_num]
    __global TYPE* output, // [input_batch, TOP_K]
    __global uint* output_index // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;
    output += batch * TOP_K;
    output_index += batch * TOP_K;
    float softmax_total = 0.0;
    float softmax_current = 0.0;
    uint sort_position = 0;

    __local TYPE local_output[VALUE_NUM];
    __local uint local_index[VALUE_NUM];
    __local TYPE local_input[VALUE_NUM];
    local_input[sort_index] = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    TYPE in_value = local_input[sort_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_cnt; i++) {
        TYPE value = local_input[i];
        softmax_total += native_exp(value);
        if(value > in_value || (value == in_value && i < sort_index)) {
            sort_position++;
        }
    }

    softmax_current = native_exp(in_value);
    local_output[sort_position] = softmax_current / softmax_total;
    local_index[sort_position] = sort_index;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sort_index==0) {
        __attribute__((opencl_unroll_hint(TOP_K)))
        for(uint i = 0; i < TOP_K; i++) {
            output[i] = local_output[i];
            output_index[i] = local_index[i];
        }
    }
}

'''

SUBGROUP_SIZE = 32
VALUE_NUM = 128


kernel = cl.kernels(src, "-D FMACNT=4 -D UNROLL=4 -D SUBGROUP_SIZE=32 -D VALUE_NUM=128")

# M = 1024
# N = 128
# TOP_K = 4

def test_softmax_topk(M,N,TOP_K):

    np.random.seed(0)
    #A0 = np.random.uniform(-2.0, 2.0, [M, N]).astype(np.float16) #input
    A0 = np.random.randint(-30, 5, [M, N]).astype(np.float16) #input
    B0 = np.zeros([M, TOP_K]).astype(np.float16) #output
    C0 = np.zeros([M, TOP_K]).astype(np.float16) #output
    D0 = np.zeros([M, TOP_K]).astype(np.uint32) #output

    print("A = ", A0)

    A = cl.tensor(A0)
    B = cl.tensor(B0)
    C = cl.tensor(C0)
    D = cl.tensor(C0)
    E1 = cl.tensor(D0)
    E2 = cl.tensor(D0)
    E3 = cl.tensor(D0)

    kernel.enqueue("softmax_topk_ref", [M, N], [1, N], A, B, E1)
    kernel.enqueue("softmax_topk_opt", [M, N], [1, N], A, C, E2)
    kernel.enqueue("softmax_topk_opt_2", [M, N], [1, N], A, D, E3)
    ret_ref = B.numpy()
    ret_opt = C.numpy()
    ret_opt_2 = D.numpy()

    ret_index_ref = E1.numpy()
    ret_index_opt = E2.numpy()
    ret_index_opt_2 = E3.numpy()

    ref_softmax_result = torch.nn.functional.softmax(torch.tensor(A0), dim=1)
    top_k_values, top_k_indices = torch.topk(ref_softmax_result, k=TOP_K, dim=1)

    print("B_ref=\n",top_k_values)
    print("ret_ref=\n", ret_ref)
    print("ret_opt=\n",ret_opt)
    print("ret_opt_2=\n",ret_opt_2)

    print()

    print("ref_index=\n", top_k_indices)
    print("ret_index_ref=\n",ret_index_ref)
    print("ret_index_opt=\n",ret_index_opt)
    print("ret_index_opt_2=\n",ret_index_opt_2)

    # if not np.allclose(ret_opt, top_k_values):
    #     print("softmax_topk_ref misMatch!!!")
    #     print("ret=\n", ret_opt)
    #     print("B_ref=\n",top_k_values)
    # else:
    #     print("softmax_topk_ref sucessfully!!!")

    profiling = cl.finish()
   
    for duration in profiling:
        print(f" cost: {duration/1000} us ")


test_softmax_topk(1, 128, 8)
# test_softmax_topk(1024, 128, 8)

