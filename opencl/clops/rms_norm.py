from . import cl
import numpy as np
import math
from .utils import *

cl_kernel_sources = '''
/***********************************************************************************
naive implementation:
    input: [batch, channels]
kernel-parallel in batch only, in 2nd-tok case, only 1 SMID lane of 1 EU is running
***********************************************************************************/
__kernel void RMSNorm_v0(__global half * input, __global half * output, __global half * weight, int size, float epsilon) {
    int row = get_global_id(0);
    input += row * size;
    output += row * size;
    float variance = 0;
    for(int i = 0; i < size; i++) {
        float x = (float)input[i];
        variance += x * x;
    }
    half scale = rsqrt((variance / size) + epsilon);
    for(int i = 0; i < size; i++) {
        output[i] = (input[i] * scale) * weight[i];
    }
}

/***********************************************************************************
SIMD optimized:
    input: [batch, channels]
 kernel-parallel in two dimensions [batch + channel-block], channels are divided into 16 blocks
 get partial variance, sync, then reduce to full variances and do normalize on each block
***********************************************************************************/
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void RMSNorm(__global half * input, __global half * output, __global half * weight, float epsilon) {
    int row = get_group_id(0);
    int id_local = get_local_id(0);
    int id_sg = get_sub_group_id();
    int id_sg_local = get_sub_group_local_id();
    input += row * SIZE;
    output += row * SIZE;
    __local float variance[16];
    float local_var = 0;

    __attribute__((opencl_unroll_hint(16)))
    for (int i = id_local; i < SIZE; i += C_WG_SIZE) {
        // half/fp16 has very limited range: ±65,504
        // which may cause square result to overflow,
        // the square must be done in fp32
        float x = (float)input[i];
        local_var = fma(x, x, local_var);
    }

    local_var = sub_group_reduce_add(local_var);
    if (id_sg_local == 0)
        variance[id_sg] = local_var;
    barrier(CLK_LOCAL_MEM_FENCE);
    local_var = variance[id_sg_local];
    float all_variance = sub_group_reduce_add(local_var);
    float scale = rsqrt((all_variance / SIZE) + epsilon);

    __attribute__((opencl_unroll_hint(16)))
    for (int i = id_local; i < SIZE; i += C_WG_SIZE) {
        output[i] = input[i] * scale * weight[i];
    }
}
'''
# choose work-group size to be 256 (16EU x SIMD-16)
C_WG_SIZE = 256

class RMSNorm:
    def __init__(self, weight, epsilon):
        self.weight = to_cl(weight.half())
        self.n_channels = weight.shape[-1]
        self.epsilon = epsilon
        assert(self.n_channels >= C_WG_SIZE)
        self.cl_kernels = kernel_cache(cl_kernel_sources, f"-D C_WG_SIZE={C_WG_SIZE} -DSIZE={self.n_channels}")

    def __call__(self, input):
        # return self.reference_impl(input)
        output = cl.tensor(input.shape, input.dtype)
        #cl_kernels.enqueue("RMSNorm_v0", [input.numel // self.n_channels], [1], input, output, self.weight, self.n_channels, self.epsilon)
        self.cl_kernels.enqueue("RMSNorm", [input.numel // self.n_channels * C_WG_SIZE], [C_WG_SIZE], input, output, self.weight, self.epsilon)
        return output

    # this is torch-cpu reference for debugging purpose
    def reference_impl(self, input):
        import torch
        input = to_torch(input)
        input_dtype = input.dtype
        # to avoid float16 overflow
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.epsilon)
        return to_cl(to_torch(self.weight) * input.to(input_dtype))