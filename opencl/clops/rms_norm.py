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
 kernel-parallel in two dimensions [batch + channel-block], channels are divided into C_WG_SIZE blocks
 C_WG_SIZE work-items get partial variance, sync, then reduce to full variances and do normalize on each block
***********************************************************************************/
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void RMSNorm(__global half * input, __global half * output, __global half * weight, int size, float epsilon) {
    int row = get_group_id(0);
    int id = get_local_id(0);
    int cur_size = size / C_WG_SIZE;
    input += row * size + id * cur_size;
    output += row * size + id * cur_size;
    __local float variance[C_WG_SIZE];
    float local_var = 0;

    __attribute__((opencl_unroll_hint(8)))
    for(int i = 0; i < cur_size; i++) {
        // half/fp16 has very limited range: ±65,504
        // which may cause square result to overflow,
        // the square must be done in fp32
        float x = (float)input[i];
        local_var += x * x;
    }

    variance[id] = local_var;
    barrier(CLK_LOCAL_MEM_FENCE);
    float all_variance = 0;

    for(int i = 0; i < C_WG_SIZE; i++) {
        all_variance += variance[i];
    }
    float scale = rsqrt((all_variance / size) + epsilon);

    weight += cur_size * id;
    __attribute__((opencl_unroll_hint(8)))
    for(int i = 0; i < cur_size; i++) {
        output[i] = input[i] * scale * weight[i];
    }
}
'''
# choose work-group size to be 128 (16EU x SIMD-8)
C_WG_SIZE = 128
cl_kernels = kernel_cache(cl_kernel_sources, f"-D C_WG_SIZE={C_WG_SIZE}")

class RMSNorm:
    def __init__(self, weight, epsilon):
        self.weight = to_cl(weight.half())
        self.n_channels = weight.shape[-1]
        self.epsilon = epsilon
        assert(self.n_channels % C_WG_SIZE == 0)

    def __call__(self, input):
        # return self.reference_impl(input)
        output = cl.tensor(input.shape, input.dtype)
        #cl_kernels.enqueue("RMSNorm_v0", [input.numel // self.n_channels], [1], input, output, self.weight, self.n_channels, self.epsilon)
        cl_kernels.enqueue("RMSNorm", [input.numel // self.n_channels * C_WG_SIZE], [C_WG_SIZE], input, output, self.weight, self.n_channels, self.epsilon)
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