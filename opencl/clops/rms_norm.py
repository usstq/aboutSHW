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
        variance += input[i] * input[i];
    }
    half scale = rsqrt((variance / size) + epsilon);
    for(int i = 0; i < size; i++) {
        output[i] = input[i] * scale * weight[i];
    }
}

/***********************************************************************************
SIMD optimized:
    input: [batch, channels]
 kernel-parallel in two dimensions [batch + channel-block], channels are divided into 16 blocks
 16 work-items get partial variance, sync, then reduce to full variances and do normalize on each block
***********************************************************************************/
//__attribute__((intel_reqd_sub_group_size(16)))
__kernel void RMSNorm(__global half * input, __global half * output, __global half * weight, int size, float epsilon) {
    int row = get_group_id(0);
    int id = get_local_id(0);
    int cur_size = size / 16;
    input += row * size + id * cur_size;
    output += row * size + id * cur_size;
    __local float variance[16];
    variance[id] = 0;
    for(int i = 0; i < cur_size; i++) {
        variance[id] += input[i] * input[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float all_variance = 0;
    for(int i = 0; i < 16; i++) {
        all_variance += variance[i];
    }
    float scale = rsqrt((all_variance / size) + epsilon);

    weight += cur_size * id;
    for(int i = 0; i < cur_size; i++) {
        output[i] = input[i] * scale * weight[i];
    }
}
'''
cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4")

class RMSNorm:
    def __init__(self, weight, epsilon):
        self.weight = to_cl(weight.half())
        self.n_channels = weight.shape[-1]
        self.epsilon = epsilon

    def __call__(self, input):
        output = cl.tensor(input.shape, input.dtype)
        cl_kernels.enqueue("RMSNorm", [input.numel // self.n_channels * 16], [16], input, output, self.weight, self.n_channels, self.epsilon)
        return output
