from . import cl
import numpy as np
import math
from .utils import *

cl_kernel_sources = '''
__kernel void RMSNorm(__global float * input, __global float * output, __global float * weight, int size, float epsilon) {
    int row = get_global_id(0);
    input += row * size;
    output += row * size;
    float variance = 0;
    for(int i = 0; i < size; i++) {
        variance += input[i] * input[i];
    }
    float scale = rsqrt((variance / size) + epsilon);
    for(int i = 0; i < size; i++) {
        output[i] = input[i] * scale * weight[i];
    }
}
'''
cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

class RMSNorm:
    def __init__(self, weight, epsilon):
        self.weight = to_cl(weight)
        self.n_channels = weight.shape[-1]
        self.epsilon = epsilon

    def __call__(self, input):
        output = cl.tensor(input.shape, input.dtype)
        cl_kernels.enqueue("RMSNorm", [input.numel], [1], input, output, self.weight, self.n_channels, self.epsilon)
        return output
