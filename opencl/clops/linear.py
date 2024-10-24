from . import cl
import numpy as np
import math
from .utils import *

cl_kernel_sources = '''
ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

// global_id [N, M]
//
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_f16(__global half * A, __global half * B, __global half * C, int M, int K, int N) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    if (m < M && n < N) {
        float sum = 0;
        __global half * pA = A + m*K;
        __global half * pB = B + n*K;
        for(int k = 0; k < K; k++) {
            sum += pA[k] * pB[k];
        }
        C[m*N + n] = sum;
    }
}

'''
cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

class Linear:
    # weight: [N, K]
    def __init__(self, weight, bias):
        assert(bias is None)
        self.weight = to_cl(weight.half())
        self.bias = to_cl(bias)

        self.N = self.weight.shape[0]
        self.K = self.weight.shape[1]

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = math.prod(i_shape) // self.K
        
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        cl_kernels.enqueue("Linear_f16", [self.N, M], [32, 32], input, self.weight, output, M, self.K, self.N)
        
        return output