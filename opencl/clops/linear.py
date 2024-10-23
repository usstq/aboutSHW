import cl
import numpy as np
import math
from .utils import *

cl_kernel_sources = '''
ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

// global_id [N, M]
//
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_f32(__global float * A, __global float * B, __global float * C, int M, int K, int N) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    if (m < M && n < N) {
        float sum = 0;
        __global float * pA = A + m*K;
        __global float * pB = B + n*K;
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
        self.weight = to_cl(weight)
        self.bias = to_cl(bias)

        self.N = self.weight.shape[0]
        self.K = self.weight.shape[1]
        assert(self.bias is None)

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, np.dtype(np.float32))

        M = math.prod(i_shape) // self.K
        
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        cl_kernels.enqueue("Linear_f32", [self.N, M], [32, 32], input, self.weight, output, M, self.K, self.N)
        
        return output