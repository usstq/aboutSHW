cl_kernel_sources = r'''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

/****************************************************************************************************************************
naive implementation:
problem in 2nd token: not enough work-global size to use all EUs when M=1
****************************************************************************************************************************/
//__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_f16(__global half * A, __global half * B, __global float * bias, __global half * C, int M, int K, int N) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    float sum = 0;
    __global half * pA = A + m*K;
    __global half * pB = B + n*K;
    for(int k = 0; k < K; k += 8) {
        for(int unroll = 0; unroll < 8; unroll++)
            sum += pA[k+unroll] * pB[k+unroll];
    }
    if (bias != NULL) sum += bias[n];
    C[m*N + n] = sum;
}
'''

from . import cl
import numpy as np
from .utils import *

# collect linear weight shapes
Linear_shapes = {}
import atexit
def add_shape(M, K, N):
    shape = (M, K, N)
    if shape in Linear_shapes:
        Linear_shapes[shape] += 1
    else:
        Linear_shapes[shape] = 1

def show_linear_shapes():
    if len(Linear_shapes):
        print("Linear_shapes:", Linear_shapes)
atexit.register(show_linear_shapes)


class Linear_f16:
    def __init__(self, weight, bias):
        self.N, self.K = weight.shape # weight: [N, K]
        self.weight = to_cl(weight.half())
        self.bias = to_cl(bias)
        self.cl_kernels = kernel_cache(cl_kernel_sources, options="")

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        add_shape(M, self.K, self.N)
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)
        self.cl_kernels.enqueue("Linear_f16", [self.N, M], [128, 1], input, self.weight, self.bias, output, M, self.K, self.N)
        return output