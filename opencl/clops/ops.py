from . import cl
import numpy as np
from .utils import *

cl_kernel_sources = '''
ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

// global_id : [Hq + Hk, q_len, batch_size]
//
// inv_freq : [half_rotary_dim]
//
//    input : [batch_size, q_len, (Hq + Hk + Hv)*S)]
__kernel void ROPE(__global half * input,
                    __global float * inv_freq,
                    int half_rotary_dim,
                    int batch_size,
                    int q_len,
                    int head_cnt_qkv,
                    int head_size,
                    int pos0) {
    int h = get_global_id(0);
    int q_offset = get_global_id(1);
    int batch = get_global_id(2);
    float position_idx = pos0 + q_offset;

    __global half * src = input + ((batch * q_len + q_offset) * head_cnt_qkv +  h) * head_size;

    for(int i0=0; i0 < half_rotary_dim; i0++) {
        int i1 = i0 + half_rotary_dim;
        float xita = position_idx * inv_freq[i0];
        float cx = cos(xita);
        float sx = sin(xita);
        float x0 = src[i0];
        float x1 = src[i1];
        src[i0] = cx * x0 - sx * x1;
        src[i1] = sx * x0 + cx * x1;
    }
}

__kernel void iAdd(__global half * input, __global half * rhs, int size) {
    int i = get_global_linear_id();
    if (i < size)
        input[i] += rhs[i];
}

__kernel void iMul(__global half * input, __global half * rhs, int size) {
    int i = get_global_linear_id();
    if (i < size)
        input[i] *= rhs[i];
}

__kernel void iSilu(__global half * input, int size) {
    int i = get_global_linear_id();
    if (i < size) {
        float x = input[i];
        input[i] = x / (1.0f + exp(-x));
    }
}
'''
cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

def iAdd(input, rhs):
    cl_kernels.enqueue("iAdd", [input.numel], [1], input, rhs, input.numel)
    return input

def iSilu(input):
    cl_kernels.enqueue("iSilu", [input.numel], [1], input, input.numel)
    return input

def iMul(input, b):
    cl_kernels.enqueue("iMul", [input.numel], [1], input, b, input.numel)
    return input

# cl.tensor supports +=,*=
setattr(cl.tensor, '__iadd__', iAdd)
setattr(cl.tensor, '__imul__', iMul)
setattr(cl.tensor, 'iSilu', iSilu)
setattr(cl.tensor, 'torch', to_torch)

# GPU needs weight-compression to work : INT8 at least

class ROPE:
    def __init__(self, inv_freq, rotary_dim, head_cnt_q, head_cnt_k, head_size):
        self.inv_freq = to_cl(inv_freq)
        self.half_rotary_dim = rotary_dim//2
        self.head_cnt_q = head_cnt_q
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2*head_cnt_k
        self.head_size = head_size
    
    def __call__(self, qkv, position_id_base):
        '''
         no shape change, inplace on VRAM
         input : [batch_size, q_len, (Hq + Hk + Hv) * head_size)]
        '''
        batch_size, q_len, S = qkv.shape

        assert(S == (self.head_cnt_qkv * self.head_size))

        cl_kernels.enqueue("ROPE",
                            [self.head_cnt_q + self.head_cnt_k, q_len, batch_size],
                            [1, 1],
                            qkv,
                            self.inv_freq,
                            self.half_rotary_dim,
                            batch_size,
                            q_len,
                            self.head_cnt_qkv,
                            self.head_size,
                            position_id_base)
        return qkv

