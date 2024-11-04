from . import cl
import numpy as np
from .utils import *

cl_kernel_sources = r'''
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

//# [batch, step, ele_size] => [batch, 1, ele_size]
//# global/local [ele_size]/[16*8?]
//#
__kernel void Slice(__global half * input, __global half * output,
                    int batch_size,
                    int axis_size,
                    int ele_size,
                    int start,
                    int step,
                    int stop) {
    int off = get_global_id(0);
   
    int i_step = step * ele_size;
    for(int b=0; b < batch_size; b++) {
        __global half * src = input + start * ele_size;
        for(int i = start; i < stop; i += step) {
            output[off] = src[off];
            src += i_step;
            output += ele_size;
        }
        input += axis_size * ele_size;
    }
}

'''
cl_kernels = kernel_cache(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4")

def iAdd(input, rhs):
    cl_kernels.enqueue("iAdd", [input.numel], [8*8], input, rhs, input.numel)
    return input

def iSilu(input):
    cl_kernels.enqueue("iSilu", [input.numel], [8*8], input, input.numel)
    return input

def iMul(input, b):
    cl_kernels.enqueue("iMul", [input.numel], [8*8], input, b, input.numel)
    return input

def Slice(input, axis, start, step, stop=-1):
    oshape = list(input.shape)
    axis_size = oshape[axis]
    if axis_size == 1:
        return input
    if stop < 0 or stop > axis_size:
        stop = axis_size
    oshape[axis] = 1
    output = cl.tensor(oshape, input.dtype)
    elesize = 1
    for dim in oshape[axis:]:
        elesize *= dim
    batch = output.numel//elesize
    cl_kernels.enqueue("Slice", [elesize], [8],
                        input, output,
                        batch,
                        axis_size,
                        elesize,
                        start,
                        step,
                        stop)
    return output

# cl.tensor supports +=,*=
setattr(cl.tensor, '__iadd__', iAdd)
setattr(cl.tensor, '__imul__', iMul)
setattr(cl.tensor, 'iSilu', iSilu)
setattr(cl.tensor, 'torch', to_torch)
setattr(cl.tensor, 'slice', Slice)


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

