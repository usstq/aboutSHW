from .cl import csrc as cl
import numpy as np
from .utils import *

cl_kernel_sources = r'''
ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

__kernel void iAdd(__global half * input, __global half * rhs, int size) {
    int i = get_global_linear_id();
    if (i < size)
        input[i] += rhs[i];
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iAdd_opt(__global half * input, __global half * rhs, int size) {
    int i = get_group_id(0) * get_local_size(0) + get_sub_group_id() * 16;
    input += i;
    half x0 = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    half x1 = as_half(intel_sub_group_block_read_us((const __global ushort*)(rhs + i)));
    intel_sub_group_block_write_us((__global ushort*)(input), as_ushort(x0 + x1));
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


__kernel void Embedding(__global uint * input_ids, 
                        __global half * weight,
                        __global half * output,
                        int num_tokens,
                        int hidden_states) {
    int s = get_global_id(0);   // which states
    int i = get_global_id(1);   // which token
    
    uint id = input_ids[i];
    output[i*hidden_states + s] = weight[id*hidden_states + s];
}

'''
cl_kernels = kernel_cache(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4")

def iAdd(input, rhs):
    cl_kernels.enqueue("iAdd_opt", [input.numel], [16*8], input, rhs, input.numel)
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

class Embedding:
    def __init__(self, weight):
        self.weight = to_cl(weight.half())
        self.vocab_size, self.hidden_states = self.weight.shape

    def __call__(self, input):
        # [batch, length] int64
        # [batch, length, hidden_states] half
        B, L = input.shape
        o_shape = [B, L, self.hidden_states]
        output = cl.tensor(o_shape, np.dtype(np.float16))
        cl_kernels.enqueue("Embedding", [self.hidden_states, L*B], [128, 1], input, self.weight, output, B*L, self.hidden_states)
        return output
