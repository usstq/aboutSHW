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

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void iAdd_opt_asm(__global half * input, __global half * rhs, int size) {
    // NOTE: must use 32 items/subgroup to compute the offset
    int i = (get_group_id(0) * get_local_size(0) + get_sub_group_id() * 16) * 2;
    input += i;
    float x0 = as_float(intel_sub_group_block_read((const __global uint*)(input)));
    float x1 = as_float(intel_sub_group_block_read((const __global uint*)(rhs + i)));
    float result;
    // NOTE: must use MX_NM to override the execution mask
    // change subgroup size from 16 to 32
    __asm__(
        "{\n"
        ".decl add_result v_type=G type=hf num_elts=32 align=GRF alias=<%0,0>\n"
        ".decl add_x0 v_type=G type=hf num_elts=32 align=GRF alias=<%1,0>\n"
        ".decl add_x1 v_type=G type=hf num_elts=32 align=GRF alias=<%2,0>\n"
        "add (M1_NM, 32) add_result(0,0)<1> add_x0(0,0)<1;1,0> add_x1(0,0)<1;1,0> // add\n"
        "/*add (M1, 32) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0> */\n"
        "}\n"
        : "=rw"(result)
        : "rw"(x0), "rw"(x1)
    );
    intel_sub_group_block_write((__global uint*)(input), as_uint(result));
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

if __name__ == "__main__":
    cl.profiling(True)
    
    size=16*8

    x0 = torch.randn([size,], dtype=torch.float16)
    #print(f'{x0=}')
    x1 = torch.randn([size,], dtype=torch.float16)
    cl0 = to_cl(x0)
    cl1 = to_cl(x1)
    cl_kernels.enqueue("iAdd_opt", [cl0.numel], [16*8], cl0, cl1, cl0.numel)
    cl0_asm = to_cl(x0)
    #cl_kernels.enqueue("iAdd_opt", [cl0_asm.numel], [16*8], cl0_asm, cl1, cl0_asm.numel)
    # NOTE: due to the subgroup size changed from 32 to 16, to keep the same xve count, the local size must be divided by 2.
    cl_kernels.enqueue("iAdd_opt_asm", [cl0_asm.numel], [16*8//2], cl0_asm, cl1, cl0_asm.numel)

    durs = cl.finish()
    cl0_np = cl0.numpy()
    cl0_asm_np = cl0_asm.numpy()
    if not np.allclose(cl0_np, cl0_asm_np):
        print(f'{cl0_np=}\n{cl0_asm_np=}')
    print('done')
