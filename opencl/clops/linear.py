cl_kernel_sources = '''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

// global_id [N, M]
//
//__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_f16(__global half * A, __global half * B, __global half * C, int M, int K, int N) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    float sum = 0;
    __global half * pA = A + m*K;
    __global half * pB = B + n*K;
    for(int k = 0; k < K; k += 8) {
        for(int unroll = 0; unroll < 8; unroll++)
            sum += pA[k+unroll] * pB[k+unroll];
    }
    C[m*N + n] = sum;
}
'''


from . import cl
import numpy as np
from .utils import *

cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

# collect linear weight shapes
Linear_shapes = set()
import atexit
def show_linear_shapes():
    if len(Linear_shapes):
        print("Linear_shapes:", Linear_shapes)
atexit.register(show_linear_shapes)

class Linear:
    # weight: [N, K]
    def __init__(self, weight, bias):
        assert(bias is None)
        self.N, self.K = weight.shape

        # self.weight = to_cl(weight.half().transpose(1, 0).contiguous()) # internally transpose to [K, N]
        self.weight = to_cl(weight.half())
        self.bias = to_cl(bias)

        # Linear_shapes.add(weight.shape)

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        cl_kernels.enqueue("Linear_f16", [self.N, M], [128, 1], input, self.weight, output, M, self.K, self.N)
        
        return output

cl_kernel_sources = '''
__kernel void Linear_quant_I4(__global half * src, __global uchar * dst, __global half * scales, __global char * zps, int N, int K) {
    int kg = get_global_id(0);
    int n = get_global_id(1);
    int k = kg * GROUP_SIZE;

    src += n * K + k;
    dst += n * K/2 + k/2;
    scales += n * (K/GROUP_SIZE) + kg;
    zps += n * (K/GROUP_SIZE) + kg;

    half vmin = src[0];
    half vmax = src[0];
    for(int i = 1; i < GROUP_SIZE; i++) {
        vmin = min(src[i], vmin);
        vmax = max(src[i], vmax);
    }

    //
    //  vmin = (0 + zp)*s
    //  vmax = (15 + zp)*s
    //
    // zero-point is choosen as interger
    char zp = round(15.0f * vmin/(vmax - vmin));
    half s = (vmax - vmin)/15.0f;
    scales[0] = s;
    zps[0] = zp;

    half rs = 1.0f/s;
    for(int i = 0; i < GROUP_SIZE; i+=2, dst++) {
        uchar v0 = clamp((uchar)round(src[i] * rs - zp), (uchar)0, (uchar)15);
        uchar v1 = clamp((uchar)round(src[i+1] * rs - zp), (uchar)0, (uchar)15);
        *dst = (v1 << 4) | v0;
    }
}


__kernel void Linear_woq_I4(__global half * A, __global half * C, __global uchar * B, __global half * scales, __global char * zps, int M, int N, int K) {
    int m = get_global_id(1);
    int n = get_global_id(0);

    A += m*K;
    B += n*K/2;
    scales += n*K/GROUP_SIZE;
    zps += n*K/GROUP_SIZE;

    float sum = 0;
    for(int k = 0; k < K; k += GROUP_SIZE, scales++, zps++, A += GROUP_SIZE) {
        // process next K group
        half scale = *scales;
        char zp = *zps;
        for(int g = 0; g < GROUP_SIZE; g+= 2, B++) {
            half b0 = ((char)(B[0] & 0xF) + zp) * scale;
            half b1 = ((char)(B[0] >> 4) + zp) * scale;
            sum += A[g] * b0;
            sum += A[g + 1] * b1;
        }
    }
    C[m*N + n] = sum;
}

'''

GROUP_SIZE = 128
cl_kernels_WOQ_I4 = cl.kernels(cl_kernel_sources, f"-D GROUP_SIZE={GROUP_SIZE}", "./dump")

# weight-only-quantization to INT4
class Linear_woq_I4:
    # weight: [N, K], 
    def __init__(self, weight, bias):
        assert(bias is None)

        # quantize weight into groupped INT4 format:(sym)
        N, K = weight.shape
        assert((K % GROUP_SIZE) == 0)

        weight_half = to_cl(weight.half())  # copy to GPU
        self.weight_i4 = cl.tensor([N, K//2], np.dtype(np.uint8))         # two int4 packed into a uint8
        self.scales = cl.tensor([N, K//GROUP_SIZE], np.dtype(np.float16)) # scales
        self.zps = cl.tensor([N, K//GROUP_SIZE], np.dtype(np.int8))    # zero-points
        cl_kernels_WOQ_I4.enqueue("Linear_quant_I4", [K//GROUP_SIZE, N], [1, 1], weight_half, self.weight_i4, self.scales, self.zps, N, K)

        self.bias = to_cl(bias)
        self.N = N
        self.K = K

        assert((self.N % 128) == 0)


    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)
        cl_kernels_WOQ_I4.enqueue("Linear_woq_I4", [self.N, M], [8, 1], input, output, self.weight_i4, self.scales, self.zps, M, self.N, self.K)
        return output

if __name__ == "__main__":

    src = '''
__kernel void Linear2(__global half * A, __global half * B, __global half * C, int M, int K, int N) {
    int m = get_global_id(1);
    int n = get_global_id(0);

    float sum = 0;
    __global half * pA = A + m*K;
    __global half * pB = B + n*K;
    for(int k = 0; k < K; k += 8) {
        for(int unroll = 0; unroll < 8; unroll++)
            sum += pA[k+unroll] * pB[k+unroll];
    }
    C[m*N + n] = sum;
}
    '''
    kernels = cl.kernels(src, "-D FMACNT=4 -D UNROLL=4", "./dump")

    import torch
    M = 1
    K = 2048
    N = 16*8*32*8
    N = 4096*8
    print(f"============= N = {N}")
    cl.profiling(True)
    A = torch.rand(M, K).half()
    B = torch.rand(N, K).half()

    ABCsize = K*N*2 + M*N*2 + M*K*2
    Bcnt = int(500e6)//(ABCsize)
    weights = [to_cl(B) for _ in range(Bcnt)]

    cl.finish()
    output = to_cl(A)
    for i in range(Bcnt):
        input = output
        output = cl.tensor([M, N], np.dtype(np.float16))
        kernels.enqueue("Linear2", [N, M], [8, 1], input, output, weights[i], M, N, K)
    durs = cl.finish()
    for ns in durs:
        print(f" {ABCsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { ABCsize/ns : .2f} GB/s")
    mean_ns = sum(durs)/len(durs)
    print(f"  {mean_ns*1e-6: .3f} ms   BW: { K*N*2/mean_ns : .2f} GB/s")