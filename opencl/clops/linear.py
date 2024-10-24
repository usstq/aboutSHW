cl_kernel_sources = '''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

// global_id [N, M]
//
//__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_f16(__global half * A, __global half * B, __global half * C, int M, int K, int N) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    if (m < M && n < N) {
        float sum = 0;
        __global half * pA = A + m*K;
        __global half * pB = B + n*K;
        for(int k = 0; k < K; k += 8) {
            for(int unroll = 0; unroll < 8; unroll++)
                sum += pA[k+unroll] * pB[k+unroll];
        }
        C[m*N + n] = sum;
    }
}

__kernel void Linear_quant_I4(__global half * src, __global uchar * dst, __global half * scales, __global half * zps, int N, int K, int group_size) {
    int kg = get_global_id(0);
    int n = get_global_id(1);
    int k = kg * group_size;

    src += n * K + k;
    dst += n * K/2 + k/2;
    scales += n * (K/group_size) + kg;
    zps += n * (K/group_size) + kg;

    half vmin = src[0];
    half vmax = src[0];
    for(int i = 1; i < group_size; i++) {
        vmin = min(src[i], vmin);
        vmax = max(src[i], vmax);
    }

    //
    //  vmin = (0 + zp)*s
    //  vmax = (15 + zp)*s
    //
    half zp = 15.0f * vmin/(vmax - vmin);
    half s = (vmax - vmin)/15.0f;
    scales[0] = s;
    zps[0] = zp;

    half rs = 1.0f/s;
    for(int i = 0; i < group_size; i+=2, dst++) {
        uchar v0 = clamp((uchar)round(src[i] * rs - zp), (uchar)0, (uchar)15);
        uchar v1 = clamp((uchar)round(src[i+1] * rs - zp), (uchar)0, (uchar)15);
        //uchar v0 = round(src[i] * rs - zp);
        //uchar v1 = round(src[i + 1] * rs - zp);
        *dst = (v1 << 4) | v0;
    }
}


__kernel void Linear_woq_I4(__global half * A, __global half * C, __global uchar * B, __global half * scales, __global half * zps, int M, int N, int K, int group_size) {
    int m = get_global_id(1);
    int n = get_global_id(0);

    A += m*K;
    B += n*K/2;
    scales += n*K/group_size;
    zps += n*K/group_size;

    float sum = 0;
    for(int k = 0; k < K; k += group_size, scales++, zps++, A += group_size) {
        // process next K group
        half scale = *scales;
        half zp = *zps;
        for(int g = 0; g < group_size; g+= 2, B++) {
            half b0 = ((B[0] & 0xF) + zp) * scale;
            half b1 = ((B[0] >> 4) + zp) * scale;
            sum += A[g] * b0;
            sum += A[g + 1] * b1;
        }
    }
    C[m*N + n] = sum;
}

'''
from . import cl
import numpy as np
from .utils import *

cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

class Linear:
    # weight: [N, K]
    def __init__(self, weight, bias):
        assert(bias is None)
        self.N, self.K = weight.shape

        # self.weight = to_cl(weight.half().transpose(1, 0).contiguous()) # internally transpose to [K, N]
        self.weight = to_cl(weight.half())
        self.bias = to_cl(bias)

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


# weight-only-quantization to INT4
class Linear_woq_I4:

    # weight: [N, K], 
    def __init__(self, weight, bias, group_size = 128):
        assert(bias is None)

        # quantize weight into groupped INT4 format:(sym)
        N, K = weight.shape
        assert((K % group_size) == 0)

        weight_half = to_cl(weight.half())  # copy to GPU
        self.weight_i4 = cl.tensor([N, K//2], np.dtype(np.uint8))         # two int4 packed into a uint8
        self.scales = cl.tensor([N, K//group_size], np.dtype(np.float16)) # scales
        self.zps = cl.tensor([N, K//group_size], np.dtype(np.float16))    # zero-points
        cl_kernels.enqueue("Linear_quant_I4", [K//group_size, N], [1, 1], weight_half, self.weight_i4, self.scales, self.zps, N, K, group_size)

        self.bias = to_cl(bias)
        self.group_size = group_size
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
        cl_kernels.enqueue("Linear_woq_I4", [self.N, M], [128, 1], input, output, self.weight_i4, self.scales, self.zps, M, self.N, self.K, self.group_size)
        return output
