cl_kernel_sources = '''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );

/****************************************************************************************************************************
naive implementation:
problem in 2nd token: not enough work-global size to use all EUs when M=1
****************************************************************************************************************************/
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

/****************************************************************************************************************************
opt1: increase occupancy
   split along K dimension to generate more work-item to all EU's memory access bandwidth
****************************************************************************************************************************/
__kernel void Linear_f16_opt1(__global half * A, __global half * B, __global half * C, int M, int K, int N, int K_group_size) {
    int kg = get_local_id(2);
    int ng = get_local_id(0);
    int m = get_global_id(1);
    int n = get_global_id(0);

    int k0 = kg * K_group_size;
    int k1 = k0 + K_group_size;

    float sum = 0;
    __global half * pA = A + m*K;
    __global half * pB = B + n*K;
    for(int k = k0; k < k1; k += 8) {
        for(int unroll = 0; unroll < 8; unroll++)
            sum += pA[k+unroll] * pB[k+unroll];
    }

    __local float all_sum[8][16];
    all_sum[ng][kg] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // only the first one responsible for reduce & output
    if (kg != 0) return;

    sum = 0;
    for(int i = 0; i < 16; i++) sum += all_sum[ng][i];
    C[m*N + n] = sum;
}


/*********************************************************************************
 in : [N, K]
 out: [N/16, K, 16]
*********************************************************************************/
__kernel void ReorderB(__global half * in, __global half * out, int N, int K) {
    int n = get_global_id(1);
    int k = get_global_id(0);

    out[((n/16)*K*16) + (k*16) + (n % 16)] = in[n*K + k];
}

/****************************************************************************************************************************
opt2: blocked read & memory-layout
****************************************************************************************************************************/
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void Linear_f16_opt2(__global half * A, __global half * B, __global half * C, int M, int K, int N, int K_group_size) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    int ngid = get_group_id(0);
    int ng = get_local_id(0);
    int kg = get_local_id(2);

    int k0 = kg * K_group_size;
    int k1 = k0 + K_group_size;

    float sum = 0;
    __global half * pA = A + m*K;
    __global half * pB = B + ngid * (K*16) + k0*16;
    for(int k = k0; k < k1; k += 8) {
        ushort vAs = intel_sub_group_block_read_us((const __global ushort*)(pA + k));
        half8 vBs = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(pB))); pB += 8*16;
        sum += as_half(intel_sub_group_broadcast(vAs, 0)) * vBs.s0;
        sum += as_half(intel_sub_group_broadcast(vAs, 1)) * vBs.s1;
        sum += as_half(intel_sub_group_broadcast(vAs, 2)) * vBs.s2;
        sum += as_half(intel_sub_group_broadcast(vAs, 3)) * vBs.s3;
        sum += as_half(intel_sub_group_broadcast(vAs, 4)) * vBs.s4;
        sum += as_half(intel_sub_group_broadcast(vAs, 5)) * vBs.s5;
        sum += as_half(intel_sub_group_broadcast(vAs, 6)) * vBs.s6;
        sum += as_half(intel_sub_group_broadcast(vAs, 7)) * vBs.s7;
    }

    __local float all_sum[16][K_groups];
    all_sum[ng][kg] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // only the first one responsible for reduce & output
    if (kg != 0) return;

    sum = 0;
    for(int i = 0; i < K_groups; i++) sum += all_sum[ng][i];
    C[m*N + n] = sum;
}
'''


from . import cl
import numpy as np
from .utils import *

cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D K_groups=16", "./dump")

# collect linear weight shapes
Linear_shapes = {}
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
        weight_raw = to_cl(weight.half())
        self.weight = cl.tensor(weight_raw.shape, weight_raw.dtype)
        self.bias = to_cl(bias)
        
        cl_kernels.enqueue("ReorderB", [self.K, self.N], [1, 1], weight_raw, self.weight, self.N, self.K)

        assert(self.K % 16 == 0)
        if weight.shape in Linear_shapes:
            Linear_shapes[weight.shape] += 1
        else:
            Linear_shapes[weight.shape] = 1

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        # cl_kernels.enqueue("Linear_f16", [self.N, M], [128, 1], input, self.weight, output, M, self.K, self.N)

        # cl_kernels.enqueue("Linear_f16_opt1", [self.N, M, 16], [8, 1, 16], input, self.weight, output, M, self.K, self.N, self.K // 16)
        cl_kernels.enqueue("Linear_f16_opt2", [self.N, M, 16], [16, 1, 16], input, self.weight, output, M, self.K, self.N, self.K // 16)

        return output

cl_kernel_sources = '''
int NK8n_INT_4bit_index(int k, int n, int K, int N) {
    return (n/8)*(8*K/2) + (n % 8) + k*8/2;
}

__kernel void Linear_quant_I4(__global half * src, __global uchar * dst, __global half * scales, __global char * zps, int N, int K) {
    int kg = get_global_id(0);
    int n = get_global_id(1);
    int k = kg * GROUP_SIZE;

    src += n * K + k;
    dst += NK8n_INT_4bit_index(k, n, K, N);
    scales += n * (K/GROUP_SIZE) + kg;
    zps += n * (K/GROUP_SIZE) + kg;

    half vmin = src[0];
    half vmax = src[0];
    for(int i = 1; i < GROUP_SIZE; i++) {
        vmin = min(src[i], vmin);
        vmax = max(src[i], vmax);
    }

    // zero-point zp is choosen as interger
    //  vmin = (0 + zp)*s
    //  vmax = (15 + zp)*s
    //
    char zp = round(15.0f * vmin/(vmax - vmin));
    half s = (vmax - vmin)/15.0f;
    scales[0] = s;
    zps[0] = zp;

    half rs = 1.0f/s;
    for(int i = 0; i < GROUP_SIZE; i+=2, dst += 8) {
        uchar v0 = clamp((uchar)round(src[i] * rs - zp), (uchar)0, (uchar)15);
        uchar v1 = clamp((uchar)round(src[i+1] * rs - zp), (uchar)0, (uchar)15);
        *dst = (v1 << 4) | v0;
    }
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_woq_I4(__global half * A,
                            __global half * C,
                            __global uchar * B,
                            __global half * scales,
                            __global char * zps,
                            int M, int N, int K) {
    int n = get_global_id(0);
    int m = get_global_id(1);
    int gk = get_local_id(2);
    int gn = get_local_id(0);
    int ng = get_group_id(0);
    int K_groups = get_local_size(2);

    int k0 = gk * GROUP_SIZE;
    int k1 = k0 + GROUP_SIZE;

    A += m*K + k0;
    B += NK8n_INT_4bit_index(k0, n, K, N);
    scales += n*K/GROUP_SIZE + gk;
    zps += n*K/GROUP_SIZE + gk;

    float sum = 0;

    // process current K group
    half scale = *scales;
    char4 zpx4 = (char4)(*zps);
    char4 mask4 = (char4)0xF;
    for(int g = 0; g < GROUP_SIZE; g += 8, B += 4*8) {
        // read 8 elements of A
        ushort vAs = intel_sub_group_block_read_us((const __global ushort*)(A + g));

        // read (4x2)x8 int4
        char4 bx4 = as_char4(intel_sub_group_block_read_uc4(B));
        half4 i4x8_even = convert_half4((bx4 & mask4) + zpx4);
        half4 i4x8_odd = convert_half4(as_char4(as_uchar4(bx4) >> 4) + zpx4);

        sum += as_half(sub_group_broadcast(vAs, 0)) * (i4x8_even.s0);
        sum += as_half(sub_group_broadcast(vAs, 1)) * (i4x8_odd.s0);
        sum += as_half(sub_group_broadcast(vAs, 2)) * (i4x8_even.s1);
        sum += as_half(sub_group_broadcast(vAs, 3)) * (i4x8_odd.s1);
        sum += as_half(sub_group_broadcast(vAs, 4)) * (i4x8_even.s2);
        sum += as_half(sub_group_broadcast(vAs, 5)) * (i4x8_odd.s2);
        sum += as_half(sub_group_broadcast(vAs, 6)) * (i4x8_even.s3);
        sum += as_half(sub_group_broadcast(vAs, 7)) * (i4x8_odd.s3);
    }
    __local float all_sum[8][256];

    // scales applied once
    all_sum[gn][gk] = sum * scale;

    barrier(CLK_LOCAL_MEM_FENCE);

    // only the first one responsible for reduce & output
    // local memory bandwidth is shared by all EUs, so we only allow
    // 1 work-item to do the final reduction
    if (gk != 0) return;

    sum = 0;

    // unroll hint can reduce loop-overhead & ALU-latency
    __attribute__((opencl_unroll_hint(8)))
    for(int i = 0; i < K_groups; i++) {
        sum += all_sum[gn][i];
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
        assert((K % GROUP_SIZE) == 0)   # 2 element packed into a INT8
        assert((N % 8) == 0)            # 8 element in N in one sub-group

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
        cl_kernels_WOQ_I4.enqueue("Linear_woq_I4",
                                  [self.N, M, self.K//GROUP_SIZE],
                                  [8, 1, self.K//GROUP_SIZE],
                                  input, output, self.weight_i4, self.scales, self.zps,
                                  M, self.N, self.K)
        return output


def tune_kernel_f16():
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

/*********************************************************************************
 in : [N, K]
 out: [N/16, K, 16]
*********************************************************************************/
__kernel void ReorderB(__global half * in, __global half * out, int N, int K) {
    int n = get_global_id(1);
    int k = get_global_id(0);

    out[((n/16)*K*16) + (k*16) + (n % 16)] = in[n*K + k];
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void Linear3(__global half * A, __global half * B, __global half * C, int M, int K, int N, int K_group_size) {
    int m = get_global_id(1);
    int n = get_global_id(0);
    int ngid = get_group_id(0);
    int ng = get_local_id(0);
    int kg = get_local_id(2);

    int k0 = kg * K_group_size;
    int k1 = k0 + K_group_size;

    float sum = 0;
    __global half * pA = A + m*K;
    __global half * pB = B + ngid * (K*16) + k0*16;
    for(int k = k0; k < k1; k += 8) {
        ushort vAs = intel_sub_group_block_read_us((const __global ushort*)(pA + k));
        half8 vBs = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(pB))); pB += 8*16;
        sum += as_half(intel_sub_group_broadcast(vAs, 0)) * vBs.s0;
        sum += as_half(intel_sub_group_broadcast(vAs, 1)) * vBs.s1;
        sum += as_half(intel_sub_group_broadcast(vAs, 2)) * vBs.s2;
        sum += as_half(intel_sub_group_broadcast(vAs, 3)) * vBs.s3;
        sum += as_half(intel_sub_group_broadcast(vAs, 4)) * vBs.s4;
        sum += as_half(intel_sub_group_broadcast(vAs, 5)) * vBs.s5;
        sum += as_half(intel_sub_group_broadcast(vAs, 6)) * vBs.s6;
        sum += as_half(intel_sub_group_broadcast(vAs, 7)) * vBs.s7;
    }

    __local float all_sum[16][K_groups];
    all_sum[ng][kg] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // only the first one responsible for reduce & output
    if (kg != 0) return;

    sum = 0;
    for(int i = 0; i < K_groups; i++) sum += all_sum[ng][i];
    C[m*N + n] = sum;
}

    '''
    K_groups = 16
    kernels = cl.kernels(src, f"-D K_groups={K_groups}", "./dump")

    import torch
    M = 1
    N, K = 2048, 5632 # down_proj
    N, K = 2048, 2048 # o_proj
    N, K = 2048, 2048 # o_proj
    #N, K = 5632, 2048 # up_proj
    #N, K = 32768, 2048 # N is big enough to fill all EU

    print(f"============= N = {N}")
    cl.profiling(True)

    A = torch.randint(-2, 2, [M, K]).half()
    B = torch.randint(-2, 2, [N, K]).half()
    ref = torch.matmul(A, B.transpose(1,0))

    Bsize = K*N*2
    Bcnt = int(500e6)//(Bsize)
    weights = [to_cl(B) for _ in range(Bcnt)]

    cl.finish()
    input = to_cl(A)
    output = cl.tensor([M, N], np.dtype(np.float16))
    for i in range(Bcnt):
        kernels.enqueue("Linear2", [N, M], [128, 1], input, weights[i], output, M, K, N)

    durs = cl.finish()
    for ns in durs:
        print(f" {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    mean_ns = sum(durs)/len(durs)
    print(f"  {mean_ns*1e-6: .3f} ms   BW: { K*N*2/mean_ns : .2f} GB/s")
    res = to_torch(output)
    if not torch.allclose(res, ref):
        print(res)
        print(ref)
        diff = res - ref
        print(diff[diff.abs() > 1])
        assert(False)

    K_group_size = (K // K_groups)
    assert(K_group_size * K_groups == K)
    assert(K_group_size % 8 == 0)
    kernels.enqueue("ReorderB", [K, N], [1, 1], weights[0], weights[Bcnt-1], N, K)

    for i in range(Bcnt):
        kernels.enqueue("Linear3", [N, M, K_groups], [16, 1, K_groups], input, weights[i], output, M, K, N, K_group_size)

    durs = cl.finish()
    for ns in durs:
        print(f" {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    mean_ns = sum(durs)/len(durs)
    print(f"  {mean_ns*1e-6: .3f} ms   BW: { K*N*2/mean_ns : .2f} GB/s")

    res = to_torch(output)
    if not torch.allclose(res, ref):
        print(res)
        print(ref)
        diff = res - ref
        print(diff[diff.abs() > 1])
        assert(False)

def tune_kernel_w4a():
    src = '''

    int NK8n_INT_4bit_index(int k, int n, int K, int N) {
        return (n/8)*(8*K/2) + (n % 8) + k*8/2;
    }

    __kernel void Linear_quant_I4(__global half * src, __global uchar * dst, __global half * scales, __global char * zps, int N, int K) {
        int kg = get_global_id(0);
        int n = get_global_id(1);
        int k = kg * GROUP_SIZE;

        src += n * K + k;
        dst += NK8n_INT_4bit_index(k, n, K, N);
        scales += n * (K/GROUP_SIZE) + kg;
        zps += n * (K/GROUP_SIZE) + kg;

        half vmin = src[0];
        half vmax = src[0];
        for(int i = 1; i < GROUP_SIZE; i++) {
            vmin = min(src[i], vmin);
            vmax = max(src[i], vmax);
        }

        // zero-point zp is choosen as interger
        //  vmin = (0 + zp)*s
        //  vmax = (15 + zp)*s
        //
        char zp = round(15.0f * vmin/(vmax - vmin));
        half s = (vmax - vmin)/15.0f;
        scales[0] = s;
        zps[0] = zp;

        half rs = 1.0f/s;
        for(int i = 0; i < GROUP_SIZE; i+=2, dst += 8) {
            uchar v0 = clamp((uchar)round(src[i] * rs - zp), (uchar)0, (uchar)15);
            uchar v1 = clamp((uchar)round(src[i+1] * rs - zp), (uchar)0, (uchar)15);
            *dst = (v1 << 4) | v0;
        }
    }

    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void Linear_woq_I4(__global half * A,
                                __global half * C,
                                __global uchar * B,
                                __global half * scales,
                                __global char * zps,
                                int M, int N, int K, int B_STEP) {
        int n = get_global_id(0);
        int m = get_global_id(1);
        int gk = get_local_id(2);
        int gn = get_local_id(0);
        int ng = get_group_id(0);
        int K_groups = get_local_size(2);

        int k0 = gk * GROUP_SIZE;
        int k1 = k0 + GROUP_SIZE;

        A += m*K + k0;

        if (B_STEP > 0){
            B += NK8n_INT_4bit_index(k0, n, K, N);
            scales += n*K/GROUP_SIZE + gk;
            zps += n*K/GROUP_SIZE + gk;
        }
        float sum = 0;

        // process current K group
        half scale = *scales;
        char4 zpx4 = (char4)(*zps);
        char4 mask4 = (char4)0xF;
        for(int g = 0; g < GROUP_SIZE; g += 8, B += 4*8) {
            // read 8 elements of A
            ushort vAs = intel_sub_group_block_read_us((const __global ushort*)(A + g));

            // read (4x2)x8 int4
            char4 bx4 = as_char4(intel_sub_group_block_read_uc4(B));
            half4 i4x8_even = convert_half4((bx4 & mask4) + zpx4);
            half4 i4x8_odd = convert_half4(as_char4(as_uchar4(bx4) >> 4) + zpx4);

            sum += as_half(sub_group_broadcast(vAs, 0)) * (i4x8_even.s0);
            sum += as_half(sub_group_broadcast(vAs, 1)) * (i4x8_odd.s0);
            sum += as_half(sub_group_broadcast(vAs, 2)) * (i4x8_even.s1);
            sum += as_half(sub_group_broadcast(vAs, 3)) * (i4x8_odd.s1);
            sum += as_half(sub_group_broadcast(vAs, 4)) * (i4x8_even.s2);
            sum += as_half(sub_group_broadcast(vAs, 5)) * (i4x8_odd.s2);
            sum += as_half(sub_group_broadcast(vAs, 6)) * (i4x8_even.s3);
            sum += as_half(sub_group_broadcast(vAs, 7)) * (i4x8_odd.s3);
        }
        __local float all_sum[8][256];

        // scales applied once
        all_sum[gn][gk] = sum * scale;

        barrier(CLK_LOCAL_MEM_FENCE);

        // only the first one responsible for reduce & output
        // local memory bandwidth is shared by all EUs, so we only allow
        // 1 work-item to do the final reduction
        if (gk != 0) return;

        sum = 0;

        // unroll hint can reduce loop-overhead & ALU-latency
        __attribute__((opencl_unroll_hint(8)))
        for(int i = 0; i < K_groups; i++) {
            sum += all_sum[gn][i];
        }
        C[m*N + n] = sum;
    }

    '''
    cl.profiling(True)

    GROUP_SIZE = 128
    kernels = cl.kernels(src, f"-D GROUP_SIZE={GROUP_SIZE}", "./dump")

    import torch
    M = 1
    N, K = 2048, 5632 # down_proj
    N, K = 2048, 2048 # o_proj
    N, K = 2048, 2048 # o_proj

    A = torch.randint(-2, 2, [M, K]).half()
    B = torch.randint(-2, 2, [N, K]).half()

    weight_half = to_cl(B)  # copy to GPU
    weight_i4 = cl.tensor([N, K//2], np.dtype(np.uint8))         # two int4 packed into a uint8
    scales = cl.tensor([N, K//GROUP_SIZE], np.dtype(np.float16)) # scales
    zps = cl.tensor([N, K//GROUP_SIZE], np.dtype(np.int8))    # zero-points
    kernels.enqueue("Linear_quant_I4", [K//GROUP_SIZE, N], [1, 1], weight_half, weight_i4, scales, zps, N, K)

    Bsize = K*N//2
    Bcnt = int(500e6)//(Bsize)
    weights_q = [to_cl(weight_i4.torch()) for _ in range(Bcnt)]
    weights_s = [to_cl(scales.torch()) for _ in range(Bcnt)]
    weights_z = [to_cl(zps.torch()) for _ in range(Bcnt)]

    input = to_cl(A)
    output = cl.tensor([M, N], np.dtype(np.float16))
    for i in range(Bcnt):
        kernels.enqueue("Linear_woq_I4",
                        [N, M, K//GROUP_SIZE],
                        [8, 1, K//GROUP_SIZE],
                        input, output, weights_q[i], weights_s[i], weights_z[i],
                        M, N, K, 4*8)

    durs = cl.finish()
    for ns in durs:
        print(f" {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    mean_ns = sum(durs)/len(durs)
    print(f"  {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")

    res = output.torch()
    ref = torch.matmul(A, B.transpose(1,0))
    if not torch.allclose(res, ref):
        print(res)
        print(ref)
        diff = res - ref
        print(diff[diff.abs() > 1])
        assert(False)
    else:
        print("ACCURACY is good")

if __name__ == "__main__":
    #tune_kernel_f16()
    tune_kernel_w4a()