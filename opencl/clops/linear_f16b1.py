cl_kernel_sources = r'''ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
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
__kernel void Linear_f16_b1(
    __global half * A,
    __global half * B,
    __global float * bias,
    __global half * C,
    int M, int K, int N, int K_group_size) {
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
    if (bias) sum += bias[n];
    C[m*N + n] = sum;
}

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
    if (bias) sum += bias[n];
    C[m*N + n] = sum;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void Linear_f16_b1_opt(__global half * A, __global half * B, __global float * bias, __global half * C, int M, int K, int N, int K_group_size) {
    int n = get_global_id(0);
    int m = get_global_id(1);

    int ngid = get_group_id(0);

    int n_local_id = get_local_id(0);
    int k_local_id = get_local_id(2);

    __global half * pA = A + m*K;
    __global half * pB = B + ngid * (K*16);

    float sum = 0;
    for(int k = k_local_id*8; k < K; k += 8*K_groups) {
        // load src, sharing for all sub-groups
        // half vB = as_half(intel_sub_group_block_read_us((const __global ushort*)(pB + k*16)));
        ushort vAs = intel_sub_group_block_read_us((const __global ushort*)(pA + k));
        half8 vBs = as_half8(intel_sub_group_block_read_us8((const __global ushort*)(pB + k*16)));
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
    all_sum[n_local_id][k_local_id] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // only the first one responsible for reduce & output
    if (k_local_id != 0) return;

    sum = 0;
    for(int i = 0; i < K_groups; i++) sum += all_sum[n_local_id][i];
    if (bias) sum += bias[n];
    C[m*N + n] = sum;
}

'''

from . import cl
import numpy as np
from .utils import *

K_groups = 16
cl_kernels = kernel_cache(cl_kernel_sources, f"-D K_groups={K_groups}")

class Linear_f16b1:
    def __init__(self, weight, bias = None):
        self.N, self.K = weight.shape # weight: [N, K]

        assert(self.K % K_groups == 0)
        assert(self.N % 16 == 0)

        self.weight0 = to_cl(weight.half())
        weight_raw = to_cl(weight.half())
        self.weight = cl.tensor(weight_raw.shape, weight_raw.dtype)
        self.bias = to_cl(bias)

        cl_kernels.enqueue("ReorderB", [self.K, self.N], [1, 1], self.weight0, self.weight, self.N, self.K)

    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, input.dtype)
        #output0 = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)
        #cl_kernels.enqueue("Linear_f16", [self.N, M], [128, 1], input, self.weight0, output, M, self.K, self.N)
        cl_kernels.enqueue("Linear_f16_b1", # "Linear_f16_b1",
                           [self.N, M, K_groups],
                           [16, 1, K_groups], 
                           input,
                           self.weight,
                           self.bias,
                           output,
                           M,
                           self.K,
                           self.N,
                           self.K // K_groups)

        return output

'''
from tracing: 

   cliloader -dv -cdt  --dump-dir ./dump/  python -m clops.tests.llama -hf ../../Qwen2-0.5B-Instruct/ -p 大家  -n 32 -q f16b1

(1, 896, 1152) 35 us     59 GB/s
(1, 896, 896)  26 us     61 GB/s
(1, 896, 4864) 99 us     89 GB/s
(1, 4864, 896) 97 us     89 GB/s
(1, 896, 151936) 2910 us 93 GB/s


 M,K,N
we can observe that for some reason GPU needs some time to get into it's best performance

Linear_shapes: {(1, 896, 1152): 768, (1, 896, 896): 768, (1, 896, 4864): 1536, (1, 4864, 896): 768, (1, 896, 151936): 32}

...
[1, 2048, 2048] 8.397 MB 4.381 ms, BW:  1.92 GB/s
[1, 2048, 2048] 8.397 MB 4.060 ms, BW:  2.07 GB/s
[1, 2048, 2048] 8.397 MB 4.225 ms, BW:  1.99 GB/s
[1, 2048, 2048] 8.397 MB 4.066 ms, BW:  2.07 GB/s
[1, 2048, 2048] 8.397 MB 0.115 ms, BW:  73.08 GB/s
[1, 2048, 2048] 8.397 MB 0.115 ms, BW:  72.75 GB/s
[1, 2048, 2048] 8.397 MB 0.115 ms, BW:  73.08 GB/s
....
[1, 2048, 2048] 8.397 MB 0.114 ms, BW:  73.95 GB/s
[1, 2048, 2048] 8.397 MB 0.114 ms, BW:  73.35 GB/s
[1, 2048, 2048] 8.397 MB 0.111 ms, BW:  75.69 GB/s
[1, 2048, 2048] 8.397 MB 0.094 ms, BW:  88.97 GB/s
[1, 2048, 2048] 8.397 MB 0.096 ms, BW:  87.24 GB/s
[1, 2048, 2048] 8.397 MB 0.095 ms, BW:  88.00 GB/s
'''

if __name__ == "__main__":
    import sys
    cl.profiling(True)
    def test_acc(shape, Bcnt = 0):
        M, K, N = shape
        A = torch.randint(-8, 8, [M, K]).half()
        B = torch.randint(-8, 8, [N, K]).half()
        ref = torch.matmul(A, B.transpose(1,0)).numpy()

        Bsize = K*N*2 + M*K*2 + M*N*2
        if (Bcnt <= 0): Bcnt = int(500e6)//(Bsize)
        linears = [Linear_f16b1(B) for _ in range(Bcnt)]

        input = to_cl(A)
        for l in linears:
            output = l(input)

        durs = cl.finish()
        for ns in durs:
            print(f"{shape} {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
        if (len(durs)):
            mean_ns = sum(durs)/len(durs)
            print(f"  {mean_ns*1e-6: .3f} ms   BW: { K*N*2/mean_ns : .2f} GB/s")

        res = output.numpy()
        compare(ref, res)

    test_acc([1, 896, 4864], 100); sys.exit(0)

    for b in [7, 1]:
        test_acc([b, 2048, 2560], 10)
        test_acc([b, 2048, 2048], 10)
        test_acc([b, 2048, 5632], 10)
        test_acc([b, 5632, 2048], 10)
        test_acc([b, 2048, 32000], 10)

