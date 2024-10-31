cl_kernel_sources = '''
int NK8n_INT_4bit_index(int k, int n, int K, int N) {
    return (n/8)*(8*K/2) + (n % 8) + (k/2)*8;
}


__kernel void Linear_quant_I4(__global half * src, __global uchar * dst, __global half * scales, __global char * zps, int N, int K) {
    // ndrange: [K//GROUP_SIZE, N], [1, 1]
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
    //  vmin = (0 - zp)*s
    //  vmax = (15 - zp)*s
    //
    char zp = (15.0f * vmin/(vmax - vmin));
    half s = (vmax - vmin)/15.0f;

    //s = 1;
    //zp = vmin;
    
    scales[0] = s;
    zps[0] = zp;

    half rs = 1.0f/s;
    for(int i = 0; i < GROUP_SIZE; i+=2, dst += 8) {
        uchar v0 = (uchar)clamp((char)round(src[i] * rs - zp), (char)0, (char)15);
        uchar v1 = (uchar)clamp((char)round(src[i+1] * rs - zp), (char)0, (char)15);
        *dst = (v1 << 4) | v0;
    }
}

__kernel void Linear_w4a_ref(__global half * A,
                             __global half * C,
                             __global uchar * B,
                             __global half * scales,
                             __global char * zps,
                             __global float * bias,
                             int M, int N, int K) {
    // global: [N, M]
    // local:  [1, 1]
    int n = get_global_id(0);
    int m = get_global_id(1);

    A += m*K;
    B += NK8n_INT_4bit_index(0, n, K, N);
    scales += n*K/GROUP_SIZE;
    zps += n*K/GROUP_SIZE;

    float sum = 0;
    for(int k = 0; k < K; k += GROUP_SIZE) {
        half s =  *scales++;
        char z =  *zps++;
        for(int ki = 0; ki < GROUP_SIZE; ki+=2) {
            uchar v = B[(k+ki) * 4];
            half w1 = ((v >> 4) + z) * s;
            half w0 = ((v & 0xF) + z) * s;

            sum += (*A++) * w0;
            sum += (*A++) * w1;
        }
    }
    if (bias) sum += bias[n];
    C[m*N + n] = sum;
}

void splitter(const int n, const int team, const int tid, int* n_start, int* n_end) {
    if (team <= 1 || n == 0) {
        *n_start = 0;
        *n_end = n;
    } else {
        int n1 = (n + team - 1) / team;
        int n2 = n1 - 1;
        int T1 = n - n2 * team;
        *n_end = tid < T1 ? n1 : n2;
        *n_start = tid <= T1 ? tid * n1 : T1 * n1 + (tid - T1) * n2;
    }
    *n_end += *n_start;
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void Linear_w4a(__global half * A_base,
                            __global half * C,
                            __global uchar * B_base,
                            __global half * scales,
                            __global char * zps,
                            __global float * bias,
                            int M, int N, int K) {
    int n = get_global_id(0);
    int m = get_global_id(1);
    int gn = get_local_id(0);
    int ng = get_group_id(0);
    int ithr = get_local_id(2);
    int nthr = get_local_size(2);

    int K_groups = K/GROUP_SIZE;
    int gk0, gk1;
    splitter(K_groups, nthr, ithr, &gk0, &gk1);

    __local float all_sum[8][256];

    scales += n*K/GROUP_SIZE;
    zps += n*K/GROUP_SIZE;

    for(int gk = gk0; gk < gk1; gk++) {
        __global half *A = A_base + m*K + gk*GROUP_SIZE;
        __global uchar *B = B_base + NK8n_INT_4bit_index(gk*GROUP_SIZE, n, K, N);

        float sum = 0;
        half scale = scales[gk];
        char4 zpx4 = (char4)(zps[gk]);
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

        // scales applied once
        all_sum[gn][gk] = sum * scale;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // only the first one responsible for reduce & output
    // local memory bandwidth is shared by all EUs, so we only allow
    // 1 work-item to do the final reduction
    if (gk0 != 0) return;

    float sum = 0;
    // unroll hint can reduce loop-overhead & ALU-latency
    __attribute__((opencl_unroll_hint(8)))
    for(int i = 0; i < K_groups; i++) {
        sum += all_sum[gn][i];
    }
    if (bias) sum += bias[n];
    C[m*N + n] = sum;
}
'''

from . import cl
import numpy as np
from .utils import *

GROUP_SIZE = 128

# weight-only-quantization to INT4
class Linear_w4a:
    # weight: [N, K], 
    def __init__(self, weight, bias=None):
        # quantize weight into groupped INT4 format:(sym)
        N, K = weight.shape
        assert (K % GROUP_SIZE) == 0, f"{K} % {GROUP_SIZE} != 0"   # 2 element packed into a INT8
        assert (N % 8) == 0, f"{N} % 8 != 0"            # 8 element in N in one sub-group

        self.cl_kernels_WOQ_I4 = kernel_cache(cl_kernel_sources, f"-D GROUP_SIZE={GROUP_SIZE}", "./dump")

        weight_half = to_cl(weight.half())  # copy to GPU
        self.weight_i4 = cl.tensor([N//8, K//2, 8], np.dtype(np.uint8))         # two int4 packed into a uint8
        self.scales = cl.tensor([N, K//GROUP_SIZE], np.dtype(np.float16)) # scales
        self.zps = cl.tensor([N, K//GROUP_SIZE], np.dtype(np.float16))    # zero-points
        self.cl_kernels_WOQ_I4.enqueue("Linear_quant_I4", [K//GROUP_SIZE, N], [1, 1], weight_half, self.weight_i4, self.scales, self.zps, N, K)

        self.bias = to_cl(bias)
        self.N = N
        self.K = K


    def __call__(self, input):
        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        
        output = cl.tensor(o_shape, input.dtype)

        M = input.numel // self.K
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        if 1:
            # local size dimension & group-size (product of local_size) is limited by platform capabilities
            # (256 on UHD-620, 1024 on A770), so we must choose carefully
            # 
            # this also means, each work-thread (sub-group running on one EU thread) should handle
            # a few k-groups based on the actual k-group count
            self.cl_kernels_WOQ_I4.enqueue("Linear_w4a",
                                    [self.N, M, 32],
                                    [8, 1, 32],
                                    input, output, self.weight_i4, self.scales, self.zps, self.bias,
                                    M, self.N, self.K)
        else:
            self.cl_kernels_WOQ_I4.enqueue("Linear_w4a_ref",
                                    [self.N, M], [1, 1],
                                    input, output, self.weight_i4, self.scales, self.zps, self.bias,
                                    M, self.N, self.K)
        return output

if __name__ == "__main__":
    import sys
    cl.profiling(True)
    def test_acc(shape, Bcnt = 0):
        M, K, N = shape

        RA = 1
        RB = 1
        torch.manual_seed(0)
        A = torch.randint(low=-RA, high=RA+1, size=[M, K], dtype=torch.half)
        B = torch.randint(low=-RB, high=RB+1, size=[N, K], dtype=torch.half)

        #A[:, 0] *= 5
        ref = torch.matmul(A, B.transpose(1,0)).numpy()

        Bsize = K*N/2 + N*(K//GROUP_SIZE)*2 + N*(K//GROUP_SIZE)
        if (Bcnt <= 0): Bcnt = int(500e6)//(Bsize)
        linears = [Linear_w4a(B) for _ in range(Bcnt)]

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
        compare(ref, res, atol=0.1, rtol=0.1)

    test_acc([1, 4096, 4096], 100); sys.exit(0)

    Bcnt = 1
    for b in [7, 1]:
        test_acc([b, 4096, 12288], Bcnt)
        test_acc([b, 4096, 4096], Bcnt)
        test_acc([b, 4096, 11008], Bcnt)
        test_acc([b, 11008, 4096], Bcnt)
        test_acc([b, 4096, 32000], Bcnt)

    sys.exit(0)

