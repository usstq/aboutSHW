import cl
import math
import numpy as np
import torch


def to_cltensor(input):
    if isinstance(input, cl.tensor):
        return input
    if input is None:
        return None
    if isinstance(input, torch.nn.parameter.Parameter):
        return cl.tensor(input.detach().numpy())
    if isinstance(input, torch.Tensor):
        return cl.tensor(input.detach().numpy())
    assert("to_cltensor(): Unexpected input ", input)


cl_kernel_sources = '''
    ulong __attribute__((overloadable)) intel_get_cycle_counter( void );


    // global_id [N, M]
    //
    __attribute__((intel_reqd_sub_group_size(8)))
    __kernel void Linear_f32(__global float * A, __global float * B, __global float * C, int M, int K, int N) {
        int m = get_global_id(1);
        int n = get_global_id(0);
        if (m < M && n < N) {
            float sum = 0;
            __global float * pA = A + m*K;
            __global float * pB = B + n*K;
            for(int k = 0; k < K; k++) {
                sum += pA[k] * pB[k];
            }
            C[m*N + n] = sum;
        }
    }

    // global_id : [Hq + Hk, q_len, batch_size]
    //
    // inv_freq : [half_rotary_dim]
    //
    //    input : [batch_size, q_len, (Hq + Hk + Hv)*S)]
    __kernel void ROPE(__global float * input,
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

        __global float * src = input + ((batch * q_len + q_offset) * head_cnt_qkv +  h) * head_size;

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
'''

cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

# GPU needs weight-compression to work : INT8 at least

class Linear:
    # weight: [N, K]
    def __init__(self, weight, bias):
        self.weight = to_cltensor(weight)
        self.bias = to_cltensor(bias)

        self.N = self.weight.shape[0]
        self.K = self.weight.shape[1]
        assert(self.bias is None)

    def __call__(self, input):
        input = to_cltensor(input)

        # shape inference
        i_shape = input.shape
        o_shape = list(i_shape)
        o_shape[-1] = self.N
        output = cl.tensor(o_shape, np.dtype(np.float32))

        M = math.prod(i_shape) // self.K
        
        #print(M, self.K, self.N,  input.shape, self.weight.shape, output.shape)

        cl_kernels.enqueue("Linear_f32", [self.N, M], [32, 32], input, self.weight, output, M, self.K, self.N)

        return torch.from_numpy(output.numpy())


class ROPE:
    def __init__(self, inv_freq, rotary_dim, head_cnt_q, head_cnt_k, head_size):
        self.inv_freq = to_cltensor(inv_freq)
        self.half_rotary_dim = rotary_dim//2
        self.head_cnt_q = head_cnt_q
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2*head_cnt_k
        self.head_size = head_size
    
    def __call__(self, input, position_id_base):
        input = to_cltensor(input)
        # no shape change, inplace on VRAM
        # input is 4D, layout is 
        batch_size, q_len, S = input.shape

        assert(S == (self.head_cnt_qkv * self.head_size))

        cl_kernels.enqueue("ROPE",
                            [self.head_cnt_q + self.head_cnt_k, q_len, batch_size],
                            [1, 1],
                            input,
                            self.inv_freq,
                            self.half_rotary_dim,
                            batch_size,
                            q_len,
                            self.head_cnt_qkv,
                            self.head_size,
                            position_id_base)
        return torch.from_numpy(input.numpy())
