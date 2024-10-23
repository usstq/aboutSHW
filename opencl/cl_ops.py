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

// global_id [Hq, batch_size]
__kernel void MHA(__global float * param_qkv,         // [batch_size, L, (Hq + Hk + Hv) * head_size)]
                    int batch_size,
                    int L,                        // input seq-length
                    __global float * param_output,      // [batch_size, L, Hq * head_size]
                    __global float * param_cache_k,     // [batch_size, Hk, max_kv_len, head_size]
                    __global float * param_cache_v,     // [batch_size, Hk, max_kv_len, head_size]
                    __global float * param_attn_score,  // [batch_size, Hq, max_kv_len]
                    int kv_seq_len0,
                    int Hq,
                    int Hk,
                    int max_kv_len,
                    int head_size,
                    int head_group_size,
                    float head_size_scaler
                    ) {
    
    int h = get_global_id(0);
    int b = get_global_id(1);

    if (h >= Hq) return;
    if (b >= batch_size) return;

    for(b = 0; b < batch_size; b++) {
        for(h=0; h<Hq; h++) {

    int Hqkv = Hq + Hk + Hk;
    int hkv = h / head_group_size;

    __global float * cache_k = param_cache_k + (b * Hk + hkv)*max_kv_len*head_size;
    __global float * cache_v = param_cache_v + (b * Hk + hkv)*max_kv_len*head_size;
    __global float * attn_score = param_attn_score + (b * Hq + h)*max_kv_len;

    // q: head_size
    // k: head_size
    // v: head_size

    // concat k & v into cache
    __global float * dst_k = cache_k + kv_seq_len0*head_size;
    __global float * dst_v = cache_v + kv_seq_len0*head_size;

    for(int l = 0; l < L; l++) {
        __global float * cur_k = param_qkv + ((b * L + l) * Hqkv + Hq + hkv) * head_size;
        __global float * cur_v = cur_k + Hk*head_size;
        for(int i = 0; i < head_size; i++) {
            *dst_k++ = cur_k[i];
            *dst_v++ = cur_v[i];
        }
    }

    __global float * q = param_qkv + ((b * L + 0) * Hqkv + h) * head_size;
    __global float * output = param_output + ((b * L + 0)*Hq + h)*head_size;

    for(int l = 0; l < L; l++, q += (Hqkv * head_size), output += (Hq*head_size)) {
        //////////// q * cache_k
        __global float * pk = cache_k;
        float max_attn_score = -1e9f;

        for(int p = 0; p <= kv_seq_len0; p++, pk += head_size) {
            float dot_prod = 0;
            for(int i = 0; i < head_size; i++) {
                dot_prod += q[i] * pk[i];
            }
            dot_prod *= head_size_scaler;
            attn_score[p] = dot_prod;
            if (max_attn_score < dot_prod) max_attn_score = dot_prod;
        }

        //////////// softmax
        float softmax_scale = 0;
        for(int p = 0; p <= kv_seq_len0; p++) {
            float a = exp(attn_score[p] - max_attn_score);
            attn_score[p] = a;
            softmax_scale += a;
        }
        softmax_scale = 1.0f / softmax_scale;

        //////////// attn_score * cache_v
        for(int i = 0; i < head_size; i++) output[i] = 0.0f;

        __global float * pv = cache_v;
        for(int p = 0; p <= kv_seq_len0; p++, pv += head_size) {
            float scale = attn_score[p] * softmax_scale;
            for(int i = 0; i < head_size; i++)
                output[i] += scale * pv[i];
        }
    }
    }
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
    
    def __call__(self, qkv, position_id_base):
        '''
         no shape change, inplace on VRAM
         input : [batch_size, q_len, (Hq + Hk + Hv) * head_size)]
        '''
        qkv = to_cltensor(qkv)
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
        return torch.from_numpy(qkv.numpy())


class MHA:
    def __init__(self, head_cnt_q, head_cnt_k, head_size, max_kv_len):
        '''
            kv-cache is allocated internally here
        '''
        self.head_cnt_q = head_cnt_q  # query heads
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2*head_cnt_k
        self.head_size = head_size
        self.max_kv_len = max_kv_len
        self.head_group_size = head_cnt_q//head_cnt_k
        self.cache_k = None
        self.cache_v = None
        self.head_size_scaler = 1.0/(head_size ** 0.5)

    def __call__(self, qkv, attention_mask):

        '''
            qkv : [batch_size, q_len, (Hq + Hk + Hv) * head_size)]
            output: [batch_size, q_len, Hq * head_size]
        '''
        qkv = to_cltensor(qkv)
        # shape infer
        kv_seq_len = attention_mask.shape[-1]
        batch_size, q_len, S = qkv.shape
        assert(S == (self.head_cnt_qkv * self.head_size))
        kv_seq_len0 = kv_seq_len - q_len

        output = cl.tensor([batch_size, q_len, self.head_cnt_q * self.head_size], np.dtype(np.float32))

        if self.cache_k is None:
            kv_cache_size = [batch_size, self.head_cnt_k, self.max_kv_len, self.head_size]
            self.cache_k = cl.tensor(kv_cache_size, np.dtype(np.float32))
            self.cache_v = cl.tensor(kv_cache_size, np.dtype(np.float32))

        # print(self.head_size_scaler, kv_seq_len0, qkv.shape, self.head_cnt_q, self.head_cnt_k, self.head_size,  self.head_group_size, self.cache_k.shape, self.attn_score.shape)

        # attn_score is just a temp/scratch cl-buffer
        attn_score = cl.tensor([batch_size, self.head_cnt_q, self.max_kv_len], np.dtype(np.float32))
        cl_kernels.enqueue("MHA",
                           #[self.head_cnt_q, batch_size],
                           [1, 1],
                           [1, 1],
                           qkv,
                           batch_size,
                           q_len,
                           output,
                           self.cache_k,
                           self.cache_v,
                           attn_score,
                           kv_seq_len0,
                           self.head_cnt_q,
                           self.head_cnt_k,
                           self.max_kv_len,
                           self.head_size,
                           self.head_group_size,
                           self.head_size_scaler
                           )
        return torch.from_numpy(output.numpy())
