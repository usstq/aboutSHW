from . import cl
import numpy as np

cl_kernel_sources = '''
// global_id [Hq, batch_size]
__kernel void MHA(__global half * param_qkv,         // [batch_size, L, (Hq + Hk + Hv) * head_size)]
                    int batch_size,
                    int L,                        // input seq-length
                    __global half * param_output,      // [batch_size, L, Hq * head_size]
                    __global half * param_cache_k,     // [batch_size, Hk, max_kv_len, head_size]
                    __global half * param_cache_v,     // [batch_size, Hk, max_kv_len, head_size]
                    __global half * param_attn_score,  // [batch_size, Hq, max_kv_len]
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

    int Hqkv = Hq + Hk + Hk;
    int hkv = h / head_group_size;

    __global half * cache_k = param_cache_k + (b * Hk + hkv)*max_kv_len*head_size;
    __global half * cache_v = param_cache_v + (b * Hk + hkv)*max_kv_len*head_size;
    __global half * attn_score = param_attn_score + (b * Hq + h)*max_kv_len;

    // q: head_size
    // k: head_size
    // v: head_size

    // concat k & v into cache
    __global half * dst_k = cache_k + kv_seq_len0*head_size;
    __global half * dst_v = cache_v + kv_seq_len0*head_size;

    for(int l = 0; l < L; l++) {
        __global half * cur_k = param_qkv + ((b * L + l) * Hqkv + Hq + hkv) * head_size;
        __global half * cur_v = cur_k + Hk*head_size;
        for(int i = 0; i < head_size; i++) {
            *dst_k++ = cur_k[i];
            *dst_v++ = cur_v[i];
        }
    }

    __global half * q = param_qkv + ((b * L + 0) * Hqkv + h) * head_size;
    __global half * output = param_output + ((b * L + 0)*Hq + h)*head_size;

    for(int ql = 0; ql < L; ql++, q += (Hqkv * head_size), output += (Hq*head_size)) {
        //////////// q * cache_k
        __global half * pk = cache_k;
        half max_attn_score = -1e9f;

        // causal_mask & attn_mask is not applied explicitly
        int kv_len = ql + kv_seq_len0;

        for(int p = 0; p <= kv_len; p++, pk += head_size) {
            float dot_prod = 0;
            for(int i = 0; i < head_size; i++)
                dot_prod += q[i] * pk[i];

            dot_prod *= head_size_scaler;
            attn_score[p] = dot_prod;
            if (max_attn_score < dot_prod) max_attn_score = dot_prod;
        }

        //////////// softmax
        half softmax_scale = 0;
        for(int p = 0; p <= kv_len; p++) {
            half a = exp(attn_score[p] - max_attn_score);
            attn_score[p] = a;
            softmax_scale += a;
        }
        softmax_scale = 1.0f / softmax_scale;

        //////////// attn_score * cache_v
        for(int i = 0; i < head_size; i++) output[i] = 0.0f;

        __global half * pv = cache_v;
        for(int p = 0; p <= kv_len; p++, pv += head_size) {
            half scale = attn_score[p] * softmax_scale;
            for(int i = 0; i < head_size; i++)
                output[i] += scale * pv[i];
        }
    }
}
'''
cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4", "./dump")

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
        # shape infer
        kv_seq_len = attention_mask.shape[-1]
        batch_size, q_len, S = qkv.shape
        assert(S == (self.head_cnt_qkv * self.head_size))
        kv_seq_len0 = kv_seq_len - q_len

        output = cl.tensor([batch_size, q_len, self.head_cnt_q * self.head_size], qkv.dtype)

        if self.cache_k is None:
            kv_cache_size = [batch_size, self.head_cnt_k, self.max_kv_len, self.head_size]
            self.cache_k = cl.tensor(kv_cache_size, np.dtype(np.float16))
            self.cache_v = cl.tensor(kv_cache_size, np.dtype(np.float16))

        # print(self.head_size_scaler, kv_seq_len0, qkv.shape, self.head_cnt_q, self.head_cnt_k, self.head_size,  self.head_group_size, self.cache_k.shape, self.attn_score.shape)

        # attn_score is just a temp/scratch cl-buffer
        attn_score = cl.tensor([batch_size, self.head_cnt_q, self.max_kv_len], np.dtype(np.float16))
        cl_kernels.enqueue("MHA",
                           [self.head_cnt_q, batch_size],
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

        return output