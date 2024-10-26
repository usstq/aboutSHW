from . import cl
import numpy as np
from .utils import *

cl_kernel_sources_ref = '''
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
cl_kernels_ref = cl.kernels(cl_kernel_sources_ref, "-D FMACNT=4 -D UNROLL=4")

cl_kernel_sources = '''
__kernel void concat(__global half * param_qkv,         // [B, L1, (Hq + Hk + Hv) * S)]
                    int B,
                    int L1,                             // input seq-length
                    __global half * param_cache_k,     // [B, Hk, max_kv_len, S]
                    __global half * param_cache_v,     // [B, Hk, max_kv_len, S]
                    int L0,
                    int Hq,
                    int Hk,
                    int max_kv_len,
                    int S
                    ) {
    int hkv = get_global_id(0);
    int b = get_global_id(1);
    int l = get_global_id(2);

    int Hqkv = Hq + Hk + Hk;

    // concat k & v into cache
    __global half * dst_k = param_cache_k + 
        ((b * Hk + hkv) * max_kv_len + L0 + l) * S;
    __global half * dst_v = param_cache_v +
        ((b * Hk + hkv) * max_kv_len + L0 + l) * S;

    __global half * cur_k = param_qkv + ((b * L1 + l) * Hqkv + Hq + hkv) * S;
    __global half * cur_v = cur_k + Hk*S;
    for(int i = 0; i < S; i++) {
        dst_k[i] = cur_k[i];
        dst_v[i] = cur_v[i];
    }
}

// global_id [Hq, B]
__kernel void MHA(__global half * param_qkv,           // [B, L1, (Hq + Hk + Hv) * S)]
                    int B,
                    int L1,                             // input seq-length
                    __global half * param_output,      // [B, L1, Hq * S]
                    __global half * param_cache_k,     // [B, Hk, max_kv_len, S]
                    __global half * param_cache_v,     // [B, Hk, max_kv_len, S]
                    __global half * param_attn_score,  // [B, Hq, max_kv_len]
                    int L0,
                    int Hq,
                    int Hk,
                    int max_kv_len,
                    int S,
                    int head_group_size,
                    float head_size_scaler
                    ) {
    
    int h = get_global_id(0);
    int b = get_global_id(1);

    if (h >= Hq) return;
    if (b >= B) return;

    int Hqkv = Hq + Hk + Hk;
    int hkv = h / head_group_size;

    __global half * cache_k = param_cache_k + (b * Hk + hkv)*max_kv_len*S;
    __global half * cache_v = param_cache_v + (b * Hk + hkv)*max_kv_len*S;
    __global half * attn_score = param_attn_score + (b * Hq + h)*max_kv_len;

    __global half * q = param_qkv + ((b * L1 + 0) * Hqkv + h) * S;
    __global half * output = param_output + ((b * L1 + 0)*Hq + h)*S;

    for(int ql = 0; ql < L1; ql++, q += (Hqkv * S), output += (Hq*S)) {
        //////////// q * cache_k
        __global half * pk = cache_k;
        half max_attn_score = -1e9f;

        // causal_mask & attn_mask is not applied explicitly
        int kv_len = ql + L0;

        for(int p = 0; p <= kv_len; p++, pk += S) {
            float dot_prod = 0;
            for(int i = 0; i < S; i++)
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
        for(int i = 0; i < S; i++) output[i] = 0.0f;

        __global half * pv = cache_v;
        for(int p = 0; p <= kv_len; p++, pv += S) {
            half scale = attn_score[p] * softmax_scale;
            for(int i = 0; i < S; i++)
                output[i] += scale * pv[i];
        }
    }
}
'''
cl_kernels = cl.kernels(cl_kernel_sources, "-D FMACNT=4 -D UNROLL=4")

class MHA:
    def __init__(self, head_cnt_q, head_cnt_k, head_size, max_kv_len, use_ref=False):
        '''
            kv-cache is allocated internally here
        '''
        self.head_cnt_q = head_cnt_q  # query heads
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2*head_cnt_k
        self.S = head_size
        self.max_kv_len = max_kv_len
        self.head_group_size = head_cnt_q//head_cnt_k
        self.cache_k = None
        self.cache_v = None
        self.head_size_scaler = 1.0/(head_size ** 0.5)
        self.use_ref = use_ref

    def __call__(self, qkv, attention_mask):

        '''
            qkv : [B, q_len, (Hq + Hk + Hv) * S)]
            output: [B, q_len, Hq * S]
        '''
        #return self.reference_impl(qkv, attention_mask)
        # shape infer
        kv_seq_len = attention_mask.shape[-1]
        B, L1, S = qkv.shape
        assert(S == (self.head_cnt_qkv * self.S))
        L0 = kv_seq_len - L1

        output = cl.tensor([B, L1, self.head_cnt_q * self.S], qkv.dtype)

        if self.cache_k is None:
            kv_cache_size = [B, self.head_cnt_k, self.max_kv_len, self.S]
            self.cache_k = cl.tensor(kv_cache_size, np.dtype(np.float16))
            self.cache_v = cl.tensor(kv_cache_size, np.dtype(np.float16))

        # print(self.head_size_scaler, L0, qkv.shape, self.head_cnt_q, self.head_cnt_k, self.S,  self.head_group_size, self.cache_k.shape, self.attn_score.shape)

        # attn_score is just a temp/scratch cl-buffer
        attn_score = cl.tensor([B, self.head_cnt_q, self.max_kv_len], np.dtype(np.float16))
        if self.use_ref:
            cl_kernels_ref.enqueue("MHA",
                                   [self.head_cnt_q, B],
                                   [1, 1],
                                   qkv,
                                   B,
                                   L1,
                                   output,
                                   self.cache_k,
                                   self.cache_v,
                                   attn_score,
                                   L0,
                                   self.head_cnt_q,
                                   self.head_cnt_k,
                                   self.max_kv_len,
                                   self.S,
                                   self.head_group_size,
                                   self.head_size_scaler
                                   )
        else:
            cl_kernels.enqueue("concat",
                               [self.head_cnt_k, B, L1],
                               [1, 1, 1],
                               qkv,
                               B,
                               L1,
                               self.cache_k,
                               self.cache_v,
                               L0,
                               self.head_cnt_q,
                               self.head_cnt_k,
                               self.max_kv_len,
                               self.S
                               )
            cl_kernels.enqueue("MHA",
                               [self.head_cnt_q, B],
                               [1, 1],
                               qkv,
                               B,
                               L1,
                               output,
                               self.cache_k,
                               self.cache_v,
                               attn_score,
                               L0,
                               self.head_cnt_q,
                               self.head_cnt_k,
                               self.max_kv_len,
                               self.S,
                               self.head_group_size,
                               self.head_size_scaler
                               )
        return output

    # reference torch-cpu implementation for debugging purpose
    def repeat_kv(self, hidden_states, n_rep: int):
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def reference_impl(self, qkv, attention_mask):
        import torch, math
        qkv = to_torch(qkv)
        attention_mask = to_torch(attention_mask)
        if not hasattr(self, "kv_cache"):
            B, L1, S = qkv.shape
            self.kv_cache = torch.zeros(2, B, self.head_cnt_k, self.max_kv_len, self.S, dtype=torch.float16)

        hidden_size = self.head_cnt_q * self.S
        num_heads = self.head_cnt_q
        num_key_value_heads = self.head_cnt_k
        head_dim = self.S
        # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/models/llama/modeling_llama.py#L331
        # query_states : B, L, (H*S)+(H*S)+(H*S)
        bsz, q_len, _ = qkv.size()

        q_size = self.head_cnt_q * self.S
        kv_size = self.head_cnt_k * self.S
        query_states = qkv[:,:,:q_size].view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = qkv[:,:,q_size:q_size+kv_size].view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = qkv[:,:,q_size+kv_size:].view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        # q/k/v states : [batch, nHead, q_len, head_dim]
        kv_seq_len = attention_mask.shape[-1]
        kv_seq_len0 = kv_seq_len - q_len

        # kv_cache [2 * n_layers, batch, n_head, max_kv_len, head_size]
        self.kv_cache[0, :, :, kv_seq_len0:kv_seq_len, :] = key_states
        self.kv_cache[1, :, :, kv_seq_len0:kv_seq_len, :] = value_states

        # us beam_idx to gather(reorder kv cache), skipped in greedy case
        key_states = self.kv_cache[0, :, :, :kv_seq_len, :]
        value_states = self.kv_cache[1, :, :, :kv_seq_len, :]

        num_key_value_groups = num_heads // num_key_value_heads

        key_states = self.repeat_kv(key_states, num_key_value_groups)
        value_states = self.repeat_kv(value_states, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        attn_weights = attn_weights + attention_mask

        # apply causal mask, only the last q-token can access all kv-cache
        # [batch, num_heads, q_len ,kv_len] 
        for k in range(q_len-1):
            attn_weights[:, :, k, (k - q_len + 1):] = torch.finfo(torch.float32).min

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        return to_cl(attn_output)


if __name__ == "__main__":
    import torch
    from . import utils
    H = 32
    HK = 32
    S = 128
    MAX_KV_LEN = 256
    B = 1
    ref = MHA(H, HK, S, MAX_KV_LEN, True)
    opt = MHA(H, HK, S, MAX_KV_LEN, False)
    kv_lens = [
        (0, 10),    # first token
        (10, 1)     # second token
    ]
    for L0, L1 in kv_lens:
        KV_LEN = L0 + L1
        attention_mask = torch.randn([B, KV_LEN], dtype=torch.float32)
        # [B, L1, (Hq + Hk + Hv) * S)]
        qkv = utils.to_cl(torch.randn([B, L1, (H + HK + HK) * S], dtype=torch.float16))
        result_ref1 = ref(qkv, attention_mask)
        result_opt1 = opt(qkv, attention_mask)
        result_ref1_cpu = result_ref1.numpy()
        result_opt1_cpu = result_opt1.numpy()
        if not np.allclose(result_ref1_cpu, result_opt1_cpu):
            pos = np.where(np.abs(result_ref1_cpu - result_opt1_cpu) > 0.0001)
            print(f'failed at {L0=} {L1=}, shape = {result_opt1_cpu.shape}')
            print(f'pos = {pos}')
            print(f'ref_val = {result_ref1_cpu[pos]}\nopt_val={result_opt1_cpu[pos]}')
            raise("failed.")
