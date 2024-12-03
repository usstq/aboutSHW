from . import cl
import numpy as np
from .utils import *

cl_kernel_sources_ref = r'''
// global_id [Hq, batch_size]
__kernel void MHA(__global half * param_qkv,         // [batch_size, L, (Hq + Hk + Hv) * head_size)]
                    int batch_size,
                    int L,                        // input seq-length
                    __global half * param_output,      // [batch_size, L, Hq * head_size]
                    __global half * param_cache_k,     // [batch_size, Hk, max_kv_len, head_size]
                    __global half * param_cache_v,     // [batch_size, Hk, max_kv_len, head_size]
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

    int HQKV = Hq + Hk + Hk;
    int hkv = h / head_group_size;

    __global half * cache_k = param_cache_k + (b * Hk + hkv)*max_kv_len*head_size;
    __global half * cache_v = param_cache_v + (b * Hk + hkv)*max_kv_len*head_size;
    __global float * attn_score = param_attn_score + (b * Hq + h)*max_kv_len;

    // q: head_size
    // k: head_size
    // v: head_size

    // concat k & v into cache
    __global half * dst_k = cache_k + kv_seq_len0*head_size;
    __global half * dst_v = cache_v + kv_seq_len0*head_size;

    for(int l = 0; l < L; l++) {
        __global half * cur_k = param_qkv + ((b * L + l) * HQKV + Hq + hkv) * head_size;
        __global half * cur_v = cur_k + Hk*head_size;
        for(int i = 0; i < head_size; i++) {
            *dst_k++ = cur_k[i];
            *dst_v++ = cur_v[i];
        }
    }

    __global half * q = param_qkv + ((b * L + 0) * HQKV + h) * head_size;
    __global half * output = param_output + ((b * L + 0)*Hq + h)*head_size;

    for(int ql = 0; ql < L; ql++, q += (HQKV * head_size), output += (Hq*head_size)) {
        //////////// q * cache_k
        __global half * pk = cache_k;
        float max_attn_score = -1e9f;

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
        float softmax_scale = 0;
        for(int p = 0; p <= kv_len; p++) {
            float a = exp(attn_score[p] - max_attn_score);
            attn_score[p] = a;
            softmax_scale += a;
        }
        softmax_scale = 1.0f / softmax_scale;

        //////////// attn_score * cache_v
        for(int i = 0; i < head_size; i++) {
            __global half * pv = cache_v;
            float sum = 0;
            for(int p = 0; p <= kv_len; p++, pv += head_size) {
                float scale = attn_score[p] * softmax_scale;
                sum += scale * pv[i];
            }
            output[i] = sum;
        }
    }
}
'''

class MHA:
    def __init__(self, head_cnt_q, head_cnt_k, head_size, max_kv_len, use_ref=False, kv_block=64):
        '''
            kv-cache is allocated internally here
        '''
        self.head_cnt_q = head_cnt_q  # query heads
        self.head_cnt_k = head_cnt_k
        self.head_cnt_qkv = head_cnt_q + 2 * head_cnt_k
        self.S = head_size
        self.max_kv_len = max_kv_len
        self.head_group_size = head_cnt_q // head_cnt_k
        self.cache_k = None
        self.cache_v = None
        self.head_size_scaler = 1.0 / (head_size ** 0.5)
        self.use_ref = use_ref
        self.kv_block = kv_block
        self.sub_group_size = 16
        self.concat_block = 256
        options = f'-D FMACNT=4 -DUNROLL=4 -DS={head_size} -DSGS={self.sub_group_size} -DHQ={head_cnt_q} -DHK={head_cnt_k} \
                    -DMAX_KV_LEN={max_kv_len} -DKV_BLOCK={self.kv_block} -DCONCAT_BLOCK={self.concat_block}'
        
        with open("cl_kernels/MHAFirst.cl", "r") as file:
            # Read the entire file content into a string
            cl_kernel_sources = file.read()
        self.cl_kernels = kernel_cache(cl_kernel_sources, options)
        self.cl_kernels_ref = kernel_cache(cl_kernel_sources_ref)

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
        assert kv_seq_len <= self.max_kv_len, f"kv cache length {self.max_kv_len} is not enough for kv_seq_len {kv_seq_len}"

        output = cl.tensor([B, L1, self.head_cnt_q * self.S], qkv.dtype)

        if self.cache_k is None:
            kv_cache_size = [B, self.head_cnt_k, self.max_kv_len, self.S]
            self.cache_k = cl.tensor(kv_cache_size, np.dtype(np.float16))
            self.cache_v = cl.tensor(kv_cache_size, np.dtype(np.float16))

        # print(self.head_size_scaler, L0, qkv.shape, self.head_cnt_q, self.head_cnt_k, self.S,  self.head_group_size, self.cache_k.shape, self.attn_score.shape)

        # attn_score is just a temp/scratch cl-buffer
        if self.use_ref:
            attn_score = cl.tensor([B, self.head_cnt_q, self.max_kv_len], np.dtype(np.float32))
            self.cl_kernels_ref.enqueue("MHA",
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
            cl_kernels = self.cl_kernels
            cl_kernels.enqueue("concat",
                               [self.head_cnt_k, B, L1 * self.concat_block],
                               [1, 1, self.concat_block],
                               qkv,
                               L1,
                               self.cache_k,
                               self.cache_v,
                               L0
                               )
            if L1 == 1:
                part_num = (L0 + L1 + self.kv_block - 1) // self.kv_block
                part_num_aligned = (part_num + 15) // 16 * 16
                part_sum = cl.tensor([B, self.head_cnt_q, part_num_aligned], np.dtype(np.float32))
                part_max = cl.tensor([B, self.head_cnt_q, part_num_aligned], np.dtype(np.float32))
                if part_num > 1:
                    part_output = cl.tensor([B, L1, part_num_aligned, self.head_cnt_q * self.S], qkv.dtype)
                cl_kernels.enqueue("MHASecond",
                                    [B, self.head_cnt_q, self.S * part_num],
                                    [1, 1, self.S],
                                    qkv,
                                    output if part_num == 1 else part_output,
                                    self.cache_k,
                                    self.cache_v,
                                    part_max,
                                    part_sum,
                                    L0 + L1
                                    )
                if part_num > 1:
                    cl_kernels.enqueue("MHAReduce",
                                       [B, self.head_cnt_q, self.S],
                                       [1, 1, self.S],
                                       part_output,
                                       output,
                                       part_max,
                                       part_sum,
                                       part_num)
            else:
                mb_blocks_num = (L1 + self.sub_group_size - 1) // self.sub_group_size
                print(cl_kernels.info("MHAFirst", [1, 1, self.S], 16))
                cl_kernels.enqueue("MHAFirst",
                                   [B, self.head_cnt_q, self.S * mb_blocks_num],
                                   [1, 1, self.S],
                                   qkv,
                                   output,
                                   L1
                                   )
        return output


if __name__ == "__main__":
    import torch
    from . import utils
    H = 32
    HK = 32
    S = 128
    MAX_KV_LEN = 256*8
    B = 16
    KV_BLOCK = 32
    ref = MHA(H, HK, S, MAX_KV_LEN, True)
    opt = MHA(H, HK, S, MAX_KV_LEN, False, kv_block=KV_BLOCK)
    first_token = 128*2-4  #KV_BLOCK - 2
    second_token_num = 80
    KV_LEN = 0
    for idx in range(second_token_num):
        L0 = KV_LEN
        if idx == 0:
            L1 = first_token
        else:
            L1 = 1
        KV_LEN += L1
        total_part_num = (KV_LEN + KV_BLOCK - 1) // KV_BLOCK
        print(f'test {L0=} {L1=} {KV_LEN=} {total_part_num=}')
        attention_mask = torch.randn([B, KV_LEN], dtype=torch.float32)
        # [B, L1, (Hq + Hk + Hv) * S)]
        qkv = utils.to_cl(torch.randn([B, L1, (H + HK + HK) * S], dtype=torch.float16))
        result_opt1 = opt(qkv, attention_mask)
        result_ref1 = ref(qkv, attention_mask)
        result_ref1_cpu = result_ref1.numpy()
        result_opt1_cpu = result_opt1.numpy()
        if not np.allclose(result_ref1_cpu, result_opt1_cpu, atol=0.02, rtol=0.02):
            print(f'{result_ref1_cpu=}\n{result_opt1_cpu=}')
            pos = np.where(np.abs(result_ref1_cpu - result_opt1_cpu) > 0.02)
            print(f'failed at {L0=} {L1=}, shape = {result_opt1_cpu.shape}')
            print(f'pos = {pos}')
            print(f'ref_val = {result_ref1_cpu[pos]}\nopt_val={result_opt1_cpu[pos]}')
            raise("failed.")
    print('done.')
