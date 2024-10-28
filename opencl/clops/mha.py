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
cl_kernels_ref = cl.kernels(cl_kernel_sources_ref, "-D FMACNT=4 -D UNROLL=4")

cl_kernel_sources = r'''
__kernel void concat(__global half * param_qkv,         // [B, L1, (Hq + Hk + Hv) * S)]
                    int L1,                             // input seq-length
                    __global half * param_cache_k,     // [B, Hk, max_kv_len, S]
                    __global half * param_cache_v,     // [B, Hk, max_kv_len, S]
                    int L0
                    ) {
    int hkv = get_global_id(0);
    int b = get_global_id(1);
    int l = get_global_id(2);

    int HQKV = HQ + HK + HK;

    // concat k & v into cache
    __global half * dst_k = param_cache_k + 
        ((b * HK + hkv) * MAX_KV_LEN + L0 + l) * S;
    __global half * dst_v = param_cache_v +
        ((b * HK + hkv) * MAX_KV_LEN + L0 + l) * S;

    __global half * cur_k = param_qkv + ((b * L1 + l) * HQKV + HQ + hkv) * S;
    __global half * cur_v = cur_k + HK * S;
    for(int i = 0; i < S; i++) {
        dst_k[i] = cur_k[i];
        dst_v[i] = cur_v[i];
    }
}

// global_id [Hq, B]
__kernel void MHA(__global half * param_qkv,           // [B, L1, (Hq + Hk + Hv) * S)]
                    int L1,                            // input seq-length
                    __global half * param_output,      // [B, L1, Hq * S]
                    __global half * param_cache_k,     // [B, Hk, max_kv_len, S]
                    __global half * param_cache_v,     // [B, Hk, max_kv_len, S]
                    __global float* param_attn_score,  // [B, Hq, max_kv_len]
                    int L0
                    ) {
    int h = get_global_id(0);
    int b = get_global_id(1);

    int HQKV = HQ + HK + HK;
    int head_group_size = HQ / HK;
    int hkv = h / head_group_size;
    const float head_size_scaler = 1.0f / sqrt(convert_float(S));

    __global half * cache_k = param_cache_k + (b * HK + hkv) * MAX_KV_LEN * S;
    __global half * cache_v = param_cache_v + (b * HK + hkv) * MAX_KV_LEN * S;
    __global float * attn_score = param_attn_score + (b * HQ + h) * MAX_KV_LEN;

    __global half * q = param_qkv + ((b * L1 + 0) * HQKV + h) * S;
    __global half * output = param_output + ((b * L1 + 0) * HQ + h) * S;

    for(int ql = 0; ql < L1; ql++, q += (HQKV * S), output += (HQ * S)) {
        //////////// q * cache_k
        __global half * pk = cache_k;
        float max_attn_score = -1e9f;

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
        float softmax_scale = 0;
        for(int p = 0; p <= kv_len; p++) {
            float a = exp(attn_score[p] - max_attn_score);
            attn_score[p] = a;
            softmax_scale += a;
        }
        softmax_scale = 1.0f / softmax_scale;

        //////////// attn_score * cache_v
        for(int i = 0; i < S; i++) {
            __global half * pv = cache_v;
            float sum = 0;
            for(int p = 0; p <= kv_len; p++, pv += S) {
                float scale = attn_score[p] * softmax_scale;
                sum += scale * pv[i];
            }
            output[i] = sum;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SGS)))
__kernel void MHASecond(__global half * param_qkv,         // [B, 1, (HQ + HK + HK) * S]
                        __global half * param_output,      // [B, 1, HQ * S]/[B, 1, part_num, HQ * S]
                        __global half * param_cache_k,     // [B, HK, max_kv_len, S]
                        __global half * param_cache_v,     // [B, HK, max_kv_len, S]
                        __global float* param_max,         // [B, HQ, part_num]
                        __global float* param_sum,         // [B, HQ, part_num]
                        int kv_len
                        ) {
    // global: [B, H, S * part_num]
    // local:  [1, 1, S]
    const uint b = get_global_id(0);
    const uint h = get_global_id(1);
    const uint id_local = get_local_id(2);
    const uint id_sg = get_sub_group_id();
    const uint id_sg_local = get_sub_group_local_id();
    const uint part_num = get_num_groups(2);
    const uint cur_part_num = get_group_id(2);
    const uint cur_kv_len = cur_part_num + 1 < part_num ? KV_BLOCK : kv_len - cur_part_num * KV_BLOCK;

    int HQKV = HQ + HK + HK;
    int head_group_size = HQ / HK;
    int hkv = h / head_group_size;
    const float head_size_scaler = 1.0f / sqrt(convert_float(S));

    __global half* cache_k = param_cache_k + ((b * HK + hkv) * MAX_KV_LEN + cur_part_num * KV_BLOCK) * S;
    __global half* cache_v = param_cache_v + ((b * HK + hkv) * MAX_KV_LEN + cur_part_num * KV_BLOCK) * S;

    __global half* q = param_qkv + b * HQKV * S + h * S;

    __local half query[S];
    // 1, load query
    query[id_local] = as_half(intel_sub_group_block_read_us((const __global ushort*)q + id_sg * SGS));
    barrier(CLK_LOCAL_MEM_FENCE);

    // 2, compute Q*K(one token) dot product in each sgw, S/SGS tokens at the same time in wg
    __local float qk_dot[KV_BLOCK];
    for (int i = id_sg; i < cur_kv_len; i += S / SGS) {
        __global half* pk = cache_k + i * S;
        float sum = 0;
        for (int j = 0; j < S; j += SGS) {
            half q_val = as_half(intel_sub_group_block_read_us((const __local ushort*)query + j));
            half k_val = as_half(intel_sub_group_block_read_us((const __global ushort*)pk + j));
            sum += q_val * k_val;
        }
        sum = sub_group_reduce_add(sum);
        if (id_sg_local == 0) {
            sum *= head_size_scaler;
            qk_dot[i] = sum;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3, compute softmax
    // 3.1 max(qk_dot)
    __local float max_qk_dot_share[S / SGS];
    float max_qk_dot = -1e9f;
    for (int i = id_local; i < cur_kv_len; i += S) {
        max_qk_dot = fmax(max_qk_dot, qk_dot[i]);
    }
    // reduce in sg
    max_qk_dot = sub_group_reduce_max(max_qk_dot);
    if (id_sg_local == 0) {
        max_qk_dot_share[id_sg] = max_qk_dot;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // reduce across sg
    if (id_sg_local < S / SGS) {
        max_qk_dot = max_qk_dot_share[id_sg_local];
    }
    max_qk_dot = sub_group_reduce_max(max_qk_dot);
    // 3.2 compute: exp(attn_score[p] - max_attn_score)
    float exp_sum = 0;
    for (int i = id_local; i < cur_kv_len; i += S) {
        float exp_val = native_exp(qk_dot[i] - max_qk_dot);
        exp_sum += exp_val;
        qk_dot[i] = exp_val;
    }
    // reduce in sg
    exp_sum = sub_group_reduce_add(exp_sum);
    if (id_sg_local == 0) {
        max_qk_dot_share[id_sg] = exp_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // reduce across sg
    exp_sum = 0;
    if (id_sg_local < S / SGS) {
        exp_sum = max_qk_dot_share[id_sg_local];
    }
    exp_sum = sub_group_reduce_add(exp_sum);
    if (part_num > 1 && id_local == 0) {
        uint offset = cur_part_num + (b * HQ + h) * part_num;
        param_max[offset] = max_qk_dot;
        param_sum[offset] = exp_sum;
    }
    // 3.3 compute: softmax_scale = 1.0f / softmax_scale; half scale = attn_score[p] * softmax_scale;
    float scale = 1.0f / exp_sum;
    __local half weights[KV_BLOCK];
    for (int i = id_local; i < cur_kv_len; i += S) {
        weights[i] = convert_half(qk_dot[i] * scale);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4 compute W*V(one token) dot product in each wg, SGS tokens at the same time in wg
    float sum = 0;
    for (int i = 0; i < cur_kv_len; i += SGS) {
        ushort weight = intel_sub_group_block_read_us((const __local ushort*)weights + i);
        __global half* pv = cache_v + i * S + id_sg * SGS;
        for (int j = 0; j < SGS; j++) {
            half v_val = as_half(intel_sub_group_block_read_us((const __global ushort*)pv + j * S));
            half w = as_half(intel_sub_group_broadcast(weight, j));
            sum += w * v_val;
        }
    }
    half acc = sum;

    // 5 output
    if (part_num == 1) {
        // [B, 1, HQ * S]
        __global half* output = param_output + b * HQ * S + h * S + id_sg * SGS;
        intel_sub_group_block_write_us((const __global ushort*)output, as_short(acc));
    } else {
        // [B, 1, part_num, HQ * S]
        __global half* output = param_output + b * part_num * HQ * S + cur_part_num * HQ * S + h * S + id_sg * SGS;
        intel_sub_group_block_write_us((const __global ushort*)output, as_short(acc));
    }
}

__attribute__((intel_reqd_sub_group_size(SGS)))
__kernel void MHAReduce(__global half * part_output,       // [B, 1, part_num, HQ * S]
                        __global half * param_output,      // [B, 1, HQ * S]
                        __global float* param_max,         // [B, HQ, part_num]
                        __global float* param_sum,         // [B, HQ, part_num]
                        uint part_num
                        ) {
    // global: [B, HQ, S]
    // local: [1, 1, S]
    const uint b = get_global_id(0);
    const uint h = get_global_id(1);
    const uint id_local = get_local_id(2);
    // sg id in workgroup
    const uint id_sg = get_sub_group_id();
    // id in subgroup
    const uint id_sg_local = get_sub_group_local_id();

    __global float* cur_param_max = param_max + b * HQ * part_num + h * part_num;
    __global float* cur_param_sum = param_sum + b * HQ * part_num + h * part_num;
    const uint MAX_PART_NUM = (MAX_KV_LEN + KV_BLOCK - 1) / KV_BLOCK;
    __local float cur_param_max_share[MAX_PART_NUM];
    // get real max val
    float max_qk_dot = -1e9f;
    for (int i = id_local; i < part_num; i += S) {
        max_qk_dot = fmax(max_qk_dot, cur_param_max[i]);
        cur_param_max_share[i] = cur_param_max[i];
    }
    // reduce in sg
    max_qk_dot = sub_group_reduce_max(max_qk_dot);
    // reduce across sg
    __local float max_qk_dot_share[S / SGS];
    if (id_sg_local == 0) {
        max_qk_dot_share[id_sg] = max_qk_dot;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    max_qk_dot = -1e9f;
    if (id_sg_local < S / SGS) {
        max_qk_dot = max_qk_dot_share[id_sg_local];
    }
    max_qk_dot = sub_group_reduce_max(max_qk_dot);

    // get real sum
    float exp_sum = 0;
    for (int i = id_local; i < part_num; i += S) {
        float exp_new = cur_param_sum[i] * native_exp(cur_param_max_share[i] - max_qk_dot);
        exp_sum += exp_new;
        cur_param_max_share[i] = exp_new;
    }
    exp_sum = sub_group_reduce_add(exp_sum);
    if (id_sg_local == 0) {
        max_qk_dot_share[id_sg] = exp_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    exp_sum = 0;
    if (id_sg_local < S / SGS) {
        exp_sum = max_qk_dot_share[id_sg_local];
    }
    exp_sum = sub_group_reduce_max(exp_sum);

    // reduce
    float rescale = 1.0f / exp_sum;
    float acc = 0;
    // [B, 1, HQ * S]
    __global half* output = param_output + b * HQ * S + h * S + id_sg * SGS;
    for (int i = 0; i < part_num; i++) {
        // [B, 1, part_num, HQ * S]
        __global half* cur_part_output = part_output + (b * part_num + i) * HQ * S + h * S;
        float compensation = cur_param_max_share[i];
        half cur_output = as_half(intel_sub_group_block_read_us((const __global ushort*)cur_part_output + id_sg * SGS));
        acc += convert_float(cur_output) * compensation;
    }
    intel_sub_group_block_write_us((const __global ushort*)output, as_short(convert_half(acc * rescale)));
}
'''

class MHA:
    def __init__(self, head_cnt_q, head_cnt_k, head_size, max_kv_len, use_ref=False, kv_block=256):
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
        options = f'-D FMACNT=4 -DUNROLL=4 -DS={head_size} -DSGS={self.sub_group_size} -DHQ={head_cnt_q} -DHK={head_cnt_k} \
                    -DMAX_KV_LEN={max_kv_len} -DKV_BLOCK={self.kv_block}'
        self.cl_kernels = cl.kernels(cl_kernel_sources, options)

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
        assert kv_seq_len <= self.max_kv_len, "kv cache length is not enough"

        output = cl.tensor([B, L1, self.head_cnt_q * self.S], qkv.dtype)

        if self.cache_k is None:
            kv_cache_size = [B, self.head_cnt_k, self.max_kv_len, self.S]
            self.cache_k = cl.tensor(kv_cache_size, np.dtype(np.float16))
            self.cache_v = cl.tensor(kv_cache_size, np.dtype(np.float16))

        # print(self.head_size_scaler, L0, qkv.shape, self.head_cnt_q, self.head_cnt_k, self.S,  self.head_group_size, self.cache_k.shape, self.attn_score.shape)

        # attn_score is just a temp/scratch cl-buffer
        attn_score = cl.tensor([B, self.head_cnt_q, self.max_kv_len], np.dtype(np.float32))
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
            cl_kernels = self.cl_kernels
            cl_kernels.enqueue("concat",
                               [self.head_cnt_k, B, L1],
                               [1, 1, 1],
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
                cl_kernels.enqueue("MHA",
                                   [self.head_cnt_q, B],
                                   [1, 1],
                                   qkv,
                                   L1,
                                   output,
                                   self.cache_k,
                                   self.cache_v,
                                   attn_score,
                                   L0
                                   )
        return output


if __name__ == "__main__":
    import torch
    from . import utils
    H = 32
    HK = 32
    S = 128
    MAX_KV_LEN = 256
    B = 1
    KV_BLOCK = 32
    ref = MHA(H, HK, S, MAX_KV_LEN, True)
    opt = MHA(H, HK, S, MAX_KV_LEN, False, kv_block=KV_BLOCK)
    first_token = KV_BLOCK - 2
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
        result_ref1 = ref(qkv, attention_mask)
        result_opt1 = opt(qkv, attention_mask)
        result_ref1_cpu = result_ref1.numpy()
        result_opt1_cpu = result_opt1.numpy()
        if not np.allclose(result_ref1_cpu, result_opt1_cpu, atol=0.01, rtol=0.01):
            print(f'{d=}\n{result_ref1_cpu=}\n{result_opt1_cpu=}')
            pos = np.where(np.abs(result_ref1_cpu - result_opt1_cpu) > 0.01)
            print(f'failed at {L0=} {L1=}, shape = {result_opt1_cpu.shape}')
            print(f'pos = {pos}')
            print(f'ref_val = {result_ref1_cpu[pos]}\nopt_val={result_opt1_cpu[pos]}')
            raise("failed.")
    print('done.')
