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

__kernel void concat(__global half * param_qkv,         // [B, L1, (Hq + Hk + Hv) * S)]
                    int L1,                             // input seq-length
                    __global half * param_cache_k,     // [B, Hk, max_kv_len, S]
                    __global half * param_cache_v,     // [B, Hk, max_kv_len, S]
                    int L0
                    ) {
    // global: [hkv, B, L1 * CONCAT_BLOCK],
    // local: [1, 1, CONCAT_BLOCK],
    int hkv = get_global_id(0);
    int b = get_global_id(1);
    int l = get_group_id(2);
    int id_local = get_local_id(2);

    int HQKV = HQ + HK + HK;

    // concat k & v into cache
    __global half * dst_k = param_cache_k + 
        ((b * HK + hkv) * MAX_KV_LEN + L0 + l) * S;
    __global half * dst_v = param_cache_v +
        ((b * HK + hkv) * MAX_KV_LEN + L0 + l) * S;

    __global half * cur_k = param_qkv + ((b * L1 + l) * HQKV + HQ + hkv) * S;
    __global half * cur_v = cur_k + HK * S;
    __attribute__((opencl_unroll_hint(16)))
    for(int i = id_local; i < S; i += CONCAT_BLOCK) {
        dst_k[i] = cur_k[i];
        dst_v[i] = cur_v[i];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void MHAFirst(__global half * param_qkv,         // [B, L1, (HQ + HK + HK) * S)]
                       __global half * param_output,      // [B, L1, HQ * S]
                       uint L1                            // input seq-length
                       ) {
    // global: [B, HQ, S * mb_blocks_num]
    // local: [1, 1, S]
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int mb_blocks_num = get_num_groups(2);
    const int cur_mb_blocks_num = get_group_id(2);
    const int id_sg = get_sub_group_id();
    const int id_sg_local = get_sub_group_local_id();
    const int id_local = get_local_id(2);

    const int HQKV = HQ + HK + HK;
    const int head_group_size = HQ / HK;
    const int hkv = h / head_group_size;
    const float head_size_scaler = 1.0f / sqrt(convert_float(S));
    const int query_len = cur_mb_blocks_num + 1 < mb_blocks_num ? SGS : L1 - cur_mb_blocks_num * SGS;  // rows for A to be handled
    const int query_start = cur_mb_blocks_num * SGS;
    const int query_end = query_start + query_len;
    const int kv_block = S / SGS * SGS;    // the number of kv rows the outer loop will be processed: sg * items number processed in sg
    const int qkv_stride = HQKV * S;

    __global half* cur_q = param_qkv + ((b * L1 + query_start) * HQKV + h) * S;
    __global half* cur_k = param_qkv + (b * L1 * HQKV + HQ + hkv) * S;
    __global half* cur_v = cur_k + HK * S;
    __global half* output = param_output + ((b * L1 + query_start) * HQ + h) * S;

    // store prev iteration state
    __local half prev_output_share[SGS][S];
    __local half prev_max_attn_score_share[SGS];
    __local half prev_exp_sum_share[SGS];

    __local half max_qk_dot_share[SGS][S / SGS];
    __local half qk_dot_share[SGS][kv_block];

    // m, n, k corresponding to loop index for M, N, K, A(query) shape: [M, K], B(key, value) shape: [N, K]
    // 1 cache query
    __local half q_share[S][SGS];                       // transpose query, prepare for dot(k, q)
    if (query_len == SGS) {
        __attribute__((opencl_unroll_hint))
        for (int m = 0; m < SGS; m++) {
            half q_val = as_half(intel_sub_group_block_read_us((const __global ushort*)cur_q + m * qkv_stride + id_sg * SGS));
            q_share[id_local][m] = q_val;
        }
    } else {
        for (int m = 0; m < SGS; m++) {
            if (m < query_len) {
                half q_val = as_half(intel_sub_group_block_read_us((const __global ushort*)cur_q + m * qkv_stride + id_sg * SGS));
                q_share[id_local][m] = q_val;
            } else {
                // avoid q*k to check if M is varying
                q_share[id_local][m] = 0;
            }
        }
    }
    if (id_local < SGS) {
        prev_max_attn_score_share[id_local] = -1e9f;
        prev_exp_sum_share[id_local] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // loop in kv_len, each sg will compute A([16, K]) * B([16, K]')
    uint kv_block_num = (L1 + kv_block - 1) / kv_block;
    for (int i = 0, key_n_start = 0; i < kv_block_num; i++, key_n_start += kv_block) {
        // 2 compute q*k, each sg will compute: dot(A([16, K]), B([L1 / kv_block, 16, K])) = C([L1 / kv_block, 16, 16])
        // each sg will process SGS key lines
        const int key_len_in_kv_block = min(kv_block, max(0, query_end - key_n_start));
        const int key_n_start_sg = key_n_start + id_sg * SGS;
        half16 dot_prod = 0;
        if (key_n_start_sg <= query_start) {
            const uint key_len_in_sg = min(key_n_start_sg + SGS, query_end) - key_n_start_sg;
            // loop whole token, 16 items each due to key can be loaded 16 items at a time by using one SIMD read
            if (key_len_in_sg == SGS) {
                for (uint k = 0; k < S; k += SGS) {
                    half query_val[SGS];
                    __attribute__((opencl_unroll_hint))
                    for (uint k_sub = 0; k_sub < SGS; k_sub++) {
                        query_val[k_sub] = q_share[k + k_sub][id_sg_local];
                        // will be slower
                        //query_val[k_sub] = as_half(intel_sub_group_block_read_us((const __local ushort*)q_share[k + k_sub]));
                    }
                    // loop key, read one key(16 items), get 16 results of dot(key, query_val[0-15]). Full reuse key_val due to it's in DDR.
                    __attribute__((opencl_unroll_hint))
                    for (uint n = 0; n < SGS; n++) {
                        ushort key_val = intel_sub_group_block_read_us((const __global ushort*)cur_k + (key_n_start_sg + n) * qkv_stride + k);
                        // accumulation along key direction
                        __attribute__((opencl_unroll_hint))
                        for (uint k_sub = 0; k_sub < SGS; k_sub++) {
                            dot_prod[n] = fma(query_val[k_sub], as_half(intel_sub_group_broadcast(key_val, k_sub)), dot_prod[n]);
                        }
                    }
                }
                half max_attn_score_in_wg = -1e9f;
                // 3 compute: dot_prod *= head_size_scaler; if (max_attn_score < dot_prod) max_attn_score = dot_prod
                if (key_n_start_sg + SGS <= query_start) {
                    __attribute__((opencl_unroll_hint))
                    for (uint n = 0; n < SGS; n++) {
                        dot_prod[n] *= head_size_scaler;
                        max_attn_score_in_wg = fmax(max_attn_score_in_wg, dot_prod[n]);
                    }
                } else {
                    __attribute__((opencl_unroll_hint))
                    for (uint n = 0; n < SGS; n++) {
                        dot_prod[n] *= head_size_scaler;
                        // casual mask: valid only if query <= kv_len
                        if (key_n_start_sg + n <= query_start + id_sg_local)
                            max_attn_score_in_wg = fmax(max_attn_score_in_wg, dot_prod[n]);
                        else
                            dot_prod[n] = -1e9f;
                    }
                }
                // change layout to M*key_len_in_sg
                max_qk_dot_share[id_sg_local][id_sg] = max_attn_score_in_wg;
                for (uint n = 0; n < SGS; n++) {
                    qk_dot_share[id_sg_local][id_sg * SGS + n] = dot_prod[n];
                }
            } else {
                for (uint k = 0; k < S; k += SGS) {
                    half query_val[SGS];
                    __attribute__((opencl_unroll_hint))
                    for (uint k_sub = 0; k_sub < SGS; k_sub++) {
                        query_val[k_sub] = q_share[k + k_sub][id_sg_local];
                        //query_val[k_sub] = as_half(intel_sub_group_block_read_us((const __local ushort*)q_share[k + k_sub]));
                    }
                    // loop key, read one key(16 items), get 16 results of dot(key, query_val[0-15]). Full reuse key_val due to it's in DDR.
                    for (uint n = 0; n < key_len_in_sg; n++) {
                        ushort key_val = intel_sub_group_block_read_us((const __global ushort*)cur_k + (key_n_start_sg + n) * qkv_stride + k);
                        // accumulation along key direction
                        __attribute__((opencl_unroll_hint))
                        for (uint k_sub = 0; k_sub < SGS; k_sub++) {
                            dot_prod[n] = fma(query_val[k_sub], as_half(intel_sub_group_broadcast(key_val, k_sub)), dot_prod[n]);
                        }
                    }
                }
                half max_attn_score_in_wg = -1e9f;
                // 3 compute: dot_prod *= head_size_scaler; if (max_attn_score < dot_prod) max_attn_score = dot_prod
                // some var layouts in the following steps:
                // assume A([M=3, S]) * B([N=3, S]), the result(dot_prod[0-15]):
                //     M0 M1 M2 M3 M4...M14 M15   apply causal->      M0   M1  M2  M3 M4...M14 M15
                //  N0 v  v  v  0  0     0  0                     N0  v    v   v   0  0     0  0
                //  N1 v  v  v  0  0     0  0                     N1 -inf  v   v   0  0     0  0 
                //  N2 v  v  v  0  0     0  0                     N2 -inf -inf v   0  0     0  0
                //  N3 ?  ?  ?  ?  ?     ?  ?                     N3  ?    ?   ?   ?  ?     ?  ?
                //  ...                                           ...
                // N14 ?  ?  ?  ?  ?     ?  ?                     N14 ?    ?   ?   ?  ?     ?  ?
                // N15 ?  ?  ?  ?  ?     ?  ?                     N15 ?    ?   ?   ?  ?     ?  ?
                //
                // max_attn_score_in_wg:
                // M0   M1  M2  M3 M4...M14 M15
                // v    v   v   0  0 ... 0  0
                //
                // max_qk_dot_share[M][sg_num]:
                //     sg0
                // M0  v
                // M1  v
                // M2  v
                // M3  0
                // ...
                // M15 0
                //
                // qk_dot_share[M][kv_len](before softmax:):
                //     N0   N1  N2  N3 N4...N14 N15
                // M0  v  -inf -inf ?  ?     ?  ?
                // M1  v    v  -inf ?  ?     ?  ? 
                // M2  v    v   v   ?  ?     ?  ?
                // M3  0    0   0   ?  ?     ?  ?
                // ...
                // M14 0    0   0   ?  ?     ?  ?
                // M15 0    0   0   ?  ?     ?  ?
                //
                // qk_dot_share[M][kv_len](after softmax+scale):
                //     N0   N1  N2  N3 N4...N14 N15
                // M0  v'   0   0   ?  ?     ?  ?
                // M1  v'   v'  0   ?  ?     ?  ? 
                // M2  v'   v'  v'  ?  ?     ?  ?
                // M3  0    0   0   ?  ?     ?  ?
                // ...
                // M14 0    0   0   ?  ?     ?  ?
                // M15 0    0   0   ?  ?     ?  ?
                for (uint n = 0; n < key_len_in_sg; n++) {
                    dot_prod[n] *= head_size_scaler;
                    // casual mask: valid only if query <= kv_len
                    if (key_n_start_sg + n <= query_start + id_sg_local)
                        max_attn_score_in_wg = fmax(max_attn_score_in_wg, dot_prod[n]);
                    else
                        dot_prod[n] = -1e9f;
                }
                // change layout to M*key_len_in_sg
                max_qk_dot_share[id_sg_local][id_sg] = max_attn_score_in_wg;
                for (uint n = 0; n < key_len_in_sg; n++) {
                    qk_dot_share[id_sg_local][id_sg * SGS + n] = dot_prod[n];
                }
            }
        } else {
            max_qk_dot_share[id_sg_local][id_sg] = -1e9f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // each sg will compute a whole row of query
        for (uint m = id_sg; m < query_len; m += S / SGS) {
            // reduce across sg
            float max_qk_dot;
            if (id_sg_local < S / SGS) {
                max_qk_dot = max_qk_dot_share[m][id_sg_local];
            } else {
                max_qk_dot = -1e9f;
            }
            max_qk_dot = sub_group_reduce_max(max_qk_dot);
            float prev_max_attn_score = prev_max_attn_score_share[m];
            max_qk_dot = fmax(max_qk_dot, prev_max_attn_score);

            // 4 softmax
            float exp_sum = 0;
            for (uint k = id_sg_local; k < key_len_in_kv_block; k += SGS) {
                float a = native_exp(qk_dot_share[m][k] - max_qk_dot);
                qk_dot_share[m][k] = a;
                exp_sum += a;
            }
            // reduce in sg
            exp_sum = sub_group_reduce_add(exp_sum);

            // compensation
            float pre_exp_sum = prev_exp_sum_share[m];
            float pre_exp_sum_fixed = pre_exp_sum * native_exp(prev_max_attn_score - max_qk_dot);
            exp_sum += pre_exp_sum_fixed;
            float scale_fixed = pre_exp_sum_fixed / exp_sum;
            float scale = 1.0f / exp_sum;
            for (uint k = id_sg_local; k < key_len_in_kv_block; k += SGS) {
                qk_dot_share[m][k] = convert_half(qk_dot_share[m][k] * scale);
            }
            for (uint k = id_sg_local; k < S; k += SGS) {
                prev_output_share[m][k] = convert_half(prev_output_share[m][k] * scale_fixed);
            }

            prev_max_attn_score_share[m] = max_qk_dot;
            prev_exp_sum_share[m] = exp_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 5 w*v
        uint value_len_in_kv_block = key_len_in_kv_block;
        half16 sum = 0.0h;
        uint k = 0;
        for (; k + SGS <= value_len_in_kv_block; k += SGS) {
            const uint v_row = key_n_start + k;
            half16 cur_weights;
            __attribute__((opencl_unroll_hint))
            for (uint m = 0; m < SGS; m++) {
                //cur_weights[m] = qk_dot_share[m][k + id_sg_local];
                cur_weights[m] = as_half(intel_sub_group_block_read_us((const __local ushort*)qk_dot_share + m * kv_block + k));
            }
            __attribute__((opencl_unroll_hint))
            for (uint k_sub = 0; k_sub < SGS; k_sub++) {
                half v_val = as_half(intel_sub_group_block_read_us((const __global ushort*)cur_v + (v_row + k_sub) * qkv_stride + id_sg * SGS));
                __attribute__((opencl_unroll_hint))
                for (uint m = 0; m < SGS; m++) {
                    sum[m] = fma(v_val, sub_group_broadcast(cur_weights[m], k_sub), sum[m]);
                }
            }
        }
        if (k < value_len_in_kv_block) {
            const uint v_row = key_n_start + k;
            half16 cur_weights;
            __attribute__((opencl_unroll_hint))
            for (uint m = 0; m < SGS; m++) {
                //cur_weights[m] = qk_dot_share[m][k + id_sg_local];
                cur_weights[m] = as_half(intel_sub_group_block_read_us((const __local ushort*)qk_dot_share + m * kv_block + k));
            }
            for (uint k_sub = 0; k_sub < value_len_in_kv_block - k; k_sub++) {
                half v_val = as_half(intel_sub_group_block_read_us((const __global ushort*)cur_v + (v_row + k_sub) * qkv_stride + id_sg * SGS));
                __attribute__((opencl_unroll_hint))
                for (uint m = 0; m < SGS; m++) {
                    sum[m] = fma(v_val, sub_group_broadcast(cur_weights[m], k_sub), sum[m]);
                }
            }
        }
        for (uint m = 0; m < query_len; m++) {
            sum[m] += as_half(intel_sub_group_block_read_us((const __local ushort*)prev_output_share + m * S + id_sg * SGS));
            intel_sub_group_block_write_us((const __local ushort*)prev_output_share + m * S + id_sg * SGS, as_short(sum[m]));
        }
        // 6 output
        if (i == kv_block_num - 1) {
            for (uint m = 0; m < query_len; m++) {
                __global half* dst = output + m * HQ * S + id_sg * SGS;
                intel_sub_group_block_write_us((__global ushort*)dst, as_short(sum[m]));
            }
        }

        // protect slm_qk_vals as it is read in w*v stage and write in next round q*k stage.
        barrier(CLK_LOCAL_MEM_FENCE);
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
    // global: [B, HQ, S * part_num]
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
    __local half qk_dot[KV_BLOCK];
    for (int i = id_sg; i < cur_kv_len; i += S / SGS) {
        __global half* pk = cache_k + i * S;
        half sum = 0;
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < S; j += SGS) {
            // intel_sub_group_block_read_us cannot take __local pointer according to:
            // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html
            half q_val = as_half(intel_sub_group_block_read_us((const __local ushort*)query + j));
            //half q_val = query[j + id_sg_local];
            half k_val = as_half(intel_sub_group_block_read_us((const __global ushort*)pk + j));
            sum = fma(q_val, k_val, sum);
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
    __local half max_qk_dot_share[S / SGS];
    half max_qk_dot = -1e9f;
    if (cur_kv_len == KV_BLOCK) {
        __attribute__((opencl_unroll_hint))
        for (int i = id_local; i < KV_BLOCK; i += S) {
            max_qk_dot = fmax(max_qk_dot, qk_dot[i]);
        }
    } else {
        for (int i = id_local; i < cur_kv_len; i += S) {
            max_qk_dot = fmax(max_qk_dot, qk_dot[i]);
        }
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
    half exp_sum = 0;
    if (cur_kv_len == KV_BLOCK) {
        __attribute__((opencl_unroll_hint))
        for (int i = id_local; i < KV_BLOCK; i += S) {
            half exp_val = native_exp(qk_dot[i] - max_qk_dot);
            exp_sum += exp_val;
            qk_dot[i] = exp_val;
        }
    } else {
        for (int i = id_local; i < cur_kv_len; i += S) {
            half exp_val = native_exp(qk_dot[i] - max_qk_dot);
            exp_sum += exp_val;
            qk_dot[i] = exp_val;
        }
    }
    // reduce in sg
    __local half exp_sum_share[S / SGS];
    exp_sum = sub_group_reduce_add(exp_sum);
    if (id_sg_local == 0) {
        exp_sum_share[id_sg] = exp_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // reduce across sg
    exp_sum = 0;
    if (id_sg_local < S / SGS) {
        exp_sum = exp_sum_share[id_sg_local];
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
    if (cur_kv_len == KV_BLOCK) {
        __attribute__((opencl_unroll_hint))
        for (int i = id_local; i < KV_BLOCK; i += S) {
            weights[i] = convert_half(qk_dot[i] * scale);
        }
    } else {
        for (int i = id_local; i < cur_kv_len; i += S) {
            weights[i] = convert_half(qk_dot[i] * scale);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4 compute W*V(one token) dot product in each wg, SGS tokens at the same time in wg
    half sum = 0;
    for (int i = 0; i < cur_kv_len; i += SGS) {
        // intel_sub_group_block_read_us cannot take __local pointer according to:
        // https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroups_short.html
        //ushort weight = intel_sub_group_block_read_us((const __local ushort*)weights + i);
        half weight = weights[i + id_sg_local];
        __global half* pv = cache_v + i * S + id_sg * SGS;
        if (i + SGS <= cur_kv_len) {
            __attribute__((opencl_unroll_hint))
            for (int j = 0; j < SGS; j++) {
                half v_val = as_half(intel_sub_group_block_read_us((const __global ushort*)pv + j * S));
                half w = as_half(intel_sub_group_broadcast(as_ushort(weight), j));
                sum += w * v_val;
            }
        } else {
            for (int j = 0; j < cur_kv_len - i; j++) {
                half v_val = as_half(intel_sub_group_block_read_us((const __global ushort*)pv + j * S));
                half w = as_half(intel_sub_group_broadcast(as_ushort(weight), j));
                sum += w * v_val;
            }
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
