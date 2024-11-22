__attribute__((intel_reqd_sub_group_size(16)))
__kernel void GQAFirst(const __global half * param_qkv,         // [B, L1, (HQ + HK + HK) * S)]
                       const __global half * param_output,      // [B, L1, HQ * S]
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
        half16 sum;
        for (uint m = 0; m < query_len; m++) {
            sum[m] = as_half(intel_sub_group_block_read_us((const __local ushort*)prev_output_share + m * S + id_sg * SGS));
        }
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
            intel_sub_group_block_write_us((const __local ushort*)prev_output_share + m * S + id_sg * SGS, as_short(sum[m]));
        }
        // 6 output
        if (i == kv_block_num - 1) {
            for (uint m = 0; m < query_len; m++) {
                __global half* dst = output + m * HQ * S + id_sg * SGS;
                intel_sub_group_block_write_us((__global ushort*)dst, as_short(sum[m]));
            }
        }
    }
}