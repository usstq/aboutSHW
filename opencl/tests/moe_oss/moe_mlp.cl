
#define EACH_EXPERT_WEIGHTS_OFFSET_SIZE 64

#define oss_alpha 1.702
#define oss_limit 7.0
#define oss_min_none -3e38
#define oss_negative_limit -7.0

#define unroll_for __attribute__((opencl_unroll_hint)) for
inline void gemv_n2x(const __global uchar* weight,
                    __global half* scales,
                    __global uchar* zps,
                    __global half* bias,
                    const __global half* x,
                    __global half* y, int N, int K,
                    half* x2,
                    float* xg_sum,
                    const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    //# interleaving x into x2
    half * px = x + id_sg*2*GROUP_SIZE;
    half * px2 = x2 + id_sg*2*GROUP_SIZE;
    for(int i = id_sg; i < HIDDEN_SIZE/(2*GROUP_SIZE); i += num_sg, px += num_sg*2*GROUP_SIZE, px2 += num_sg*2*GROUP_SIZE) {
        //# quantization group
        float2 x_group_sum = 0;
        for(int j = id_local; j < 2 * GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + 2*GROUP_SIZE/2] = odd;
            x_group_sum[j/SUBGROUP_SIZE] += even + odd;
        }
        x_group_sum[0] = sub_group_reduce_add(x_group_sum[0]);
        x_group_sum[1] = sub_group_reduce_add(x_group_sum[1]);
        if (id_local == 0) {
            xg_sum[2*i] = x_group_sum[0] / SUBGROUP_SIZE;
            xg_sum[2*i+1] = x_group_sum[1] / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for(int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        // process 2 groups each iteration
        for(int gk = 0; gk < K / GROUP_SIZE; gk+=2, S += 2 * N, Z += 2 * N / 2) {
            half s00 = S[0];
            half s10 = S[1];
            half s01 = S[N];
            half s11 = S[N+1];
            //ushort z0 = Z[0];
            //ushort z1 = Z[N/2];
            //half z_hf00 = convert_half(z0 & 0xf);
            //half z_hf10 = convert_half(z0 >> 4);
            //half z_hf01 = convert_half(z1 & 0xf);
            //half z_hf11 = convert_half(z1 >> 4);

#if SUBGROUP_SIZE == 16
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s0 = fma(a.s2, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (convert_half(b2.s1 >> 4)), sum1.s1);

            sum_all0 += (sum0.s0 - xg_sum[gk] * (half)8.0) * s00 + (sum0.s1 - xg_sum[gk+1] * (half)8.0) * s01;
            sum_all1 += (sum1.s0 - xg_sum[gk] * (half)8.0) * s10 + (sum1.s1 - xg_sum[gk+1] * (half)8.0) * s11;
#else
            printf("WARNING: SUBGROUP_SIZE != 16, please verify the correctness!\n");
#endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0) + bias[n];
        sum_all1 = sub_group_reduce_add(sum_all1) + bias[n+1];
        if (id_local == 0) {
            if (silu) { // gate
                sum_all0 = clamp(sum_all0, (float)oss_min_none, (float)oss_limit);
                sum_all1 = clamp(sum_all1, (float)oss_min_none, (float)oss_limit);
                y[n] = (y[n]+1) * (sum_all0 / (1 + exp(-sum_all0 * oss_alpha)));
                y[n+1] = (y[n+1]+1) * (sum_all1 / (1 + exp(-sum_all1 * oss_alpha)));
            } else { // up
                y[n] = clamp(sum_all0, (float)oss_negative_limit, (float)oss_limit);
                y[n+1] = clamp(sum_all1, (float)oss_negative_limit, (float)oss_limit);
            }
        }
    }
}

inline void gemv_n4x(const __global uchar* weight,
                    __global half* scales,
                    __global uchar* zps,
                    __global half* bias,
                    const __global half* x,
                    __global half* y, int N, int K,
                    half* x2,
                    float* xg_sum,
                    const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    //# interleaving x into x2
    half * px = x + id_sg*2*GROUP_SIZE;
    half * px2 = x2 + id_sg*2*GROUP_SIZE;
    for(int i = id_sg; i < HIDDEN_SIZE/(2*GROUP_SIZE); i += num_sg, px += num_sg*2*GROUP_SIZE, px2 += num_sg*2*GROUP_SIZE) {
        //# quantization group
        float2 x_group_sum = 0;
        for(int j = id_local; j < 2 * GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + 2*GROUP_SIZE/2] = odd;
            x_group_sum[j/SUBGROUP_SIZE] += even + odd;
        }
        x_group_sum[0] = sub_group_reduce_add(x_group_sum[0]);
        x_group_sum[1] = sub_group_reduce_add(x_group_sum[1]);
        if (id_local == 0) {
            xg_sum[2*i] = x_group_sum[0] / SUBGROUP_SIZE;
            xg_sum[2*i+1] = x_group_sum[1] / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for(int n = n_start; n < n_end; n+=4) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        float sum_all2 = 0;
        float sum_all3 = 0;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        // process 2 groups each iteration
        for(int gk = 0; gk < K / GROUP_SIZE; gk+=2, S += 2 * N, Z += 2 * N / 2) {
            half s00 = S[0];
            half s10 = S[1];
            half s20 = S[2];
            half s30 = S[3];
            half s01 = S[N];
            half s11 = S[N+1];
            half s21 = S[N+2];
            half s31 = S[N+3];
            // ushort z0 = Z[0];
            // ushort z2 = Z[1];
            // ushort z1 = Z[N/2];
            // ushort z3 = Z[N/2+1];

            // half z_hf00 = convert_half(z0 & 0xf);
            // half z_hf10 = convert_half(z0 >> 4);
            // half z_hf01 = convert_half(z1 & 0xf);
            // half z_hf11 = convert_half(z1 >> 4);
            // half z_hf20 = convert_half(z2 & 0xf);
            // half z_hf30 = convert_half(z2 >> 4);
            // half z_hf21 = convert_half(z3 & 0xf);
            // half z_hf31 = convert_half(z3 >> 4);

#if SUBGROUP_SIZE == 16
            half2 sum0;
            half2 sum1;
            half2 sum2;
            half2 sum3;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)B + 1*(K/2) + gk*GROUP_SIZE/2);
            uchar2 b3 = intel_sub_group_block_read_uc2((const __global uchar*)B + 2*(K/2) + gk*GROUP_SIZE/2);
            uchar2 b4 = intel_sub_group_block_read_uc2((const __global uchar*)B + 3*(K/2) + gk*GROUP_SIZE/2);

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s0 = fma(a.s2, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (convert_half(b2.s1 >> 4)), sum1.s1);

            sum2.s0 = fma(a.s0, (convert_half(b3.s0 & 0x0F)), 0);
            sum2.s1 = fma(a.s1, (convert_half(b3.s1 & 0x0F)), 0);
            sum2.s0 = fma(a.s2, (convert_half(b3.s0 >> 4)), sum2.s0);
            sum2.s1 = fma(a.s3, (convert_half(b3.s1 >> 4)), sum2.s1);

            sum3.s0 = fma(a.s0, (convert_half(b4.s0 & 0x0F)), 0);
            sum3.s1 = fma(a.s1, (convert_half(b4.s1 & 0x0F)), 0);
            sum3.s0 = fma(a.s2, (convert_half(b4.s0 >> 4)), sum3.s0);
            sum3.s1 = fma(a.s3, (convert_half(b4.s1 >> 4)), sum3.s1);

            sum_all0 += (sum0.s0 - xg_sum[gk] * (half)8.0) * s00 + (sum0.s1 - xg_sum[gk+1] * (half)8.0) * s01;
            sum_all1 += (sum1.s0 - xg_sum[gk] * (half)8.0) * s10 + (sum1.s1 - xg_sum[gk+1] * (half)8.0) * s11;
            sum_all2 += (sum2.s0 - xg_sum[gk] * (half)8.0) * s20 + (sum2.s1 - xg_sum[gk+1] * (half)8.0) * s21;
            sum_all3 += (sum3.s0 - xg_sum[gk] * (half)8.0) * s30 + (sum3.s1 - xg_sum[gk+1] * (half)8.0) * s31;
#else
            printf("WARNING: SUBGROUP_SIZE != 16, please verify the correctness!\n");
#endif
        }

        sum_all0 = sub_group_reduce_add(sum_all0) + bias[n];
        sum_all1 = sub_group_reduce_add(sum_all1) + bias[n+1];
        sum_all2 = sub_group_reduce_add(sum_all2) + bias[n+2];
        sum_all3 = sub_group_reduce_add(sum_all3) + bias[n+3];
        if (id_local == 0) {
            if (silu) { // gate
                sum_all0 = clamp(sum_all0, (float)oss_min_none, (float)oss_limit);
                sum_all1 = clamp(sum_all1, (float)oss_min_none, (float)oss_limit);
                sum_all2 = clamp(sum_all2, (float)oss_min_none, (float)oss_limit);
                sum_all3 = clamp(sum_all3, (float)oss_min_none, (float)oss_limit);
                y[n] = (y[n]+1) * (sum_all0 / (1 + exp(-sum_all0 * oss_alpha)));
                y[n+1] = (y[n+1]+1) * (sum_all1 / (1 + exp(-sum_all1 * oss_alpha)));
                y[n+2] = (y[n+2]+1) * (sum_all2 / (1 + exp(-sum_all2 * oss_alpha)));
                y[n+3] = (y[n+3]+1) * (sum_all3 / (1 + exp(-sum_all3 * oss_alpha)));
            } else { // up
                y[n] = clamp(sum_all0, (float)oss_negative_limit, (float)oss_limit);
                y[n+1] = clamp(sum_all1, (float)oss_negative_limit, (float)oss_limit);
                y[n+2] = clamp(sum_all2, (float)oss_negative_limit, (float)oss_limit);
                y[n+3] = clamp(sum_all3, (float)oss_negative_limit, (float)oss_limit);
            }
        }
    }
}


__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_gate_up(
    const __global int* expert_list,
    const __global uchar* weight_base_addr,
    const __global uchar* weight_offset,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    y += expert_no * INTERMEDIATE_SIZE;
    __global unsigned int *addr_offset = (__global unsigned int *)(weight_offset + expert_list[expert_no] * EACH_EXPERT_WEIGHTS_OFFSET_SIZE);

    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* gate_weight = (__global uchar*)(weight_base_addr + addr_offset[0]);
    __global half* gate_scale = (__global half*)(weight_base_addr + addr_offset[1]);
    __global uchar* gate_zp = (__global uchar*)(weight_base_addr + addr_offset[2]);
    __global half* gate_bias = (__global half*)(weight_base_addr + addr_offset[3]);

    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* up_weight = (__global uchar*)(weight_base_addr + addr_offset[4]);
    __global half* up_scale = (__global half*)(weight_base_addr + addr_offset[5]);
    __global uchar* up_zp = (__global uchar*)(weight_base_addr + addr_offset[6]);
    __global half* up_bias = (__global half*)(weight_base_addr + addr_offset[7]);

    #define XG_SIZE (HIDDEN_SIZE/GROUP_SIZE)
    __local half x2[HIDDEN_SIZE];
    //__local float xg_sum[HIDDEN_SIZE/32];
    __local float xg_sum[XG_SIZE];
    gemv_n4x(up_weight, up_scale, up_zp, up_bias, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gemv_n4x(gate_weight, gate_scale, gate_zp, gate_bias, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
}

#if 0
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_down(
    const __global int* expert_list,
    const __global uchar* weight_base_addr,
    const __global uchar* weight_offset,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;
    __global unsigned int *addr_offset = (__global unsigned int *)(weight_offset + expert_list[expert_no] * EACH_EXPERT_WEIGHTS_OFFSET_SIZE);

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* weight = (__global uchar*)(weight_base_addr + addr_offset[8]);
    __global half* scales = (__global half*)(weight_base_addr + addr_offset[9]);
    __global uchar* zps = (__global uchar*)(weight_base_addr + addr_offset[10]);
    __global half* bias = (__global half*)(weight_base_addr + addr_offset[11]);

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    #define XG_SIZE (HIDDEN_SIZE/GROUP_SIZE)
    __local half x2[INTERMEDIATE_SIZE];
    //__local float xg_sum[INTERMEDIATE_SIZE/32];
    __local float xg_sum[XG_SIZE];


    //# interleaving x into x2
    __global half * px = x + id_sg*2*GROUP_SIZE;
    __local half * px2 = x2 + id_sg*2*GROUP_SIZE;
    for(int i = id_sg; i < INTERMEDIATE_SIZE/(2*GROUP_SIZE); i += num_sg, px += num_sg*2*GROUP_SIZE, px2 += num_sg*2*GROUP_SIZE) {
        //# quantization group
        float2 x_group_sum = 0;
        for(int j = id_local; j < 2*GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + 2*GROUP_SIZE/2] = odd;
            x_group_sum[j/SUBGROUP_SIZE] += even + odd;
        }
        x_group_sum[0] = sub_group_reduce_add(x_group_sum[0]);
        x_group_sum[1] = sub_group_reduce_add(x_group_sum[1]);
        if (id_local == 0) {
            xg_sum[2*i] = x_group_sum[0] / SUBGROUP_SIZE;
            xg_sum[2*i+1] = x_group_sum[1] / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //# global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for(int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        // process 2 groups each iteration
        for(int gk = 0; gk < K / GROUP_SIZE; gk+=2, S += 2 * N, Z += 2 * N / 2) {
            half s00 = S[0];
            half s10 = S[1];
            half s01 = S[N];
            half s11 = S[N+1];
            //ushort z0 = Z[0];
            //ushort z1 = Z[N/2];
            //half z_hf00 = convert_half(z0 & 0xf);
            //half z_hf10 = convert_half(z0 >> 4);
            //half z_hf01 = convert_half(z1 & 0xf);
            //half z_hf11 = convert_half(z1 >> 4);

#if SUBGROUP_SIZE == 16
            half2 sum0;
            half2 sum1;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s0 = fma(a.s2, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (convert_half(b2.s1 >> 4)), sum1.s1);

            sum_all0 += (sum0.s0 - xg_sum[gk] * (half)8.0) * s00 + (sum0.s1 - xg_sum[gk+1] * (half)8.0) * s01;
            sum_all1 += (sum1.s0 - xg_sum[gk] * (half)8.0) * s10 + (sum1.s1 - xg_sum[gk+1] * (half)8.0) * s11;
#else
            printf("WARNING: SUBGROUP_SIZE != 16, please verify the correctness!\n");
#endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0) + bias[n];
        sum_all1 = sub_group_reduce_add(sum_all1) + bias[n+1];
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n+1] = sum_all1 * routing_weights[expert_no];
        }
    }
}
#else
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_down(
    const __global int* expert_list,
    const __global uchar* weight_base_addr,
    const __global uchar* weight_offset,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;
    __global unsigned int *addr_offset = (__global unsigned int *)(weight_offset + expert_list[expert_no] * EACH_EXPERT_WEIGHTS_OFFSET_SIZE);

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* weight = (__global uchar*)(weight_base_addr + addr_offset[8]);
    __global half* scales = (__global half*)(weight_base_addr + addr_offset[9]);
    __global uchar* zps = (__global uchar*)(weight_base_addr + addr_offset[10]);
    __global half* bias = (__global half*)(weight_base_addr + addr_offset[11]);

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    #define XG_SIZE (HIDDEN_SIZE/GROUP_SIZE)
    __local half x2[INTERMEDIATE_SIZE];
    //__local float xg_sum[INTERMEDIATE_SIZE/32];
    __local float xg_sum[XG_SIZE];


    //# interleaving x into x2
    __global half * px = x + id_sg*2*GROUP_SIZE;
    __local half * px2 = x2 + id_sg*2*GROUP_SIZE;
    for(int i = id_sg; i < INTERMEDIATE_SIZE/(2*GROUP_SIZE); i += num_sg, px += num_sg*2*GROUP_SIZE, px2 += num_sg*2*GROUP_SIZE) {
        //# quantization group
        float2 x_group_sum = 0;
        for(int j = id_local; j < 2*GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + 2*GROUP_SIZE/2] = odd;
            x_group_sum[j/SUBGROUP_SIZE] += even + odd;
        }
        x_group_sum[0] = sub_group_reduce_add(x_group_sum[0]);
        x_group_sum[1] = sub_group_reduce_add(x_group_sum[1]);
        if (id_local == 0) {
            xg_sum[2*i] = x_group_sum[0] / SUBGROUP_SIZE;
            xg_sum[2*i+1] = x_group_sum[1] / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //# global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for(int n = n_start; n < n_end; n += 4) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        float sum_all2 = 0;
        float sum_all3 = 0;
        // process 2 groups each iteration
        for(int gk = 0; gk < K / GROUP_SIZE; gk += 2, S += 2 * N, Z += 2 * N / 2) {
            half s00 = S[0];
            half s10 = S[1];
            half s20 = S[2];
            half s30 = S[3];
            half s01 = S[N];
            half s11 = S[N+1];
            half s21 = S[N+2];
            half s31 = S[N+3];
            // ushort z0 = Z[0];
            // ushort z2 = Z[1];
            // ushort z1 = Z[N/2];
            // ushort z3 = Z[N/2+1];

            // half z_hf00 = convert_half(z0 & 0xf);
            // half z_hf10 = convert_half(z0 >> 4);
            // half z_hf01 = convert_half(z1 & 0xf);
            // half z_hf11 = convert_half(z1 >> 4);
            // half z_hf20 = convert_half(z2 & 0xf);
            // half z_hf30 = convert_half(z2 >> 4);
            // half z_hf21 = convert_half(z3 & 0xf);
            // half z_hf31 = convert_half(z3 >> 4);

#if SUBGROUP_SIZE == 16
            half2 sum0;
            half2 sum1;
            half2 sum2;
            half2 sum3;
            half4 a = as_half4(intel_sub_group_block_read_us4((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar2 b = intel_sub_group_block_read_uc2((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar2 b2 = intel_sub_group_block_read_uc2((const __global uchar*)B + 1*(K/2) + gk*GROUP_SIZE/2);
            uchar2 b3 = intel_sub_group_block_read_uc2((const __global uchar*)B + 2*(K/2) + gk*GROUP_SIZE/2);
            uchar2 b4 = intel_sub_group_block_read_uc2((const __global uchar*)B + 3*(K/2) + gk*GROUP_SIZE/2);

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s0 = fma(a.s2, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s3, (convert_half(b2.s1 >> 4)), sum1.s1);

            sum2.s0 = fma(a.s0, (convert_half(b3.s0 & 0x0F)), 0);
            sum2.s1 = fma(a.s1, (convert_half(b3.s1 & 0x0F)), 0);
            sum2.s0 = fma(a.s2, (convert_half(b3.s0 >> 4)), sum2.s0);
            sum2.s1 = fma(a.s3, (convert_half(b3.s1 >> 4)), sum2.s1);

            sum3.s0 = fma(a.s0, (convert_half(b4.s0 & 0x0F)), 0);
            sum3.s1 = fma(a.s1, (convert_half(b4.s1 & 0x0F)), 0);
            sum3.s0 = fma(a.s2, (convert_half(b4.s0 >> 4)), sum3.s0);
            sum3.s1 = fma(a.s3, (convert_half(b4.s1 >> 4)), sum3.s1);

            sum_all0 += (sum0.s0 - xg_sum[gk] * (half)8.0) * s00 + (sum0.s1 - xg_sum[gk+1] * (half)8.0) * s01;
            sum_all1 += (sum1.s0 - xg_sum[gk] * (half)8.0) * s10 + (sum1.s1 - xg_sum[gk+1] * (half)8.0) * s11;
            sum_all2 += (sum2.s0 - xg_sum[gk] * (half)8.0) * s20 + (sum2.s1 - xg_sum[gk+1] * (half)8.0) * s21;
            sum_all3 += (sum3.s0 - xg_sum[gk] * (half)8.0) * s30 + (sum3.s1 - xg_sum[gk+1] * (half)8.0) * s31;
#else
            printf("WARNING: SUBGROUP_SIZE != 16, please verify the correctness!\n");
#endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0) + bias[n];
        sum_all1 = sub_group_reduce_add(sum_all1) + bias[n+1];
        sum_all2 = sub_group_reduce_add(sum_all2) + bias[n+2];
        sum_all3 = sub_group_reduce_add(sum_all3) + bias[n+3];
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n+1] = sum_all1 * routing_weights[expert_no];
            y[n+2] = sum_all2 * routing_weights[expert_no];
            y[n+3] = sum_all3 * routing_weights[expert_no];
        }
    }
}
#endif

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_reduce(const __global TYPE* x,                // [MAX_TOPK, HIDDEN_SIZE]
    __global TYPE* y) {                                    // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    half sum[MAX_TOPK] = {0};
    __attribute__((opencl_unroll_hint(MAX_TOPK)))
    unroll_for (int i = 0; i < MAX_TOPK; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + i*HIDDEN_SIZE + n)));
    }
    unroll_for (int i = 1; i < MAX_TOPK; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((const __global ushort*)(y + n), as_ushort(sum[0]));
}

