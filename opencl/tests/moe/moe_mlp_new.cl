
#if 0
// mlp common
#define N_BLOCK      4
#define SUBGROUP_NUM 8
MAX_TOPK
MAX_TOPK
HIDDEN_SIZE
INTERMEDIATE_SIZE
N_BLOCK
SUBGROUP_SIZE  16/32
SUBGROUP_NUM
GROUP_SIZE
TYPE
TYPE_SIZE
#endif

//#define MAX_TOPK 8
//#define HIDDEN_SIZE 2048
//#define INTERMEDIATE_SIZE 768
//#define GROUP_SIZE 128
//#define N_BLOCK_SIZE 64
//#define SUBGROUP_SIZE 32

#define TYPE half
#define TYPE_SIZE 2

// 4bytes for each offset, there are 4 offset for each expert
#define EACH_EXPERT_WEIGHTS_OFFSET_SIZE 16

//#if GATE_UP_ENABLE
inline void gemv_n2x(const __global uchar* weight,
                    __global half* scales,
                    __global uchar* zps,
                    const __global half* x,
                    __global half* y, int N, int K,
                    half* x2,
                    float* xg_sum,
                    const bool silu) {
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    //# interleaving x into x2
    half * px = x + id_sg*GROUP_SIZE;
    half * px2 = x2 + id_sg*GROUP_SIZE;
    for(int i = id_sg; i < HIDDEN_SIZE/GROUP_SIZE; i += num_sg, px += num_sg*GROUP_SIZE, px2 += num_sg*GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        for(int j = id_local; j < GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + GROUP_SIZE/2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //  if(silu == false /*&& get_global_id(0)==0 && get_global_id(1)==0 && get_global_id(2)==0*/) {
    //     printf("x2 = [%f, %f, %f, %f, %f, %f, %f, %f,%f, %f, %f, %f,%f, %f, %f, %f]\n", 
    //         x2[0], x2[1], x2[2], x2[3], x2[4], x2[5], x2[6], x2[7],x2[8], x2[9], x2[10], x2[11], x2[12], x2[13], x2[14], x2[15]);
    //  }
    //  barrier(CLK_LOCAL_MEM_FENCE);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for (int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
#if SZ_LAYOUT == 0
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half s0 = S[0];
            half s1 = S[1];
            ushort z = Z[0];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);
#else
        __global half* S = scales + n*K/GROUP_SIZE;
        __global uchar* Z = zps + n*K/GROUP_SIZE/2;
        
        half scale_values = as_half(intel_sub_group_block_read_us((const __global ushort*)S));
        uchar zp_values = intel_sub_group_block_read_uc((const __global uchar*)Z);
        half zp_even = convert_half(zp_values & 0xF);
        half zp_odd = convert_half(zp_values >> 4);

        for (int gk = 0; gk < K / GROUP_SIZE; gk++) {
            half s0 = sub_group_broadcast(scale_values, 2*gk + 0);
            half s1 = sub_group_broadcast(scale_values, 2*gk + 1);
            half z_hf0 = sub_group_broadcast(zp_even, gk);
            half z_hf1 = sub_group_broadcast(zp_odd, gk);
#endif

#if SUBGROUP_SIZE == 32
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

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s2 = fma(a.s2, (convert_half(b.s2 & 0x0F)), 0);
            sum0.s3 = fma(a.s3, (convert_half(b.s3 & 0x0F)), 0);

            sum0.s0 = fma(a.s4, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s5, (convert_half(b.s1 >> 4)), sum0.s1);
            sum0.s2 = fma(a.s6, (convert_half(b.s2 >> 4)), sum0.s2);
            sum0.s3 = fma(a.s7, (convert_half(b.s3 >> 4)), sum0.s3);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s2 = fma(a.s2, (convert_half(b2.s2 & 0x0F)), 0);
            sum1.s3 = fma(a.s3, (convert_half(b2.s3 & 0x0F)), 0);

            sum1.s0 = fma(a.s4, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s5, (convert_half(b2.s1 >> 4)), sum1.s1);
            sum1.s2 = fma(a.s6, (convert_half(b2.s2 >> 4)), sum1.s2);
            sum1.s3 = fma(a.s7, (convert_half(b2.s3 >> 4)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#endif
            // if(silu == false /*&& get_global_id(0)==0 && get_global_id(1)==0 && get_global_id(2)==0*/) {
            //     printf("[%d,%d,%d]: n =%d, gk = %d,a = [%f,%f,%f,%f]\n", get_global_id(0), get_global_id(1), get_global_id(2),n, gk, a.s0, a.s1, a.s2, a.s3);
            //     // printf("b = [%d,%d], b2=[%d,%d]\n", b.s0, b.s1, b2.s0, b2.s1);
            //     // printf("sum0[0] = %f, sum0[1] = %f, sum0[2] = %f, sum0[3] = %f\n", sum0[0], sum0[1], sum0[2], sum0[3]);
            //     // printf("sum1[0] = %f, sum1[1] = %f, sum1[2] = %f, sum1[3] = %f\n", sum1[0], sum1[1], sum1[2], sum1[3]);
            //     printf("n =%d, gk = %d, sum_all0 = %f, sum_all1 = %f, xg_sum[gk] = %f, z_hf0 = %f, z_hf1 = %f, s0 = %f, s1 = %f\n",
            //         n, gk, sum_all0, sum_all1, xg_sum[gk], z_hf0, z_hf1, s0, s1);
            // }
        }

        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            if (silu) {
                y[n] *= sum_all0 / (1 + exp(-sum_all0));
                y[n+1] *= sum_all1 / (1 + exp(-sum_all1));
            } else {
                y[n] = sum_all0;
                y[n+1] = sum_all1;
            }
        }
        // if(silu == false && id_local == 0) {
        //     printf("[%d,%d,%d]: y[%d] = %f, y[%d] = %f\n", get_global_id(0), get_global_id(1), get_global_id(2),
        //     n + get_global_id(0) * INTERMEDIATE_SIZE, y[n], n + get_global_id(0) * INTERMEDIATE_SIZE + 1, y[n+1]);
        // }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_gate_up(
    const __global int* expert_list,
    const __global uchar* gate_weight_base_addr,
    const __global uchar* gate_weight_offset,
    const __global uchar* up_weight_base_addr,
    const __global uchar* up_weight_offset,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    y += expert_no * INTERMEDIATE_SIZE;

    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global unsigned int *gate_addr_offset = (__global unsigned int *)(gate_weight_offset + expert_list[expert_no] * EACH_EXPERT_WEIGHTS_OFFSET_SIZE);
    __global uchar* gate_weight = (__global uchar*)(gate_weight_base_addr + gate_addr_offset[0]);
    __global half* gate_scale = (__global half*)(gate_weight_base_addr + gate_addr_offset[1]);
    __global uchar* gate_zp = (__global uchar*)(gate_weight_base_addr + gate_addr_offset[2]);

    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global unsigned int *up_addr_offset = (__global unsigned int *)(up_weight_offset + expert_list[expert_no] * EACH_EXPERT_WEIGHTS_OFFSET_SIZE);
    __global uchar* up_weight = (__global uchar*)(up_weight_base_addr + up_addr_offset[0]);
    __global half* up_scale = (__global half*)(up_weight_base_addr + up_addr_offset[1]);
    __global uchar* up_zp = (__global uchar*)(up_weight_base_addr + up_addr_offset[2]);

    // if(expert_no==0 && get_global_id(1)==0 && get_global_id(2)==0) {
    //     printf("mlp_gate_up: topk_id = %d, expert_id = %d, gate_wei_off = %d, gate_scale_off = %d, gate_zp_off = %d, up_wei_off = %d, up_scale_off = %d, up_zp_off = %d\n",
    //             expert_no, expert_list[expert_no], addr_offset[0], addr_offset[1], addr_offset[2], addr_offset[3], addr_offset[4], addr_offset[5]);
        
    //     printf("x = [ %f, %f, %f, %f, %f, %f, %f, %f,%f, %f, %f, %f,%f, %f, %f, %f]\n",x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
    //     printf("up_weight = [ %d, %d, %d, %d, %d, %d, %d, %d,%d, %d, %d, %d,%d, %d, %d, %d]\n",
    //                 up_weight[0], up_weight[1], up_weight[2], up_weight[3], up_weight[4], up_weight[5], up_weight[6], up_weight[7],
    //                 up_weight[8], up_weight[9], up_weight[10], up_weight[11], up_weight[12], up_weight[13], up_weight[14], up_weight[15]);
        
    //     printf("up_weight = [ %d, %d, %d, %d, %d, %d, %d, %d,%d, %d, %d, %d,%d, %d, %d, %d]\n",
    //                 up_weight[0+32], up_weight[1+32], up_weight[2+32], up_weight[3+32], up_weight[4+32], up_weight[5+32], up_weight[6+32], up_weight[7+32],
    //                 up_weight[8+32], up_weight[9+32], up_weight[10+32], up_weight[11+32], up_weight[12+32], up_weight[13+32], up_weight[14+32], up_weight[15+32]);
    // }

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE/32];
    gemv_n2x(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
    gemv_n2x(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
}

//#elif DOWN_ENABLE
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_down(
    const __global int* expert_list,
    const __global uchar* down_weight_base_addr,
    const __global uchar* down_weight_offset,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;
    __global unsigned int *addr_offset = (__global unsigned int *)(down_weight_offset + expert_list[expert_no] * EACH_EXPERT_WEIGHTS_OFFSET_SIZE);

    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* weight = (__global uchar*)(down_weight_base_addr + addr_offset[0]);
    __global half* scales = (__global half*)(down_weight_base_addr + addr_offset[1]);
    __global uchar* zps = (__global uchar*)(down_weight_base_addr + addr_offset[2]);

    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    __local half x2[INTERMEDIATE_SIZE];
    __local float xg_sum[INTERMEDIATE_SIZE/32];

    //# interleaving x into x2
    __global half * px = x + id_sg*GROUP_SIZE;
    __local half * px2 = x2 + id_sg*GROUP_SIZE;
    for(int i = id_sg; i < INTERMEDIATE_SIZE/GROUP_SIZE; i += num_sg, px += num_sg*GROUP_SIZE, px2 += num_sg*GROUP_SIZE) {
        //# quantization group
        float x_group_sum = 0;
        for(int j = id_local; j < GROUP_SIZE/2; j += SUBGROUP_SIZE) {
            half even = px[2*j + 0];
            half odd = px[2*j + 1];
            px2[j] = even;
            px2[j + GROUP_SIZE/2] = odd;
            x_group_sum += even + odd;
        }
        x_group_sum = sub_group_reduce_add(x_group_sum);
        if (id_local == 0) {
            xg_sum[i] = x_group_sum / SUBGROUP_SIZE;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //# global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    for (int n = n_start; n < n_end; n+=2) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half s0 = S[0];
            half s1 = S[1];
            ushort z = Z[0];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

#if SUBGROUP_SIZE == 32
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

            sum_all0 += (sum0[0] + sum0[1] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] - xg_sum[gk] * z_hf1) * s1;
#else
            half4 sum0;
            half4 sum1;
            half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2 + gk*GROUP_SIZE));
            uchar4 b = intel_sub_group_block_read_uc4((const __global uchar*)B + gk*GROUP_SIZE/2);
            uchar4 b2 = intel_sub_group_block_read_uc4((const __global uchar*)(B + (K/2) + gk*GROUP_SIZE/2));

            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), 0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), 0);
            sum0.s2 = fma(a.s2, (convert_half(b.s2 & 0x0F)), 0);
            sum0.s3 = fma(a.s3, (convert_half(b.s3 & 0x0F)), 0);

            sum0.s0 = fma(a.s4, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s5, (convert_half(b.s1 >> 4)), sum0.s1);
            sum0.s2 = fma(a.s6, (convert_half(b.s2 >> 4)), sum0.s2);
            sum0.s3 = fma(a.s7, (convert_half(b.s3 >> 4)), sum0.s3);

            sum1.s0 = fma(a.s0, (convert_half(b2.s0 & 0x0F)), 0);
            sum1.s1 = fma(a.s1, (convert_half(b2.s1 & 0x0F)), 0);
            sum1.s2 = fma(a.s2, (convert_half(b2.s2 & 0x0F)), 0);
            sum1.s3 = fma(a.s3, (convert_half(b2.s3 & 0x0F)), 0);

            sum1.s0 = fma(a.s4, (convert_half(b2.s0 >> 4)), sum1.s0);
            sum1.s1 = fma(a.s5, (convert_half(b2.s1 >> 4)), sum1.s1);
            sum1.s2 = fma(a.s6, (convert_half(b2.s2 >> 4)), sum1.s2);
            sum1.s3 = fma(a.s7, (convert_half(b2.s3 >> 4)), sum1.s3);

            sum_all0 += (sum0[0] + sum0[1] + sum0[2] + sum0[3] - xg_sum[gk] * z_hf0) * s0;
            sum_all1 += (sum1[0] + sum1[1] + sum1[2] + sum1[3] - xg_sum[gk] * z_hf1) * s1;
#endif
        }
        sum_all0 = sub_group_reduce_add(sum_all0);
        sum_all1 = sub_group_reduce_add(sum_all1);
        if (id_local == 0) {
            y[n] = sum_all0 * routing_weights[expert_no];
            y[n+1] = sum_all1 * routing_weights[expert_no];
        }
    }
}

//#else
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_reduce(const __global TYPE* x,                // [MAX_TOPK, HIDDEN_SIZE]
    __global TYPE* y) {                                    // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    half sum[MAX_TOPK] = {0};
    __attribute__((opencl_unroll_hint(MAX_TOPK)))
    for (int i = 0; i < MAX_TOPK; i++) {
        sum[i] = as_half(intel_sub_group_block_read_us((const __global ushort*)(x + i*HIDDEN_SIZE + n)));
    }
    for (int i = 1; i < MAX_TOPK; i++) {
        sum[0] += sum[i];
    }
    intel_sub_group_block_write_us((const __global ushort*)(y + n), as_ushort(sum[0]));
}
//#endif