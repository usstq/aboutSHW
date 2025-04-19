#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys


def pack_i8_to_i4(B_q):
    assert B_q.ndim == 2
    K = B_q.shape[0]
    N = B_q.shape[1]
    B_q4 = np.zeros([K, N//2], dtype=np.int8)
    for k in range(K):
        for n in range(N//2):
            even = (B_q[k,2*n]) & 0xF
            odd = (B_q[k,2*n+1]) & 0xF
            B_q4[k,n] = even | (odd << 4)
    return B_q4

mlp_sources=r'''
__kernel void (mlp_down_ref)(
    //const __global FUNC(expert_info)* info_ptrs,
    __global uchar* down_weight,
    __global half* down_scale,
    __global uchar* down_zp,
    const __global half* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    //__global half* routing_weights,                       // [MAX_TOPK]
    __global half* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    //int routing_offset) {
    //int expert_no = get_global_id(0);
    int n = get_global_id(1);
    //x += expert_no * INTERMEDIATE_SIZE;
    //y += expert_no * HIDDEN_SIZE;
    //const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    //__global uchar* down_weight = (__global uchar*)info_ptr->weight[2] + n / 2;
    //__global half* down_scale = (__global half*)info_ptr->scale[2] + n;
    //__global uchar* down_zp = (__global uchar*)info_ptr->zp[2] + n / 2;
    down_weight += n / 2;
    down_scale += n;
    down_zp += n / 2;
    float down_sum = 0;
    for (int k = 0; k < INTERMEDIATE_SIZE; k += GROUP_SIZE) {
        half down_scale_v = down_scale[k / GROUP_SIZE * HIDDEN_SIZE];
        uchar down_zp_org = down_zp[k / GROUP_SIZE * HIDDEN_SIZE / 2];
        half down_zp_v = convert_half((n & 1) ? (down_zp_org >> 4) : (down_zp_org & 0xf));
        float down_sum_group = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            uchar down_weight_org = down_weight[(k + i) * HIDDEN_SIZE / 2];
            half down_w_v = convert_half((n & 1) ? (down_weight_org >> 4) : (down_weight_org & 0xf));
            down_sum_group += (down_w_v - down_zp_v) * x[k + i];
        }
        down_sum += down_sum_group * down_scale_v;
    }
    // down * routing
    y[n] = down_sum; // * routing_weights[routing_offset];
}

#define TYPE half
void splitter(const int n, const int team, const int tid, int* n_start, int* n_end) {
    if (team <= 1 || n == 0) {
        *n_start = 0;
        *n_end = n;
    } else {
        int n1 = (n + team - 1) / team;
        int n2 = n1 - 1;
        int T1 = n - n2 * team;
        *n_end = tid < T1 ? n1 : n2;
        *n_start = tid <= T1 ? tid * n1 : T1 * n1 + (tid - T1) * n2;
    }
    *n_end += *n_start;
}
#if 0
// x: [1, K]
// y: [1, N]
// all_sum: [N, sg_num]
inline void gemv_64n(const __global uchar* weight, __global half* scales, __global uchar* zps, const __global half* x,
    __global half* y, int N, int K, __local float* all_sum) { // __local float all_sum[SUBGROUP_SIZE * 2][SUBGROUP_NUM]
    // global: [expert, N/2, SUBGROUP_NUM], local: [1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int id_sg = get_local_id(2);
    int id_local = get_local_id(1);
    int ithr = get_local_id(2);     // 0~7
    int nthr = get_local_size(2);   // 8

    int K_groups = K / GROUP_SIZE;
    int gk0, gk1;
    splitter(K_groups, nthr, ithr, &gk0, &gk1);

    half2 sum_all = 0;
    for(int gk = gk0; gk < gk1; gk++) {
        const int k = gk * GROUP_SIZE;
        const __global half* A = x + k;
        const __global uchar* B = weight + k * N / 2;

        half8 sum = 0;
        half2 scale = as_half2(((const __global ushort2*)(scales + gk * N))[0]);
        uchar zp = zps[gk * N / 2];
        char zp_even = (zp & 0xf);
        char zp_odd = (char)(zp >> 4);
        __attribute__((opencl_unroll_hint(4)))
        for(int g = 0; g < GROUP_SIZE; g += SUBGROUP_SIZE, B += SUBGROUP_SIZE * N / 2) {
            // read 32/2 elememts of A
            ushort vAs = as_short(intel_sub_group_block_read_us((const __global ushort*)(A + g)));
            // read 32 int4
            uchar b;
            half i4_even, i4_odd;
#define ACC(B_row, sum_idx) \
            b = B[B_row * N / 2];   \
            i4_even = convert_half((b & 0xf) - zp_even);  \
            i4_odd = convert_half((char)(b >> 4) - zp_odd);  \
            sum[sum_idx] = fma(as_half(sub_group_broadcast(vAs, B_row)), i4_even, sum[sum_idx]);    \
            sum[sum_idx + 1] = fma(as_half(sub_group_broadcast(vAs, B_row)), i4_odd, sum[sum_idx + 1]);
            ACC(0, 0); ACC(1, 2); ACC(2, 4);  ACC(3, 6); ACC(4, 0); ACC(5, 2); ACC(6, 4); ACC(7, 6);
            ACC(8, 0); ACC(9, 2); ACC(10, 4); ACC(11, 6); ACC(12, 0); ACC(13, 2); ACC(14, 4); ACC(15, 6);
#if SUBGROUP_SIZE == 32
            ACC(16, 0); ACC(17, 2); ACC(18, 4); ACC(19, 6); ACC(20, 0); ACC(21, 2); ACC(22, 4); ACC(23, 6);
            ACC(24, 0); ACC(25, 2); ACC(26, 4); ACC(27, 6); ACC(28, 0); ACC(29, 2); ACC(30, 4); ACC(31, 6);
#endif
#undef ACC
        }

        // scales applied once
        sum_all[0] += (sum[0] + sum[2] + sum[4] + sum[6]) * scale[0];
        sum_all[1] += (sum[1] + sum[3] + sum[5] + sum[7]) * scale[1];
    }
    all_sum[id_local * 2 * SUBGROUP_NUM + id_sg] = sum_all[0];
    all_sum[(id_local * 2 + 1) * SUBGROUP_NUM + id_sg] = sum_all[1];

    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint()))
    for (int i = 0; i < SUBGROUP_SIZE * 2; i += SUBGROUP_NUM) {
        float sum_all = 0;
        if (id_local < SUBGROUP_NUM) {
            sum_all = all_sum[(i + id_sg) * SUBGROUP_NUM + id_local];
        }
        sum_all = sub_group_reduce_add(sum_all);
        if (id_local == 0) {
            y[i + id_sg] = sum_all;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void (mlp_down_n64)(
    //const __global FUNC(expert_info)* info_ptrs,
    __global uchar* down_weight,
    __global half* down_scale,
    __global uchar* down_zp,
    const __global half* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    //__global half* routing_weights,                       // [MAX_TOPK]
    __global half* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    //int routing_offset) {
    // global: [expert, N/2, SUBGROUP_NUM], local: [1, SUBGROUP_SIZE, SUBGROUP_NUM]
    //int expert_no = get_global_id(0);
    int n = get_global_id(1)*2;
    //x += expert_no * INTERMEDIATE_SIZE;
    //y += expert_no * HIDDEN_SIZE;
    //const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    //__global uchar* down_weight = (__global uchar*)info_ptr->weight[2] + n / 2;
    //__global half* down_scale = (__global half*)info_ptr->scale[2] + n;
    //__global uchar* down_zp = (__global uchar*)info_ptr->zp[2] + n / 2;
    ///////////////////////////////////////////////////////////////////////////
    down_weight += n / 2;
    down_scale += n;
    down_zp += n / 2;
    y += n;
    __local float all_sum[SUBGROUP_SIZE * 2][SUBGROUP_NUM];
    gemv_64n(down_weight, down_scale, down_zp, x, y, HIDDEN_SIZE, INTERMEDIATE_SIZE, all_sum);
    /*
    float down_sum = 0;
    for (int k = 0; k < INTERMEDIATE_SIZE; k += GROUP_SIZE) {
        half down_scale_v = down_scale[k / GROUP_SIZE * HIDDEN_SIZE];
        uchar down_zp_org = down_zp[k / GROUP_SIZE * HIDDEN_SIZE / 2];
        half down_zp_v = convert_half((n & 1) ? (down_zp_org >> 4) : (down_zp_org & 0xf));
        float down_sum_group = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            uchar down_weight_org = down_weight[(k + i) * HIDDEN_SIZE / 2];
            half down_w_v = convert_half((n & 1) ? (down_weight_org >> 4) : (down_weight_org & 0xf));
            down_sum_group += (down_w_v - down_zp_v) * x[k + i];
        }
        down_sum += down_sum_group * down_scale_v;
    }
    // down * routing
    y[n] = down_sum; // * routing_weights[routing_offset];*/
}
#endif
// x: [1, K]
// y: [1, N]
inline void gemv_n(const __global uchar* weight, __global half* scales, __global uchar* zps, const __global half* x,
    __global half* y, int N, int K, __local half* x_cache) { // __local half x_cache[K]
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    int id_sg = get_local_id(2);
    int id_local = get_local_id(1);

    int n_start = get_global_id(2) * N_BLOCK;
    int n_end = n_start + N_BLOCK;
    for (int n = n_start; n < n_end; n++) {
        const __global uchar* B = weight + n * K / 2;
        __global half* S = scales + n;
        __global uchar* Z = zps + n / 2;
        half sum_all = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half4 sum = 0;
            half s = S[0];
            ushort z = Z[0];
            half z_hf = convert_half((n & 1) ? (z >> 4) : (z & 0xf));
            // read 1 group(32*2 or 16*4 bytes)
            ushort bs;
            half b_even, b_odd;
#define ACC(idx, idx_sum)    \
            bs = b[idx]; b_even = convert_half(bs & 0xf); b_odd = convert_half(bs >> 4);    \
            sum[2 * idx_sum + 0] = fma(a[2 * idx + 0], b_even - z_hf, sum[2 * idx_sum + 0]);    \
            sum[2 * idx_sum + 1] = fma(a[2 * idx + 1], b_odd - z_hf,  sum[2 * idx_sum + 1]);

#if SUBGROUP_SIZE == 32
            half4 a = ((const __global half4*)(x + gk * GROUP_SIZE))[id_local];
            uchar2 b = ((const __global uchar2*)(B + gk * GROUP_SIZE / 2))[id_local];
            //bs = b[0]; b_even = convert_half(bs & 0xf); b_odd = convert_half(bs >> 4);
            //sum[0] = fma(a[0], b_even - z_hf, sum[0]);
            //sum[1] = fma(a[1], b_odd - z_hf, sum[1]);
            ACC(0, 0); ACC(1, 1);
            sum_all += (sum[0] + sum[1] + sum[2] + sum[3]) * s;
#else
            half8 a = ((const __global half8*)(x + gk * GROUP_SIZE))[id_local];
            uchar4 b = ((const __global uchar4*)(B + gk * GROUP_SIZE / 2))[id_local];
            ACC(0, 0); ACC(1, 1); ACC(2, 0); ACC(3, 1);
            sum_all += (sum[0] + sum[1] + sum[2] + sum[3]) * s;
            //if (get_global_id(1) == 0 && id_local == 0 && gk < 2)
            //     printf("s0 = %f s1 = %f s2 = %f s3 = %f z=%f s=%f gk=%d\n", sum[0], sum[1], sum[2], sum[3], z_hf, s, gk);
#endif
#undef ACC
        }
        //if (get_global_id(1) == 0 && n < n_start + 3)
        //     printf("sum_all = %f id_local=%d n = %d\n", sum_all, id_local, n);

        sum_all = sub_group_reduce_add(sum_all);
        //if (get_global_id(1) == 0 && n < n_start + 3)
        //     printf("--sum_all = %f id_local=%d n = %d\n", sum_all, id_local, n);
        if (id_local == 0)
            y[n] = sum_all;
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void (mlp_down_n)(
    //const __global FUNC(expert_info)* info_ptrs,
    __global uchar* down_weight,
    __global half* down_scale,
    __global uchar* down_zp,
    const __global half* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    //__global half* routing_weights,                       // [MAX_TOPK]
    __global half* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    //int routing_offset) {
    // global: [expert, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM]
    //int expert_no = get_global_id(0);
    int n = get_global_id(2) * N_BLOCK;
    //x += expert_no * INTERMEDIATE_SIZE;
    //y += expert_no * HIDDEN_SIZE;
    //const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    //__global uchar* down_weight = (__global uchar*)info_ptr->weight[2] + n / 2;
    //__global half* down_scale = (__global half*)info_ptr->scale[2] + n;
    //__global uchar* down_zp = (__global uchar*)info_ptr->zp[2] + n / 2;
    ///////////////////////////////////////////////////////////////////////////
    //down_weight += n / 2;
    //down_scale += n;
    //down_zp += n / 2;
    //y += n;
    __local half x_cache[HIDDEN_SIZE];
    gemv_n(down_weight, down_scale, down_zp, x, y, HIDDEN_SIZE, INTERMEDIATE_SIZE, x_cache);
    /*
    float down_sum = 0;
    for (int k = 0; k < INTERMEDIATE_SIZE; k += GROUP_SIZE) {
        half down_scale_v = down_scale[k / GROUP_SIZE * HIDDEN_SIZE];
        uchar down_zp_org = down_zp[k / GROUP_SIZE * HIDDEN_SIZE / 2];
        half down_zp_v = convert_half((n & 1) ? (down_zp_org >> 4) : (down_zp_org & 0xf));
        float down_sum_group = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            uchar down_weight_org = down_weight[(k + i) * HIDDEN_SIZE / 2];
            half down_w_v = convert_half((n & 1) ? (down_weight_org >> 4) : (down_weight_org & 0xf));
            down_sum_group += (down_w_v - down_zp_v) * x[k + i];
        }
        down_sum += down_sum_group * down_scale_v;
    }
    // down * routing
    y[n] = down_sum; // * routing_weights[routing_offset];*/
}

'''
GROUP_SIZE = 128
INTERMEDIATE_SIZE = 768
HIDDEN_SIZE = 2048
MAX_TOPK = 8
SUBGROUP_SIZE = 16
SUBGROUP_NUM = 8
N_BLOCK = 8

def test_mm(M, K, N, K_group_size, w_dtype):
    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    tA = cl.tensor(A)
    tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
    tP1 = cl.tensor()

    if w_dtype == cl.onednn_dtype.u4:
        assert (K % K_group_size) == 0
        K_groups = K // K_group_size
        B_q = np.random.randint(0,3,[K, N]).astype(np.int8)
        B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float16)
        B_zp = np.random.randint(0,3,[K_groups, N]).astype(np.int8)

        BinarySrc = np.random.randint(-2,3,[M, N]).astype(np.float16)
        
        B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)

        C = A @ B
        #C = C * BinarySrc
        print("ref is calculated!")

        # pack weight into 4bit
        B_q4 = pack_i8_to_i4(B_q.transpose().copy())
        #B_q4 = pack_i8_to_i4(B_q.copy())
        B_zp4 = pack_i8_to_i4(B_zp)

        tBq4 = cl.tensor(B_q4)
        tBs = cl.tensor(B_scale)
        tBz = cl.tensor(B_zp4)
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, K_group_size, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.u4, tBq4, tBs, tBz)
        ocl_kernels = cl.kernels(mlp_sources, f"-DN_BLOCK={N_BLOCK} -DSUBGROUP_NUM={SUBGROUP_NUM} -DSUBGROUP_SIZE={SUBGROUP_SIZE} -D GROUP_SIZE={GROUP_SIZE} -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} -D HIDDEN_SIZE={HIDDEN_SIZE} -D MAX_TOPK={MAX_TOPK}", "./dump")

        tP1 = cl.tensor(BinarySrc)

    # linear.forward(tA, tC, tP1)

    # ocl_kernels.enqueue("mlp_down_ref", [1, HIDDEN_SIZE],[1, 32],
    #                                 tBq4, tBs, tBz, tA, tC)
    # ocl_kernels.enqueue("mlp_down_n64", [1, HIDDEN_SIZE//2, SUBGROUP_NUM],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
    #                                 tBq4, tBs, tBz, tA, tC)
    ocl_kernels.enqueue("mlp_down_n", [1, SUBGROUP_SIZE, HIDDEN_SIZE//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                    tBq4, tBs, tBz, tA, tC)
    cl.finish()

    C1 = tC.numpy()

    if not np.allclose(C, C1):
        print(f'{C=}\n')
        print(f'{C1=}')
        print(f'{C.shape=} {C1.shape=}')
        sys.exit(0)
    else:
        print("================ PASSED ==================" , M, K, N, w_dtype)
    
    cl.profiling(True)
    for l in range(0, 10):
        # ocl_kernels.enqueue("mlp_down_ref", [1, HIDDEN_SIZE],[1, 32],
        #                             tBq4, tBs, tBz, tA, tC)
        # ocl_kernels.enqueue("mlp_down_n64", [1, HIDDEN_SIZE//2, SUBGROUP_NUM],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
        #                                 tBq4, tBs, tBz, tA, tC)
        ocl_kernels.enqueue("mlp_down_n", [1, SUBGROUP_SIZE, HIDDEN_SIZE//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                        tBq4, tBs, tBz, tA, tC)
    durs = cl.finish()
    Bsize = N*K//2
    for ns in durs:
        print(f"{K=} {N=} {Bsize*1e-6:.3f} MB {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    if (len(durs)):
        mean_ns = sum(durs)/len(durs)
        print(f"  {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")

np.set_printoptions(linewidth=1024)

# cl.profiling(True)

# test_moe(1)
# test_moe(1024)
# #test_moe(4096, running_experts = 1)

# sys.exit()

if 1:
    #test_mm(M = 1, K = 768, N = 2048, K_group_size = 0, w_dtype = cl.onednn_dtype.f16)
    #test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.s8)
    test_mm(M = 1, K = INTERMEDIATE_SIZE, N = HIDDEN_SIZE, K_group_size = GROUP_SIZE, w_dtype = cl.onednn_dtype.u4)

#test_mlp(K_group_size = 128, w_dtype = cl.onednn_dtype.f16)
