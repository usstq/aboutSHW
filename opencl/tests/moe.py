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
SUBGROUP_SIZE = 32
SUBGROUP_NUM = 8
N_BLOCK = 4

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

        ocl_kernels = cl.kernels(mlp_sources, f"-DN_BLOCK={N_BLOCK} -DSUBGROUP_NUM={SUBGROUP_NUM} -DSUBGROUP_SIZE={SUBGROUP_SIZE} -D GROUP_SIZE={GROUP_SIZE} -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} -D HIDDEN_SIZE={HIDDEN_SIZE} -D MAX_TOPK={MAX_TOPK}", "./dump")

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

test_mm(M = 1, K = INTERMEDIATE_SIZE, N = HIDDEN_SIZE, K_group_size = GROUP_SIZE, w_dtype = cl.onednn_dtype.u4)
