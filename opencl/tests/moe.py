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
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_down_n2(
    const __global uintptr_t* weight_ptrs, int base_id,
    const __global half* x,
    __global half* y) {
    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    int wid = (get_group_id(0) + base_id) * 3;

    const __global uchar* weight = (const __global uchar*)weight_ptrs[wid + 0];
    const __global half* scales = (const __global half*)weight_ptrs[wid + 1];
    const __global uchar* zps = (const __global uchar*)weight_ptrs[wid + 2];

    y += (get_group_id(0) + base_id) * HIDDEN_SIZE;

    __local half x2[INTERMEDIATE_SIZE];
    __local float xg_sum[INTERMEDIATE_SIZE/32];

    const __global half * px = x + id_sg*GROUP_SIZE;
    __local half * px2 = x2 + id_sg*GROUP_SIZE;
    for(int i = id_sg; i < INTERMEDIATE_SIZE/GROUP_SIZE; i += num_sg, px += num_sg*GROUP_SIZE, px2 += num_sg*GROUP_SIZE) {
        //# quantization group : 128/64
        //# put elements of even index to first half, odd index to second half
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
        const __global half* S = scales + n;
        const __global uchar* Z = zps + n / 2;
        float sum_all0 = 0;
        float sum_all1 = 0;
        float sum_all2 = 0;
        float sum_all3 = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
#if SUBGROUP_SIZE == 32
            half2 sum0;
            half2 sum1;
#else
            half4 sum0;
            half4 sum1;
#endif
            half s0 = S[0];
            half s1 = S[1];
            ushort z = Z[0];
            half z_hf0 = convert_half(z & 0xf);
            half z_hf1 = convert_half(z >> 4);

            // sum(a[i] * (b[i]-z)*s) = sum(a[i] * b[i] - sum(a[i])*z) * s

#if SUBGROUP_SIZE == 32
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
            y[n] = sum_all0;
            y[n+1] = sum_all1;
        }
    }
}
'''

GROUP_SIZE = 128
INTERMEDIATE_SIZE = 768
HIDDEN_SIZE = 2048
MAX_TOPK = 8
SUBGROUP_SIZE = 16 
#SUBGROUP_SIZE = 32 # on Xe2 system, use sub-group 32
SUBGROUP_NUM = 8
N_BLOCK = 8
N_EXPERTS = 8
N_LAYERS = 24*8

import sys
kernel_name = "mlp_down_n2" #sys.argv[1]

def test_mm(M, K, N, K_group_size, w_dtype):
    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    tA = cl.tensor(A)
    tC = cl.tensor(np.zeros([M, N*N_EXPERTS], dtype=np.float16))
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

        weight_tensors = [] # for holding reference counter
        weight_ptrs = []    # [24, 128]
        for i in range(N_LAYERS):
            for k in range(N_EXPERTS):
                t_weight = cl.tensor(B_q4)
                t_scales =  cl.tensor(B_scale)
                t_zero_points = cl.tensor(B_zp4)
                weight_tensors.append([t_weight, t_scales, t_zero_points])
                weight_ptrs.append([t_weight.addr, t_scales.addr, t_zero_points.addr])

        weight_ptrs = cl.tensor(np.array(weight_ptrs, dtype=np.uint64))
        
        linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype,
                                  M, K, N,
                                  K_group_size, cl.onednn_matmul_type.none,
                                  cl.onednn_dtype.u4, tBq4, tBs, tBz)

        ocl_kernels = cl.kernels(mlp_sources, 
                                 f"-DN_BLOCK={N_BLOCK} -DSUBGROUP_NUM={SUBGROUP_NUM} -DSUBGROUP_SIZE={SUBGROUP_SIZE} -D GROUP_SIZE={GROUP_SIZE} -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} -D HIDDEN_SIZE={HIDDEN_SIZE} -D MAX_TOPK={MAX_TOPK}", "./dump")

        tP1 = cl.tensor(BinarySrc)

    ocl_kernels.enqueue(kernel_name, [N_EXPERTS, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                    weight_ptrs, 0, tA, tC)
    cl.finish()

    C1 = tC.numpy()

    # performance
    cl.profiling(True)
    for r in range(0, 10):
        cl.finish()
        for i in range(N_LAYERS):
            ocl_kernels.enqueue(kernel_name,
                                [N_EXPERTS, SUBGROUP_SIZE, HIDDEN_SIZE//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                weight_ptrs, i*N_EXPERTS, tA, tC)
        durs = cl.finish()
        Bsize = N_EXPERTS*N*K//2
        for i,ns in enumerate(durs):
            print(f"  {r&1}   {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
        mean_ns = sum(durs)/len(durs)

        print(f" {kernel_name} {K=} {N=} {Bsize*1e-6:.3f} MB {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")

    # 
    C = np.concatenate([C]*N_EXPERTS, axis=1)
    if not np.allclose(C, C1):
        print(f'{C=}\n')
        print(f'{C1=}')
        print(f'{C.shape=} {C1.shape=}')
        #sys.exit(0)
    else:
        print("================ PASSED ==================" , M, K, N, w_dtype)
    print(f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} HIDDEN_SIZE={HIDDEN_SIZE}")

np.set_printoptions(linewidth=1024)


#test_mm(M = 1, K = 768, N = 2048, K_group_size = 0, w_dtype = cl.onednn_dtype.f16)
#test_mm(M = 1, K = 768, N = 2048, K_group_size = 32, w_dtype = cl.onednn_dtype.s8)
test_mm(M = 1, K = INTERMEDIATE_SIZE, N = HIDDEN_SIZE, K_group_size = GROUP_SIZE, w_dtype = cl.onednn_dtype.u4)

#for info in cl.dev_info()["CL_DEVICE_EXTENSIONS"]:
#    if "subgroup" in info: print("\t", info)
#test_mlp(K_group_size = 128, w_dtype = cl.onednn_dtype.f16)
