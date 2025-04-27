#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys



mlp_sources= r'''
#include <cm/cm.h>
#include <cm/cmtl.h>

//# WG_THREADS_NUM threads run as a work-group
//# each HW thread do N_BLOCK columns of work 
//# GWS=[N_EXPERTS, HIDDEN_SIZE//N_BLOCK]
//# LWS=[1, WG_THREADS_NUM]

extern "C" _GENX_MAIN_ void mlp_down_n2(
            svmptr_t weight_ptrs [[type("svmptr_t")]],
            int base_id,
            svmptr_t x [[type("svmptr_t")]],
            svmptr_t y [[type("svmptr_t")]],
            svmptr_t sg_info [[type("svmptr_t")]]) {
    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;

    cm_slm_init(INTERMEDIATE_SIZE * sizeof(half) + INTERMEDIATE_SIZE/32 * sizeof(float));

    uint slmX = cm_slm_alloc(INTERMEDIATE_SIZE * sizeof(half));
    uint slmXSUM = cm_slm_alloc(INTERMEDIATE_SIZE/32 * sizeof(float));
    auto thr_id = cm_local_id(1);
    auto thr_cnt = cm_local_size(1);
#define SIMDW 32
    //#pragma unroll (4)
    for(int i = thr_id; i < INTERMEDIATE_SIZE/GROUP_SIZE; i += thr_cnt) {
        int offset = i*GROUP_SIZE*sizeof(half);

        vector<half, SIMDW*2> a;
        vector<half, SIMDW*2> a2;
        vector<float, SIMDW*2> a_sum = 0;

        for(int k = 0; k < GROUP_SIZE; k += SIMDW*2, offset += SIMDW*2*sizeof(half)) {
            cm_svm_block_read(x + offset, a);

            a_sum += a;
            a2.select<SIMDW, 1>(0)       = a.select<SIMDW, 2>(0);
            a2.select<SIMDW, 1>(SIMDW) = a.select<SIMDW, 2>(1);

            cm_slm_block_write(slmX, offset, a2);
        }
        vector<float, 1> v = cm_sum<float>(a_sum)*(1.0f/SIMDW);
        cm_slm_block_write(slmXSUM, i*sizeof(float), v);
    }
    cm_barrier();

    //# INTERMEDIATE_SIZE/GROUP_SIZE < 64
    vector<float, INTERMEDIATE_SIZE/GROUP_SIZE> xsum;
    cm_slm_block_read(slmXSUM, GENX_DWALIGNED, 0, xsum);

    int wid = (cm_global_id(0) + base_id) * 3;
    auto weight = ((const __global uint8_t**)weight_ptrs)[wid + 0];
    auto scales = ((const __global half**)weight_ptrs)[wid + 1];
    auto zps = ((const __global uint8_t**)weight_ptrs)[wid + 2];

    int n_start = cm_global_id(1) * N_BLOCK;
    int n_end = n_start + N_BLOCK;

    #pragma unroll (1)
    for (int n = n_start; n < n_end; n+=2) {
        auto B = weight + n * K/2;

        vector<float, SIMDW> sum0 = 0;
        vector<float, SIMDW> sum1 = 0;
        auto S = scales + n;
        auto Z = zps + n / 2;
        float sumzp0 = 0;
        float sumzp1 = 0;
        for (int gk = 0; gk < K / GROUP_SIZE; gk++, S += N, Z += N / 2) {
            half s0 = S[0];
            half s1 = S[1];
            uint8_t z = Z[0];
            uint8_t z0 = (z & 0xf);
            uint8_t z1 = (z >> 4);

            vector<half, SIMDW*2> a;
            vector<uint8_t, SIMDW> bi40;
            vector<uint8_t, SIMDW> bi41;
            vector<uint8_t, SIMDW> bi40_even;
            vector<uint8_t, SIMDW> bi40_odd;
            vector<uint8_t, SIMDW> bi41_even;
            vector<uint8_t, SIMDW> bi41_odd;
            vector<float, SIMDW> sum_p0 = 0;
            vector<float, SIMDW> sum_p1 = 0;

            for(int i = 0; i < GROUP_SIZE; i += SIMDW*2) {
                //cm_svm_block_read(x + gk*GROUP_SIZE*sizeof(half), a);
                cm_slm_block_read(slmX, GENX_DWALIGNED, (gk*GROUP_SIZE + i)*sizeof(half), a);

                cm_svm_block_read((svmptr_t)B + (gk*GROUP_SIZE + i)/2, bi40);
                cm_svm_block_read((svmptr_t)B + (K + gk*GROUP_SIZE + i)/2, bi41);

                bi40_even = bi40 & (uint8_t)0xF;
                bi40_odd  = bi40 >> (uint8_t)4;
                sum_p0 += (a.select<SIMDW, 1>(0) * bi40_even.format<int8_t>() +
                           a.select<SIMDW, 1>(SIMDW) * bi40_odd.format<int8_t>());

                bi41_even = bi41 & (uint8_t)0xF;
                bi41_odd  = bi41 >> (uint8_t)4;
                sum_p1 += (a.select<SIMDW, 1>(0) * bi41_even.format<int8_t>() +
                           a.select<SIMDW, 1>(SIMDW) * bi41_odd.format<int8_t>());
            }
            sum0 += (sum_p0 - xsum[gk]*z0)*s0;
            sum1 += (sum_p1 - xsum[gk]*z1)*s1;
        }
        ((__global half*)y)[n] = cm_sum<float>(sum0);
        ((__global half*)y)[n+1] = cm_sum<float>(sum1);
    }
}

'''

GROUP_SIZE = 128
K = INTERMEDIATE_SIZE = 768
N = HIDDEN_SIZE = 2048
MAX_TOPK = 8
WG_THREADS_NUM = 8
N_BLOCK = 16
N_EXPERTS = 8
N_LAYERS = 24*4
SZ_LAYOUT = 0
PERF_TEST_ROUNDS = N_LAYERS
kernel_name = "mlp_down_n2" #sys.argv[1]

M = 1

A = np.random.randint(-1,2,[M, K]).astype(np.float16)
tA = cl.tensor(A)
tC = cl.tensor(np.zeros([M, N], dtype=np.float16))

assert (K % GROUP_SIZE) == 0
K_groups = K // GROUP_SIZE

class WeightQ4A:
    def __init__(self, K, N, K_group_size):
        self.K = K
        self.N = N
        self.K_group_size = K_group_size
        assert K % K_group_size == 0
        self.K_groups = K // K_group_size
        self.raw_weight_q = np.random.randint(0,3,[K, N]).astype(np.int8)
        self.raw_weight_s = np.random.randint(-4,5,[self.K_groups, N]).astype(np.float16)/32
        self.raw_weight_z = np.random.randint(0,3,[self.K_groups, N]).astype(np.int8)
        #self.raw_weight_z *= 0
        
        self.weight = (self.raw_weight_q.astype(np.float32) - self.raw_weight_z.astype(np.float32).repeat(K_group_size, axis=0)) * self.raw_weight_s.repeat(K_group_size, axis=0)
        # [N, K]
        self.weight_q4 = self.pack_i8_to_i4(self.raw_weight_q.transpose().copy())
        self.weight_scale = self.relayout_sz(self.raw_weight_s)
        self.weight_zp4 = self.pack_i8_to_i4(WeightQ4A.relayout_sz(self.raw_weight_z))

    @staticmethod
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

    @staticmethod
    def relayout_sz(sz):
        if SZ_LAYOUT == 0:
            return sz
        k_groups, N = sz.shape
        assert N % 2 == 0
        new_sz = np.zeros([N//2, k_groups*2], sz.dtype)
        for n in range(N//2):
            for k in range(k_groups):
                new_sz[n, 2*k + 0] = sz[k, 2*n+0]
                new_sz[n, 2*k + 1] = sz[k, 2*n+1]
        return new_sz

w_gate = WeightQ4A(HIDDEN_SIZE, INTERMEDIATE_SIZE, GROUP_SIZE)
w_up = WeightQ4A(HIDDEN_SIZE, INTERMEDIATE_SIZE, GROUP_SIZE)
w_down = WeightQ4A(INTERMEDIATE_SIZE, HIDDEN_SIZE, GROUP_SIZE)


C = A @ w_down.weight
print("ref is calculated!")

tBq4 = cl.tensor(w_down.weight_q4)
tBs = cl.tensor(w_down.weight_scale)
tBz = cl.tensor(w_down.weight_zp4)

weight_tensors = [] # for holding reference counter
weight_ptrs = []    # [24, 128]
for i in range(N_LAYERS):
    for k in range(N_EXPERTS):
        t_weight = cl.tensor(w_down.weight_q4)
        t_scales =  cl.tensor(w_down.weight_scale)
        t_zero_points = cl.tensor(w_down.weight_zp4)
        weight_tensors.append([t_weight, t_scales, t_zero_points])
        weight_ptrs.append([t_weight.addr, t_scales.addr, t_zero_points.addr])

weight_ptrs = cl.tensor(np.array(weight_ptrs, dtype=np.uint64))

ocl_kernels = cl.kernels(mlp_sources, 
                            #"-cmc  -mdump_asm -g2"
                            "-cmc"
                            f" -D N_BLOCK={N_BLOCK} \
                              -D WG_THREADS_NUM={WG_THREADS_NUM} \
                              -D GROUP_SIZE={GROUP_SIZE} \
                              -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} \
                              -D HIDDEN_SIZE={HIDDEN_SIZE} \
                              -D MAX_TOPK={MAX_TOPK} \
                              -D SZ_LAYOUT={SZ_LAYOUT}", "./dump")

# WG_THREADS_NUM threads run as a work-group
# each HW thread do N_BLOCK columns of work 
GWS=[N_EXPERTS, HIDDEN_SIZE//N_BLOCK]
LWS=[1, WG_THREADS_NUM]


cl.profiling(True)
for r in range(0, PERF_TEST_ROUNDS):
    cl.finish()
    for i in range(N_LAYERS):
        ocl_kernels.enqueue(kernel_name,
                            GWS,LWS,
                            weight_ptrs, i*N_EXPERTS, tA, tC, None)
    durs = cl.finish()
    Bsize = N_EXPERTS*N*K//2
    #for i,ns in enumerate(durs):
    #    print(f"  {r&1}   {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    mean_ns = sum(durs)/len(durs)
    print(f"\t{kernel_name} {K=} {N=} {Bsize*1e-6:.3f} MB {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")

t_sg_info = cl.tensor(np.zeros([N_EXPERTS*(N//N_BLOCK), 3], np.uint64))
ocl_kernels.enqueue(kernel_name, GWS,LWS,
                                weight_ptrs, 0, tA, tC, t_sg_info)
sg_info = t_sg_info.numpy()
cl.SGTracer.dump(sg_info)
print(f"{sg_info.shape[0]} sub-groups   [{N_EXPERTS=}, {N//N_BLOCK}],[1, {WG_THREADS_NUM}]")

C1 = tC.numpy()
if not np.allclose(C, C1):
    print(f'{C=}\n')
    print(f'{C1=}')
    #print(f'{C.shape=} {C1.shape=}')
    #sys.exit(0)
    print("================ ERROR ==================" , M, K, N)
else:
    print(f'{C=}\n')
    print(f'{C1=}')
    print("================ PASSED ==================" , M, K, N)
print(f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} HIDDEN_SIZE={HIDDEN_SIZE}")
