#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys



mlp_sources= cl.SGTracer.code + r'''

// x: [1, K]
// y: [1, N]
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_gate_up(
    const __global uintptr_t* weight_ptrs, int base_id,
    const __global half* x,
    __global half* y,
    __global ulong* sg_info) {

    int N = INTERMEDIATE_SIZE;
    int K = HIDDEN_SIZE;
    
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    SGTracer_begin(&sg_info);

    //# GWS: [expert, SUBGROUP_SIZE, 2*N//N_BLOCK]
    //# LWS: [1,      SUBGROUP_SIZE, SUBGROUP_NUM]

    //# 
    uint seq_nb = get_global_id(2);
    int wid = (get_group_id(0) + base_id) * 6;
    
    if (seq_nb & 1) {
        // [x*up]
        wid += 3;
    }

    const __global uchar* weight = (const __global uchar*)weight_ptrs[wid + 0];
    const __global half* scales = (const __global half*)weight_ptrs[wid + 1];
    const __global uchar* zps = (const __global uchar*)weight_ptrs[wid + 2];

    //# temp results
    __local half act_gate[N_BLOCK * SUBGROUP_NUM/2];
    __local half act_up[N_BLOCK * SUBGROUP_NUM/2];
    __local half * p_out = (seq_nb & 1) ? act_up : act_gate;
    //# where gate/up to put in SLM
    p_out += (id_sg/2) * N_BLOCK;

    __local half x2[HIDDEN_SIZE];
    __local float xg_sum[HIDDEN_SIZE/32];

    __global half * px = x + id_sg*GROUP_SIZE;
    __local half * px2 = x2 + id_sg*GROUP_SIZE;
    for(int i = id_sg; i < HIDDEN_SIZE/GROUP_SIZE; i += num_sg, px += num_sg*GROUP_SIZE, px2 += num_sg*GROUP_SIZE) {
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
    int n_start = (seq_nb/2) * N_BLOCK;
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
#else
            half4 sum0;
            half4 sum1;
#endif

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
            p_out[0] = sum_all0;
            p_out[1] = sum_all1;
        }
        p_out += 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //
    //   combine gate & up within work-group
    //
    // __local half act_gate[N_BLOCK * SUBGROUP_NUM/2];
    // __local half act_up[N_BLOCK * SUBGROUP_NUM/2];
    // __local half * p_out = (seq_nb & 1) ? act_up : act_gate;
    //# where gate/up to put in SLM
    // p_out += (id_sg/2) * N_BLOCK;

#if 0
    if (id_sg == 0 && id_local == 0) {
        y += INTERMEDIATE_SIZE * get_group_id(0) + get_group_id(2)*(N_BLOCK*SUBGROUP_NUM/2);
        for(int i = 0; i < N_BLOCK * SUBGROUP_NUM/2; i++) {
            half v_gate = act_gate[i];
            half v_up = act_up[i];
            half v_silu = v_gate * (1 / (1 + exp(-v_gate)));
            half v_final = v_silu * v_up;
            y[i] = v_final;
        }
    }
#endif

#if 1
    y += INTERMEDIATE_SIZE * get_group_id(0) + get_group_id(2)*(N_BLOCK*SUBGROUP_NUM/2);
    for(uint i = get_local_linear_id(); i < N_BLOCK * SUBGROUP_NUM/2; i += SUBGROUP_SIZE*SUBGROUP_NUM) {
        half v_gate = act_gate[i];
        half v_up = act_up[i];
        half v_silu = v_gate * (1 / (1 + exp(-v_gate)));
        half v_final = v_silu * v_up;
        y[i] = v_final;
    }
#endif

#if 0
    y += INTERMEDIATE_SIZE * get_group_id(0) + get_group_id(2)*(N_BLOCK*SUBGROUP_NUM/2);
    for(int i = id_sg*SUBGROUP_SIZE; i < (N_BLOCK * SUBGROUP_NUM/2); i += SUBGROUP_SIZE*SUBGROUP_NUM) {
        half v_gate = as_half(intel_sub_group_block_read_us((const __local ushort*)act_gate + i));
        half v_up = as_half(intel_sub_group_block_read_us((const __local ushort*)act_up + i));
        half v_silu = v_gate * (1 / (1 + exp(-v_gate)));
        half v_final = v_silu * v_up;

        intel_sub_group_block_write_us((const __global ushort*)y + i, as_short(v_final));
    }
#endif
    SGTracer_end(&sg_info);
}

'''

class WeightQ4A:
    def __init__(self, K, N, K_group_size, SZ_LAYOUT):
        self.K = K
        self.N = N
        self.K_group_size = K_group_size
        assert K % K_group_size == 0
        self.K_groups = K // K_group_size
        self.raw_weight_q = np.random.randint(0,3,[K, N]).astype(np.int8)
        self.raw_weight_s = np.random.randint(-4,5,[self.K_groups, N]).astype(np.float16) / 32
        self.raw_weight_z = np.random.randint(0,3,[self.K_groups, N]).astype(np.int8)
        self.weight = (self.raw_weight_q.astype(np.float32) - self.raw_weight_z.astype(np.float32).repeat(K_group_size, axis=0)) * self.raw_weight_s.repeat(K_group_size, axis=0)
        # [N, K]
        self.weight_q4 = self.pack_i8_to_i4(self.raw_weight_q.transpose().copy())
        self.weight_scale = self.relayout_sz(self.raw_weight_s, SZ_LAYOUT)
        self.weight_zp4 = self.pack_i8_to_i4(WeightQ4A.relayout_sz(self.raw_weight_z, SZ_LAYOUT))
        self.nbytes = self.weight_q4.nbytes + self.weight_scale.nbytes + self.weight_zp4.nbytes

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
    def relayout_sz(sz, SZ_LAYOUT):
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


# given problem
GROUP_SIZE = 128
M = 1
K = HIDDEN_SIZE = 2048
N = INTERMEDIATE_SIZE = 768
assert (K % GROUP_SIZE) == 0
N_EXPERTS = 8

kernel_name = "mlp_gate_up" #sys.argv[1]

# tunable params
SUBGROUP_SIZE = 16
SUBGROUP_SIZE = 32
SUBGROUP_NUM = 8
SZ_LAYOUT = 1
N_BLOCK = 16


# each sub-group do N_BLOCK columns
# but how each sub-group find their N_BLOCK ?
#   

A = np.random.randint(-1,2,[M, K]).astype(np.float16)
tA = cl.tensor(A)
tC = cl.tensor(np.zeros([M, N*N_EXPERTS], dtype=np.float16))

w_gate = WeightQ4A(HIDDEN_SIZE, INTERMEDIATE_SIZE, GROUP_SIZE, SZ_LAYOUT)
w_up = WeightQ4A(HIDDEN_SIZE, INTERMEDIATE_SIZE, GROUP_SIZE, SZ_LAYOUT)

gate = (A @ w_gate.weight).astype(np.float16)
up = (A @ w_up.weight).astype(np.float16)
def silu(x):
    return x * (1 / (1 + np.exp(-x)))
C = silu(gate) * up
C = np.concat([C]*N_EXPERTS, axis=1)
#C = np.concat((gate, up), axis=1)
print("ref is calculated!")


weight_tensors = [] # for holding reference counter
weight_ptrs = []    # [24, 128]
total_nbytes = 0
N_LAYERS = 0
while total_nbytes < 0.5e9:
    for k in range(N_EXPERTS):
        gate_wq4a = [cl.tensor(w_gate.weight_q4), cl.tensor(w_gate.weight_scale), cl.tensor(w_gate.weight_zp4)]
        up_wq4a = [cl.tensor(w_up.weight_q4), cl.tensor(w_up.weight_scale), cl.tensor(w_up.weight_zp4)]
        weight_tensors.append(gate_wq4a)
        weight_tensors.append(up_wq4a)
        # each expert's gate-up are combined into a unit of 6 pointers
        weight_ptrs.append([gate_wq4a[0].addr, gate_wq4a[1].addr, gate_wq4a[2].addr,
                            up_wq4a[0].addr, up_wq4a[1].addr, up_wq4a[2].addr])
        total_nbytes += w_gate.nbytes + w_up.nbytes
    N_LAYERS += 1
PERF_TEST_ROUNDS = N_LAYERS
weight_ptrs = cl.tensor(np.array(weight_ptrs, dtype=np.uint64))

print(f"{N_LAYERS=}....")

#====================== runtime ===================================
cl.profiling(True)

for round in range(4):
    print(f"{round=} ====================================")
    for N_BLOCK in range(2, 34, 2):
        if N % N_BLOCK: continue

        GWS = [N_EXPERTS, SUBGROUP_SIZE, 2*N//N_BLOCK]
        LWS = [1, SUBGROUP_SIZE, SUBGROUP_NUM]
        ocl_kernels = cl.kernels(mlp_sources, 
                                    f"-D N_BLOCK={N_BLOCK} \
                                    -D SUBGROUP_NUM={SUBGROUP_NUM} \
                                    -D SUBGROUP_SIZE={SUBGROUP_SIZE}\
                                    -D GROUP_SIZE={GROUP_SIZE} \
                                    -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} \
                                    -D HIDDEN_SIZE={HIDDEN_SIZE} \
                                    -D SZ_LAYOUT={SZ_LAYOUT}", "./dump")

        for r in range(0, PERF_TEST_ROUNDS):
            cl.finish()
            for i in range(N_LAYERS):
                ocl_kernels.enqueue(kernel_name, GWS, LWS,
                                    weight_ptrs, i*N_EXPERTS, tA, tC, None)
            durs = cl.finish()
            Bsize = N_EXPERTS*2*N*K//2
            #for i,ns in enumerate(durs):
            #    print(f"  {r&1}   {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
            mean_ns = sum(durs)/len(durs)
            # print(f"\t{kernel_name} {K=} {N=} {Bsize*1e-6:.3f} MB {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")

        if 0:
            t_sg_info = cl.tensor(np.zeros([GWS[0]*GWS[2], 3], np.uint64))
            ocl_kernels.enqueue(kernel_name, GWS, LWS,
                                weight_ptrs, 0, tA, tC, t_sg_info)
            durs = cl.finish()
            mean_ns = sum(durs)/len(durs)
            sg_info = t_sg_info.numpy()
            cl.SGTracer.dump(sg_info)
            print(f"{sg_info.shape[0]} sub-groups {GWS=} {LWS=} {mean_ns*1e-6: .3f} ms")

        C1 = tC.numpy()
        if not np.allclose(C, C1, rtol=1e-02, atol=1e-04):
            print(f'\t{C[0,:]=}')
            print(f'\t{C1[0,:]=}')
            sys.exit(0)
            #print(f'{C.shape=} {C1.shape=}')
            acc_info = "ERROR"
        else:
            acc_info = "PASSED"

        print(f" [{N_BLOCK}] {GWS=} {LWS=} Accuracy:{acc_info}  {Bsize*1e-6:.3f} MB {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s ")
