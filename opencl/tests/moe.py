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
ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
uint __attribute__((overloadable)) intel_get_active_channel_mask( void);
uint __attribute__((overloadable)) intel_get_hw_thread_id( void );
uint __attribute__((overloadable)) intel_get_slice_id( void );
uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
uint __attribute__((overloadable)) intel_get_subslice_id( void );
uint __attribute__((overloadable)) intel_get_eu_id( void );
uint __attribute__((overloadable)) intel_get_eu_thread_id( void );
void __attribute__((overloadable)) intel_eu_thread_pause( uint value );

// x: [1, K]
// y: [1, N]
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mlp_down_n2(
    const __global uintptr_t* weight_ptrs, int base_id,
    const __global half* x,
    __global half* y,
    __global uint* stats,
    __global ulong* sg_info) {
    int N = HIDDEN_SIZE;
    int K = INTERMEDIATE_SIZE;
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    ulong cycle_start = intel_get_cycle_counter();
    
    int wid = (get_group_id(0) + base_id) * 3;

    const __global uchar* weight = (const __global uchar*)weight_ptrs[wid + 0];
    const __global half* scales = (const __global half*)weight_ptrs[wid + 1];
    const __global uchar* zps = (const __global uchar*)weight_ptrs[wid + 2];

    __local half x2[INTERMEDIATE_SIZE];
    __local float xg_sum[INTERMEDIATE_SIZE/32];

    __global half * px = x + id_sg*GROUP_SIZE;
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
        float sum_all0 = 0;
        float sum_all1 = 0;
        float sum_all2 = 0;
        float sum_all3 = 0;
#if 0
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
            y[n] = sum_all0;
            y[n+1] = sum_all1;
        }
    }

    if (stats && id_local == 0) {
        uint hw_tid = intel_get_hw_thread_id();
        uint slice_id = intel_get_slice_id();
        uint sub_slice_id = intel_get_subslice_id();
        uint eu_id = intel_get_eu_id();
        uint eu_tid = intel_get_eu_thread_id();
        uint geuid = (slice_id & 0xF);               // Render-Slice
        geuid = (geuid << 4) | (sub_slice_id & 0xF); // xe-core
        geuid = (geuid << 4) | (eu_id & 0xF);
        geuid = (geuid << 4) | (eu_tid & 0xF);
        atomic_add(stats + geuid, 1);
        //printf(" %d, %d  slice: %d.%d.%d.%d \n", global_sg_id, hw_tid, slice_id, sub_slice_id, eu_id, eu_tid);
    }

    if (sg_info && id_local == 0) {
        uint wg_id = get_group_id(0) + (get_group_id(1) + get_group_id(2)*get_num_groups(1)) * get_num_groups(0);
        uint sg_id = wg_id * get_num_sub_groups() + get_sub_group_id();
        sg_info += sg_id * 3;

        uint hw_tid = intel_get_hw_thread_id();
        uint slice_id = intel_get_slice_id();
        uint sub_slice_id = intel_get_subslice_id();
        uint eu_id = intel_get_eu_id();
        uint eu_tid = intel_get_eu_thread_id();
        ulong geuid = (slice_id & 0xF);               // Render-Slice
        geuid = (geuid << 4) | (sub_slice_id & 0xF); // xe-core
        geuid = (geuid << 4) | (eu_id & 0xF);
        geuid = (geuid << 4) | (eu_tid & 0xF);

        sg_info[0] = geuid;
        sg_info[1] = cycle_start;
        sg_info[2] = intel_get_cycle_counter();
    }
}

'''
np.random.seed(0)
np.set_printoptions(linewidth=1024)

GROUP_SIZE = 128
INTERMEDIATE_SIZE = 2048
HIDDEN_SIZE = 768
MAX_TOPK = 8
SUBGROUP_SIZE = 16
SUBGROUP_SIZE = 16
SUBGROUP_NUM = 8
N_BLOCK = 8
N_EXPERTS = 16
N_LAYERS = 24*8

import sys
kernel_name = "mlp_down_n2" #sys.argv[1]

M = 1
K = INTERMEDIATE_SIZE
N = HIDDEN_SIZE

A = np.random.randint(-1,2,[M, K]).astype(np.float16)
tA = cl.tensor(A)
tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
tP1 = cl.tensor()

assert (K % GROUP_SIZE) == 0
K_groups = K // GROUP_SIZE
B_q = np.random.randint(0,3,[K, N]).astype(np.int8)
B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float16)
B_zp = np.random.randint(0,3,[K_groups, N]).astype(np.int8)

B = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(GROUP_SIZE, axis=0)) * B_scale.repeat(GROUP_SIZE, axis=0)
C = A @ B
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

ocl_kernels = cl.kernels(mlp_sources, 
                            f"-DN_BLOCK={N_BLOCK} -DSUBGROUP_NUM={SUBGROUP_NUM} -DSUBGROUP_SIZE={SUBGROUP_SIZE} -D GROUP_SIZE={GROUP_SIZE} -D INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} -D HIDDEN_SIZE={HIDDEN_SIZE} -D MAX_TOPK={MAX_TOPK}", "./dump")

t_stats = cl.tensor(np.zeros([16,16,16,16], np.uint32))
ocl_kernels.enqueue(kernel_name, [N_EXPERTS, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                weight_ptrs, 0, tA, tC, t_stats, None)
cl.finish()

C1 = tC.numpy()


t_sg_info = cl.tensor(np.zeros([N_EXPERTS*(N//N_BLOCK), 3], np.uint64))
ocl_kernels.enqueue(kernel_name, [N_EXPERTS, SUBGROUP_SIZE, N//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                                weight_ptrs, N_EXPERTS*8, tA, tC, None, t_sg_info)
cl.finish()
sg_info = t_sg_info.numpy()

with cl.ChromeTraceDumpper("ocl.json") as ctd:
    #ctd.phb("name","cat", 1, "100", "1", 0, 1000, {"EV": 1.78, "xxx":"hwllo"})
    #ctd.phb("name","cat", 2, "100", "1", 100, 1200)
    #ctd.phX("name","catXXX", "100", "1", 100, 1200)
    ts_base = sg_info[:,1].min()
    sg_info[:,1:] -= ts_base

    for sg_id in range(sg_info.shape[0]):
        loc = int(sg_info[sg_id,0])
        eu_tid = loc & 0xF; loc = loc >> 4
        eu_id  = loc & 0xF; loc = loc >> 4
        sub_slice_id  = loc & 0xF; loc = loc >> 4
        slice_id  = loc & 0xF; loc = loc >> 4
        wg_id = sg_id // SUBGROUP_NUM

        cycle_start = sg_info[sg_id,1]
        cycle_end = sg_info[sg_id,2]
        if 0:
            ctd.phb(name = f"{wg_id}",
                    cat = f"{wg_id}",
                    pid = f"SubSlice:{slice_id}.{sub_slice_id}",
                    tid = f"EU-thread:{eu_id}.{eu_tid}",
                    begin_us = float(cycle_start)/1e3,
                    end_us = float(cycle_end)/1e3,
                    args = {
                        "cycle_start":int(cycle_start),
                        "cycle_end":int(cycle_end),
                        "loc" : f"{slice_id}.{sub_slice_id}.{eu_id}.{eu_tid}",
                        "work-group":wg_id,
                        "sub-group" : sg_id
                    })
        else:
            ctd.phX(name = f"{wg_id}",
                    cat = f"{wg_id}",
                    pid = f"SubSlice:{slice_id}.{sub_slice_id}",
                    tid = f"EU-thread:{eu_id}.{eu_tid}",
                    begin_us = float(cycle_start)/1e3,
                    end_us = float(cycle_end)/1e3,
                    args = {
                        "cycle_start":int(cycle_start),
                        "cycle_end":int(cycle_end),
                        "loc" : f"{slice_id}.{sub_slice_id}.{eu_id}.{eu_tid}",
                        "work-group":wg_id,
                        "sub-group" : sg_id
                    })
        if slice_id == 0 and sub_slice_id == 0 and eu_id == 0:
            print(f" >>>>>>>>>> {wg_id=} {sg_id=}  loc:{slice_id}.{sub_slice_id}.{eu_id}.{eu_tid}  {cycle_start} ~ {cycle_end}  {cycle_end-cycle_start} ")

print(f"{sg_info.shape[0]} sub-groups   [{N_EXPERTS=}, {SUBGROUP_SIZE=}, {N//N_BLOCK}],[1, {SUBGROUP_SIZE}, {SUBGROUP_NUM}]")

#assert 0

cl.profiling(True)
for r in range(0, 10):
    cl.finish()
    for i in range(N_LAYERS):
        ocl_kernels.enqueue(kernel_name,
                            [N_EXPERTS, SUBGROUP_SIZE, HIDDEN_SIZE//N_BLOCK],[1, SUBGROUP_SIZE, SUBGROUP_NUM],
                            weight_ptrs, i*N_EXPERTS, tA, tC, None, None)
    durs = cl.finish()
    Bsize = N_EXPERTS*N*K//2
    #for i,ns in enumerate(durs):
    #    print(f"  {r&1}   {ns*1e-6:.3f} ms, BW: { Bsize/ns : .2f} GB/s")
    mean_ns = sum(durs)/len(durs)

    print(f" {kernel_name} {K=} {N=} {Bsize*1e-6:.3f} MB {mean_ns*1e-6: .3f} ms   BW: { Bsize/mean_ns : .2f} GB/s")

if not np.allclose(C, C1):
    #print(f'{C=}\n')
    #print(f'{C1=}')
    #print(f'{C.shape=} {C1.shape=}')
    #sys.exit(0)
    print("================ ERROR ==================" , M, K, N)
else:
    print("================ PASSED ==================" , M, K, N)
print(f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} HIDDEN_SIZE={HIDDEN_SIZE}")

stats = t_stats.numpy()
for slice_id in range(16):
    for sub_slice_id in range(16):
        if (stats[slice_id, sub_slice_id,:,:].sum() == 0):
            continue
        eu_info = ""
        for eu_id in range(16):
            info = []
            for eu_tid in range(16):
                count = stats[slice_id, sub_slice_id, eu_id, eu_tid]
                if (count > 0):
                    info.append(f"{count}")
            if len(info):
                eu_info += f"EU{eu_id}:[{','.join(info)}] "
        print(f"Slice {slice_id}.{sub_slice_id}: {eu_info}")

