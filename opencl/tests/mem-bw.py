#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys



mlp_sources=r'''

// https://github.com/intel/pti-gpu/blob/master/chapters/binary_instrumentation/OpenCLBuiltIn.md

ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
uint __attribute__((overloadable)) intel_get_hw_thread_id( void );
uint __attribute__((overloadable)) intel_get_slice_id( void );
uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
uint __attribute__((overloadable)) intel_get_subslice_id( void );
uint __attribute__((overloadable)) intel_get_eu_id( void );
uint __attribute__((overloadable)) intel_get_eu_thread_id( void );
void __attribute__((overloadable)) intel_eu_thread_pause( uint value );

//https://www.anandtech.com/show/20046/intel-unveils-meteor-lake-architecture-intel-4-heralds-the-disaggregated-future-of-mobile-cpus/7
// Intel includes 8 x Xe graphics cores with 128 vector engines (12 per Xe core)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void mem_bw(
        const __global uchar* weight,
        int N, int N_BLK, int expect,
        __global ulong* sg_info) {
    int num_sg = get_num_sub_groups();
    int id_sg = get_sub_group_id();
    int id_local = get_sub_group_local_id();

    ulong cycle_start = intel_get_cycle_counter();

    __local half x2[128];

    int global_sg_id = get_global_id(1);
    
    uint4 sum0 = 0;
    
    int n_start = global_sg_id * N_BLK;
    int n_end = n_start + N_BLK;
    const __global uchar* B = weight + n_start*K_SIZE/2;

    half8 a = as_half8(intel_sub_group_block_read_us8((const __local ushort*)x2));

    for (int n = n_start; n < n_end; n++) {

        // SUBGROUP_SIZE    : 16
        // quant-group size : 128
        // 
        //      (SUBGROUP_SIZE*8) elements => (SUBGROUP_SIZE*4) char
        //
        const __global uchar* pB = B;

        __attribute__((opencl_unroll_hint(6)))
        for(int k = 0; k < K_SIZE; k += SUBGROUP_SIZE*8, pB += SUBGROUP_SIZE*4) {
            uchar4 b = intel_sub_group_block_read_uc4(pB);
#if 1
            sum0.s0 = fma(a.s0, (convert_half(b.s0 & 0x0F)), sum0.s0);
            sum0.s1 = fma(a.s1, (convert_half(b.s1 & 0x0F)), sum0.s1);
            sum0.s2 = fma(a.s2, (convert_half(b.s2 & 0x0F)), sum0.s2);
            sum0.s3 = fma(a.s3, (convert_half(b.s3 & 0x0F)), sum0.s3);

            sum0.s0 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s1 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);
            sum0.s2 = fma(a.s2, (convert_half(b.s0 >> 4)), sum0.s0);
            sum0.s3 = fma(a.s3, (convert_half(b.s1 >> 4)), sum0.s1);
#else
            sum0 = sum0 + convert_uint4(b);
#endif
        }
        B += K_SIZE/2;
    }

    uint all_sum = sub_group_reduce_add(sum0.s0 + sum0.s1 + sum0.s2 + sum0.s3);

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

    if (expect >= 0) {
        if (id_local == 0 && all_sum == expect-1)
            printf(" ERROR  sg:[%d] %d != %d\n", global_sg_id, all_sum, expect);
    }
}
'''
cl.profiling(True)

np.random.seed(0)

SUBGROUP_SIZE = 16
wg_sg_cnt = 16  # how many sub-groups are joined as a work-group
N = 2048*8
K = 768

# each sub-group do N_BLK rows
N_BLK = int(sys.argv[1])
total_sg_cnt = N // N_BLK

weight = np.ones([N, K//2], np.int8)

t_stats = cl.tensor(np.zeros([16,16,16,16], np.uint32))
t_weight = cl.tensor(weight)

ocl_kernels = cl.kernels(mlp_sources, 
                        f" -D SUBGROUP_SIZE={SUBGROUP_SIZE} -D K_SIZE={K}", "./dump")

print(f"GWS [{SUBGROUP_SIZE=}, {total_sg_cnt=}]  LWS [{SUBGROUP_SIZE=},{wg_sg_cnt=}]")

expect = N_BLK * K
expect = 122880

t_sg_info = cl.tensor(np.zeros([total_sg_cnt, 3], np.uint64))

ocl_kernels.enqueue("mem_bw",
                    [SUBGROUP_SIZE, total_sg_cnt],  # SUBGROUP_SIZE * total_sg_cnt = work-item count
                    [SUBGROUP_SIZE, wg_sg_cnt],     # 
                    t_weight, N, N_BLK, expect, t_sg_info)

cl.finish()

sg_info = t_sg_info.numpy()
with cl.ChromeTraceDumpper("ocl.json") as ctd:
    ts_base = sg_info[:,1].min()
    sg_info[:,1:] -= ts_base
    for sg_id in range(sg_info.shape[0]):
        loc = int(sg_info[sg_id,0])
        eu_tid = loc & 0xF; loc = loc >> 4
        eu_id  = loc & 0xF; loc = loc >> 4
        sub_slice_id  = loc & 0xF; loc = loc >> 4
        slice_id  = loc & 0xF; loc = loc >> 4
        wg_id = sg_id // wg_sg_cnt

        cycle_start = sg_info[sg_id,1]
        cycle_end = sg_info[sg_id,2]
        if 1:
            ctd.phb(name = f"{wg_id}",
                    cat = f"{wg_id}",
                    _id = f"{wg_id}",
                    #pid = f"SubSlice:{slice_id}.{sub_slice_id}",
                    pid = 0,
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

weight_bytes = N*K//2

all_weights = []
total_size = 0
while total_size < 500e6:
    total_size += weight_bytes
    all_weights.append(cl.tensor(weight))

t_stats = cl.tensor()
for r in range(5):
    for w in all_weights:
        ocl_kernels.enqueue("mem_bw",
                    [SUBGROUP_SIZE, total_sg_cnt],
                    [SUBGROUP_SIZE, wg_sg_cnt],
                     w, N, N_BLK, expect, t_stats)

durs = cl.finish()
ns = sum(durs)/len(durs)
print()
print(f" {ns*1e-3:.3f} us  {weight_bytes/1e6:.2f} MB {weight_bytes / ns : .3f} GB/s")
print(f" {K=} {N=} {SUBGROUP_SIZE=}, {total_sg_cnt=} {wg_sg_cnt=} work-groups={total_sg_cnt//wg_sg_cnt}")
print(f" {weight_bytes/1e6:.2f}MB x {len(all_weights)} = {weight_bytes*len(all_weights)/1e6:.1f} MB")
