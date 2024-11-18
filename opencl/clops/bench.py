from . import cl
import numpy as np
from .utils import *

cl_kernel_sources = r'''
ulong __attribute__((overloadable)) intel_get_cycle_counter( void );
uint __attribute__((overloadable)) intel_get_slice_id( void );
uint __attribute__((overloadable)) intel_get_subslice_id( void );
uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
uint __attribute__((overloadable)) intel_get_eu_id( void );
uint __attribute__((overloadable)) intel_get_eu_thread_id( void );

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void mul_f32(float val, int idx, int size, __global float *result, __global ulong* ts//, __global uint* eu_id, __global uint* dss_id
    ) {
    __local float tmp[64 * 1024 / 4];
    int id_local = get_local_id(0);
    int id_global = get_global_id(0);
    int id_sg_local = get_sub_group_local_id();
    tmp[id_local*100] = val * 10;
    barrier(CLK_LOCAL_MEM_FENCE);
    float x = tmp[id_local + idx] * 10;
    float16 s = 0, s1 = 0, s2 = 0, s3 = 0;
    ulong beg = intel_get_cycle_counter();
    __attribute__((opencl_unroll_hint(1)))
    for (int i = 0; i < size; i++) {
        s.s0 += x;
        s.s1 += x;
        s.s2 += x;
        s.s3 += x;
        s.s4 += x;
        s.s5 += x;
        s.s6 += x;
        s.s7 += x;
        s.s8 += x;
        s.s9 += x;
        s.sa += x;
        s.sb += x;
        s.sc += x;
        s.sd += x;
        s.se += x;
        s.sf += x;
#define ADD(n) \
        s##n.s0 += x; \
        s##n.s1 += x; \
        s##n.s2 += x; \
        s##n.s3 += x; \
        s##n.s4 += x; \
        s##n.s5 += x; \
        s##n.s6 += x; \
        s##n.s7 += x; \
        s##n.s8 += x; \
        s##n.s9 += x; \
        s##n.sa += x; \
        s##n.sb += x; \
        s##n.sc += x; \
        s##n.sd += x; \
        s##n.se += x; \
        s##n.sf += x;

        ADD(1);
        ADD(2);
        ADD(3);
    }
    ulong end = intel_get_cycle_counter();
    for (int i = 1; i < 16; i++)
        s[0] += s[i];
    for (int i = 0; i < 16; i++) {
        s[0] += s1[i];
        s[0] += s2[i];
        s[0] += s3[i];
    }
    *result = s[0];
    if (id_sg_local == 0) {
        //ts[id_global / 8] = end - beg;
        ts[0 / 8] = end - beg;
        //eu_id[id_global / 8] = intel_get_eu_id();
        //dss_id[id_global / 8] = intel_get_dual_subslice_id();
    }
}
'''
cl_kernels = kernel_cache(cl_kernel_sources, "-D FMACNT=4", dump='./dump')

if __name__ == "__main__":
    from . import utils
    import torch
    cl.profiling(True)
    # __global float val, int idx, int size, float *result
    
    result = cl.tensor(np.ones([128,], np.float32))
    # [64 * 1024 * 128,]:
    # i=    32 global=    32x128 local_size=128      2.6 Gflop /    1.6 ms =   1651.1 GFlops/s ts=1,825,088 CPI(EU)=2.9 CPI(ALL)=6.0
    # i=    64 global=    64x128 local_size=128      5.2 Gflop /    3.1 ms =   1690.1 GFlops/s ts=1,778,609 CPI(EU)=2.8 CPI(ALL)=5.8
    # i=   128 global=   128x128 local_size=128     10.5 Gflop /    3.1 ms =   3398.5 GFlops/s ts=1,778,497 CPI(EU)=2.8 CPI(ALL)=2.9
    # [64 * 512 * 128,]:
    # i=    32 global=    32x128 local_size=128      2.6 Gflop /    0.7 ms =   3528.1 GFlops/s ts=1,580,208 CPI(EU)=2.5 CPI(ALL)=2.8
    # i=    64 global=    64x128 local_size=128      5.2 Gflop /    0.9 ms =   6075.8 GFlops/s ts=1,864,548 CPI(EU)=2.9 CPI(ALL)=1.6
    # i=   128 global=   128x128 local_size=128     10.5 Gflop /    1.5 ms =   6796.1 GFlops/s ts=1,824,997 CPI(EU)=2.9 CPI(ALL)=1.4

    ts = cl.tensor(np.ones([64 * 512 * 128,], np.int64))
    # eu_id = cl.tensor(np.ones([64 * 1024 * 128,], np.int32))
    # dss_id = cl.tensor(np.ones([64 * 1024 * 128,], np.int32))
    count = 1_0_000
    print(f'{count=:,}, one xecore={2.4*16*8}GFLOPS, peak={2.4*16*8*32}GLOPS')
    fops_per_iter = 64
    for thread_per_eu in [1, 2, 4, 8]:
        for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 8192*2, 8192*8]:
            local_size = 8 * 16 * thread_per_eu
            global_size = i * local_size
            cl_kernels.enqueue("mul_f32", [global_size], [local_size],
                                1.0, 20*1024,
                                count,
                                result, ts, #eu_id, dss_id
                                )
            latency_ns = cl.finish()
            for ns in latency_ns:
                tmp = ts.numpy()[:global_size // 8][0]
                # tmp_eu = eu_id.numpy()[:global_size // 8]
                # tmp_dss = dss_id.numpy()[:global_size // 8]
                # xxx = np.ones([global_size//8, 3], dtype=np.int32)
                # xxx[:,0]=tmp.astype(np.int32)
                # xxx[:,1]=tmp_eu
                # xxx[:,2]=tmp_dss
                #print(tmp)
                # tmp = np.average(tmp)
                flops = 8 * fops_per_iter * count * global_size / 8
                cpi = 2.4*ns / (fops_per_iter * count) / ((i + 31) // 32) / (local_size // 128) #flops/ns/i/16/2.4 # (global_size/8)/2.4
                print(f"{i=:6} global={global_size//local_size:6}x{local_size} {local_size=} {flops*1e-9:8.1f} Gflop / {ns*1e-6:6.1f} ms = {flops/ns:8.1f} GFlops/s ts={tmp:,.0f} CPI(EU)={tmp/fops_per_iter/count:.1f} CPI(ALL)={cpi:.1f}")

        # qkv = utils.to_cl(torch.randn([B, L1, (H + HK + HK) * S], dtype=torch.float16))


