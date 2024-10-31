#!/usr/bin/python3
from clops import cl
import numpy as np
import sys

print(dir(cl))

def test_basic():
    z = np.zeros([2, 3], dtype=np.float32)
    z[0,0] = 1.3
    z[0,1] = 2.3
    z[1,0] = 1.4
    z[1,1] = 2.4
    print(z)

    t = cl.tensor(z)
    print("+++++++++++")
    print(f"t.shape={t.shape}, t.numpy()={t.numpy()}")

    src = '''
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void set_ids(__global float * x) {
            int i = get_global_linear_id();
            x[i] = i;
        }
    '''
    k = cl.kernels(src, "")
    k.enqueue("set_ids", t.shape,[1,1], t)
    print("+++++++++++2222")
    print(t.numpy())

def test_buffer_pool():
    def test():
        t1 = cl.tensor([10,20], np.dtype(np.float32))
        t2 = cl.tensor([20,20], np.dtype(np.float32))
        return t1

    a = test()
    test()
    test()

def test_max_gflops():

    src = '''
        __attribute__((intel_reqd_sub_group_size(8)))
        __kernel void max_fma(__global float * x, int K) {
            //int i = get_global_linear_id();
            float c[FMACNT];
            for(int j = 0; j < FMACNT; j++) c[j] = j;
            float a = get_local_id(0);
            float b = get_local_id(1);
            for(int k = 0; k < K; k += FMACNT*UNROLL) {
                // following loop will be unrolled
                for(int unroll = 0; unroll < UNROLL; unroll ++)
                    for(int j = 0; j < FMACNT; j++)
                        c[j] = fma(a, b, c[j]);
            }

            // prevent optimization
            float sum_c = 0;
            for(int j = 0; j < FMACNT; j++) sum_c += c[j];
            if (sum_c == 0) x[(int)c[0]] = 0;
        }
    '''
    k = cl.kernels(src, "-D FMACNT=4 -D UNROLL=4")
    t = cl.tensor([16], np.dtype(np.float32))
    K = 4096000
    M = 44096
    N = 32
    k.enqueue("max_fma", [M, N],[1, N], t, K)
    profiling = cl.finish()
    print(profiling)
    dur_ns = profiling[0]

    print(f" {dur_ns}ns  {2*M*N*K*1e-3/dur_ns: .2f} TFLOPS")


def test_scheduler(global_size, local_size, subgroup_size):
    src = r'''
    uint __attribute__((overloadable)) intel_get_active_channel_mask( void );
    uint __attribute__((overloadable)) intel_get_hw_thread_id( void );
    uint __attribute__((overloadable)) intel_get_eu_thread_id( void );
    uint __attribute__((overloadable)) intel_get_slice_id( void );
    uint __attribute__((overloadable)) intel_get_dual_subslice_id( void );
    uint __attribute__((overloadable)) intel_get_subslice_id( void );
    uint __attribute__((overloadable)) intel_get_eu_id( void );

    __attribute__((intel_reqd_sub_group_size(subgroup_size)))
    __kernel void test(__global ulong * info) {
        
        int gi0 = get_global_id(0);
        int gi1 = get_global_id(1);
        int gi2 = get_global_id(2);

        int li0 = get_local_id(0);
        int li1 = get_local_id(1);

        int gri0 = get_group_id(0);
        int gri1 = get_group_id(1);

        ulong execmask = intel_get_active_channel_mask();
        ulong slice = intel_get_slice_id();
        ulong dual_slice = intel_get_dual_subslice_id();
        ulong sub_slice = intel_get_subslice_id();
        ulong gtid = intel_get_hw_thread_id();
        ulong euid = intel_get_eu_id();
        ulong tid = intel_get_eu_thread_id();
        int sg_id = get_sub_group_id();

        ulong i = slice;
        i= (i << 8) | (dual_slice);
        i= (i << 8) | (gtid);
        i= (i << 8) | (euid);
        i= (i << 8) | (tid);

        info[get_global_linear_id()] = i;

        //printf("Hello global_id:%d,%d  group.local_id: (%d,%d) (%d,%d)  gtid.slice.dual_slice.euid.tid: %d.%d.%d.%d.%d sg_id:%d exec_mask: %x\n", 
        //    gi0, gi1, gri0, gri1, li0, li1, gtid, slice, dual_slice, euid, tid, sg_id, execmask);
    }
    '''
    info = cl.tensor(global_size, np.dtype(np.uint64))
    k = cl.kernels(src, f"-Dsubgroup_size={subgroup_size}")
    k.enqueue("test", global_size, local_size, info) # [2*(4096), 2],[2, 3])
    info = info.numpy()

    ghtid_summary = {}
    slice_summary = {}
    print(type(info), info.dtype, info.shape)
    for u in np.nditer(info):
        u = u.item()
        tid = (u & 0xFF); u >>= 8
        euid = (u & 0xFF); u >>= 8
        gtid = (u & 0xFF); u >>= 8
        slice_sub_slice = (u & 0xFFFF); u >>= 16
        
        if gtid in ghtid_summary:
            ghtid_summary[gtid] += 1
        else:
            ghtid_summary[gtid] = 1

        if slice_sub_slice in slice_summary:
            slice_summary[slice_sub_slice] += 1
        else:
            slice_summary[slice_sub_slice] = 1
    
    total = 0
    for k in slice_summary:
        total += slice_summary[k]
        print(f"sub_slice 0x{k:04x} : {slice_summary[k]}")
    print(f"=== total = {total} ")

    total = 0
    keys = list(ghtid_summary.keys())
    keys.sort()
    for k in keys:
        total += ghtid_summary[k]
        print(f"hw tid {k:4d} : {ghtid_summary[k]}")
    print(f"=== total = {total} in {len(keys)} hwids")
    cl.finish()


cl.profiling(True)
# work-item from different work-group will not be executed within same sub-group (or on same EU), hence the name `sub-group`
test_scheduler([2048*2, 1, 16],[8, 1, 16], 16); sys.exit(1)

test_basic()
test_buffer_pool()
test_max_gflops()