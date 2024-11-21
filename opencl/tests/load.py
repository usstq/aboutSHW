src=r'''
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void test(__global int * p) {
    p += get_global_id(0)*8;
    int ch = get_sub_group_local_id();
    int8 x = *(__global int8 *)p;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    printf("ch %d: %d,%d,%d,%d,%d,%d,%d,%d\n",
            ch,
            x.s0, x.s1, x.s2, x.s3,
            x.s4, x.s5, x.s6, x.s7);
}
'''

from clops import cl
import numpy as np

k = cl.kernels(src, options="")

t0 = np.random.randint(-3,3,[8, 8], dtype=np.int32)
t1 = cl.tensor(t0)

print(t0)
k.enqueue("test", [8], [8], t1)
