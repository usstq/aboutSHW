#!/usr/bin/python3
import cl
import numpy as np
import sys

print(dir(cl))

z = np.zeros([2, 3], dtype=np.float32)
z[0,0] = 1.3
z[0,1] = 2.3
z[1,0] = 1.4
z[1,1] = 2.4
print(z)

t = cl.tensor_f32(z)
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


sys.exit(0)

def test():
    t1 = cl.tensor_f32([10,20])
    t2 = cl.tensor_f32([20,20])
    return t1



a = test()
test()
test()