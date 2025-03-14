#!/usr/bin/python3
from clops import cl
import numpy as np
import sys

z = np.zeros([2, 3], dtype=np.float32)
z[0,0] = 1.3
z[0,1] = 2.3
z[1,0] = 1.4
z[1,1] = 2.4

t = cl.tensor(z)
print(f"t.shape={t.shape}, t.numpy()=\n{t.numpy()}")

src = r'''
    #include <cm/cm.h>
    #include <cm/cmtl.h>
    extern "C" _GENX_MAIN_ void cm_check(float* pt) {
        unsigned int id = cm_linear_global_id();
        printf("============%d, %f\n",id, pt[id]);
    }
'''
k = cl.kernels(src, "-cmc")
k.enqueue("cm_check", [2,3],[1,1], t)
cl.finish()

