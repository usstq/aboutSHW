#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import time, sys

# on lunarlake, it seems the gpu memory which uses device memory should be easy swapped
#  but gpu memory with malloc_host will not
# set CLDEBUG=1 python moe-mem.py # using malloc_device, should: RuntimeError: clEnqueueNDRangeKernel("mymemset",...) failed with CL_OUT_OF_RESOURCES (-5)
# set CLDEBUG=5 python moe-mem.py # using malloc_host, should be ok
mytensors = []
N=2048
K=768
shape_w=[N, K//2]
shape_scale=[N, K//128*2]
shape_zp=[N, K//128//2]
# 48 layers, 128 experts each, 3 fc each, total about 15GB
size = 48*128*3
# simulate 8G additional memory
fake = np.ones([1024*1024*1024*2]).astype(np.float32)

for i in range(size):
    w = np.ones(shape_w).astype(np.uint8)
    tw = cl.tensor(w)
    scale = np.ones(shape_scale).astype(np.uint8)
    ts = cl.tensor(scale)
    zp = np.ones(shape_zp).astype(np.uint8)
    tz = cl.tensor(zp)
    mytensors.append([tw, ts, tz])

def direct():
    code = r'''
    __kernel void mymemset(
        __global uchar* y) {
        int n = get_global_id(0);
        y[n] = 0;
    }
    '''
    ocl_kernels = cl.kernels(code)

    input("press to continue")
    fake[:] = 0

    for j in range(10):
        print(f'infer no {j} ...')
        for i in range(size):
            if i % 1024 == 0:
                fake[:] = 0
            layer = mytensors[i]
            ocl_kernels.enqueue('mymemset', [sum(shape_w)], [512], layer[0])
            ocl_kernels.enqueue('mymemset', [sum(shape_scale)], [512], layer[1])
            ocl_kernels.enqueue('mymemset', [sum(shape_zp)], [512], layer[2])

def indirect():
    code = r'''
    __kernel void mymemset(
        const __global uintptr_t* weight_ptrs, int base_id) {
        __global uchar* y = (const __global uchar*)weight_ptrs[base_id];
        int n = get_global_id(0);
        y[n] = 0;
    }
    '''
    ocl_kernels = cl.kernels(code)

    add_w = np.ones([size, 3]).astype(np.uint64)
    for i in range(size):
        add_w[i, 0] = mytensors[i][0].addr
        add_w[i, 1] = mytensors[i][1].addr
        add_w[i, 2] = mytensors[i][2].addr
    ta = cl.tensor(add_w)

    input("press to continue")
    fake[:] = 0

    for j in range(10):
        print(f'infer no {j} ...')
        for i in range(size):
            if i % 1024 == 0:
                fake[:] = 0
            ocl_kernels.enqueue('mymemset', [sum(shape_w)], [512], ta, i * 3 + 0)
            ocl_kernels.enqueue('mymemset', [sum(shape_scale)], [512], ta, i * 3 + 1)
            ocl_kernels.enqueue('mymemset', [sum(shape_zp)], [512], ta, i * 3 + 2)

#direct()
indirect()
print('done.')