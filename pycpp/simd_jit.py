import numpy as np
import pycpp

@pycpp.clib("-march=core-avx2 -g")
def mylib():
    return r'''
#define JIT_DEBUG 1
#include "simd_jit.hpp"

'''

np.random.seed(0)
np.set_printoptions(linewidth = 180)

M = 4
N = 3
src1 = np.random.rand(M, N).astype(np.float32)
src2 = np.random.rand(M, N).astype(np.float32)*0 + 1
dst = np.random.rand(M, N).astype(np.float32) * 0 + 0.123456

print(src1)
print(src2)

mylib.test(src1, src2, dst, M, N)

print(dst)
