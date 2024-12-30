import numpy as np
import pycpp
import sys
import platform

extra_flags = ""

if platform.machine() == "aarch64":
    extra_flags += "-L../thirdparty/xbyak_aarch64/lib/ -lxbyak_aarch64"

@pycpp.clib(f"-std=c++11 -march=native -g {extra_flags}")
def mylib():
    return r'''
// show predefined macro
// echo | gcc -dM -E -march=native -
#include "simd_jit_tests.hpp"

'''


mylib.test()
sys.exit(0)

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
