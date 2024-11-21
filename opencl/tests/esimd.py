#!/usr/bin/python3
from clops import cl
import numpy as np
import sys

print(dir(cl))

vRANGE = 2
np.random.seed(0)
M = 8
N = 8
K = 16
A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [N, K]).astype(np.float16)
C = np.matmul(A, B.transpose(1,0))

tA = cl.tensor(A)
tB = cl.tensor(B)
cl.test_esimd(tA, tB)

assert np.allclose(tA.numpy(), A*B)


tC = cl.test_dpas(tA, tB)
print(tA.numpy())
print(tB.numpy())
print(tC.numpy())
print(C)