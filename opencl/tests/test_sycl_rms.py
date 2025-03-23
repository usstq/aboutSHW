#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import torch

#print(dir(cl))

vRANGE = 2
np.random.seed(0)
B = 8
# TODO: esimd support tails(1~31)
HS = 1024 + 32*1
A = np.random.randint(-vRANGE, vRANGE+1, [B, HS]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [1, HS]).astype(np.float16)

eps = 0.01
tA = cl.tensor(A)
tB = cl.tensor(B)
tC = cl.rms(tA, tB, eps)

def ref_rms(a, b, eps):
    a1 = torch.from_numpy(a)
    b1 = torch.from_numpy(b)
    variance = a1.pow(2).mean(-1, keepdim=True)
    return (a1 * torch.rsqrt(variance + eps) * b1).cpu().detach().numpy()

ref = ref_rms(A, B, eps)
cur = tC.numpy()
if not np.allclose(cur, ref, rtol=0.01, atol=0.01):
    print(f'{cur=}\n{ref=}')
    assert(False)
print('done')