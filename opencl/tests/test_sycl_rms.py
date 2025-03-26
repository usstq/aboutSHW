#!/usr/bin/python3
import clops
from clops import cl
import numpy as np
import sys
import torch

#print(dir(cl))
cl.profiling(True)

vRANGE = 2
np.random.seed(0)
B = 1
# TODO: esimd support tails(1~31)
# TODO: xetla support tails(1~255)
HS = 4096 + 32*0
#HS = 128
A = np.random.randint(-vRANGE, vRANGE+1, [B, HS]).astype(np.float16)
B = np.random.randint(-vRANGE, vRANGE+1, [1, HS]).astype(np.float16)

eps = 0.01
tA = cl.tensor(A)
tB = cl.tensor(B)
tC = cl.tensor(A*0)

cl_norm = clops.RMSNorm(weight=tB, epsilon = eps)
cl_norm._profile(tA, tB, tC, eps)

cl.rms(tA, tB, tC, eps)
cl.finish()

def ref_rms(a, b, eps):
    a1 = torch.from_numpy(a)
    b1 = torch.from_numpy(b)
    variance = a1.pow(2).mean(-1, keepdim=True)
    return (a1 * torch.rsqrt(variance + eps) * b1).cpu().detach().numpy()

ref = ref_rms(A, B, eps)
cur = tC.numpy()
if not np.allclose(cur, ref, rtol=0.01, atol=0.01):
    print(f'{cur=}\n{ref=}')
    pos = np.where(np.abs(cur - ref > 0.01))
    print(f"{pos=}\ncur_bad={cur[pos]}\nref_bad={ref[pos]}")
    assert(False)
else:
    print(f'accuracy: good')


cl_norm = clops.RMSNorm(weight=tB, epsilon = eps)

for _ in range(10):
    cl.rms(tA, tB, tC, eps)

latencies = cl.finish()
for ns in latencies:
    print(f'{ns/1e3:.2f} uS')

print("===============================")
for _ in range(10):
    cl_norm._profile(tA, tB, tC, eps)
latencies = cl.finish()
for ns in latencies:
    print(f'{ns/1e3:.2f} uS')
