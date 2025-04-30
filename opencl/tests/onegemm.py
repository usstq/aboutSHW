#!/usr/bin/python3
from clops import cl
import numpy as np
import sys
import torch
import time, sys

def pack_i8_to_i4(B_q):
    assert B_q.ndim == 2
    K = B_q.shape[0]
    N = B_q.shape[1]
    B_q4 = np.zeros([K, N//2], dtype=np.int8)
    for k in range(K):
        for n in range(N//2):
            even = (B_q[k,2*n]) & 0xF
            odd = (B_q[k,2*n+1]) & 0xF
            B_q4[k,n] = even | (odd << 4)
    return B_q4

class Weight:
    def __init__(self, K, N, K_group_size, w_dtype):
        if w_dtype == cl.onednn_dtype.f16:
            self.weight = np.random.randint(-1,2,[K, N]).astype(np.float16)
            self.weight_q = self.weight.transpose().copy()
            self.nbytes = self.weight_q.nbytes
        elif w_dtype == cl.onednn_dtype.u8:
            assert (K % K_group_size) == 0
            K_groups = K // K_group_size
            B_q = np.random.randint(0,256,[K, N]).astype(np.uint8)
            B_scale = np.random.randint(-1,2,[K_groups, N]).astype(np.float16)
            B_zp = np.random.randint(126,130,[K_groups, N]).astype(np.uint8)
            self.weight = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)
            self.weight_q = B_q.transpose().copy()
            self.weight_scale = B_scale
            self.weight_zp = B_zp
            self.nbytes = self.weight_q.nbytes + self.weight_scale.nbytes + self.weight_zp.nbytes
        elif w_dtype == cl.onednn_dtype.u4:
            assert (K % K_group_size) == 0
            K_groups = K // K_group_size
            B_q = np.random.randint(0,3,[K, N]).astype(np.int8)
            B_scale = np.random.randint(-4,5,[K_groups, N]).astype(np.float16)
            B_zp = np.random.randint(0,3,[K_groups, N]).astype(np.int8)
            self.weight = (B_q.astype(np.float32) - B_zp.astype(np.float32).repeat(K_group_size, axis=0)) * B_scale.repeat(K_group_size, axis=0)
            self.weight_q = pack_i8_to_i4(B_q.transpose().copy())
            self.weight_scale = B_scale
            self.weight_zp = pack_i8_to_i4(B_zp)
            self.nbytes = self.weight_q.nbytes + self.weight_scale.nbytes + self.weight_zp.nbytes
        else:
            assert False


def cm_xmx_int4(x, weight, scale, zp, K_group_size):
    '''
    weight: [N, K//2, 2] int4
    scale: [K_groups, N] float16
    zp   : [K_groups, N] int8
    '''
    
    pass


def test_mm(M, K, N, K_group_size, w_dtype):
    if K_group_size <= 0: K_group_size = K
    np.random.seed(0)
    A = np.random.randint(-1,2,[M, K]).astype(np.float16)
    tA = cl.tensor(A)
    tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
    tP1 = cl.tensor()
    weight = Weight(K, N, K_group_size, w_dtype)
    Layers = []
    if w_dtype == cl.onednn_dtype.f16:
        weight_bytes = 0
        while weight_bytes < 1e9:
            tB = cl.tensor(weight.weight_q)
            linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, -1, cl.onednn_matmul_type.none,
                                      cl.onednn_dtype.f16, tB, cl.tensor(), cl.tensor())
            Layers.append(linear)
            weight_bytes += weight.nbytes
    elif w_dtype == cl.onednn_dtype.u8:
        weight_bytes = 0
        while weight_bytes < 1e9:
            tB = cl.tensor(weight.weight_q)
            tBs = cl.tensor(weight.weight_scale)
            tBz = cl.tensor(weight.weight_zp)
            linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, K_group_size, cl.onednn_matmul_type.none,
                                      cl.onednn_dtype.u8, tB, tBs, tBz)
            Layers.append(linear)
            weight_bytes += weight.nbytes
    elif w_dtype == cl.onednn_dtype.u4:
        weight_bytes = 0
        while weight_bytes < 1e9:
            tBq4 = cl.tensor(weight.weight_q)
            tBs = cl.tensor(weight.weight_scale)
            tBz = cl.tensor(weight.weight_zp)
            linear = cl.onednn_linear(cl.onednn_dtype.f16, w_dtype, M, K, N, K_group_size, cl.onednn_matmul_type.none,
                                      cl.onednn_dtype.u4, tBq4, tBs, tBz)
            Layers.append(linear)
            weight_bytes += weight.nbytes

    tP1 = cl.tensor()
    if M < 256:
        C = A @ weight.weight
        print("ref is calculated!")
        linear.forward(tA, tC, tP1)
        cl.finish()
        C1 = tC.numpy()
        if not np.allclose(C, C1):
            print(C)
            print(C1)
        else:
            print("================ PASSED ==================" , M, K, N, w_dtype)
    durs = []
    for r in range(10):
        t0 = time.time()
        for linear in Layers:
            linear.forward(tA, tC, tP1)
        cl.finish()
        durs.append((time.time() - t0)/len(Layers))
    dt = sum(durs[3:])/len(durs[3:])
    print(f"{M=} {K=} {N=} {w_dtype} {weight_bytes*1e-9:.2f}GB : [{r}] {dt*1e3:.3f} ms {M*K*N*2*1e-12/dt:.3f} TFLOPS")


#for M in [200,400,600,1000]: test_mm(M=M, K=2048, N=768, K_group_size=128, w_dtype=cl.onednn_dtype.f16)
#for M in [200,400,600,1000]: test_mm(M=M, K=2048, N=768, K_group_size=-1, w_dtype=cl.onednn_dtype.u8)
for M in [200,400,600,1000]: test_mm(M=M, K=2048, N=768, K_group_size=128, w_dtype=cl.onednn_dtype.u4)

log = r'''
================ PASSED ================== 200 2048 768 onednn_dtype.f16
M=200 K=2048 N=768 onednn_dtype.f16 1.00GB : [9] 0.019 ms 32.999 TFLOPS
M=400 K=2048 N=768 onednn_dtype.f16 1.00GB : [9] 0.043 ms 28.988 TFLOPS
M=600 K=2048 N=768 onednn_dtype.f16 1.00GB : [9] 0.044 ms 43.311 TFLOPS
M=1000 K=2048 N=768 onednn_dtype.f16 1.00GB : [9] 0.037 ms 85.407 TFLOPS
ref is calculated!
M=200 K=2048 N=768 onednn_dtype.u8 1.00GB : [9] 0.024 ms 25.793 TFLOPS
M=400 K=2048 N=768 onednn_dtype.u8 1.00GB : [9] 0.027 ms 47.319 TFLOPS
M=600 K=2048 N=768 onednn_dtype.u8 1.00GB : [9] 0.027 ms 71.013 TFLOPS
M=1000 K=2048 N=768 onednn_dtype.u8 1.00GB : [9] 0.037 ms 84.749 TFLOPS
ref is calculated!
================ PASSED ================== 200 2048 768 onednn_dtype.u4
M=200 K=2048 N=768 onednn_dtype.u4 1.00GB : [9] 0.040 ms 15.619 TFLOPS
M=400 K=2048 N=768 onednn_dtype.u4 1.00GB : [9] 0.040 ms 31.190 TFLOPS
M=600 K=2048 N=768 onednn_dtype.u4 1.00GB : [9] 0.040 ms 46.722 TFLOPS
M=1000 K=2048 N=768 onednn_dtype.u4 1.00GB : [9] 0.040 ms 77.732 TFLOPS
'''
