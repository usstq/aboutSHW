#!/usr/bin/python3
from clops import cl
import numpy as np
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

class mmTest:
    def __init__(self, M, K, N):
        np.random.seed(0)
        self.M, self.K, self.N = M, K, N
        self.A = np.random.randint(-1,2,[M, K]).astype(np.float16)
        self.tA = cl.tensor(self.A)
        self.tC = cl.tensor(np.zeros([M, N], dtype=np.float16))
        self.tP1 = cl.tensor()
        self.w_dtype = cl.onednn_dtype.f16
        if self.w_dtype == cl.onednn_dtype.f16:
            self.B = np.random.randint(-1,2,[N, K]).astype(np.float16)
            self.tB = cl.tensor(self.B)
            self.linear = cl.onednn_linear(cl.onednn_dtype.f16, self.w_dtype, 
                                          M, K, N, -1, cl.onednn_matmul_type.none,
                                          cl.onednn_dtype.f16, self.tB, cl.tensor(), cl.tensor())

    def test_perf(self, rounds=30, layers=100):
        for r in range(rounds):
            cl.finish()
            t0 = time.time()
            for l in range(layers):
                self.linear.forward(self.tA, self.tC, self.tP1)
            cl.finish()
            dt = time.time() - t0
            tflops = self.M*self.K*self.N*2/dt*1e-12
            mem_bw = (self.M*self.K + self.N*self.K)*2/dt*1e-6
            print(f" [{r}] : {dt*1e3:.3f} ms  {tflops:.3f} TFLOPS   {mem_bw:.1f} GB/s")
        
    def test_acc(self):
        if self.w_dtype == cl.onednn_dtype.f16:
            print("calculate ref...")
            C = self.A @ self.B.transpose()
            print("calculate impl")
            self.linear.forward(self.tA, self.tC, self.tP1)
            print("compare")
            cl.finish()
            C1 = self.tC.numpy()
            print("compare")
            if not np.allclose(C, C1):
                print(C)
                print(C1)
            else:
                print("================ PASSED ==================" , self.M, self.K, self.N, self.w_dtype)
            
mm = mmTest(32, 768, 2048)
mm.test_acc()
mm.test_perf()