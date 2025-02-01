import subprocess
import numpy as np
import sys

num_hidden_layers = 26
hidden_size = 3200
intermediate_size = 8640


# make csrc & import lib
subprocess.check_output(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Debug"], shell=False)
subprocess.check_output(["cmake", "--build", "build", "--config", "Debug"], shell=False)
from build import lut_gemm

print(dir(lut_gemm))
a = np.random.randint(low=-1, high=2, size=(3, 4), dtype=np.int8)
c = lut_gemm.pack_i2s(a)
print(c.shape, c.strides, c.dtype, c)

def test_mbench_i2s():
    lut_gemm.mbench()

class LUTGemm(object):
    def __init__(self, K, N, M=None):
        self.K = K
        self.N = N
        self.B = np.random.randint(low=-1, high=2, size=(K, N), dtype=np.int8) # -1, 0, 1
        # here we also do offline transformation of the weights
        self.resetM(M)

    def resetM(self, M):
        if M is None:
            self.A = None
        else:
            self.A = np.random.randint(low=-128, high=127, size=(M, self.K), dtype=np.int8)

    def run_ref(self):
        return self.A @ self.B

def test_lut_baisc_ref():
    q = LUTGemm(2, 3, 1)
    C = q.run_ref()
    print(q.A.shape, q.A)
    print(q.B.shape, q.B)
    print(C.shape, C)
    assert(C.shape == (1,3))

def test_lut_accuarcy():
    M = 1
    K = 3200
    N = 3200
    gemm = LUTGemm(K, N, M)
    ref = gemm.run_ref()
    
    pass


def xxx():
    # raw weights are int8
    # transform ternary weights (-1, 0, 1) into optimized layout
    mylib.transform_weight()

    # input is per-token int8-quantized
    # and dequantization is fused so output is fp32
    mylib.lut_gemm()

    func = getattr(mylib, "test")
    func()
