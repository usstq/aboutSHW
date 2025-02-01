import subprocess
import numpy as np

# make csrc & import lib
subprocess.check_output(["cmake", "-B", "build"], shell=False)
subprocess.check_output(["cmake", "--build", "build", "--config", "Release"], shell=False)
from build import lut_gemm

def test_simd_basic():
    lut_gemm.simd_test_basic()

def test_simd_tput():
    lut_gemm.simd_test_tput("all", 50)

def test_simd_printreg():
    lut_gemm.simd_test_printreg()

#  pytest -k i2s
def test_mbench_i2s8_M1():
    lut_gemm.mbench_i2s8()

def test_mbench_i2s_M1():
    lut_gemm.mbench_i2s(1)

def test_mbench_i2s_M2():
    lut_gemm.mbench_i2s(2)

def test_mbench_i2s_M3():
    lut_gemm.mbench_i2s(3)

def test_mbench_w4_M1():
    lut_gemm.mbench_w4(1)

def test_mbench_w4_M2():
    lut_gemm.mbench_w4(2)

def test_mbench_w8_M1():
    lut_gemm.mbench_w8(1)

def test_mbench_w8_M2():
    lut_gemm.mbench_w8(2)

def test_mbench_w8_M3():
    lut_gemm.mbench_w8(3)

def test_mbench_tl1_M1():
    lut_gemm.mbench_tl1()

def test_mm_i2s():
    K = 4096
    N = 4096*4
    layers = 100
    np.random.seed(0)

    W = np.random.randint(low=-1, high=2, size=(K, N), dtype=np.int8) # -1, 0, 1
    
    print(W)
    PW = lut_gemm.pack_i2s(W)

    with np.printoptions(formatter={'int':hex}):
        print(PW)

    M = 1
    X = np.random.randint(low=-128, high=128, size=(M, K), dtype=np.int8)
    Yref = X.astype(np.int32) @ (W.astype(np.int32) + 1)

    Y = lut_gemm.mm_i2s(X, PW)
    assert np.allclose(Y, Yref)

    PWS = [PW.copy() for i in range(layers)]
    #for pw in PWS:
    #    Y = lut_gemm.mm_i2s(X, pw)
    # const std::string& title, uint64_t rounds, uint64_t kernel_loops, uint64_t kernel_flops, uint64_t kernel_mem_rbytes
    print(f"MKN=({M},{K},{N},{layers}) {X.nbytes*1e-3:.1f} KB + ({PW.nbytes*1e-3:.1f} KB x {layers})  = {(X.nbytes + layers * PW.nbytes)*1e-6:.1f} MB")
    with lut_gemm.perf().verbose("mm_i2s", len(PWS), K, M*N*K*2, X.nbytes + PW.nbytes):
        for pw in PWS:
            Y = lut_gemm.mm_i2s(X, pw)

test_mm_i2s()
#test_mbench_tl1()
#test_pack_i2s()
#test_mbench_i2s8a_M1()