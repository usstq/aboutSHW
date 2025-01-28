import subprocess

# make csrc & import lib
subprocess.check_output(["cmake", "-B", "build"], shell=False)
subprocess.check_output(["cmake", "--build", "build", "--config", "Debug"], shell=False)
from build import lut_gemm

def test_simd_basic():
    lut_gemm.simd_test_basic()

def test_simd_tput():
    lut_gemm.simd_test_tput("all", 50)

def test_simd_printreg():
    lut_gemm.simd_test_printreg()
