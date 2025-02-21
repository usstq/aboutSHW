import csrc
import torch
import numpy as np
import time, sys
import numa

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
# clear && numactl -C48-95 -m1 python test.py

def test_amx_repack_B():
    np.random.seed(0)
    src = np.random.randint(low=-100, high=100, size=(160, 160)).astype(np.float32)
    dst = csrc.test_amx_repack_B(src)
    dst = csrc.test_amx_repack_B(src)
    dst = csrc.test_amx_repack_B(src)
    if (dst != np.transpose(src[:16, :16])).any():
        print(src)
        print(dst)
        assert False, "amx_repack_B failed!"

class testcase:
    def __init__(self, x_dtype, w_dtype,  M, K, N) -> None:
        self.N = N
        self.M = M
        self.K = K
        self.W0 = torch.randint(low=-1, high=2, size=(N[0], K)).to(dtype = w_dtype).detach() # -1, 0, 1
        self.W1 = torch.randint(low=-1, high=2, size=(N[1], K)).to(dtype = w_dtype).detach() # -1, 0, 1
        self.W2 = torch.randint(low=-1, high=2, size=(N[2], K)).to(dtype = w_dtype).detach() # -1, 0, 1
        self.X = torch.randint(low=-1, high=2, size=(M, K)).to(dtype = x_dtype).detach() # -1, 0, 1

        assert x_dtype == torch.int8 or x_dtype == torch.float16 or x_dtype == torch.bfloat16

        if x_dtype == torch.int8:
            acc_dtype = torch.int32
            o_dtype = torch.int32
        if x_dtype == torch.float16 or x_dtype == torch.bfloat16:
            acc_dtype = torch.float32
            o_dtype = torch.float32

        self.x_dtype = x_dtype
        self.w_dtype = w_dtype
        self.o_dtype = o_dtype
        
        self.Y0 = (self.X.to(dtype=acc_dtype) @ self.W0.transpose(0,1).to(dtype=acc_dtype)).to(dtype=o_dtype).detach()
        self.Y1 = (self.X.to(dtype=acc_dtype) @ self.W1.transpose(0,1).to(dtype=acc_dtype)).to(dtype=o_dtype).detach()
        self.Y2 = (self.X.to(dtype=acc_dtype) @ self.W2.transpose(0,1).to(dtype=acc_dtype)).to(dtype=o_dtype).detach()
        self.Z0 = torch.zeros([M, N[0]], dtype=o_dtype).detach()
        self.Z1 = torch.zeros([M, N[1]], dtype=o_dtype).detach()
        self.Z2 = torch.zeros([M, N[2]], dtype=o_dtype).detach()

        self.X = self.to_numpy(self.X)
        self.Z0 = self.to_numpy(self.Z0)
        self.Z1 = self.to_numpy(self.Z1)
        self.Z2 = self.to_numpy(self.Z2)
        self.W0 = self.to_numpy(self.W0)
        self.W1 = self.to_numpy(self.W1)
        self.W2 = self.to_numpy(self.W2)

    def to_torch(self, n):
        t = torch.from_numpy(n)
        if t.dtype == torch.int16:
            t = t.view(dtype = torch.bfloat16)
        return t
            
    def to_numpy(self, t):
        if t.dtype == torch.bfloat16:
            return t.detach().view(dtype=torch.int16).numpy()
        return t.detach().numpy()

    def check(self):
        Z0 = self.to_torch(self.Z0)
        Z1 = self.to_torch(self.Z1)
        Z2 = self.to_torch(self.Z2)
        if not torch.allclose(self.Y0, Z0):
            print(self.Y0)
            print(Z0)
            print(np.where((self.Y0 != Z0).numpy()))
            assert False, "Y0 != Z0"
        if not torch.allclose(self.Y1, Z1):
            print(self.Y1)
            print(Z1)
            print(np.where((self.Y1 != Z1).numpy()))
            assert False, "Y1 != Z1"
        if not torch.allclose(self.Y2, Z2):
            print(self.Y2)
            print(Z2)
            print(np.where((self.Y2 != Z2).numpy()))
            assert False, "Y2 != Z2"

    def dtname(self, dt):
        return str(dt).replace("torch","").replace("bfloat16","bf16").replace("float32","fp32")

    def test(self, nthr, layers, expect_latency_ms = 0):
        csrc.set_nthr(nthr)
        t0 = time.time()
        qkv_projs = [csrc.AMXQKVLinear(self.W0, self.W1, self.W2) for _ in range(layers)]
        build_ms = (time.time() - t0)/layers * 1e3
        latency_ms = []
        for r in range(5):
            t0 = time.time()
            for qkv in qkv_projs:
                qkv.forward(self.X, self.Z0, self.Z1, self.Z2)
            latency_ms.append((time.time() - t0)/layers * 1e3)
        self.check()
        min_latency = float(np.amin(latency_ms))
        std_latency = float(np.std(latency_ms[1:]))
        mem_bw_GB = (self.W0.nbytes + self.W1.nbytes + self.W2.nbytes) * 1e-6 / min_latency
        GFLOPS = self.M * self.K * np.sum(self.N) * 2 * 1e-6 / min_latency
        tag = f"{self.dtname(self.x_dtype)}{self.dtname(self.w_dtype)}{self.dtname(self.o_dtype)},t{nthr}M{self.M}K{self.K}N{self.N}L{layers}"
        ColorsCode = Colors.GREEN
        if expect_latency_ms > 0:
            if min_latency > expect_latency_ms:
                ColorsCode = Colors.RED
        print(f"{tag:60s} : build-{build_ms : .2f}ms  exec:{ColorsCode}min{min_latency : .2f}{Colors.END}+SD{std_latency:.2f}ms   {mem_bw_GB:.2f} GB/s   {GFLOPS/nthr:.2f} GFLOPS/core")
        return self


#test_amx_repack_B()

def test_perf_with_ncores():
    layers = 100
    ttt = testcase(torch.int8, torch.int8, 256, 4096, [4096, 4096, 4096])
    ttt.test(24, layers)
    ttt.test(24, layers)
    ttt.test(24, layers)
    ttt.test(24, layers)
    for nthr in range(1, 48):
        ttt.test(48, layers)
        ttt.test(nthr, layers)

def testall(layers = 100):
    testcase(torch.int8, torch.int8, 1, 4096, [4096, 4096, 4096]).test(24, layers)
    testcase(torch.int8, torch.int8, 8, 4096, [4096, 4096, 4096]).test(24, layers)
    testcase(torch.int8, torch.int8, 16, 4096, [4096, 4096, 4096]).test(24, layers)
    testcase(torch.int8, torch.int8, 32, 4096, [4096, 4096, 4096]).test(24, layers)

numa.schedule.run_on_nodes(1)
numa.memory.set_membind_nodes(1)

testcase(torch.bfloat16, torch.bfloat16, 2500, 1024, [1024, 1024, 1024]).test(48, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 2500, 1024, [1024, 1024, 1024]).test(47, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 2500, 1024, [1024, 1024, 1024]).test(47, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 2500, 1024, [1024, 1024, 1024]).test(47, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 10000, 512, [512, 512, 512]).test(48, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 10000, 512, [512, 512, 512]).test(48, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 10000, 512, [512, 512, 512]).test(48, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 10000, 512, [512, 512, 512]).test(48, 100, 0.45)
sys.exit(0)

testcase(torch.int8, torch.int8, 1, 4096, [4096, 4096, 4096]).test(48, 100, 0.3)
testcase(torch.int8, torch.int8, 256, 4096, [4096, 4096, 4096]).test(48, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 1, 4096, [4096, 4096, 4096]).test(48, 100, 0.45)
testcase(torch.bfloat16, torch.bfloat16, 256, 4096, [4096, 4096, 4096]).test(48, 100, 0.7)

sys.exit(0)

#testall();sys.exit(0)
test_perf_with_ncores(); sys.exit(0)
