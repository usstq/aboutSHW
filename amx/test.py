import csrc
import torch
import numpy as np
import time, sys
import numa
import inspect
import argparse

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
    src = np.random.randint(low=-100, high=100, size=(16, 16)).astype(np.float32)
    dst = csrc.test_amx_repack_B(src)
    dst = csrc.test_amx_repack_B(src)
    dst = csrc.test_amx_repack_B(src)
    
    if (dst != np.transpose(src[:16, :16])).any():
        np.set_printoptions(linewidth=200)
        print(src[:16, :16])
        print(dst)
        assert False, "amx_repack_B failed!"

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--ncores', type=int, default=0)
parser.add_argument('-N', '--node', type=int, default=1)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('-r', '--repeat', type=int, default=5)
args = parser.parse_args()


## compute bound is set according to
## https://github.com/usstq/mm_amx?tab=readme-ov-file#with-ab-sub-block-prefetched-amx-mm
sys_cap_mem_bw_GBPS = 250
sys_cap_amx_core_GFLOPS = {2:1242, 1:1242*2}  # item-size 2(bf16/fp16) 1(int8)

class testcase:
    def __init__(self, x_dtype, w_dtype, M, K, Ns) -> None:
        self.Ns = Ns
        self.M = M
        self.K = K
        torch.manual_seed(0)
        
        self.W = []
        for N in Ns:
            self.W.append(torch.randint(low=-1, high=2, size=(N, K)).to(dtype = w_dtype).detach()) # -1, 0, 1
        self.X = torch.randint(low=-1, high=2, size=(M, K)).to(dtype = x_dtype).detach() # -1, 0, 1

        assert x_dtype == torch.int8 or x_dtype == torch.float16 or x_dtype == torch.bfloat16

        if x_dtype == torch.int8:
            acc_dtype = torch.int32
            o_dtype = torch.int32
        if x_dtype == torch.float16 or x_dtype == torch.bfloat16:
            acc_dtype = torch.float32
            o_dtype = x_dtype

        self.x_dtype = x_dtype
        self.w_dtype = w_dtype
        self.o_dtype = o_dtype
        
        self.Y = []
        self.Z = []
        for i, N in enumerate(Ns):
            YRef = (self.X.to(dtype=acc_dtype) @ self.W[i].transpose(0,1).to(dtype=acc_dtype)).to(dtype=o_dtype)
            self.Y.append(YRef)
            self.Z.append(self.to_numpy(torch.zeros([M, N], dtype=o_dtype)))
            self.W[i] = self.to_numpy(self.W[i])

        self.X = self.to_numpy(self.X)

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
        for i, (Y, Z) in enumerate(zip(self.Y, self.Z)):
            Z = self.to_torch(Z)
            if not torch.allclose(Y, Z):
                print(Y)
                print(Z)
                print(torch.nonzero(Y != Z))
                #assert False, f"Y{i} != Z{i}"
                return False
        return True

    def dtname(self, dt):
        return str(dt).replace("torch","").replace("bfloat16","bf16").replace("float32","fp32")

    def test(self, nthr, layers, expect_latency_ms = 0):
        csrc.set_nthr(nthr)
        t0 = time.time()
        qkv_projs = [csrc.AMXQKVLinear(self.W) for _ in range(layers)]
        build_ms = (time.time() - t0)/layers * 1e3
        latency_ms = []
        for r in range(5):
            t0 = time.time()
            for qkv in qkv_projs:
                qkv.forward(self.X, self.Z)
            latency_ms.append((time.time() - t0)/layers * 1e3)
        correct = self.check()
        min_latency = float(np.amin(latency_ms))
        std_latency = float(np.std(latency_ms[1:]))
        mem_bw_GB = np.sum([w.nbytes for w in self.W]) * 1e-6 / min_latency
        GFLOPS = self.M * self.K * np.sum(self.Ns) * 2 * 1e-6 / min_latency
        tag = f"{self.dtname(self.x_dtype)}{self.dtname(self.w_dtype)}{self.dtname(self.o_dtype)},t{nthr}M{self.M}K{self.K}N{self.Ns}L{layers}"
        ColorsCode = Colors.GREEN
        if expect_latency_ms > 0:
            if min_latency > expect_latency_ms:
                ColorsCode = Colors.RED

        peak_GFLOPS = sys_cap_amx_core_GFLOPS[self.w_dtype.itemsize]
        print(f"{tag:60s} : build-{build_ms : .2f}ms  exec:{ColorsCode}min{min_latency : .2f}{Colors.END}+SD{std_latency:.2f}ms "
              f"  {mem_bw_GB:.2f} GB/s ({mem_bw_GB*100/sys_cap_mem_bw_GBPS:.0f}%)   {GFLOPS/nthr:.2f} GFLOPS/core ({GFLOPS*100/nthr/peak_GFLOPS:.0f}%) ... {'CORRECT' if correct else 'WRONG'}")
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

ncores = 0

def numactl(target_node, cores):
    global ncores
    numa.schedule.run_on_nodes(target_node)
    numa.memory.set_membind_nodes(target_node)
    node_num_cpus = len(numa.info.node_to_cpus(target_node))
    if cores > 0:
        ncores = cores
    else:
        ncores = node_num_cpus//2
    print(f"numactl ncores {ncores} on node {target_node} with {node_num_cpus} cpus")

numactl(args.node, args.ncores)

def test_mem_bounds(repeat):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    # need to repeat to see if memory bound can reach due to random system mem-accessing noise
    for i in range(repeat):
        testcase(torch.int8, torch.int8, 1, 4096, [4096, 4096, 4096]).test(ncores, 100, 0.3)
        testcase(torch.bfloat16, torch.bfloat16, 1, 4096, [4096, 4096, 4096]).test(ncores, 100, 0.45)

def test_compute_bounds(repeat, batch_size):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(repeat):
        testcase(torch.int8, torch.int8, batch_size, 4096, [4096, 4096, 4096]).test(ncores, 100, 0.43)
        testcase(torch.bfloat16, torch.bfloat16, batch_size, 4096, [4096, 4096, 4096]).test(ncores, 100, 0.68)

def test_compute_bounds_bigM(repeat):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(repeat):
        testcase(torch.bfloat16, torch.bfloat16, 2500, 1024, [1024, 1024, 1024]).test(ncores, 100, 0.48)
        testcase(torch.bfloat16, torch.bfloat16, 10000, 512, [512, 512, 512]).test(ncores, 100, 0.42)

def test_k_groups(repeat, batch_size):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(repeat):
        testcase(torch.bfloat16, torch.bfloat16, batch_size, 11008, [4096]).test(ncores, 100, 0.58)
        testcase(torch.bfloat16, torch.bfloat16, batch_size, 4096, [4096]).test(ncores, 100, 0.28)

#ncores = 4
#testcase(torch.bfloat16, torch.bfloat16, 256, 4096, [4096, 4096, 4096]).test(ncores, 100, 0.45)
#testcase(torch.bfloat16, torch.bfloat16, 256, 4096, [11008*2]).test(ncores, 100, 0.45)
#sys.exit(0)
#test_k_groups(); sys.exit(0)

#test_amx_repack_B(); sys.exit(0)

test_mem_bounds(args.repeat)
test_compute_bounds(args.repeat, args.batch_size)
test_compute_bounds_bigM(args.repeat)
test_k_groups(args.repeat,args.batch_size)

sys.exit(0)

test_perf_with_ncores(); sys.exit(0)
