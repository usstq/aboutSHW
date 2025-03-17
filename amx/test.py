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

np.set_printoptions(linewidth=400)
torch.set_printoptions(linewidth=400)

def to_torch(n):
    t = torch.from_numpy(n)
    if t.dtype == torch.int16:
        t = t.view(dtype = torch.bfloat16)
    return t

def to_numpy(t):
    if t.dtype == torch.bfloat16:
        return t.detach().view(dtype=torch.int16).numpy()
    return t.detach().numpy()

def test_amx_repack_B():
    np.random.seed(0)
    src = np.random.randint(low=-100, high=100, size=(16, 16)).astype(np.float32)
    dst = csrc.test_amx_repack_B(src)
    dst = csrc.test_amx_repack_B(src)
    dst = csrc.test_amx_repack_B(src)
    
    if (dst != np.transpose(src[:16, :16])).any():
        print(src[:16, :16])
        print(dst)
        assert False, "amx_repack_B failed!"

#test_quant_i8(torch.bfloat16); sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--ncores', type=int, default=0)
parser.add_argument('-N', '--node', type=int, default=1)
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('-r', '--repeat', type=int, default=5)
parser.add_argument('-q', '--quanti8', action="store_true")
parser.add_argument('-dq', '--dynquant', action="store_true")
parser.add_argument('-e', '--expr', action="store_true")

args = parser.parse_args()


## compute bound is set according to
## https://github.com/usstq/mm_amx?tab=readme-ov-file#with-ab-sub-block-prefetched-amx-mm
sys_cap_mem_bw_GBPS = 250
sys_cap_amx_core_GFLOPS = {2:1242, 1:1242*2}  # item-size 2(bf16/fp16) 1(int8)

def quant_tensor(weight, dim):
    abs_max = weight.abs().max(dim=dim, keepdim=True).values.to(torch.float32)
    scale = 127.0/abs_max
    return (weight * scale).clamp(-128, 127).to(dtype = torch.int8), abs_max/127

def test_quant_tensor():
    weight = torch.randint(low=-100, high=100, size=(5, 8)).to(dtype = torch.float16).detach()
    w, s = quant_tensor(weight, 1)
    print(weight)
    print(w, s, w.shape, s.shape)
    print((w * s - weight).abs().max())
    sys.exit(0)

AMXQKVLinear = csrc.AMXQKVLinear

class testcase:
    def __init__(self, x_dtype, w_dtype, M, K, Ns, dyn_quant=False) -> None:
        self.Ns = Ns
        self.M = M
        self.K = K
        torch.manual_seed(0)
        
        self.W = []
        self.Wscales = []
        for N in Ns:
            weight = torch.randint(low=-1, high=2, size=(N, K)).to(dtype = w_dtype).detach()
            if dyn_quant:
                # quantize w into int8-per-OC
                weight, scale = quant_tensor(weight, 1)
                self.Wscales.append(scale)
            self.W.append(weight) # -1, 0, 1

        self.X = torch.randint(low=-1, high=2, size=(M, K)).to(dtype = x_dtype).detach() # -1, 0, 1
        
        self.x_quant, self.x_scale = quant_tensor(self.X, 1)
        
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
            if dyn_quant:
                self.Wscales[i] = self.Wscales[i].transpose(0,1)
                YRef = ((self.x_quant.to(dtype=torch.int32) @ self.W[i].transpose(0,1).to(dtype=torch.int32))\
                            .to(dtype=torch.float) * self.x_scale * self.Wscales[i])\
                            .to(dtype=o_dtype)
            else:
                YRef = (self.X.to(dtype=acc_dtype) @ self.W[i].transpose(0,1).to(dtype=acc_dtype)).to(dtype=o_dtype)
            self.Y.append(YRef)
            self.Z.append(to_numpy(torch.zeros([M, N], dtype=o_dtype)))
            self.W[i] = to_numpy(self.W[i])

        self.X = to_numpy(self.X)
        self.dyn_quant = dyn_quant
        self.Wscales = [to_numpy(w) for w in self.Wscales]

    def check(self):
        for i, (Y, Z) in enumerate(zip(self.Y, self.Z)):
            Z = to_torch(Z)
            if not torch.allclose(Y, Z):
                return False
                print(f"refernce: Y[{i}] ", Y.dtype, Y.shape)
                print(Y)
                print(f"actual: Z[{i}] ", Z.dtype, Z.shape)
                print(Z)
                if self.dyn_quant:
                    for ws in self.Wscales:
                        print("ws:", ws)
                print("diff-location")
                print(torch.nonzero(Y != Z))
                assert False, f"Y{i} != Z{i}"
                return False
        return True

    def dtname(self, dt):
        return str(dt).replace("torch","").replace("bfloat16","bf16").replace("float32","fp32")

    def test(self, nthr, layers, ms = 0, gbps = 0, gflops=0):
        csrc.set_nthr(nthr)
        t0 = time.time()
        qkv_projs = [AMXQKVLinear(self.W) for _ in range(layers)]
        build_ms = (time.time() - t0)/layers * 1e3
        latency_ms = []
        for r in range(5):
            t0 = time.time()
            if self.dyn_quant:
                for qkv in qkv_projs:
                    qkv.forward_dyn_quant_i8(self.X, self.Z, self.Wscales)
            else:
                for qkv in qkv_projs:
                    qkv.forward(self.X, self.Z)
            latency_ms.append((time.time() - t0)/layers * 1e3)
        correct = self.check()
        min_latency = float(np.amin(latency_ms))
        std_latency = float(np.std(latency_ms[1:]))
        mem_bw_GB = np.sum([w.nbytes for w in self.W]) * 1e-6 / min_latency
        GFLOPS = self.M * self.K * np.sum(self.Ns) * 2 * 1e-6 / min_latency
        tag = f"{self.dtname(self.x_dtype)}{self.dtname(self.w_dtype)}{self.dtname(self.o_dtype)},t{nthr}M{self.M}K{self.K}N{self.Ns}L{layers}"
        if ms > 0:
            if min_latency > ms:
                min_latency = f"{Colors.RED}min{min_latency : .2f}{Colors.END}"
            else:
                min_latency = f"{Colors.GREEN}min{min_latency : .2f}{Colors.END}"
        else:
            min_latency = f"min{min_latency : .2f}"
        
        gbperc = f"({mem_bw_GB*100/sys_cap_mem_bw_GBPS:.0f}%)"
        if gbps > 0:
            if mem_bw_GB > gbps:
                mem_bw_GB = f"{Colors.GREEN}{mem_bw_GB : .2f}{Colors.END}GB/s {gbperc}"
            else:
                mem_bw_GB = f"{Colors.RED}{mem_bw_GB : .2f}{Colors.END}GB/s {gbperc}"
        else:
            mem_bw_GB = f"min{mem_bw_GB : .2f}GB/s {gbperc}"
        
        GFLOPS = GFLOPS/nthr
        peak_GFLOPS = sys_cap_amx_core_GFLOPS[self.w_dtype.itemsize]
        GFLOPS_perc = f"({GFLOPS*100/peak_GFLOPS:.0f}%)"
        if gflops > 0:
            if GFLOPS > gflops:
                GFLOPS = f"{Colors.GREEN}{GFLOPS : .2f}{Colors.END}GFLOPS/core {GFLOPS_perc}"
            else:
                GFLOPS = f"{Colors.RED}{GFLOPS : .2f}{Colors.END}GFLOPS/core {GFLOPS_perc}"
        else:
            GFLOPS = f"{GFLOPS : .2f}GFLOPS/core {GFLOPS_perc}"
        
        print(f"{tag:60s} : build-{build_ms : .2f}ms  exec:min{min_latency}+SD{std_latency:.2f}ms {mem_bw_GB}  {GFLOPS} ... {'CORRECT' if correct else 'WRONG'}")
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


def test_mem_bounds(repeat):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    # need to repeat to see if memory bound can reach due to random system mem-accessing noise
    for i in range(repeat):
        testcase(torch.int8, torch.int8, 1, 4096, [4096, 4096, 4096]).test(ncores, 100, gbps=250)
        testcase(torch.bfloat16, torch.bfloat16, 1, 4096, [4096, 4096, 4096]).test(ncores, 100, gbps=250)

def test_compute_bounds(repeat, batch_size):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(repeat):
        testcase(torch.int8, torch.int8, batch_size, 4096, [4096, 4096, 4096]).test(ncores, 100, gflops=1200)
        testcase(torch.bfloat16, torch.bfloat16, batch_size, 4096, [4096, 4096, 4096]).test(ncores, 100, gflops=800)

def test_compute_bounds_bigM(repeat):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(repeat):
        testcase(torch.bfloat16, torch.bfloat16, 2500, 1024, [1024, 1024, 1024]).test(ncores, 100, gflops=700)
        testcase(torch.bfloat16, torch.bfloat16, 10000, 512, [512, 512, 512]).test(ncores, 100, gflops=900)

def test_k_groups(repeat, batch_size):
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(repeat):
        testcase(torch.bfloat16, torch.bfloat16, batch_size, 11008, [4096]).test(ncores, 100, 0.58, gflops=850)
        testcase(torch.bfloat16, torch.bfloat16, batch_size, 4096, [4096]).test(ncores, 100, 0.28, gflops=650)

def test_dyn_quant():
    print(f"======================{inspect.currentframe().f_code.co_name}======================")
    for i in range(args.repeat):
        testcase(torch.bfloat16, torch.bfloat16, 1, 4096, [4096, 4096, 4096], dyn_quant=True).test(ncores, 100, 0.45)
    for i in range(args.repeat):
        testcase(torch.bfloat16, torch.bfloat16, args.batch_size, 4096, [4096, 4096, 4096], dyn_quant=True).test(ncores, 100, 0.68)
    for i in range(args.repeat):
        testcase(torch.bfloat16, torch.bfloat16, args.batch_size, 11008, [4096], dyn_quant=True).test(ncores, 100, 0.58)

#ncores = 4
#testcase(torch.bfloat16, torch.bfloat16, 256, 4096, [4096, 4096, 4096]).test(ncores, 100, 0.45)
#testcase(torch.bfloat16, torch.bfloat16, 256, 4096, [11008*2]).test(ncores, 100, 0.45)
#sys.exit(0)
#test_k_groups(); sys.exit(0)

#test_amx_repack_B(); sys.exit(0)

numactl(args.node, args.ncores)

if args.expr:
    # experimental tests
    AMXQKVLinear = csrc.AMXQKVLinear2
    for i in range(args.repeat):
        testcase(torch.bfloat16, torch.bfloat16, 10000+16, 512, [512, 512, 512], dyn_quant=False).test(ncores, 100, 0.45)
    sys.exit(0)

if args.dynquant:
    test_dyn_quant()
    sys.exit(0)

def test_quant_i8(dtype):
    csrc.set_nthr(ncores)
    torch.manual_seed(0)
    src = torch.randint(low=-100, high=100, size=(256, 4096+15)).to(dtype = dtype).detach()
    npsrc = to_numpy(src)
    layers = 100
    all_src = [npsrc.copy() for _ in range(layers)]
    q = to_numpy(torch.randint(low=-100, high=100, size=src.shape).to(dtype = torch.int8).detach())
    scale = to_numpy(torch.randint(low=-100, high=100, size=(src.shape[0], 1)).to(dtype = torch.float32).detach())
    # warm-up    
    for s in all_src:
        csrc.test_quant_i8(s, q, scale)

    latency_ms = []
    for r in range(5):
        t0 = time.time()
        for s in all_src:
            csrc.test_quant_i8(s, q, scale)
        latency_ms.append((time.time() - t0)/layers * 1e3)
    min_latency = float(np.amin(latency_ms))
    std_latency = float(np.std(latency_ms[1:]))

    #print(q)
    #print(scale[:16, :])
    deq = to_torch(q*scale).to(dtype)
    #print(src)
    abs_diff = (deq-src).abs()
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    print(f" diff_max:{max_abs_diff:.2f} diff_mean:{mean_abs_diff:.2f} min_latency:{min_latency:.2f} + std{std_latency:.2f}ms")

if args.quanti8:
    test_quant_i8(torch.bfloat16)
    test_quant_i8(torch.float16)
    sys.exit(0)

test_mem_bounds(args.repeat)
test_compute_bounds(args.repeat, args.batch_size)
test_compute_bounds_bigM(args.repeat)
test_k_groups(args.repeat,args.batch_size)

sys.exit(0)

test_perf_with_ncores(); sys.exit(0)
