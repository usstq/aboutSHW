'''
#########################################################
numactl -C0 python ./test.py
============== torch.nn.Linear ============ 
95.727 ms/linear
95.443 ms/linear
95.432 ms/linear
============== torch.matmul ============ 
94.658 ms/linear
94.424 ms/linear
94.310 ms/linear
============== my gemm ============ 
118.018 ms/linear
117.265 ms/linear
116.250 ms/linear

#########################################################
numactl -C0,2 python ./test.py 
============== torch.nn.Linear ============ 
48.996 ms/linear
48.996 ms/linear
48.906 ms/linear
============== torch.matmul ============ 
49.078 ms/linear
48.866 ms/linear
48.784 ms/linear
============== my gemm ============ 
59.755 ms/linear
59.388 ms/linear
59.510 ms/linear

#########################################################
numactl -C0,2,4,6,8,10,12,14 python ./test.py
============== torch.nn.Linear ============ 
13.978 ms/linear
14.504 ms/linear
14.710 ms/linear
============== torch.matmul ============ 
14.172 ms/linear
14.612 ms/linear
14.769 ms/linear
============== my gemm ============ 
18.405 ms/linear
17.833 ms/linear
17.936 ms/linear

'''
import pycpp
import torch
import time
import psutil
import gc
import sys
import numpy as np

# pycpp.test_basic(2560);sys.exit(0)

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
    YELLOW = "\033[33m"
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

@pycpp.clib
def mylib():
    return r'''
#include "common.hpp"
extern "C" void test(const char * str) {
    std::cout << str << std::endl;
    int64_t a = std::strtoull(str, NULL, 0);
    std::cout << a << std::endl;
}
'''
mylib.test("0xfffffffffffffe00")


print(dir(pycpp))

print(torch.backends.mkldnn.is_available())
print(*torch.__config__.show().split("\n"), sep="\n")

class MemTracer:
    def __init__(self):
        self.process = psutil.Process()
        self.mi = self.process.memory_info()
        print(f"VMS/RSS  base: {self.mi.vms*1e-6: .3f} / {self.mi.rss*1e-6: .3f} MB")
    def update(self, info):
        mi = self.process.memory_info()
        print(f"[VMS/RSS] after {info}: {(mi.vms - self.mi.vms)*1e-6: .3f} / {(mi.rss - self.mi.rss)*1e-6: .3f} MB ")
        self.mi = mi

mtracer = MemTracer()

def compare(ref, opt, atol=0.01, rtol=0.01):
    if not np.allclose(ref, opt, atol=atol, rtol=rtol):
        pos = np.where(np.isnan(opt))
        if len(pos[0]) > 0:
            print(f'========================================================')
            print(f'pos nan = {len(pos)}x{len(pos[0])} {pos}')
            print(f'opt nan = {opt[pos]}')
            raise Exception("failed.")

        pos = np.where(np.abs(ref - opt) > atol)
        print(f'========================================================')
        print(f'compare failed (ref={ref.dtype} {ref.shape}, opt={opt.dtype} {opt.shape}, atol={atol}, rtol={rtol})')
        print(f'pos = {len(pos)}x{len(pos[0])} {pos}')
        print(f'ref_val = {ref[pos]}')
        print(f'opt_val = {opt[pos]}')
        raise Exception("failed.")
    else:
        print(f"allclose for {ref.dtype}{ref.shape} vs {opt.dtype}{opt.shape}")

from torch.utils import mkldnn as mkldnn_utils;

import atexit
output_log_str = ""
def show_final_logs():
    if len(output_log_str):
        print("========================================:")
        print(output_log_str)
atexit.register(show_final_logs)

def test(M, K, N, REPEAT=3, LAYERS=-1):
    global output_log_str
    with torch.no_grad():
        device = torch.device("cpu")
        A = torch.randint(-8, 8, [M, K]).float().to(device)
        B = torch.randint(-8, 8, [K, N]).float().to(device)
        C = torch.randint(-8, 8, [M, N]).float().to(device)

        # number of layers are limited by max weight memory footprint & max gflops tested
        if LAYERS <= 0:
            total_gflops = 300e9
            LAYERS = min(int(total_gflops//(M*K*N*2)),
                        int(2e9//(K*N*4)))

        print(f" ******* test M={M}, K={K}, N={N}, LAYERS={LAYERS} weights:{K*N*4*LAYERS/1024:.1f}(KB) ******* ")
        
        mtracer.update("init")

        if False:
            net = []
            for i in range(LAYERS):
                node = torch.nn.Linear(K, N)
                #node = mkldnn_utils.to_mkldnn(node)
                net.append(node)
        
            mtracer.update("torch.nn.Linear initialize")
            #batch = A.to_mkldnn()
            batch = A
            for r in range(REPEAT):
                t0 = time.time()
                for n in net:
                    with pycpp.perf("torch.nn.Linear"):
                        n(batch)
                t1 = time.time()
                dt = (t1-t0)/len(net)
                
                print(f"{dt*1e3:.3f} ms/linear {M*K*N*2/dt*1e-9:.3f} GFLOPS")

            mtracer.update("torch.nn.Linear infer")

        print("============== torch.matmul ============ ")
        net2 = []
        for i in range(LAYERS):
            w = torch.randint(-8, 8, [K, N]).float().to(device)
            net2.append(w)
        mtracer.update("torch.matmul duplicates weight")

        avg_gflops = 0
        for r in range(REPEAT):
            t0 = time.time()
            for wei in net2:
                with pycpp.perf("torch.matmul"):
                    Cref = torch.matmul(A, wei)
            t1 = time.time()
            dt = (t1-t0)/len(net2)
            gflops = M*K*N*2/dt*1e-9
            if (r > 0): avg_gflops += gflops
            print(f"{dt*1e3:.3f} ms/linear {gflops:.3f} GFLOPS")
        avg_gflops_matmul = avg_gflops / (REPEAT-1)

        mtracer.update("torch.matmul execution")
        print("============== my gemm ============ ")
        src = A.numpy()
        dst = C.numpy()
        avg_gflops = 0
        for r in range(REPEAT):
            t0 = time.time()
            for wei in net2:
                with pycpp.perf("pycpp.gemm"):
                    pycpp.gemm(src, wei.numpy(), dst)
            t1 = time.time()
            dt = (t1-t0)/len(net2)
            gflops = M*K*N*2/dt*1e-9
            if (r > 0): avg_gflops += gflops
            print(f"{dt*1e3:.3f} ms/linear {M*K*N*2/dt*1e-9:.3f} GFLOPS")
        avg_gflops_mygemm = avg_gflops / (REPEAT-1)

        mtracer.update("my gemm execution")

        compare(Cref.numpy(), C.numpy())
        
        if avg_gflops_mygemm > avg_gflops_matmul * 0.95:
            color = Colors.LIGHT_CYAN
        else:
            color = Colors.LIGHT_RED
        
        log_str = f"{color}avg_gflops_matmul / avg_gflops_mygemm :  test({M},{K},{N})  {avg_gflops_matmul:.3f} / {avg_gflops_mygemm:.3f}  GFLOPS {Colors.END}\n"
        print(log_str)
        output_log_str += log_str


test(6*8,89200,1024, LAYERS=-1)
sys.exit(0)

test(1,892,896)
test(4,892,896)
test(6,892,896)
test(6*2,892,896)
test(6*4,892,896)
test(6*8,892,896)
sys.exit(0)


#test(4000, 4096, 4096);sys.exit(0)

test(4, 892, 896)
test(40, 892, 896)
test(400, 892, 896)

test(4, 892, 896*4)
test(40, 892, 896*4)
test(400, 892, 896*4)

test(4, 892, 896*8)
test(40, 892, 896*8)
test(400, 892, 896*8)

test(1, 4096, 4096)
test(4, 4096, 4096)
test(40, 4096, 4096)
test(400, 4096, 4096)
test(4000, 4096, 4096)
