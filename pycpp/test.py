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

src = r'''
#include "common.hpp"
extern "C" void test(std::vector<KArg>& kargs) {
    std::cout << kargs[0].str << std::endl;
    int64_t a = std::strtoull(kargs[0].str, NULL, 0);
    std::cout << a << std::endl;
}
'''
#kernel = pycpp.kernels(src)
#kernel.call("test", "0xffffffffffffffe0")
#sys.exit(0)

print(dir(pycpp))
M = 400+3
K = 4096
N = 4800+8

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
with torch.no_grad():
    device = torch.device("cpu")
    A = torch.randint(-8, 8, [M, K]).float().to(device)
    B = torch.randint(-8, 8, [K, N]).float().to(device)
    C = torch.randint(-8, 8, [M, N]).float().to(device)

    REPEAT = 3
    LAYERS = 30

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
        print("============== torch.nn.Linear ============ ")
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

    for r in range(REPEAT):
        t0 = time.time()
        for wei in net2:
            with pycpp.perf("torch.matmul"):
                Cref = torch.matmul(A, wei)
        t1 = time.time()
        dt = (t1-t0)/len(net2)
        print(f"{dt*1e3:.3f} ms/linear {M*K*N*2/dt*1e-9:.3f} GFLOPS")

    mtracer.update("torch.matmul execution")
    print("============== my gemm ============ ")
    src = A.numpy()
    dst = C.numpy()
    for r in range(REPEAT):
        t0 = time.time()
        for wei in net2:
            with pycpp.perf("pycpp.gemm"):
                pycpp.gemm(src, wei.numpy(), dst)
        t1 = time.time()
        dt = (t1-t0)/len(net2)
        print(f"{dt*1e3:.3f} ms/linear {M*K*N*2/dt*1e-9:.3f} GFLOPS")

    mtracer.update("my gemm execution")

    compare(Cref.numpy(), C.numpy())

#pycpp.gemm()