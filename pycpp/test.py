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

if False:
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


pycpp.test_reg_blocking(2560)
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

def test(M, K, N, REPEAT=3, LAYERS=-1, name=""):
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
        if LAYERS <= 0:
            LAYERS = 4

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
        avg_dt = 0
        for r in range(REPEAT):
            t0 = time.time()
            for wei in net2:
                with pycpp.perf("torch.matmul"):
                    Cref = torch.matmul(A, wei)
            t1 = time.time()
            dt = (t1-t0)/len(net2)
            gflops = M*K*N*2/dt*1e-9
            if (r > 0):
                avg_gflops += gflops
                avg_dt += dt
            print(f"{dt*1e3:.3f} ms/linear {gflops:.3f} GFLOPS")
        avg_gflops_matmul = avg_gflops / (REPEAT-1)
        avg_dt_matmul = avg_dt / (REPEAT-1)

        mtracer.update("torch.matmul execution")
        print("============== my gemm ============ ")
        src = A.numpy()
        dst = C.numpy()
        avg_gflops = 0
        avg_dt = 0
        for r in range(REPEAT):
            t0 = time.time()
            for wei in net2:
                with pycpp.perf("pycpp.gemm"):
                    pycpp.gemm(src, wei.numpy(), dst)
            t1 = time.time()
            dt = (t1-t0)/len(net2)
            gflops = M*K*N*2/dt*1e-9
            if (r > 0):
                avg_gflops += gflops
                avg_dt += dt
            print(f"{dt*1e3:.3f} ms/linear {M*K*N*2/dt*1e-9:.3f} GFLOPS")
        avg_gflops_mygemm = avg_gflops / (REPEAT-1)
        avg_dt_mygemm = avg_dt / (REPEAT-1)

        mtracer.update("my gemm execution")

        compare(Cref.numpy(), C.numpy())
        
        dt_ratio = avg_dt_mygemm / avg_dt_matmul

        if dt_ratio < 1.05:
            color = Colors.LIGHT_CYAN
            info = f"[ PASS : {dt_ratio:.2f}]"
        else:
            color = Colors.LIGHT_RED
            info = f"[FAILED: {dt_ratio:.2f}]"

        log_str = f"{info} {name} {color}avg_gflops_matmul / avg_gflops_mygemm :  test({M},{K},{N})  {avg_gflops_matmul:.3f} / {avg_gflops_mygemm:.3f}  GFLOPS   {avg_dt_matmul*1e6:.0f}/{avg_dt_mygemm*1e6:.0f} us {Colors.END}\n"
        print(log_str)
        output_log_str += log_str

if 0:
    import timeit
    for n in range(1, 100, 8):
        pycpp.test_brgemm_6x2(2560, 16*n, 20000)
        #print(n, timeit.timeit(lambda: pycpp.test_brgemm_6x2(2560, 16*n, 20000), number=1))
    sys.exit(0)


#cs.test(892, 64*16*4)
#cs.test(892, 7168*4)
#cs.test(892, (7168+16)*4)
sys.exit(0)
#def test_stride_access(stride, cache_size=2048*1024, way=16):

#test(6*8,892,16*100, LAYERS=1);sys.exit(0)
#test(6*2,8920,896)
#test(6*8,8920,896)
#test(6*4,8920,896)
#sys.exit(0)
#for n in [59,60,61]:test(6,400,16*n)
#sys.exit(0)
if 0:
    for k in [127, 160, 195, 255]:
        test(12,k,240*1)
        test(12,k,240*3)
        test(12,k,240*5)
        test(12,k,240*7)
        test(12,k,240*9)
        output_log_str += "\n"
    sys.exit(0)

    for m in [6, 12, 48, 60]:
        test(m,8920,896)
    sys.exit(0)

    for m in [6, 12, 24, 48, 49, 50, 51, 52, 53, 54, 60]:
        test(m,8920,896)
    sys.exit(0)

    for m in [1, 6, 12, 24, 48, 60, 80, 100, 200, 300, 400, 500, 1000]:
        test(m,8920,896)
    sys.exit(0)


    for m in range(1, 60):
        test(m,8920,896)

    sys.exit(0)

    test(1,8920,896)
    test(4,8920,896)
    test(6,8920,896)
    test(6*2,8920,896)
    test(6*4,8920,896)
    test(6*8,8920,896)
    test(6*10,8920,896)
    sys.exit(0)


def test_llm_gemm(name, Ms, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads):
    head_size = hidden_size//num_attention_heads
    q_proj_size = head_size * num_attention_heads
    k_proj_size = head_size * num_key_value_heads
    v_proj_size = head_size * num_key_value_heads
    qkv_proj_size = q_proj_size + k_proj_size + v_proj_size
    print(f"======= {name} ========")
    for m in Ms:
        test(m, hidden_size, hidden_size, name=name) # o_proj
        test(m, hidden_size, q_proj_size, name=name)
        test(m, hidden_size, k_proj_size, name=name)
        test(m, hidden_size, qkv_proj_size, name=name)
        test(m, hidden_size, intermediate_size, name=name)
        test(m, hidden_size, intermediate_size * 2, name=name) # gate-up fused
        test(m, intermediate_size, hidden_size, name=name)

Ms = [1, 4, 40, 400, 1024, 2048]
test_llm_gemm("llama-3-8b", Ms, hidden_size = 4096, intermediate_size = 14336, num_attention_heads = 32, num_key_value_heads = 8)
test_llm_gemm("mistral-7b-v0.1", Ms, hidden_size = 4096, intermediate_size = 14336, num_attention_heads = 32, num_key_value_heads = 8)
test_llm_gemm("Qwen2-0.5B-Instruct", Ms, hidden_size = 896, intermediate_size = 4864, num_attention_heads = 14, num_key_value_heads = 2)
test_llm_gemm("Qwen2-7B-Instruct", Ms, hidden_size = 3584, intermediate_size = 18944, num_attention_heads = 28, num_key_value_heads = 4)
test_llm_gemm("llama-2-7b-chat", Ms, hidden_size = 4096, intermediate_size = 11008, num_attention_heads = 32, num_key_value_heads = 32)
test_llm_gemm("llama-2-13b-chat", Ms, hidden_size = 5120, intermediate_size = 13824, num_attention_heads = 40, num_key_value_heads = 40)
test_llm_gemm("glm4-4b", Ms, hidden_size = 3072, intermediate_size = 8192, num_attention_heads = 24, num_key_value_heads = 6)
test_llm_gemm("chatglm-6b", Ms, hidden_size = 4096, intermediate_size = 16384, num_attention_heads = 32, num_key_value_heads = 32)
test_llm_gemm("chatglm2-6b", Ms, hidden_size = 4096, intermediate_size = 13696, num_attention_heads = 32, num_key_value_heads = 32)

sys.exit(0)

#test(4000, 4096, 4096);sys.exit(0)
#test(40,892,7168)
#test(40,892,7168+16)
#sys.exit(0)

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
