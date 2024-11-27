import pycpp
import torch
import time
import psutil
import gc
import sys
import numpy as np

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

import os

LINUX_PERF = ['dump']
# LINUX_PERF.append("STALLS_TOTAL=0xa3-0x04-0x04")
# LINUX_PERF.append("STALLS_L3_MISS=0xa3-0x06-0x06")
# LINUX_PERF.append("STALLS_L2_MISS=0xa3-0x05-0x05")
# LINUX_PERF.append("BOUND_ON_LOADS=0xa6-0x21-0x05")
# LINUX_PERF.append("BOUND_ON_STORES=0xa6-0x40-0x02")
# LINUX_PERF.append("SPLIT_LOADS=0xd0-0x41")
# LINUX_PERF.append("SPLIT_STORES=0xd0-0x42")
LINUX_PERF.append("L1_MISS=0xd1-0x08")
LINUX_PERF.append("L2_MISS=0xd1-0x10")

os.environ['LINUX_PERF']=':'.join(LINUX_PERF)

print(os.environ['LINUX_PERF'])

def test_strided_6x2(M, N, strideN, K):
    vRANGE = 3
    np.random.seed(0)
    A = np.random.randint(-vRANGE, vRANGE+1, [M, K]).astype(np.float32)
    B0 = np.random.randint(-vRANGE, vRANGE+1, [K, strideN]).astype(np.float32)
    B = B0[:,:N] # now B is strided tensor
    C0 = np.matmul(A, B)
    C1 = np.zeros([M, N], dtype=np.float32)

    assert (N % 8) == 0

    # there are python framework overheads, so we
    # repeat measurements within CPP function many rounds
    ROUNDS = 20000
    # warn-up
    pycpp.test_regblk(A, B, C1, ROUNDS)
    with pycpp.perf("") as p:
        pycpp.test_regblk(A, B, C1, ROUNDS)
        events = p.finish()
        for key in events:
            events[key] = events[key]/ROUNDS
        dt_ns = events["ns"]
        cycles = events["HW_CPU_CYCLES"]
        info = ",".join([f'{k}={events[k]:.0f}' for k in events])
        print(f"[{strideN/16}] regblk-{M}x{N//8} {dt_ns*1e-6:.3f} ms   {cycles/dt_ns:.1f} GHz  {info}  CPI:{cycles/K:.1f}  {M*K*N*2/dt_ns:.0f} GFLOPS ")
    #compare(C0, C1)

for strideN in range(16, 16*100, 16):
    test_strided_6x2(M=6, N=2*8, strideN=strideN, K=2560)
