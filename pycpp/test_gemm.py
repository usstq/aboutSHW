import pycpp
import contextlib
import sys
import torch, time
import numpy as np

from utils import Colors, last_log, compare


###############################
# perf = pycpp.perf(["HW_CPU_CYCLES", "SPLIT_LOADS=0xd0,0x41"])
perf = contextlib.nullcontext()

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

        print("============== torch.matmul ============ ")
        net2 = []
        for i in range(LAYERS):
            w = torch.randint(-8, 8, [K, N]).float().to(device)
            net2.append(w)

        avg_gflops = 0
        avg_dt = 0
        for r in range(REPEAT):
            t0 = time.time()
            for wei in net2:
                with perf:
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

        print("============== my gemm ============ ")
        src = A.numpy()
        dst = C.numpy()
        avg_gflops = 0
        avg_dt = 0
        for r in range(REPEAT):
            t0 = time.time()
            for wei in net2:
                with perf:
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
        last_log(log_str)


test(1,8920,896)
test(4,8920,896)
test(6,8920,896)
test(6*2,8920,896)
test(6*4,8920,896)
test(6*8,8920,896)
test(6*10,8920,896)
test(400, 4096, 4096)
test(4000, 4096, 4096)
sys.exit(0)

################################################################
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
