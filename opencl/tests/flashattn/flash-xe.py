import argparse
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from clops import cl
import numpy as np
import time

torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=1)
parser.add_argument('-b', "--batch", type=int, default=1)
parser.add_argument('-nh', "--num-heads", type=int, default=28)
parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=4)
parser.add_argument('-ql', "--q-len", type=int, default=8192)
parser.add_argument('-kvl', "--kv-len", type=int, default=8192)
parser.add_argument('-hs', "--head-size", type=int, default=128)
parser.add_argument('-v', "--verbose", type=int, default=-1)
parser.add_argument('-c', "--causal-mask", action="store_true")

def get_xe_version():
    cm_kernels = cl.kernels(r'''
    extern "C" _GENX_MAIN_ void cm_check_platform(int * info [[type("svmptr_t")]]) {
        info[0] = CM_GRF_WIDTH;
    #ifdef CM_HAS_LSC_UNTYPED_2D
        info[1] = 1;
    #else
        info[1] = 0;
    #endif
    }''', f"-cmc")
    t_info = cl.tensor([2], np.dtype(np.int32))
    cm_kernels.enqueue("cm_check_platform", [1], [1], t_info)
    t_info = t_info.numpy()

    assert(t_info[0] == 512 or t_info[0] == 256)
    xe_version = 1
    if t_info[0] == 512:
        assert(t_info[1] == 1)
        xe_version = 2
    return xe_version

xe_version = get_xe_version()

args = parser.parse_args()
print(args, f"xe_version={xe_version}")

enable_vprint = False
def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)

batch = args.batch
q_len, q_step = args.q_len, (8 if xe_version == 1 else 16)
kv_len, kv_step = args.kv_len, 16
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
if num_kv_heads <= 0:
    num_kv_heads = num_heads
head_size = args.head_size
enable_gqa = num_heads > num_kv_heads
causal_mask = int(args.causal_mask)

#q_len, q_step = 160, 16
#kv_len, kv_step = 800, 16
low = -1
high = 2
act_dtype = torch.float16
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)
k = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)
v = torch.randint(low, high, [batch, kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)

# causal attn-mask
if causal_mask:
    for i in range(q_len):
        attention_mask[:, :, i, 0:(kv_len - q_len + i + 1)] = 0
    print(attention_mask[0,0,:10, :10])
else:
    attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0
    attention_mask[...] = 0

# BLHS=>BHLS
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

#q[:,:,:,:] = q[:,:,2,:]
#attention_mask[:,:,:,:] = attention_mask[:,:,2,:]

print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
print("attention_mask:", attention_mask.shape, attention_mask.dtype)

def get_org(Q, K, V, attention_mask):
    B,H,L,S = Q.shape
    _,Hkv,_,_ = K.shape
    out = torch.zeros([B,H,L,S], dtype=Q.dtype)
    scale_factor = S**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            attn_score = Q[b, h, :, :].to(dtype=torch.float32) @ (K[b, hkv, :,:].transpose(0,1)).to(dtype=torch.float32)
            attn_score *= scale_factor
            attn_score += attention_mask[b,0,:,:]
            #print(attn_score.shape)
            attn_weights = F.softmax(attn_score, 1)
            out[b,h,:,:] = attn_weights @ V[b, hkv, :, :].to(dtype=torch.float32)
    return out

ref = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0, enable_gqa = enable_gqa)
org = get_org(q,k,v,attention_mask)

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    print(f"[check_close] rtol_max: {rtol_max}")
    print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        assert 0

check_close(ref, org, atol=1e-3, rtol=1e-2)

# [batch, seq-len, heads, size] BLHS
print("ref:", ref.shape, ref.dtype)

# blocking on kv-len dimension with online-softmax
def get_flash0(query, key, value, attention_mask):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]
            K = key[b, hkv, :, :]
            V = value[b, hkv, :, :]
            mask = attention_mask[b,0,:,:]
            for i in range(0, q_len, q_step):
                i1 = min(i + q_step, q_len)
                # online softmax states:
                #     per-row max-value     : [q_step, 1]
                #     per-row sum           : [q_step, 1]
                #     current accumulated V : [q_step, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                # we prefer transposed S since:
                #   1. per-column max/sum is easier than per-row in register, 
                #   2. loop in K direction is inner-most, so load K in noraml
                #      instead of VNNI is faster, load Q in transpose-VNNI is 
                #      done only once at loop begin.
                #   3. 
                rQt = Q[i:i1, :].transpose(0,1)  # sub Q block only transposed & VNNI packed once

                for j in range(0, kv_len, kv_step):
                    j1 = min(j + kv_step, kv_len)

                    if (args.verbose >= 0):
                        print(f"======== i={i} j={j} ==========")

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rK = K[j:j1,:]
                    St = (rK @ rQt).to(dtype=torch.float32)
                    MaskT = mask[i:i1, j:j1].transpose(0,1)

                    vprint("rK=", rK.shape, rK)
                    vprint("rQt=",rQt.shape, rQt[:16,:])
                    vprint("St=",St.shape, St)
                    vprint("MaskT=",MaskT.shape, MaskT)

                    St *= scale_factor
                    St += mask[i:i1, j:j1].transpose(0,1)
                    vprint("St=",St.shape, St)

                    rowmax = St.max(0, keepdim=True).values
                    if j == 0:
                        cur_max = rowmax
                    else:
                        rowmax = torch.maximum(cur_max, rowmax)
                    vprint("rowmax=", rowmax)

                    # compute in local SRAM
                    St = torch.exp(St - rowmax)
                    vprint("St(Pt)=", St.shape, St)

                    rowsumP = St.sum(0, keepdim=True)
                    vprint("rowsumP=", rowsumP)

                    # corrected sum of previous block
                    if j > 0:
                        max_comp = torch.exp(cur_max - rowmax)

                    if j == 0:
                        cur_sum = rowsumP
                    else:
                        cur_sum = cur_sum * max_comp + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = St.to(dtype=torch.float16).transpose(0,1)
                    
                    vprint("P=", partial_attn_weight.shape, partial_attn_weight)

                    rV = V[j:j1, :]
                    vprint("rV=",rV.shape, rV)

                    # correct last Output to current statistics
                    
                    if j == 0:
                        cur_O = partial_attn_weight @ rV
                    else:
                        cur_O = (cur_O * max_comp.transpose(0, 1))
                        vprint("cur_O1=", cur_O)
                        cur_O += partial_attn_weight @ rV
                    vprint("cur_O2=", cur_O.shape, cur_O)

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                cur_O_f16 = (cur_O/cur_sum.transpose(0, 1)).to(torch.float16)

                if (i == args.verbose):
                    enable_vprint = True
                    vprint("cur_O_f16=", cur_O_f16.shape, cur_O_f16)
                    assert 0

                out[b, h, i:i1, :] = cur_O_f16
    return out

if args.impl == 0:
    f0 = get_flash0(q,k,v,attention_mask)
    check_close(org, f0, atol=1e-2, rtol=1e-3)
    print("=========== PASS ===========")
    sys.exit(0)

#====================================================================================================
# using the same parameter & inputs, develop cm kernels which produces the same output
# prototyping CM kernels


# transpose back to orginal shape: [batch, q_len, num_heads, head_size]
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)
print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
print("attention_mask:", attention_mask.shape, attention_mask.dtype)

def pyeval(src):
    result_src = ""
    for line in src.splitlines():
        if line.startswith("#pyeval"):
            new_line = eval(line[8:])
            result_src += new_line + "\n"
            # print(f"[pyeval] {new_line}")
        else:
            result_src += line + "\n"
    return result_src

scale_factor = 1.0/(head_size**0.5)

# adjacent query-heads may share same KV-heads
# dispatch them together can help increase cache-usage
WG_SIZE = 16
q_group_size = WG_SIZE * q_step
def divide_round_up(numerator, denominator):
    return (numerator + denominator - 1)//denominator * denominator
q_threads = divide_round_up(q_len, q_group_size) // q_step
if (q_threads < WG_SIZE): q_threads = WG_SIZE

GWS=[batch, num_heads, q_threads]
#WG_SIZE = min(GWS[-1], 16 if xe_version == 1 else 8)

LWS=[1, 1, WG_SIZE]

dim_batch = 0
dim_head = 1
dim_q_batch = 2

if False:
    # for GQA/multi-query, invoke all heads sharing same KV can further help cache-hit-rate
    # but it's not helping, WHY?
    num_heads_share_kv = num_heads//num_kv_heads
    GWS=[batch, num_heads_share_kv, q_len//q_step]
    WG_SIZE = min(q_len//q_step, 16)
    LWS=[1, 1, WG_SIZE]

    dim_batch = 0
    dim_head = 1
    dim_q_batch = 2

print("GWS=", GWS)
print("LWS=", LWS)
print(f"total HW threads: {batch} x {num_heads} x {q_threads} = {batch * num_heads * q_threads}")
#========================================================================
# Optimization Log
r'''
increase WG_SIZE to 16 (-70ms)
use GRF to store temp Output(rO) instead of SLM, requires `-Qxcm_register_file_size=256` (-40ms)
avoid type-promotion:   St = cm_mul<float>(St, scale_factor);    =>   St = cm_mul<float>(St, (float)scale_factor);   (-10ms) 
change dtype of attention_mask from float to half (-14ms)
avoid indirect register access: unroll for loop which access matrix rows using loop-index
use GRF to store temp Input rQ instead of SLM, this allows more Work-Groups to be packed into same Xe-core!!!

in Xe1, GRF is only half size of Xe2, so reduce q_step to fit rO/rQ into GRF
reading V into SLM is slow due to smaller REG_N of Xe1 DPAS (8 vs 16), increase the consecutive read width to 16 helps cache-line usage

keep all other parameter unchanged, if head-size is 128, latency is 11ms, if head-size is 144, latency increases to 24ms


 head-size : latency(ms) @kv-length 1024/512
     96 : 8
    112 : 8     5
    128 : 11    6
    144 : 24    12
    160 : 69    34
    176 : 63

ql 8192: 115ms
   4096: 58ms
   2048: 29ms <===========
   1024: 14ms <===========
    512: 8.6
    256: 4.5
    128: 2.0

kvl 8192: 118
    4096: 59
    2048: 23.5 <===========
    1024: 11.3 <===========
     512: 5.99
     256: 3.7
     128: 2.5 (error)

nkvh 28: 145ms
     14: 122ms
      7: 118 <=============
      4: 118 <=============
      2: 125 ?????
      1: 130 ?????

nh   28: 145
     14: 66.8
      7: 30
      4: 18
      2: 9.3
      1: 5.1

no-attn-mask: 100 ms 

query/output-size : 8192*128*28*2   ~  56 MB
KV-cache size     : 8192 * 128*4 *2 ~   8 MB
attn-mask size    : 8192 * 8192  *2 ~ 128 MB

'''
#========================================================================

src1 = cl.CMTracer.code + r'''
//# CM kernel for flash attn, reference

#pyeval f"#define num_heads {num_heads}"
#pyeval f"#define num_kv_heads {num_kv_heads}"
#pyeval f"#define head_size {head_size}"
#pyeval f"#define q_step {q_step}"
#pyeval f"#define kv_step {kv_step}"
#pyeval f"#define scale_factor {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"

#pyeval f"#define dim_batch {dim_batch}"
#pyeval f"#define dim_head {dim_head}"
#pyeval f"#define dim_q_batch {dim_q_batch}"

#pyeval f"#define WG_SIZE_HINT {WG_SIZE}"


#pyeval f"#define causal_mask {causal_mask}"

#include "cm_sdpa_xe.hpp"
'''

cl.profiling(True)

t_q = cl.tensor(q.detach().numpy())
t_k = cl.tensor(k.detach().numpy())
t_v = cl.tensor(v.detach().numpy())
t_out = cl.tensor([batch, q_len, num_heads, head_size], np.dtype(np.float16))
t_cminfo = cl.tensor([GWS[0]*GWS[1]*GWS[2]//WG_SIZE, 3], np.dtype(np.uint64))

# f"-cmc -mdump_asm -g2 "
cwd = os.path.dirname(os.path.realpath(__file__))
print(f"compiling {cwd} ...")
cm_kernels = cl.kernels(pyeval(src1), f"-cmc -Qxcm_register_file_size=256  -mCM_printregusage -I{cwd} -mdump_asm -g2")
print("first call ...")

cm_kernels.enqueue("cm_sdpa", GWS, LWS, 0, q_len, kv_len, t_q, t_k, t_v, t_out, t_cminfo)
f1 = torch.from_numpy(t_out.numpy())

if args.verbose >= 0:
    check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
    print(f"=========== cm_sdpa PASS GWS={GWS} LWS={LWS}  ===========")
    sys.exit(0)


all_layers = []
mem_size = 0
while len(all_layers) < 100 and mem_size < 4e9:
    all_layers.append([
        cl.tensor(q.detach().numpy()),
        cl.tensor(k.detach().numpy()),
        cl.tensor(v.detach().numpy()),
        cl.tensor([batch, q_len, num_heads, head_size], np.dtype(np.float16)),
    ])
    mem_size += q.numel() * q.element_size()
    mem_size += k.numel() * k.element_size()
    mem_size += v.numel() * v.element_size()
    print(f"nlayers={len(all_layers)} mem_size={mem_size*1e-9:.3f} GB")

for i in range(100):
    j  = i % len(all_layers)
    cm_kernels.enqueue("cm_sdpa", GWS, LWS,
                    0, q_len, kv_len,
                    all_layers[j][0],
                    all_layers[j][1],
                    all_layers[j][2],
                    all_layers[j][3],
                    t_cminfo)
latency = cl.finish()
for i,ns in enumerate(latency):
    print(f"[{i}]  {ns*1e-6:.3f} ms")
print(f" average latency: {sum(latency[10:])/len(latency[10:])*1e-6:.3f} ms")

cl.CMTracer.dump(t_cminfo.numpy(), 2250e6, "cm.json")

check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
print(f"=========== cm_sdpa PASS GWS={GWS} LWS={LWS}  ===========")
sys.exit(0)
