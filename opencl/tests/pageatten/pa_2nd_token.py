import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

parser = argparse.ArgumentParser('')
parser.add_argument('-i', "--impl", type=int, default=1)
parser.add_argument('-b', "--batch", type=int, default=1)
parser.add_argument('-nh', "--num-heads", type=int, default=32)
parser.add_argument('-nkvh', "--num-kv-heads", type=int, default=8)
parser.add_argument('-ql', "--q-len", type=int, default=1)
parser.add_argument('-kvl', "--kv-len", type=int, default=32769)
parser.add_argument('-hs', "--head-size", type=int, default=128)
parser.add_argument('-rkv', "--reset_kv_cache", type=int, default=1)
parser.add_argument('-v', "--verbose", type=int, default=-1)
args = parser.parse_args()
print(args)

enable_vprint = False

def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)

batch = args.batch
q_len, q_step = args.q_len, 32
kv_len, kv_step = args.kv_len, 8
num_heads = args.num_heads
num_kv_heads = args.num_kv_heads
head_size = args.head_size
enable_gqa = num_heads > num_kv_heads

# define KV_BLOCK_SIZE = 32,64,128,256
kv_block_size = 256

enable_kvcache_compression = 1
kv_cache_quantization_mode = os.environ.get("KV_CACHE_QUANT_MODE", "by_token")
kv_cache_quantization_mode = "by_channel"

def _validate_quant_mode(mode: str) -> str:
    mode = mode.strip().lower()
    if mode not in {"by_token", "by_channel"}:
        raise ValueError(f"Unsupported kv-cache quantization mode: {mode}")
    return mode

kv_cache_quantization_mode = _validate_quant_mode(kv_cache_quantization_mode)
kvcache_quantization_by_token = int(kv_cache_quantization_mode == "by_token")
print(f"{kv_cache_quantization_mode=}, {kvcache_quantization_by_token=}")

enable_clean_unused_kvcache = args.reset_kv_cache

def get_tensor(name, dtype=np.float16):
    with open(name, 'rb') as f:
        data = f.read()
        np_data = np.frombuffer(data, dtype=dtype).copy()
        return torch.from_numpy(np_data)

#xe_arch: 1: xe, 2: xe2
xe_arch=2

if xe_arch == 1:
    kv_step = 8
else:
    kv_step = 16

# dpas number for each split_len
# split_subblock_num = kv_partition_size // kv_step

# reduce step size
reduce_split_step = 8

low = -127
high = 128
act_dtype = torch.float16
new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
k = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, new_kv_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high

run_real_data_test=False

if run_real_data_test:
    q = get_tensor("./qwen3_q_f16_1_4096.bin").reshape([1,1,32,128])
    k = get_tensor("./qwen3_k_f16_129_8_16_128.bin").reshape([129,8,16,128]).transpose(1,2).reshape([1,129*16,8,128])[:,:128*16+1,:,:]
    v = get_tensor("./qwen3_v_f16_129_8_16_128.bin").reshape([129,8,16,128]).transpose(1,2).reshape([1,129*16,8,128])[:,:128*16+1,:,:]
    kv_len = 128*16+1
    # q = torch.randint(low, high, [1, 1, 32, 128]).to(dtype=act_dtype)/high
    # k = torch.randint(low, high, [1, kv_len, 8, 128]).to(dtype=act_dtype)/high
    # v = torch.randint(low, high, [1, kv_len, 8, 128]).to(dtype=act_dtype)/high
    print("q.shape = ", q.shape)
    print("k.shape = ", k.shape)
    print("v.shape = ", v.shape)

# sdpa split size, must be multiple of kv_step and kv_len should be multiple of kv_partition_size
k_partition_block_num = kv_len//8192
if k_partition_block_num < 1:
    k_partition_block_num = 1
k_partition_block_num = 1  # test cm_sdpa_2nd
# k_partition_block_num = 0.5  # test cm_sdpa_2nd_half_block, now output is wrong for enable_kvcache_compression
kv_partition_size = int(kv_block_size * k_partition_block_num)

print("kv_step:", kv_step)
print("k_partition_block_num:", k_partition_block_num)
print("kv_partition_size:", kv_partition_size)

new_kv_len = (kv_len + kv_block_size - 1) // kv_block_size * kv_block_size
kv_partition_num = (new_kv_len + kv_partition_size - 1) // kv_partition_size
total_partition_num = kv_partition_num * num_heads
# lse_num = batch * kv_partition_num

# assert((kv_len//kv_partition_size)%(kv_partition_size//kv_step)==0)
# assert(kv_len%kv_partition_size == 0)
assert(kv_partition_size % kv_step == 0)

# # test code
# q = torch.arange(0, q_len* num_heads * head_size, dtype=act_dtype).reshape(batch, q_len, num_heads, head_size)/10000.0
# k = torch.arange(0, kv_len* num_kv_heads * head_size, dtype=act_dtype).reshape(batch, kv_len, num_kv_heads, head_size)/10000.0
# v = torch.arange(0, kv_len* num_kv_heads * head_size, dtype=act_dtype).reshape(batch, kv_len, num_kv_heads, head_size)/10000.0

# print("v.shape:", v.shape, v.dtype)
# for i in range(batch):
#     for j in range(kv_len):
#         for kk in range(num_kv_heads):
#             for h in range(head_size):
#                 v[i][j][kk][h] = (h + j*head_size)/100.0

# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min).to(dtype=act_dtype)
#attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0
attention_mask[...] = 0
att = torch.from_numpy(attention_mask.numpy())
#print("attention_mask:", att)

# support single sequence
seq_num = 1
total_blk_num = new_kv_len//kv_block_size

past_lens=torch.tensor([kv_len-1]).to(torch.int32)

# the first block of each sequence
block_indices_begins=torch.tensor([0, total_blk_num]).to(torch.int32)

# block physical indices from logic index 
block_indices =  torch.randperm(total_blk_num).to(torch.int32)
# print("block_indices:", block_indices)

subsequence_begins=torch.tensor([0, seq_num]).to(torch.int32)
gws_subseq_mapping=torch.tensor([0]).to(torch.int32)

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

print("k = ", k.shape, k.dtype)
ref = F.scaled_dot_product_attention(q, k[:,:,:kv_len,:], v[:,:,:kv_len,:], attention_mask, dropout_p=0.0, enable_gqa = enable_gqa)
org = get_org(q, k[:,:,:kv_len,:], v[:,:,:kv_len,:], attention_mask)

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    #print("ref = ", input)
    #print("res = ", other)
    rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    # print(f"[check_close] rtol_max: {rtol_max}")
    # print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        print(f"Not close indices: {not_close_indices}")
        print(f"    ref_tensor: {input[not_close_indices]}")
        print(f"    res_tensor: {other[not_close_indices]}")
        assert 0

check_close(ref, org, atol=1e-3, rtol=1e-2)

# [batch, seq-len, heads, size] BLHS
print("ref:", ref.shape, ref.dtype)

# blocking on kv-len dimension with online-softmax
# Softmax(Q@Kt)@V
def get_flash1(query, key, value, attention_mask):
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

            # loop one time
            # q_len == 1, q_step == 1
            for i in range(0, 1, 1):
                i1 = 1
                # online softmax states:
                #     per-row max-value     : [1, 1]
                #     per-row sum           : [1, 1]
                #     current accumulated V : [1, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                rQ = Q[0, :].reshape(1,hs) # [1,128] sub Q block VNNI packed

                cur_O = torch.full([kv_len // kv_step, 1, hs], 0, dtype=torch.float32)
                max_comp_0 = torch.full([kv_len // kv_step], 1, dtype=torch.float32)

                for j in range(0, kv_len, kv_step):
                    j1 = min(j + kv_step, kv_len)

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rKt = K[j:j1,:].transpose(0,1)  #[16, 128] -> [128, 16], suppose kv_step is 16
                    rS = (rQ @ rKt).to(dtype=torch.float32).reshape(1,kv_step)  # [1,16]
                    rMask = mask[i:i1, j:j1] # [1,16]

                    vprint("rK=", rKt.shape)
                    vprint("rQt=",rQ.shape)
                    vprint("rS=",rS.shape)
                    vprint("rMask=",rMask.shape)

                    rS *= scale_factor
                    rS += rMask
                    vprint("rS=",rS.shape)

                    rowmax = rS.max(1, keepdim=True).values # [1,1]
                    if j == 0:
                        cur_max = rowmax
                    else:
                        rowmax = torch.maximum(cur_max, rowmax)
                    vprint("rowmax=", rowmax.shape)

                    # compute in local SRAM
                    rS = torch.exp(rS - rowmax) # [1,16]
                    vprint("St(Pt)=", rS.shape)

                    rowsumP = rS.sum(1, keepdim=True) # [1,1]
                    vprint("rowsumP=", rowsumP.shape)

                    # corrected sum of previous block
                    if j > 0:
                        max_comp = torch.exp(cur_max - rowmax)
                        vprint("max_comp=", max_comp.shape)
                        max_comp_0[j//kv_step] = max_comp

                    if j == 0:
                        cur_sum = rowsumP
                    else:
                        cur_sum = cur_sum * max_comp + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = rS.to(dtype=torch.float16) # [1,16]
                    
                    vprint("P=", partial_attn_weight.shape)

                    rV = V[j:j1, :]  # [16,128]
                    vprint("rV=",rV.shape)

                    # correct last Output to current statistics
                    cur_O[j//kv_step,:,:] = partial_attn_weight @ rV # [:,1,128]
                    vprint("cur_O2=", cur_O.shape)

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                cur_O_f32 = cur_O[0,:,:]
                for j in range(1, kv_len//kv_step):
                    cur_O_f32 = cur_O_f32 *  max_comp_0[j] + cur_O[j,:,:]
                vprint("cur_O_f32=", cur_O_f32.shape)
                vprint("cur_sum=", cur_sum.shape)
                cur_O_f16 = (cur_O_f32/cur_sum).to(torch.float16)

                if (i == args.verbose):
                    enable_vprint = True
                    print("cur_O_f16=", cur_O_f16.shape, cur_O_f16)
                    assert 0

                out[b, h, i:i1, :] = cur_O_f16
    return out

# Split KV online-softmax
def get_flash2(query, key, value, attention_mask, real_kv_len=0):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    if real_kv_len == 0:
        real_kv_len = kv_len
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]
            K = key[b, hkv, :, :]
            V = value[b, hkv, :, :]
            mask = attention_mask[b,0,:,:]

            # loop kv_split
            cur_O=torch.full([kv_len//kv_partition_size, 1, hs], 0, dtype=torch.float32)
            cur_O_f32=torch.full([1, hs], 0, dtype=torch.float32)
            cur_O_f16=torch.full([1, hs], 0, dtype=torch.float16)
            lse=torch.full([kv_len//kv_partition_size], 0, dtype=torch.float32)
            for i in range(0, kv_len, kv_partition_size):
                i1 = min(i + kv_partition_size, kv_len)
                # online softmax states:
                #     per-row max-value     : [1, 1]
                #     per-row sum           : [1, 1]
                #     current accumulated V : [1, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                rQ = Q[0, :].reshape(1,hs) # [1,128] sub Q block VNNI packed
                vprint("rQ = ", rQ)
                vprint("mask = ", mask)

                cur_lse = 0.0
                cur_sum = 0.0
                for j in range(i, i1, kv_step):
                    j1 = min(j + kv_step, kv_len)
                    if j > kv_len:
                        break

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rKt = K[j:j1,:].transpose(0,1) #[16,128]->[128,16]
                    rS = (rQ @ rKt).to(dtype=torch.float32).reshape(1,kv_step) #[1,16]
                    rMask = mask[0:1, j:j1] #[1,16]

                    vprint("rK=", rKt)
                    vprint("rQt=",rQ)
                    vprint("rS=",rS)
                    vprint("rMask=",rMask)

                    rS *= scale_factor
                    vprint("rS * scale_factor =",rS)
                    rS += rMask

                    cur_lse += torch.exp(rS).sum(1, keepdim=True).item() # [1,1]
                    vprint("rS=", rS)
                    vprint("exp(rS)=", torch.exp(rS))
                    vprint("cur_lse=", cur_lse)

                    rowmax = rS.max(1, keepdim=True).values # [1,1]
                    if j == 0:
                        cur_max = rowmax
                    else:
                        rowmax = torch.maximum(cur_max, rowmax)
                    vprint("rowmax=", rowmax.shape)

                    vprint("rS=", rS)
                    # compute in local SRAM
                    rS = torch.exp(rS - rowmax) # [1,16]
                    vprint("rowmax = ", rowmax)
                    vprint("St(Pt)=", rS)

                    rowsumP = rS.sum(1, keepdim=True) # [1,1]
                    vprint("rowsumP=", rowsumP)

                    # corrected sum of previous block
                    if j > 0:
                        max_comp = torch.exp(cur_max - rowmax)
                        vprint("max_comp=", max_comp.shape)

                    if j == 0:
                        cur_sum = rowsumP
                    else:
                        cur_sum = cur_sum * max_comp + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = rS.to(dtype=torch.float16) # [1,16]
                    
                    vprint("P=", partial_attn_weight)

                    rV = V[j:j1, :] # [16,128]
                    vprint("rV=",rV)

                    # correct last Output to current statistics
                    if j== 0:
                        cur_O[i//kv_partition_size,:,:] = partial_attn_weight @ rV
                    else:
                        cur_O[i//kv_partition_size,:,:] = cur_O[i//kv_partition_size,:,:] * max_comp;
                        cur_O[i//kv_partition_size,:,:] += partial_attn_weight @ rV # [:,1,128]
                    vprint("j = ", j)
                    vprint("cur_O2=",  cur_O[i//kv_partition_size,:,:])

                    cur_max = rowmax
                    if (j == args.verbose): assert 0

                lse[i//kv_partition_size] = cur_lse
                if i > real_kv_len:
                    cur_O[i//kv_partition_size,:,:] = 0
                else:
                    cur_O[i//kv_partition_size,:,:] = cur_O[i//kv_partition_size,:,:] / cur_sum
                vprint("cur_sum=", cur_sum.shape)
                vprint("cur_O=",  cur_O[i//kv_partition_size,:,:])

            # reduce
            # for i in range(0, kv_len//kv_partition_size, 1):
            #     for j in range(0, hs, kv_step):
            #         stop = min(j + kv_step, hs)
            #         print("i=", i, ", j = ", j, ": cur_O[i,:,:]=", cur_O[i,0,j:stop])
            # vprint("lse=", lse)
            # print("lse=", lse.shape) # [4]
            sum_lse = lse.sum(0)
            # print("cur_O=", cur_O.shape) # 
            # print("cur_O_f32=", cur_O_f32.shape) #
            for i in range(0, kv_len//kv_partition_size, 1):
                if i * kv_partition_size > real_kv_len:
                    break
                cur_O_f32 += cur_O[i,:,:] * lse[i] / sum_lse
            cur_O_f16 = cur_O_f32.to(torch.float16)
            out[b, h, :, :] = cur_O_f16
            # print("cur_O_f16=", cur_O_f16.shape) #
            # print("out=", out.shape) #
    #print("out = ", out[0,0,0,:])
    #print("out = ", out)
    return out

# Split KV online-softmax
def get_flash3(query, key, value, attention_mask):
    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]  # [q_len, head_size]
            K = key[b, hkv, :, :]  # [kv_len, head_size]
            V = value[b, hkv, :, :]  # [kv_len, head_size]
            mask = attention_mask[b,0,:,:]  # [q_len, kv_len]

            # loop kv_split
            cur_O=torch.full([kv_len//kv_partition_size, 1, hs], 0, dtype=torch.float32)  # [kv_len//kv_partition_size, 1, head_size]
            cur_O_f32=torch.full([1, hs], 0, dtype=torch.float32)  # [1, head_size]
            cur_O_f16=torch.full([1, hs], 0, dtype=torch.float16)  # [1, head_size]
            lse=torch.full([kv_len//kv_partition_size], 0, dtype=torch.float32)  # [kv_len//kv_partition_size]
            for i in range(0, kv_len, kv_partition_size):
                i1 = min(i + kv_partition_size, kv_len)
                # online softmax states:
                #     per-row max-value     : [1, 1]
                #     per-row sum           : [1, 1]
                #     current accumulated V : [1, S]
                #cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                #cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                #cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)

                rQ = Q[0, :].reshape(1,hs) # [1,128] sub Q block VNNI packed
                if i==0:
                    vprint("rQ = ", rQ)
                vprint("mask = ", mask)

                cur_lse = 0.0
                cur_sum = 0.0
                for j in range(i, i1, kv_partition_size):
                    j1 = min(j + kv_partition_size, kv_len)

                    if (j == args.verbose): enable_vprint = True
                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    rKt = K[j:j1,:].transpose(0,1)  # [head_size, kv_partition_size]
                    rS = (rQ @ rKt).to(dtype=torch.float32).reshape(1,kv_partition_size)  #[1, kv_partition_size]
                    rMask = mask[0:1, j:j1]  #[1, kv_partition_size]

                    vprint("rK=", rKt)
                    vprint("rQt=",rQ)
                    vprint("rS=",rS)
                    vprint("rMask=",rMask)

                    rS *= scale_factor
                    vprint("rS * scale_factor =",rS)
                    rS += rMask

                    cur_lse += torch.exp(rS).sum(1, keepdim=True).item() # [1,1]
                    vprint("rS=", rS)
                    vprint("exp(rS)=", torch.exp(rS))
                    vprint("cur_lse=", cur_lse)

                    rowmax = rS.max(1, keepdim=True).values # [1,1]

                    # compute in local SRAM
                    rS = torch.exp(rS - rowmax) # [1,16]
                    vprint("rowmax = ", rowmax)
                    vprint("St(Pt)=", rS)

                    rowsumP = rS.sum(1, keepdim=True) # [1,1]
                    vprint("rowsumP=", rowsumP)

                    # corrected sum of previous block
                    cur_sum = rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = rS.to(dtype=torch.float16) # [1,16]

                    vprint("P=", partial_attn_weight)

                    rV = V[j:j1, :]  # [kv_partition_size, head_size]
                    vprint("rV=",rV)

                    # correct last Output to current statistics
                    cur_O[i//kv_partition_size,:,:] = partial_attn_weight @ rV
                    vprint("cur_O2=",  cur_O[i//kv_partition_size,:,:])

                lse[i//kv_partition_size] = cur_lse
                cur_O[i//kv_partition_size,:,:] = cur_O[i//kv_partition_size,:,:] / cur_sum
                vprint("cur_sum=", cur_sum.shape)
                vprint("Omat=",  cur_O[i//kv_partition_size,:,:])

            # reduce
            # for i in range(0, kv_len//kv_partition_size, 1):
            #     for j in range(0, hs, kv_step):
            #         stop = min(j + kv_step, hs)
            #         print("i=", i, ", j = ", j, ": cur_O[i,:,:]=", cur_O[i,0,j:stop])
            vprint("lse=", torch.log(lse))
            vprint("lse=", lse.shape) # [4]
            sum_lse = lse.sum(0)
            # print("cur_O=", cur_O.shape) # 
            # print("cur_O_f32=", cur_O_f32.shape) #
            vprint("lse = ", lse)
            vprint("sum_lse = ", sum_lse)
            for i in range(0, kv_len//kv_partition_size, 1):
                cur_O_f32 += cur_O[i,:,:] * lse[i] / sum_lse
            cur_O_f16 = cur_O_f32.to(torch.float16)
            out[b, h, :, :] = cur_O_f16
            # print("cur_O_f16=", cur_O_f16.shape) #
            # print("out=", out.shape) #
    #print("out = ", out[0,0,0,:])
    #print("out = ", out)
    return out

# transpose back to orginal shape: [batch, q_len, num_heads, head_size] for padding
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

if kv_len % kv_block_size != 0:
    # pad k,v to multiple of kv_block_size
    pad_len = ((kv_len + kv_block_size - 1)//kv_block_size)*kv_block_size - kv_len
    # k = F.pad(k, (0,0,0,0,0,pad_len,0,0), "constant", 0)
    # v = F.pad(v, (0,0,0,0,0,pad_len,0,0), "constant", 0)
    attention_mask = F.pad(attention_mask, (0,pad_len,0,0), "constant", torch.finfo(act_dtype).min)
    print(f"pad k,v from {kv_len} to {k.shape[1]}")
    new_kv_len = k.shape[1]
    total_blk_num = new_kv_len//kv_block_size
    assert(new_kv_len % kv_block_size == 0)

print("k.shape:", k.shape, k.dtype)
print("v.shape:", v.shape, v.dtype)
print("new_kv_len = ", new_kv_len)

# transpose shape: [batch, num_heads, q_len, head_size] for get_flash3
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

if args.impl == 0:
    f0 = get_flash3(q,k,v,attention_mask)
    check_close(org, f0, atol=1e-2, rtol=1e-3)
    print("=========== PASS ===========")
    sys.exit(0)

org1 = get_flash1(q,k,v,attention_mask)
check_close(ref, org1, atol=1e-3, rtol=1e-2)
print("org of get_flash1 passed !")

org2 = get_flash2(q,k,v,attention_mask, real_kv_len=kv_len)
check_close(ref, org2, atol=1e-3, rtol=1e-2)
print("org of get_flash2 passed !")

org = get_flash3(q,k,v,attention_mask)
check_close(ref, org, atol=1e-3, rtol=1e-2)
print("org of get_flash3 passed !")
check_close(org, org2, atol=1e-3, rtol=1e-2)

print()
print("GPU cm kernels for flash attn2:")

#====================================================================================================
# using the same parameter & inputs, develop cm kernels which produces the same output
# prototyping CM kernels
from clops import cl
import numpy as np
import time

# transpose back to orginal shape: [batch, q_len, num_heads, head_size]
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)
print("q:", q.shape, q.dtype)
print("k:", k.shape, k.dtype)
print("v:", v.shape, v.dtype)
#print("attention_mask:", attention_mask.shape, attention_mask.dtype)

#print("original k:", k)

# if kv_len % kv_block_size != 0:
#     # pad k,v to multiple of kv_block_size
#     pad_len = ((kv_len + kv_block_size - 1)//kv_block_size)*kv_block_size - kv_len
#     k = F.pad(k, (0,0,0,0,0,pad_len,0,0), "constant", 0)
#     v = F.pad(v, (0,0,0,0,0,pad_len,0,0), "constant", 0)
#     attention_mask = F.pad(attention_mask, (0,pad_len,0,0), "constant", torch.finfo(act_dtype).min)
#     print(f"pad k,v from {kv_len} to {k.shape[1]}")
#     new_kv_len = k.shape[1]
#     total_blk_num = new_kv_len//kv_block_size
#     assert(new_kv_len % kv_block_size == 0)

def quan_per_token(kv):
    blk_num, kv_heads, blksz, *_ = kv.shape
    kv_max = kv.amax(dim=-1, keepdim = True)
    kv_min = kv.amin(dim=-1, keepdim = True)
    qrange = kv_max - kv_min

    INTMAX = 255.0
    INTMIN = 0.0
    INTRAGNE = INTMAX - INTMIN
    kv_scale = ((INTRAGNE)/qrange).to(dtype=torch.half)
    kv_zp = ((0.0-kv_min)*kv_scale+INTMIN).to(dtype=torch.half)

    kv_INT8 = torch.round((kv*kv_scale+kv_zp)).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    # print("################################################################################")
    # print(f'KV\n:{kv.reshape(16, 32)}')
    # print(f'kv_INT8\n:{kv_INT8.reshape(16, 32)}')
    # print(f'kv_scale\n:{kv_scale.reshape( 16, 1)}')
    # print(f'kv_zp\n:{kv_zp.reshape( 16, 1)}')
    # print("################################################################################")

    # print("quant_scale =", (1.0/kv_scale).reshape(blk_num,kv_heads,-1))
    # print("quant_zp    =", kv_zp.reshape(blk_num,kv_heads,-1))

    dq_scale = (1.0/kv_scale).view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    # print("dq_scale: ", dq_scale)
    # print("kz_zp: ", kv_zp)
    return torch.concat((kv_INT8, dq_scale, kv_zp), dim=-1)

def dequant_per_token(kv, head_size, blk_size):
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:,:,:head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:,:,head_size * blk_size: (head_size * blk_size + blk_size * 2)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)
    kv_zp = kv[:,:, (head_size * blk_size + blk_size * 2):(head_size * blk_size + blk_size * 4)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, 1)

    # print("dequant_kv_u8 = ", kv_u8)
    # print("dequant_kv_scale = ", kv_scale.reshape(blk_num, kv_head_num, blk_size))
    # print("dequant_kv_zp    = ", kv_zp.reshape(blk_num, kv_head_num, blk_size))

    kv_dequant = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)

    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(blk_size):
                kv_dequant[m,n,i,:] = (kv_u8[m,n,i,:].to(dtype=torch.float16) - kv_zp[m,n,i,0].to(dtype=torch.float16)) * kv_scale[m,n,i,0].to(dtype=torch.float16)

    return kv_dequant

def quan_per_channel(kv):
    blk_num, kv_heads, blksz, head_size = kv.shape
    kv_max = kv.amax(dim=2, keepdim=True)
    kv_min = kv.amin(dim=2, keepdim=True)
    qrange = kv_max - kv_min

    INTMAX = 255.0
    INTMIN = 0.0
    INTRANGE = INTMAX - INTMIN

    # need to consider qrange equals to zero
    kv_scale = torch.zeros(blk_num, kv_heads, 1, head_size, dtype=torch.float16)
    kv_scale[qrange!=0] = (INTRANGE / qrange[qrange!=0]).to(dtype=torch.float16)
    kv_zp = ((0.0 - kv_min) * kv_scale + INTMIN).to(dtype=torch.float16)
    kv_INT8 = torch.round(kv * kv_scale + kv_zp).clamp(INTMIN, INTMAX).to(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    dq_scale = torch.zeros(blk_num, kv_heads, 1, head_size, dtype=torch.float16)
    dq_scale[kv_scale!=0] = (1.0 / kv_scale[kv_scale!=0]).to(dtype=torch.float16)
    dq_scale = dq_scale.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)
    kv_zp = kv_zp.view(dtype=torch.uint8).reshape(blk_num,kv_heads,-1)

    return torch.concat((kv_INT8, dq_scale, kv_zp), dim=-1)

def dequant_per_channel(kv, head_size, blk_size):
    blk_num, kv_head_num, _ = kv.shape
    kv_u8 = kv[:,:,:head_size * blk_size].to(dtype=torch.float16).reshape(blk_num, kv_head_num, blk_size, head_size)
    kv_scale = kv[:,:,head_size * blk_size: (head_size * blk_size + head_size * 2)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, 1, head_size)
    kv_zp = kv[:,:, (head_size * blk_size + head_size * 2):(head_size * blk_size + head_size * 4)].view(dtype=torch.float16).reshape(blk_num, kv_head_num, 1, head_size)

    # print("dequant_kv_u8 = ", kv_u8)
    # print("dequant_kv_scale = ", kv_scale.reshape(blk_num, kv_head_num, blk_size))
    # print("dequant_kv_zp    = ", kv_zp.reshape(blk_num, kv_head_num, blk_size))

    kv_dequant = torch.empty([blk_num, kv_head_num, blk_size, head_size], dtype=torch.float16)

    for m in range(blk_num):
        for n in range(kv_head_num):
            for i in range(head_size):
                kv_dequant[m,n,:,i] = (kv_u8[m,n,:,i].to(dtype=torch.float16) - kv_zp[m,n,0,i].to(dtype=torch.float16)) * kv_scale[m,n,0,i].to(dtype=torch.float16)

    return kv_dequant

if enable_clean_unused_kvcache:
    print("before blocking k:", k.shape, k.dtype)
    print("before blocking v:", v.shape, v.dtype)
    if kvcache_quantization_by_token:
        k.view(torch.uint16)[0,kv_len:,:,:] = 0xFE00
    else:
        k.view(torch.uint16)[0,kv_len:,:,:] = 0
    v.view(torch.uint16)[0,kv_len:,:,:] = 0xFE00

# change from [batch, kv_len, num_kv_heads, head_size] to [total_blk_num, num_kv_heads, kv_block_size, head_size]
k = k.reshape(total_blk_num, kv_block_size, num_kv_heads, head_size).transpose(1,2).contiguous()
v = v.reshape(total_blk_num, kv_block_size, num_kv_heads, head_size).transpose(1,2).contiguous()
print("after blocking k:", k.shape, k.dtype)
print("after blocking v:", v.shape, v.dtype)

vprint("k[0,0,0,:] = ", k[0,0,0,:])
vprint("v[0,0,0,:] = ", v[0,0,0,:])
print()

if enable_kvcache_compression:
    # print("quant = ", k.reshape(total_blk_num, num_kv_heads, kv_block_size, head_size))
    k_origin = k.clone()
    print("k = ", k.shape)
    if kvcache_quantization_by_token:
        k = quan_per_token(k)
    else:
        k = quan_per_channel(k)
    v = quan_per_token(v)
    print(f"quant k shape: {k.shape}, dtype={k.dtype}")
    print(f"quant v shape: {v.shape}, dtype={v.dtype}")

    enable_dequant_check = 1
    if enable_dequant_check:
        if kvcache_quantization_by_token:
            k_dequan = dequant_per_token(k, head_size, kv_block_size)
        else:
            k_dequan = dequant_per_channel(k, head_size, kv_block_size)
        v_dequan = dequant_per_token(v, head_size, kv_block_size)
        # print("de-quant = ", k_dequan.reshape(total_blk_num, num_kv_heads, kv_block_size, head_size))
        # print("diff = ", (k_dequan - k_origin).abs())
        # print("k_dequan = ", k_dequan.shape)

        q_input = q.transpose(1,2).contiguous()
        k_input = k_dequan.transpose(1,2).reshape(batch, new_kv_len, num_kv_heads, head_size).transpose(1,2).contiguous()
        v_input = v_dequan.transpose(1,2).reshape(batch, new_kv_len, num_kv_heads, head_size).transpose(1,2).contiguous()

        # print("q = ", q_input.shape)
        # print("k_input = ", k_input.shape)
        org = get_org(q_input, k_input[:, :, :kv_len, :], v_input[:, :, :kv_len, :], attention_mask[:, :, :, :kv_len])
        check_close(ref, org, atol=1e-3, rtol=1e-2)


# print("quanted k[0,0,16*32:16*32+2*16]:", k[0,0,16*32:16*32+2*16])
# scale_slice=k[0,0,16*32:16*32+2*16]
# print("scale_slice:", scale_slice, scale_slice.dtype)
# scale_temp=scale_slice[0:2].view(torch.half)
# zp_slice=k[0,0,16*34:16*34+2*16]
# zp_temp=zp_slice[0:2].view(torch.half)[0]

# print("quanted scale = ",scale_temp)
# print("quanted zp    = ",zp_temp)
# print()

# k_test = k[0,0,:34]
# k_test = (k_test.to(dtype=torch.half) - zp_temp).to(dtype=torch.half) * scale_temp.to(dtype=torch.half)
# print("de-quant k_test:", k_test)
# print()

# print("reshape k:", k.shape)
# print("k[8,1,:,:]=", k[8,1,:,:])

blocked_k = k.clone()
blocked_v = v.clone()

for i in range(len(block_indices)):
    k[block_indices[i],:] =  blocked_k[i,:]
    v[block_indices[i],:] =  blocked_v[i,:]

k = k.contiguous()
v = v.contiguous()

print(f"final k shape: {k.shape}")
print(f"final v shape: {v.shape}")

#print("block_indices:", block_indices)
#print("blocked k:", k)

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

'''
ugemm_qk: [q_step, head_size] x [head_size, kv_step]
ugemm_kq: [kv_step, head_size] x [head_size, q_step]
ugemm_pv: [q_step, kv_step] x [kv_step, head_size]
'''

scale_factor = 1.0/(head_size**0.5)

# each WG processes a partition
GWS=[seq_num, num_kv_heads, (new_kv_len + kv_partition_size - 1) // kv_partition_size]
WG_SIZE = 1;#max(kv_len // kv_partition_size//2, 1)
LWS=[1, 1, WG_SIZE]

# GWS=[seq_num, (new_kv_len + kv_partition_size - 1) // kv_partition_size, num_heads]
# WG_SIZE = 1;#max(kv_len // kv_partition_size//2, 1)
# LWS=[1, WG_SIZE, num_heads//num_kv_heads]
# #LWS=[1, 1, WG_SIZE]

print("GWS=", GWS)
print("LWS=", LWS)

#========================================================================
# Optimization Log
r'''
increase WG_SIZE to 16 (-70ms)
use GRF to store temp Output(rO) instead of SLM, requires `-Qxcm_register_file_size=256` (-40ms)
avoid type-promotion:   St = cm_mul<float>(St, scale_factor);    =>   St = cm_mul<float>(St, (float)scale_factor);   (-10ms) 
change dtype of attention_mask from float to half (-14ms)
avoid indirect register access: unroll for loop which access matrix rows using loop-index
use GRF to store temp Input rQ instead of SLM, this allows more Work-Groups to be packed into same Xe-core!!!
'''
#========================================================================

src1 = r'''
//# CM kernel for flash attn, reference

#pyeval f"#define HEADS_NUM {num_heads}"
#pyeval f"#define KV_HEADS_NUM {num_kv_heads}"
#pyeval f"#define HEAD_SIZE {head_size}"
#pyeval f"#define Q_STEP {q_step}"
#pyeval f"#define KV_STEP {kv_step}"
#pyeval f"#define SCALE_FACTOR {scale_factor}"
#pyeval f"#define args_verbose {args.verbose}"
#pyeval f"#define WG_SIZE {WG_SIZE}"

#pyeval f"#define KV_BLOCK_SIZE {kv_block_size}"
#pyeval f"#define KV_PARTITION_SIZE {kv_partition_size}"
#pyeval f"#define REDUCE_SPLIT_SIZE {reduce_split_step}"

#pyeval f"#define CLEAN_UNUSED_KVCACHE {enable_clean_unused_kvcache}"

#pyeval f"#define KV_CACHE_COMPRESSION {enable_kvcache_compression}"
#pyeval f"#define KV_CACHE_COMPRESSION_BY_TOKEN {kvcache_quantization_by_token}"

// #define KV_CACHE_COMPRESSION 0

// xe-1:8, xe-2:16
#pyeval f"#define XE_ARCH {xe_arch}"

#if XE_ARCH==1
#define REG_N 8
#define USE_LSC_BLOCK_2D_DESC 0
#else
#define REG_N 16
#define USE_LSC_BLOCK_2D_DESC 1
#endif

#define SystolicDepth 8
#define RepeatCount 1
#define VNNI_WIDTH 2
#define REG_K (SystolicDepth * VNNI_WIDTH)
#define REG_M RepeatCount

#define PRINT_THR_ID 1000
#define PRINT_HEAD_ID 1000

#define KV_PARTITION_STEP_NUM  (KV_PARTITION_SIZE / KV_STEP)

// static_assert(Q_STEP == 16);
static_assert(KV_STEP == 8 || KV_STEP == 16);

template<typename T, int M, int N>
void show(matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}


template<typename T, int M, int N>
void show_u8(matrix<T, M, N> mat) {
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%4d", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template<typename T, int N>
void show(vector<T, N> vec) {
    printf("\t[");
    for(int n = 0; n < N; n ++) {
        printf("%8.4f,", vec[n]);
    }
    printf("]\n");
}

CM_INLINE uint64_t get_clock() {
    auto clk = cm_clock();
    return ((uint64_t)clk[1]) << 32 | clk[0];
}

template <typename T1, typename T2>
CM_INLINE void Transpose_16x16(matrix_ref<T1, 16, 16> in,
                               matrix_ref<T2, 16, 16> out) {
  matrix<T2, 16, 16> bBuf;
  bBuf.row(0) = in.template select<4, 1, 4, 4>(0, 0);   // 0,4,8,c
  bBuf.row(1) = in.template select<4, 1, 4, 4>(4, 0);   // 0,4,8,c
  bBuf.row(2) = in.template select<4, 1, 4, 4>(8, 0);   // 0,4,8,c
  bBuf.row(3) = in.template select<4, 1, 4, 4>(12, 0);  // 0,4,8,c
  bBuf.row(4) = in.template select<4, 1, 4, 4>(0, 1);   // 1,5,9,d
  bBuf.row(5) = in.template select<4, 1, 4, 4>(4, 1);   // 1,5,9,d
  bBuf.row(6) = in.template select<4, 1, 4, 4>(8, 1);   // 1,5,9,d
  bBuf.row(7) = in.template select<4, 1, 4, 4>(12, 1);  // 1,5,9,d
  bBuf.row(8) = in.template select<4, 1, 4, 4>(0, 2);   // 2,6,a,e
  bBuf.row(9) = in.template select<4, 1, 4, 4>(4, 2);   // 2,6,a,e
  bBuf.row(10) = in.template select<4, 1, 4, 4>(8, 2);  // 2,6,a,e
  bBuf.row(11) = in.template select<4, 1, 4, 4>(12, 2); // 2,6,a,e
  bBuf.row(12) = in.template select<4, 1, 4, 4>(0, 3);  // 3,7,b,f
  bBuf.row(13) = in.template select<4, 1, 4, 4>(4, 3);  // 3,7,b,f
  bBuf.row(14) = in.template select<4, 1, 4, 4>(8, 3);  // 3,7,b,f
  bBuf.row(15) = in.template select<4, 1, 4, 4>(12, 3); // 3,7,b,f

  out.row(0) = bBuf.template select<4, 1, 4, 4>(0, 0);   // 0
  out.row(1) = bBuf.template select<4, 1, 4, 4>(4, 0);   // 1
  out.row(2) = bBuf.template select<4, 1, 4, 4>(8, 0);   // 2
  out.row(3) = bBuf.template select<4, 1, 4, 4>(12, 0);  // 3
  out.row(4) = bBuf.template select<4, 1, 4, 4>(0, 1);   // 4
  out.row(5) = bBuf.template select<4, 1, 4, 4>(4, 1);   // 5
  out.row(6) = bBuf.template select<4, 1, 4, 4>(8, 1);   // 6
  out.row(7) = bBuf.template select<4, 1, 4, 4>(12, 1);  // 7
  out.row(8) = bBuf.template select<4, 1, 4, 4>(0, 2);   // 8
  out.row(9) = bBuf.template select<4, 1, 4, 4>(4, 2);   // 9
  out.row(10) = bBuf.template select<4, 1, 4, 4>(8, 2);  // a
  out.row(11) = bBuf.template select<4, 1, 4, 4>(12, 2); // b
  out.row(12) = bBuf.template select<4, 1, 4, 4>(0, 3);  // c
  out.row(13) = bBuf.template select<4, 1, 4, 4>(4, 3);  // d
  out.row(14) = bBuf.template select<4, 1, 4, 4>(8, 3);  // e
  out.row(15) = bBuf.template select<4, 1, 4, 4>(12, 3); // f
}

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
  matrix<T2, 8, 8> temp;
  temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
  temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
  temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
  temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
  temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
  temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
  temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
  temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

  out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
  out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
  out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
  out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
  out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
  out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
  out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
  out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
}

//prepack [K, N] to [K/2, N, 2] layout.
template <typename T1, typename T2, int K, int N>
inline void prepackAsVNNIWidth2(matrix_ref<T1, K, N> input, matrix_ref<T2, K/2, N*2> out) {
    #pragma unroll
    for (int r = 0; r < K/2; r++) {
        out.row(r).select<N, 2>(0) = input.row(r*2);
        out.row(r).select<N, 2>(1) = input.row(r*2+1);
    }
}

#define Q_SLICE_NUM (HEADS_NUM / KV_HEADS_NUM)
#if Q_SLICE_NUM > 8 || Q_SLICE_NUM == 1
#define Q_RepeatCount 1
#else
#define Q_RepeatCount Q_SLICE_NUM
#endif

#if KV_CACHE_COMPRESSION
    // scale/zp is half-precision, so size = 2 * 2 = 4 bytes
    #define KV_SCALE_ZP_SIZE 4 // scale/zp bytes
    #define KV_ELEMENT_TYPE uint8_t
#else
    #define KV_SCALE_ZP_SIZE 0 // no scale/zp
    #define KV_ELEMENT_TYPE half
#endif

extern "C" _GENX_MAIN_ void cm_sdpa_2nd(
    half* query [[type("svmptr_t")]],
    KV_ELEMENT_TYPE* key [[type("svmptr_t")]],
    KV_ELEMENT_TYPE* value [[type("svmptr_t")]],
    int* past_lens [[type("svmptr_t")]],
    int* block_indices [[type("svmptr_t")]],
    int* block_indices_begins [[type("svmptr_t")]],
    int* subsequence_begins [[type("svmptr_t")]],
    float* output [[type("svmptr_t")]],
    float* lse [[type("svmptr_t")]],
    int* gws_subseq_mapping [[type("svmptr_t")]],
    int q_len// 1
    ) {
    //# batch=1, seq_num=1 or >1
    //# query [seq_idx, seq_num, head_num, head_size]
    //# output[seq_idx, seq_num, head_num, head_size]
    //#   key [block_num, kv_head_num, block_size, head_size] + [block_num, kv_head_num, block_size, 4] (scale/zp)
    //# value [block_num, kv_head_num, block_size, head_size] + [block_num, kv_head_num, block_size, 4] (scale/zp)

    //# KV_PARTITION_SIZE should be multiple of kv_block_size(KV_BLOCK_SIZE)
    //# kv_len dimision will be split into multiple partitions, each WG process a partition
    //# total_partitions_num = kv_len // KV_PARTITION_SIZE
    //# GWS=[seq_num, num_kv_heads, total_partitions_num]
    //# LWS=[1, 1, 1]

    //# Each WG processes a partition, which is KV_PARTITION_SIZE long and multiple of KV_BLOCK_SIZE.
    //# KV_BLOCK_SIZE can be 32/64/128/256, etc.
    const auto seq_idx = cm_global_id(0);
    const auto kv_head_num_idx = cm_global_id(1);
    const auto head_num_idx = kv_head_num_idx * (HEADS_NUM/KV_HEADS_NUM);
    //# KV_PARTITION_SIZE --> EU thread
    const auto wg_thread_id = cm_global_id(2);
    const uint kv_partition_num = cm_group_count(2);
    const uint kv_partition_idx = cm_group_id(2);

    //# const uint subsequence_idx = gws_subseq_mapping[seq_idx];
    //# const uint subsequence_idx = seq_idx;
    //# const uint subsequence_begin = subsequence_begins[subsequence_idx];
    //# const uint subsequence_end = subsequence_begins[subsequence_idx + 1];
    const uint kv_len = past_lens[seq_idx] + 1;
    // The code here requires KV_PARTITION_SIZE to be an integer multiple of KV_BLOCK_SIZE.
    const uint start_block_idx = block_indices_begins[seq_idx] + kv_partition_idx * (KV_PARTITION_SIZE / KV_BLOCK_SIZE);

    //if(cm_global_id(0)==0 && cm_global_id(1)==0 && cm_global_id(2)==0)
    //{
    //    printf("cm_sdpa_2nd: kv_len = %d\n", kv_len);
    //}

    if(kv_partition_idx * KV_PARTITION_SIZE > kv_len) {
        // printf("WG exit: kv_partition_idx=%d, KV_PARTITION_SIZE=%d, kv_len=%d\n", kv_partition_idx, KV_PARTITION_SIZE, kv_len);
        return;
    }
    const uint total_blocks_num = (kv_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    constexpr uint kv_pitch = HEAD_SIZE * sizeof(KV_ELEMENT_TYPE);

    //# Load Q into register(as dpas-A tile)
    const uint qo_offset = (seq_idx * HEADS_NUM * q_len + head_num_idx) * HEAD_SIZE;

    #if Q_RepeatCount != 1
        matrix <half, Q_RepeatCount, HEAD_SIZE> Qmat = 0;
        cm_svm_block_read<half, Q_RepeatCount * HEAD_SIZE>((svmptr_t)(query + qo_offset), Qmat.format<half>());
    #else
        matrix <half, Q_SLICE_NUM, HEAD_SIZE> Qmat = 0;
        cm_svm_block_read<half, Q_SLICE_NUM * HEAD_SIZE>((svmptr_t)(query + qo_offset), Qmat.format<half>());
    #endif

    //if(kv_head_num_idx==0 && kv_partition_idx == 0) {
    //    printf("Qmat loaded, kv_head_num_idx=%d\n", kv_head_num_idx);
    //    show(Qmat);
    //}

    constexpr uint per_v_block_element_num = KV_BLOCK_SIZE * KV_HEADS_NUM * (HEAD_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp
    #if KV_CACHE_COMPRESSION_BY_TOKEN
        constexpr uint per_k_block_element_num = KV_BLOCK_SIZE * KV_HEADS_NUM * (HEAD_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp
    #else
        constexpr uint per_k_block_element_num = KV_HEADS_NUM * HEAD_SIZE * (KV_BLOCK_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp
    #endif
    uint block_num = KV_PARTITION_SIZE / KV_BLOCK_SIZE;

    uint leftover_size = 0;
    if(kv_partition_idx == kv_partition_num - 1) {
        // last partition
        leftover_size = (kv_len - KV_PARTITION_SIZE * kv_partition_idx) % KV_PARTITION_SIZE;
    }
    if(block_num > total_blocks_num - start_block_idx) {
        block_num = total_blocks_num - start_block_idx;
    }

    //# rS = Q @ Kt
    //#  KV_PARTITION_STEP_NUM * [REG_M, REG_K] * [REG_K, REG_N] = KV_PARTITION_STEP_NUM * [REG_M, REG_N]
    #if Q_RepeatCount != 1
        matrix<float, Q_RepeatCount, REG_M * KV_PARTITION_STEP_NUM * REG_N> rS = 0;
    #else
        matrix<float, Q_SLICE_NUM, REG_M * KV_PARTITION_STEP_NUM * REG_N> rS = 0;
    #endif
    // # Each SG can process multiple blocks
    #pragma unroll
    for(uint block_idx = 0, ki = 0; block_idx < block_num; block_idx++) {
        // split kv_partition into multi kv_block
        uint blk_indices = block_indices[start_block_idx + block_idx];
        uint k_base_offset = blk_indices * per_k_block_element_num + kv_head_num_idx * (per_k_block_element_num / KV_HEADS_NUM);
        uint k_scale_zp_offset = k_base_offset + KV_BLOCK_SIZE * HEAD_SIZE; // scale/zp offset

        // printf("seq_idx = %d, head_num_idx = %d, kv_partition_idx = %d,  start_block_idx = %d, block_idx = %d, blk_indices = %d, KV_PARTITION_SIZE = %d, KV_BLOCK_SIZE = %d, total_blocks_num = %d, kv_pitch = %d, k_base_offset = %d\n",
        //        seq_idx, head_num_idx, kv_partition_idx, start_block_idx, block_idx, blk_indices, KV_PARTITION_SIZE, KV_BLOCK_SIZE, total_blocks_num, kv_pitch, k_base_offset);

    #if USE_LSC_BLOCK_2D_DESC
        #if KV_CACHE_COMPRESSION
            // Transpose only support dword and qwork
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/4> b2dK(reinterpret_cast<uint*>(key + k_base_offset),  KV_BLOCK_SIZE - 1, HEAD_SIZE*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
        #else
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dK(reinterpret_cast<uint*>(key + k_base_offset),  KV_BLOCK_SIZE - 1, HEAD_SIZE*sizeof(half) - 1, kv_pitch - 1, 0, 0);
        #endif
    #else
        uint kv_offset = k_base_offset;
        uint kv_stride = HEAD_SIZE;
        uint kv_x0 = 0, kv_y0 = 0;
        uint kv_x1 = HEAD_SIZE*sizeof(KV_ELEMENT_TYPE);
        uint kv_y1 = KV_BLOCK_SIZE;
    #endif

        uint kv_pos_end = KV_BLOCK_SIZE;
        if(block_idx == block_num - 1 && leftover_size > 0) {
            kv_pos_end = leftover_size % KV_BLOCK_SIZE;
            if(kv_pos_end == 0) kv_pos_end = KV_BLOCK_SIZE;
        }

        #if KV_CACHE_COMPRESSION
            #if KV_CACHE_COMPRESSION_BY_TOKEN
            // load scale/zp
            vector<half, KV_BLOCK_SIZE> scale_vec;
            vector<half, KV_BLOCK_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset + KV_BLOCK_SIZE * sizeof(half)), zp_vec);
            if(kv_pos_end < KV_BLOCK_SIZE) {
                // fill leftover with last valid scale/zp
                #pragma unroll
                for(int i = kv_pos_end; i < KV_BLOCK_SIZE; i++) {
                    scale_vec[i] = 0.0;
                    zp_vec[i] = 0.0;
                }
            }
            #else
            // load scale/zp
            vector<half, HEAD_SIZE> scale_vec;
            vector<half, HEAD_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + k_scale_zp_offset + HEAD_SIZE * sizeof(half)), zp_vec);
            #endif
        #endif

        //if(cm_global_id(0)==0 && cm_global_id(1)==0){
        //    printf("kv_partition_idx = %d, leftover_size = %d, kv_pos_end = %d, token_idx = %d\n", kv_partition_idx, leftover_size, kv_pos_end, kv_pos_end + KV_PARTITION_SIZE * kv_partition_idx + block_idx*KV_BLOCK_SIZE);
        //}

        for(int kv_pos = 0; kv_pos < kv_pos_end; kv_pos += KV_STEP, ki++) {
            // auto rSvec = rS[ki].format<float>();
            // uint kv_offset_y = kv_pos;

            #if KV_CACHE_COMPRESSION && KV_CACHE_COMPRESSION_BY_TOKEN
                vector<half, REG_N * 2> temp_scale, temp_zp;
                temp_scale.select<REG_N,2>(0) = scale_vec.select<REG_N,1>(kv_pos);
                temp_scale.select<REG_N,2>(1) = scale_vec.select<REG_N,1>(kv_pos);
                temp_zp.select<REG_N,2>(0) = zp_vec.select<REG_N,1>(kv_pos);
                temp_zp.select<REG_N,2>(1) = zp_vec.select<REG_N,1>(kv_pos);
            #endif

            #pragma unroll
            #if KV_CACHE_COMPRESSION
            for(int k = 0, ri = 0; k < HEAD_SIZE/4; k += REG_K/4, ri ++ ) {
            #else
            for(int k = 0, ri = 0; k < HEAD_SIZE/2; k += REG_K/2, ri ++ ) {
            #endif
                matrix<half, REG_K, REG_N> Kt = 0;
            #if USE_LSC_BLOCK_2D_DESC
                //# Load Kt into register & pack as VNNI(as dpas-B tile)
                //# DWORD transposed load == (transposed + VNNI) load
                b2dK.set_block_x(k);

                #if KV_CACHE_COMPRESSION
                    // dequantize
                    matrix<uint8_t, REG_K, REG_N> Kt_quant_temp, Kt_quant;
                    cm_load<lsc::Transpose>(Kt_quant_temp.format<uint>(), b2dK.set_block_y(kv_pos));
                    auto quant_src = Kt_quant_temp.format<ushort, REG_K/2, REG_N>();
                    auto quant_dst = Kt_quant.format<ushort, REG_K/2, REG_N>();

                    #pragma unroll
                    for(int r = 0; r < REG_K / 2; r += 2) {
                        quant_dst.row(r  ) = quant_src.select<2,1,8,2>(r,0);
                        quant_dst.row(r+1) = quant_src.select<2,1,8,2>(r,1);
                    }

                    #if 0
                    printf("Kt_quant_temp: k = %d\n", k);
                    show_u8(Kt_quant_temp.format<uint8_t, REG_K, REG_N>());
                    printf("Kt_quant_vnni: k = %d\n", k);
                    show_u8(Kt_quant.format<uint8_t, REG_K, REG_N>());
                    #endif

                    #if KV_CACHE_COMPRESSION_BY_TOKEN
                    #pragma unroll
                    for(int r = 0; r < REG_K; r++) {
                        Kt[r] = Kt_quant[r] - temp_zp.format<half, 2, REG_N>()[r%2]; //vector - vector
                        Kt[r] = cm_mul<half>(Kt[r], temp_scale.format<half, 2, REG_N>()[r%2]);    // vector * vector
                    }
                    #else
                    vector<half, REG_K> temp_scale, temp_zp;
                    temp_scale.select<REG_K, 1>(0) = scale_vec.select<REG_K, 1>(k * 4);
                    temp_zp.select<REG_K, 1>(0) = zp_vec.select<REG_K, 1>(k * 4);

                    // performance is very slow for this code, 37.2ms
                    // #pragma unroll
                    // for(int r = 0; r < REG_K; r+=2) {
                        // #pragma unroll
                        // for(int z = 0; z <= 1; z++) {
                        //    Kt[r].select<REG_N/2, 2>(z) = Kt_quant[r].select<REG_N/2, 2>(z) - temp_zp[r+z];
                        //    Kt[r].select<REG_N/2, 2>(z)  = cm_mul<half>(Kt[r].select<REG_N/2, 2>(z), temp_scale[r+z]);
                        //    Kt[r+1].select<REG_N/2, 2>(z) = Kt_quant[r+1].select<REG_N/2, 2>(z) - temp_zp[r+z];
                        //    Kt[r+1].select<REG_N/2, 2>(z)  = cm_mul<half>(Kt[r+1].select<REG_N/2, 2>(z), temp_scale[r+z]);
                        //}
                    // }

                    auto Kt_dequant_out = Kt.format<half, REG_K/2, 2*REG_N>();
                    auto Kt_dequant_tmp = Kt_quant.format<uint8_t, REG_K/2, 2*REG_N>();
                    #pragma unroll
                    for(int r = 0; r < REG_K/2; r++) {
                        Kt_dequant_out[r].select<REG_N, 2>(0) = Kt_dequant_tmp[r].select<REG_N, 2>(0) - temp_zp[r*2];
                        Kt_dequant_out[r].select<REG_N, 2>(0) = cm_mul<half>(Kt_dequant_out[r].select<REG_N, 2>(0), temp_scale[r*2]);
                        Kt_dequant_out[r].select<REG_N, 2>(1) = Kt_dequant_tmp[r].select<REG_N, 2>(1) - temp_zp[r*2+1];
                        Kt_dequant_out[r].select<REG_N, 2>(1) = cm_mul<half>(Kt_dequant_out[r].select<REG_N, 2>(1), temp_scale[r*2+1]);
                    }
                    #endif
                #else
                    cm_load<lsc::Transpose>(Kt.format<uint>(), b2dK.set_block_y(kv_pos));
                    #if CLEAN_UNUSED_KVCACHE & 0
                    // Not need clean K cache: 1) col write will lead huge perf drop; 2) softmax will clear unused scores
                    if(kv_pos_end < kv_pos + KV_STEP) {
                        auto KmatRef = Kt.format<half, REG_K/2, REG_N*2>();
                        uint valid_cols = kv_pos_end - kv_pos;
                        uint valid_cols_vnni = valid_cols * 2;
                        //printf("Kt: kv_pos = %d, kv_pos_end = %d, leftover = %d\n", kv_pos, kv_pos_end, KV_STEP - valid_cols);
                        //show(Kt.format<half, REG_K/2, REG_N*2>());
                        for (int r = valid_cols_vnni; r < KV_STEP * 2; r++)
                            KmatRef.select<REG_K/2,1,1,1>(0,r) = 0.0f;
                        //printf("Kt: reset unused...\n");
                        //show(Kt.format<half, REG_K/2, REG_N*2>());
                        //printf("\n\n");
                    }
                    #endif
                #endif
                //if(kv_partition_idx==kv_partition_num - 1 && kv_head_num_idx == KV_HEADS_NUM - 1) {
                //    printf("Kt: k = %d\n", k);
                //    show(Kt.format<half, REG_K, REG_N>());
                //}
            #else
                matrix<uint, REG_N, REG_K/2> temp;
                uint cur_kv_offset = kv_offset + kv_pos * kv_stride + k * 2;// uint --> half
                #pragma unroll
                for(int kk = 0; kk < REG_N; kk++) {
                    cm_svm_block_read<uint, REG_K/2>((svmptr_t)(key + cur_kv_offset + kk * kv_stride), temp[kk].format<uint>());
                }
                #if XE_ARCH==1
                Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,0));
                #else
                Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,0));
                Transpose_8x8(temp.select<8,1,8,1>(8,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,8));
                #endif
            #endif
            #if Q_RepeatCount != 1
                matrix<half, Q_RepeatCount, REG_K> Qmat_data = Qmat.select<Q_RepeatCount,1,REG_K,1>(0, ri*REG_K);
                matrix<float, Q_RepeatCount, REG_N> rS_data = 0;
                rS_data = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, Q_RepeatCount>(
                            rS_data.format<float>(),
                            Kt.format<int32_t>(),
                            Qmat_data.format<int32_t>());
                rS.select<Q_RepeatCount, 1, REG_N, 1>(0, ki*REG_N) += rS_data;

                //if(kv_partition_idx==kv_partition_num - 1 && kv_head_num_idx == KV_HEADS_NUM - 1) {
                //    show(rS_data);
                //}
            #else
                #pragma unroll
                for(int qi = 0; qi < Q_SLICE_NUM; qi ++) {
                    auto Qmat_slice = Qmat[qi].format<half, HEAD_SIZE / REG_K, REG_K>();
                    auto rSvec = rS[qi].format<float, REG_M * KV_PARTITION_STEP_NUM, REG_N>()[ki].format<float>();
                    rSvec = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            rSvec,
                            Kt.format<int32_t>(),
                            Qmat_slice[ri].format<int32_t>());
                }
            #endif
            }
        }
    }

    //if(kv_partition_idx==kv_partition_num - 1 && kv_head_num_idx == KV_HEADS_NUM - 1) {
    //    printf("rS:\n");
    //    show(rS);
    //}

    // online softmax
    vector<float, Q_SLICE_NUM> cur_sum = 0.0f;
    vector<float, Q_SLICE_NUM> cur_lse = 0.0f;
    #if XE_ARCH==1
    matrix<half, KV_PARTITION_STEP_NUM / 2 * REG_M, REG_N> Pmat = 0;
    #else
        #if Q_RepeatCount != 1
            matrix<half, Q_RepeatCount, KV_PARTITION_STEP_NUM * REG_M * REG_N> Pmat = 0;
        #else
            matrix<half, Q_SLICE_NUM, KV_PARTITION_STEP_NUM * REG_M * REG_N> Pmat = 0;
        #endif
    #endif
    #pragma unroll
    for(int qi = 0; qi < Q_SLICE_NUM; qi++) {
        auto rS_slice = rS[qi].format<float, KV_PARTITION_STEP_NUM, REG_N>();
        rS_slice = cm_mul<float>(rS_slice, (float)SCALE_FACTOR);  // convert scale_factor into (float), or it will be promoted to double

        // printf("leftover_size = %d, leftover_aligned_size = %d, XE_ARCH = %d, KV_PARTITION_STEP_NUM * REG_N = %d\n", leftover_size, leftover_aligned_size, XE_ARCH, KV_PARTITION_STEP_NUM * REG_N);
        if(leftover_size > 0) {
            auto Svec = rS_slice.format<float>();
            for(int i = leftover_size; i < KV_PARTITION_STEP_NUM * REG_N; i++){
                Svec[i] = -3e38f;
            }
        }

        // compute lse
        constexpr float log2e = 1.4426950408889634f;
        constexpr float loge2 = 0.6931471805599453f;

        // compute row_max
        auto rSv = rS_slice.format<float>();
        float row_max = rSv[0];
        // It is performance hotspot for u8,  must add unroll
        #if KV_CACHE_COMPRESSION
        #pragma unroll
        #endif
        for(int r = 1; r < rSv.n_elems(); r++)
            row_max = cm_max<float>(row_max, rSv[r]);

        // compute Pmat = exp(rS_slice - row_max)
        #if XE_ARCH==1
        Pmat[qi].format<half, KV_PARTITION_STEP_NUM / 2 * REG_M, REG_K>() = cm_exp((rS_slice.format<float, KV_PARTITION_STEP_NUM / 2 * REG_M, REG_K>() - row_max)*log2e);
        vector<float, KV_PARTITION_STEP_NUM * REG_N> rS_exp_temp = cm_exp((rS_slice.format<float>() - row_max)*log2e);
        #else
        vector<float, KV_PARTITION_STEP_NUM * REG_N> rS_exp_temp = cm_exp((rS_slice.format<float>() - row_max)*log2e);
        Pmat[qi].format<half, KV_PARTITION_STEP_NUM * REG_M, REG_N>() = rS_exp_temp;
        #endif

        cur_lse[qi] = cm_sum<float>(rS_exp_temp.format<float>());
        cur_lse[qi] = cm_log<float>(cur_lse[qi]) * loge2 + row_max; // log2(sum(exp(x))) = log2e * log(sum(exp(x)))

        // compute row sum of P
        auto rPv = Pmat[qi].format<half, 1, KV_PARTITION_STEP_NUM * REG_N>();
        cur_sum[qi] = cm_sum<float>(rPv[0]);
    }

    //if(kv_partition_idx==0 && kv_head_num_idx == 0) {
    //    printf("Pmat:\n");
    //    show(Pmat);
    //}

    //# rO = P * V
    #if Q_RepeatCount != 1
    matrix <float, Q_RepeatCount, HEAD_SIZE/REG_N * REG_M*REG_N> Omat = 0;
    #else
    matrix <float, Q_SLICE_NUM, HEAD_SIZE/REG_N * REG_M*REG_N> Omat = 0;
    #endif
    #pragma unroll
    for(uint block_idx = 0, ki = 0; block_idx < block_num; block_idx++) {
        uint blk_indices = block_indices[start_block_idx + block_idx];
        uint v_base_offset = blk_indices * per_v_block_element_num + kv_head_num_idx * (per_v_block_element_num / KV_HEADS_NUM);
        uint v_scale_zp_offset = v_base_offset + KV_BLOCK_SIZE * HEAD_SIZE; // scale/zp offset

    #if USE_LSC_BLOCK_2D_DESC
        //# vector load cannot be used for block_2d_desc
        //# note: candidate template ignored: deduced type 'details::Block2DRefTy<half, 1U, 16U, 1U, (LoadOp)0U>' (aka 'vector_ref<half,32>') of 1st parameter
        #if KV_CACHE_COMPRESSION
        lsc::block_2d_desc<uint8_t, 1, REG_K, REG_N> b2dV(value + v_base_offset,  KV_BLOCK_SIZE - 1, HEAD_SIZE*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
        #else
        lsc::block_2d_desc<half, 1, REG_K, REG_N>   b2dV(value + v_base_offset,  KV_BLOCK_SIZE - 1, HEAD_SIZE * sizeof(half) - 1, kv_pitch - 1, 0, 0);
        #endif
    #else
        uint kv_offset = v_base_offset;
        uint kv_stride = HEAD_SIZE;
        uint kv_x0 = 0, kv_y0 = 0;
        uint kv_x1 = HEAD_SIZE*sizeof(half);
        uint kv_y1 = KV_BLOCK_SIZE;
    #endif
    
        //if(kv_partition_idx==kv_partition_num - 1 && head_num_idx == HEADS_NUM - 1) {
        //    printf("leftover_size = %d, leftover_aligned_size = %d, XE_ARCH = %d, KV_BLOCK_SIZE = %d\n", leftover_size, leftover_aligned_size, XE_ARCH, KV_BLOCK_SIZE);
        //} 
        uint kv_pos_end = KV_BLOCK_SIZE;
        if(block_idx == block_num - 1 && leftover_size > 0) {
            kv_pos_end = leftover_size % KV_BLOCK_SIZE;
            if(kv_pos_end == 0) kv_pos_end = KV_BLOCK_SIZE;
        }

        #if KV_CACHE_COMPRESSION
            // load scale/zp
            vector<half, KV_BLOCK_SIZE> scale_vec;
            vector<half, KV_BLOCK_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(value + v_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(value + v_scale_zp_offset + KV_BLOCK_SIZE * sizeof(half)), zp_vec);
            if(kv_pos_end < KV_BLOCK_SIZE) {
                // fill leftover with last valid scale/zp
                for(int i = kv_pos_end; i < KV_BLOCK_SIZE; i++) {
                    scale_vec[i] = 0.0;
                    zp_vec[i] = 0.0;
                }
            }
        #endif
        #pragma unroll
        for(int kv_pos = 0; kv_pos < kv_pos_end; kv_pos += REG_K, ki++) {
            // uint kv_offset_y = kv_pos;
            #if KV_CACHE_COMPRESSION
            vector<half, REG_K> temp_scale = scale_vec.select<REG_K, 1>(kv_pos);
            vector<half, REG_K> temp_zp = zp_vec.select<REG_K, 1>(kv_pos);
            #endif
            #pragma unroll
            for(int k = 0, ri = 0; k < HEAD_SIZE; k += REG_N, ri ++ ) {
                // Load V into register & pack as VNNI(as dpas-B tile)
                matrix<half, REG_K, REG_N> VmatNormal;
                matrix<half, REG_M, REG_K*REG_N> Vmat;
            #if USE_LSC_BLOCK_2D_DESC
                b2dV.set_block_x(k); // x is the column index
                #if KV_CACHE_COMPRESSION
                    // dequantize
                    matrix<uint8_t, REG_K, REG_N> Vt_quant;
                    cm_load<lsc::Normal>(Vt_quant.format<uint8_t>(), b2dV.set_block_y(kv_pos));

                    #if 0
                    printf("Vt_quant: k = %d\n", k);
                    show_u8(Vt_quant.format<uint8_t, REG_K, REG_N>());
                    show(temp_scale);
                    show(temp_zp);
                    printf("\n");
                    #endif

                    #pragma unroll
                    for(int r = 0; r < REG_K; r++) {
                        VmatNormal[r] = Vt_quant[r] - temp_zp[r]; // vector - scalar
                        VmatNormal[r] = cm_mul<half>(VmatNormal[r], temp_scale[r]); // vector * scalar
                    }
                    // show(VmatNormal.format<half, REG_K, REG_N>());

                    if(kv_pos_end - kv_pos < KV_STEP) {
                        #pragma unroll
                        for(int r = kv_pos_end; r < KV_STEP; r++)  {
                            VmatNormal[r] = 0;
                        }
                    }
                    prepackAsVNNIWidth2(VmatNormal, Vmat.format<half, REG_K/2, REG_N*2>());
                #else
                    cm_load<lsc::VNNI>(Vmat[0].format<half>(), b2dV.set_block_y(kv_pos)); // y is the row index
                    // somtimes KV cache would be filled with random Nan, so need to clean up the unused value data.
                    #if CLEAN_UNUSED_KVCACHE
                    if(kv_pos_end - kv_pos < KV_STEP) {
                        auto VmatRef = Vmat[0].format<half, REG_K/2, REG_N*2>();
                        uint valid_rows = kv_pos_end - kv_pos;
                        uint valid_rows_vnni = (valid_rows+1)/2;
                        //printf("V: kv_pos = %d, kv_pos_end = %d, leftover = %d\n", kv_pos, kv_pos_end, KV_STEP - valid_rows);
                        //show(VmatRef.format<half, REG_K/2, REG_N*2>());
                        for (int r = valid_rows_vnni; r < KV_STEP / 2; r++)
                            VmatRef.row(r) = 0.f;
                        if (valid_rows % 2 == 1)
                            VmatRef.row(valid_rows_vnni-1).select<REG_N,2>(1) = 0.f;
                        //printf("Kt: reset unused...\n");
                        //show(VmatRef.format<half, REG_K/2, REG_N*2>());
                        //printf("\n\n");
                    }
                    #endif
                #endif
                if(0) {
                    printf("Vmat: k = %d\n", k);
                    show(Vmat.format<half, REG_K, REG_N>());
                }
            #else
                matrix<half, REG_K, REG_N> temp;
                uint cur_kv_offset = kv_offset + kv_pos * kv_stride + k;
                #pragma unroll
                for(int kk = 0; kk < REG_K; kk++) {
                    cm_svm_block_read<half, REG_N>((svmptr_t)(value + cur_kv_offset + kk * kv_stride), temp[kk].format<half>());
                }
                auto Vref = Vmat[0].format<half, REG_K/2, 2*REG_N>();
                Vref.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp.select<REG_K/2, 2, REG_N, 1>(0, 0);
                Vref.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp.select<REG_K/2, 2, REG_N, 1>(1, 0);
            #endif
                #if Q_RepeatCount != 1
                matrix<half, Q_RepeatCount, REG_K> Pmat_data = Pmat.select<Q_RepeatCount,1,REG_K,1>(0, ki*REG_K);
                matrix<float, Q_RepeatCount, REG_N> Omat_data = 0;
                Omat_data = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, Q_RepeatCount>(
                            Omat_data.format<float>(),
                            Vmat[0].format<int32_t>(),
                            Pmat_data.format<int32_t>());
                Omat.select<Q_RepeatCount,1,REG_N,1>(0, ri*REG_N) += Omat_data;
                #else
                for(int qi = 0; qi < Q_SLICE_NUM; qi ++) {
                    auto Pmat_slice = Pmat[qi].format<half, KV_PARTITION_STEP_NUM * REG_M, REG_K>();
                    auto Omat_slice = Omat[qi].format<float, HEAD_SIZE/REG_N * REG_M, REG_N>();
                    Omat_slice[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            Omat_slice[ri],
                            Vmat[0].format<int32_t>(),
                            Pmat_slice[ki].format<int32_t>());
                }
                #endif
                //if(kv_partition_idx==kv_partition_num - 1 && head_num_idx == 27) {
                //    printf("Omat[%d][%d]:\n",kv_pos, k);
                //    show(Omat);
                //}
            }
        }
    }

    //if(kv_partition_idx==5 && kv_head_num_idx == 5) {
    //    printf("Omat:\n");
    //    show(Omat);
    //}

    //# save Output
    #pragma unroll
    for (int qi = 0; qi < Q_SLICE_NUM; qi++) {
        matrix<float, REG_M, REG_N> cur_O_f32;
        uint o_offset = seq_idx * kv_partition_num * HEADS_NUM * HEAD_SIZE + kv_partition_num * (head_num_idx + qi) * HEAD_SIZE + wg_thread_id * HEAD_SIZE;
        float div_cur_sum = 1.0/cur_sum[qi];
        auto Omat_slice = Omat[qi].format<float, HEAD_SIZE/REG_N * REG_M, REG_N>();
        #pragma unroll
        for(int k = 0, ri=0; k < HEAD_SIZE; k += REG_N, ri++) {
            auto cO = Omat_slice[ri].format<float, REG_M, REG_N>();
            #if XE_ARCH==1
            cur_O_f32= cm_mul<float>(cO, div_cur_sum);
            #else
            cur_O_f32= cm_div_ieee(cO, cur_sum[qi]);
            #endif
            cm_svm_block_write<float, REG_N>((svmptr_t)(output + o_offset + k),cur_O_f32.format<float>());
        }
        uint lse_offset = seq_idx * HEADS_NUM * kv_partition_num + (head_num_idx + qi) * kv_partition_num + wg_thread_id;
        lse[lse_offset] = cur_lse[qi];
    }
}

extern "C" _GENX_MAIN_ void cm_sdpa_2nd_half_block(
    half* query [[type("svmptr_t")]],
    KV_ELEMENT_TYPE* key [[type("svmptr_t")]],
    KV_ELEMENT_TYPE* value [[type("svmptr_t")]],
    int* past_lens [[type("svmptr_t")]],
    int* block_indices [[type("svmptr_t")]],
    int* block_indices_begins [[type("svmptr_t")]],
    int* subsequence_begins [[type("svmptr_t")]],
    float* output [[type("svmptr_t")]],
    float* lse [[type("svmptr_t")]],
    int* gws_subseq_mapping [[type("svmptr_t")]],
    int q_len// 1
    ) {
    //# batch=1, seq_num=1 or >1
    //# query [seq_idx, seq_num, head_num, head_size]
    //# output[seq_idx, seq_num, head_num, head_size]
    //#   key [block_num, head_num, block_size, head_size] + [block_num, head_num, block_size, 4] (scale/zp)
    //# value [block_num, head_num, block_size, head_size] + [block_num, head_num, block_size, 4] (scale/zp)

    //# KV_PARTITION_SIZE here equals to half kv_block_size(KV_BLOCK_SIZE)
    //# kv_len dimision will be split into multiple partitions, each WG process a partition
    //# total_partitions_num = kv_len // KV_PARTITION_SIZE
    //# GWS=[seq_num, num_heads, total_partitions_num]
    //# LWS=[1, 1, 1]

    //# Each WG processes half partition, which is KV_PARTITION_SIZE long and multiple of KV_BLOCK_SIZE.
    //# KV_BLOCK_SIZE can be 32/64/128/256, etc.
    const auto seq_idx = cm_global_id(0);
    const auto kv_head_num_idx = cm_global_id(1);
    const auto head_num_idx = kv_head_num_idx * (HEADS_NUM/KV_HEADS_NUM);
    //# KV_PARTITION_SIZE --> EU thread
    const auto wg_thread_id = cm_global_id(2);
    const uint kv_partition_num = cm_group_count(2);
    const uint kv_partition_idx = cm_group_id(2);

    cm_assert(KV_BLOCK_SIZE == 2 * KV_PARTITION_SIZE);

    //# const uint subsequence_idx = gws_subseq_mapping[seq_idx];
    //# const uint subsequence_idx = seq_idx;
    //# const uint subsequence_begin = subsequence_begins[subsequence_idx];
    //# const uint subsequence_end = subsequence_begins[subsequence_idx + 1];
    const uint kv_len = past_lens[seq_idx] + 1;
    const uint start_block_idx = block_indices_begins[seq_idx] + (kv_partition_idx * KV_PARTITION_SIZE) / KV_BLOCK_SIZE;

    //if(cm_global_id(0)==0 && cm_global_id(1)==0 && cm_global_id(2)==0)
    //{
    //    printf("cm_sdpa_2nd: kv_len = %d\n", kv_len);
    //}

    if(kv_partition_idx * KV_PARTITION_SIZE > kv_len) {
        return;
    }
    // const uint total_blocks_num = (kv_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    constexpr uint kv_pitch = HEAD_SIZE * sizeof(KV_ELEMENT_TYPE);

    //# Load Q into register(as dpas-A tile)
    const uint qo_offset = (seq_idx*HEADS_NUM*q_len + head_num_idx)*HEAD_SIZE;

    #if Q_RepeatCount != 1
        matrix <half,Q_RepeatCount, HEAD_SIZE> Qmat = 0;
        cm_svm_block_read<half, Q_RepeatCount * HEAD_SIZE>((svmptr_t)(query + qo_offset), Qmat.format<half>());
    #else
        matrix <half,Q_SLICE_NUM, HEAD_SIZE> Qmat = 0;
        cm_svm_block_read<half, Q_SLICE_NUM * HEAD_SIZE>((svmptr_t)(query + qo_offset), Qmat.format<half>());
    #endif

    //if(kv_head_num_idx==0 && kv_partition_idx == 0) {
    //    printf("Qmat loaded, kv_head_num_idx=%d\n", kv_head_num_idx);
    //    show(Qmat);
    //}

    constexpr uint per_kv_block_element_num = KV_BLOCK_SIZE * KV_HEADS_NUM * (HEAD_SIZE + KV_SCALE_ZP_SIZE); // 4 bytes: scale/zp

    uint leftover_size = 0;
    if(kv_partition_idx == kv_partition_num - 2) {
        leftover_size = (kv_len - KV_PARTITION_SIZE * kv_partition_idx);
        if(leftover_size >= KV_PARTITION_SIZE)
            leftover_size = 0;
    }
    if(kv_partition_idx == kv_partition_num - 1) {
        leftover_size = (kv_len - KV_PARTITION_SIZE * kv_partition_idx);
    }

    //# rS = Q @ Kt
    //#  KV_PARTITION_STEP_NUM * [REG_M, REG_K] * [REG_K, REG_N] = KV_PARTITION_STEP_NUM * [REG_M, REG_N]
    #if Q_RepeatCount != 1
        matrix<float, Q_RepeatCount, REG_M * KV_PARTITION_STEP_NUM * REG_N> rS = 0;
    #else
        matrix<float, Q_SLICE_NUM, REG_M * KV_PARTITION_STEP_NUM * REG_N> rS = 0;
    #endif
    // Each SG can process half block
    {
        uint ki = 0;
        uint blk_indices = block_indices[start_block_idx];
        uint kv_base_offset = blk_indices * per_kv_block_element_num + kv_head_num_idx * (per_kv_block_element_num / KV_HEADS_NUM);
        uint kv_scale_zp_offset = kv_base_offset + KV_BLOCK_SIZE * HEAD_SIZE; // scale/zp offset
        if(kv_partition_idx % 2 == 1) // second half block
            kv_base_offset += KV_PARTITION_SIZE * HEAD_SIZE; // move to the 2nd half partition
            kv_scale_zp_offset += KV_PARTITION_SIZE * sizeof(half); // move to the 2nd half partition

    #if USE_LSC_BLOCK_2D_DESC
        #if KV_CACHE_COMPRESSION
            // Transpose only support dword and qwork
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/4> b2dK(reinterpret_cast<uint*>(key + kv_base_offset),  KV_PARTITION_SIZE - 1, HEAD_SIZE*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
        #else
            lsc::block_2d_desc<uint, 1, REG_N, REG_K/2> b2dK(reinterpret_cast<uint*>(key + kv_base_offset),  KV_PARTITION_SIZE - 1, HEAD_SIZE*sizeof(half) - 1, kv_pitch - 1, 0, 0);
        #endif
    #else
        uint kv_offset = kv_base_offset;
        uint kv_stride = HEAD_SIZE;
        uint kv_x0 = 0, kv_y0 = 0;
        uint kv_x1 = HEAD_SIZE*sizeof(KV_ELEMENT_TYPE);
        uint kv_y1 = KV_BLOCK_SIZE;
    #endif

        uint kv_pos_end = KV_PARTITION_SIZE;
        if(leftover_size > 0) {
            kv_pos_end = leftover_size % KV_PARTITION_SIZE;
            if(kv_pos_end == 0) kv_pos_end = KV_PARTITION_SIZE;
        }

        #if KV_CACHE_COMPRESSION
            // load scale/zp
            vector<half, KV_PARTITION_SIZE> scale_vec;
            vector<half, KV_PARTITION_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + kv_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(key + kv_scale_zp_offset + KV_BLOCK_SIZE * sizeof(half)), zp_vec);
            if(kv_pos_end < KV_PARTITION_SIZE) {
                // fill leftover with last valid scale/zp
                #pragma unroll
                for(int i = kv_pos_end; i < KV_PARTITION_SIZE; i++) {
                    scale_vec[i] = 0.0;
                    zp_vec[i] = 0.0;
                }
            }
        #endif

        //if(cm_global_id(0)==0 && cm_global_id(1)==0){
        //    printf("kv_partition_idx = %d, leftover_size = %d, kv_pos_end = %d, token_idx = %d\n", kv_partition_idx, leftover_size, kv_pos_end, kv_pos_end + KV_PARTITION_SIZE * kv_partition_idx + block_idx*KV_BLOCK_SIZE);
        //}

        for(int kv_pos = 0; kv_pos < kv_pos_end; kv_pos += KV_STEP, ki++) {
            // auto rSvec = rS[ki].format<float>();
            // uint kv_offset_y = kv_pos;

            #if KV_CACHE_COMPRESSION
                vector<half, REG_N * 2> temp_scale, temp_zp;
                temp_scale.select<REG_N,2>(0) = scale_vec.select<REG_N,1>(kv_pos);
                temp_scale.select<REG_N,2>(1) = scale_vec.select<REG_N,1>(kv_pos);
                temp_zp.select<REG_N,2>(0) = zp_vec.select<REG_N,1>(kv_pos);
                temp_zp.select<REG_N,2>(1) = zp_vec.select<REG_N,1>(kv_pos);
            #endif

            #pragma unroll
            #if KV_CACHE_COMPRESSION
            for(int k = 0, ri = 0; k < HEAD_SIZE/4; k += REG_K/4, ri ++ ) {
            #else
            for(int k = 0, ri = 0; k < HEAD_SIZE/2; k += REG_K/2, ri ++ ) {
            #endif
                matrix<half, REG_K, REG_N> Kt = 0;
            #if USE_LSC_BLOCK_2D_DESC
                //# Load Kt into register & pack as VNNI(as dpas-B tile)
                //# DWORD transposed load == (transposed + VNNI) load
                b2dK.set_block_x(k);

                #if KV_CACHE_COMPRESSION
                    // dequantize
                    matrix<uint8_t, REG_K, REG_N> Kt_quant_temp, Kt_quant;
                    cm_load<lsc::Transpose>(Kt_quant_temp.format<uint>(), b2dK.set_block_y(kv_pos));
                    auto quant_src = Kt_quant_temp.format<ushort, REG_K/2, REG_N>();
                    auto quant_dst = Kt_quant.format<ushort, REG_K/2, REG_N>();

                    #pragma unroll
                    for(int r = 0; r < REG_K / 2; r += 2) {
                        quant_dst.row(r  ) = quant_src.select<2,1,8,2>(r,0);
                        quant_dst.row(r+1) = quant_src.select<2,1,8,2>(r,1);
                    }

                    #if 0
                    printf("Kt_quant_temp: k = %d\n", k);
                    show_u8(Kt_quant_temp.format<uint8_t, REG_K, REG_N>());
                    printf("Kt_quant_vnni: k = %d\n", k);
                    show_u8(Kt_quant.format<uint8_t, REG_K, REG_N>());
                    #endif

                    #pragma unroll
                    for(int r = 0; r < REG_K; r++) {
                        Kt[r] = Kt_quant[r] - temp_zp.format<half, 2, REG_N>()[r%2]; //vector - vector
                        Kt[r] = cm_mul<half>(Kt[r], temp_scale.format<half, 2, REG_N>()[r%2]);    // vector * vector
                    }
                #else
                    cm_load<lsc::Transpose>(Kt.format<uint>(), b2dK.set_block_y(kv_pos));
                    #if CLEAN_UNUSED_KVCACHE & 0
                    // Not need clean K cache: 1) col write will lead huge perf drop; 2) softmax will clear unused scores
                    if(kv_pos_end < kv_pos + KV_STEP) {
                        auto KmatRef = Kt.format<half, REG_K/2, REG_N*2>();
                        uint valid_cols = kv_pos_end - kv_pos;
                        uint valid_cols_vnni = valid_cols * 2;
                        //printf("Kt: kv_pos = %d, kv_pos_end = %d, leftover = %d\n", kv_pos, kv_pos_end, KV_STEP - valid_cols);
                        //show(Kt.format<half, REG_K/2, REG_N*2>());
                        for (int r = valid_cols_vnni; r < KV_STEP * 2; r++)
                            KmatRef.select<REG_K/2,1,1,1>(0,r) = 0.0f;
                        //printf("Kt: reset unused...\n");
                        //show(Kt.format<half, REG_K/2, REG_N*2>());
                        //printf("\n\n");
                    }
                    #endif
                #endif
                //if(kv_partition_idx==kv_partition_num - 1 && kv_head_num_idx == KV_HEADS_NUM - 1) {
                //    printf("Kt: k = %d\n", k);
                //    show(Kt.format<half, REG_K, REG_N>());
                //}
            #else
                matrix<uint, REG_N, REG_K/2> temp;
                uint cur_kv_offset = kv_offset + kv_pos * kv_stride + k * 2;// uint --> half
                #pragma unroll
                for(int kk = 0; kk < REG_N; kk++) {
                    cm_svm_block_read<uint, REG_K/2>((svmptr_t)(key + cur_kv_offset + kk * kv_stride), temp[kk].format<uint>());
                }
                #if XE_ARCH==1
                Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,0));
                #else
                Transpose_8x8(temp.select<8,1,8,1>(0,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,0));
                Transpose_8x8(temp.select<8,1,8,1>(8,0), Kt.format<uint, REG_K/2, REG_N>().select<8,1,8,1>(0,8));
                #endif
            #endif
            #if Q_RepeatCount != 1
                matrix<half, Q_RepeatCount, REG_K> Qmat_data = Qmat.select<Q_RepeatCount,1,REG_K,1>(0, ri*REG_K);
                matrix<float, Q_RepeatCount, REG_N> rS_data = 0;
                #pragma unroll
                for(int qi = 0; qi < Q_SLICE_NUM; qi ++) {
                    Qmat_data[qi] = Qmat[qi].format<half, HEAD_SIZE / REG_K, REG_K>()[ri];
                }
                rS_data = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, Q_RepeatCount>(
                            rS_data.format<float>(),
                            Kt.format<int32_t>(),
                            Qmat_data.format<int32_t>());
                rS.select<Q_RepeatCount,1,REG_N,1>(0, ki*REG_N) += rS_data;

                //if(kv_partition_idx==kv_partition_num - 1 && kv_head_num_idx == KV_HEADS_NUM - 1) {
                //    show(rS_data);
                //}
            #else
                #pragma unroll
                for(int qi = 0; qi < Q_SLICE_NUM; qi ++) {
                    auto Qmat_slice = Qmat[qi].format<half, HEAD_SIZE / REG_K, REG_K>();
                    auto rSvec = rS[qi].format<float, REG_M * KV_PARTITION_STEP_NUM, REG_N>()[ki].format<float>();
                    rSvec = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            rSvec,
                            Kt.format<int32_t>(),
                            Qmat_slice[ri].format<int32_t>());
                }
            #endif
            }
        }
    }

    //if(kv_partition_idx==kv_partition_num - 1 && kv_head_num_idx == KV_HEADS_NUM - 1) {
    //    printf("rS:\n");
    //    show(rS);
    //}

    // online softmax
    vector<float, Q_SLICE_NUM> cur_sum = 0.0f;
    vector<float, Q_SLICE_NUM> cur_lse = 0.0f;
    #if XE_ARCH==1
    #else
        #if Q_RepeatCount != 1
            matrix<half, Q_RepeatCount, KV_PARTITION_STEP_NUM * REG_M * REG_N> Pmat = 0;
        #else
            matrix<half, Q_SLICE_NUM, KV_PARTITION_STEP_NUM * REG_M * REG_N> Pmat = 0;
        #endif
    #endif
    #pragma unroll
    for(int qi = 0; qi < Q_SLICE_NUM; qi++) {
        auto rS_slice = rS[qi].format<float, KV_PARTITION_STEP_NUM, REG_N>();
        rS_slice = cm_mul<float>(rS_slice, (float)SCALE_FACTOR);  // convert scale_factor into (float), or it will be promoted to double

        // printf("leftover_size = %d, XE_ARCH = %d, KV_PARTITION_STEP_NUM * REG_N = %d\n", leftover_size, XE_ARCH, KV_PARTITION_STEP_NUM * REG_N);
        if(leftover_size > 0) {
            auto Svec = rS_slice.format<float>();
            for(int i = leftover_size; i < KV_PARTITION_STEP_NUM * REG_N; i++){
                Svec[i] = -3e38f;
            }
        }

        // compute lse
        constexpr float log2e = 1.4426950408889634f;
        constexpr float loge2 = 0.6931471805599453f;
        vector<float, KV_PARTITION_STEP_NUM * REG_N> rS_exp = cm_exp(rS_slice.format<float>()*log2e);

        // compute row_max
        auto rSv = rS_slice.format<float>();
        float row_max = rSv[0];
        // It is performance hotspot for u8,  must add unroll
        #if KV_CACHE_COMPRESSION
        #pragma unroll
        #endif
        for(int r = 1; r < rSv.n_elems(); r++)
            row_max = cm_max<float>(row_max, rSv[r]);

        // compute Pmat = exp(rS_slice - row_max)
        #if XE_ARCH==1
        Pmat[qi].format<half, KV_PARTITION_STEP_NUM / 2 * REG_M, REG_K>() = cm_exp((rS_slice.format<float, KV_PARTITION_STEP_NUM / 2 * REG_M, REG_K>() - row_max)*log2e);
        vector<float, KV_PARTITION_STEP_NUM * REG_N> rS_exp_temp = cm_exp((rS_slice.format<float>() - row_max) * log2e);
        #else
        vector<float, KV_PARTITION_STEP_NUM * REG_N> rS_exp_temp = cm_exp((rS_slice.format<float>() - row_max) * log2e);
        Pmat[qi].format<half, KV_PARTITION_STEP_NUM * REG_M, REG_N>() = rS_exp_temp;
        #endif

        cur_lse[qi] = cm_sum<float>(rS_exp_temp.format<float>());
        cur_lse[qi] = cm_log<float>(cur_lse[qi]) * loge2 + row_max; // log2(sum(exp(x))) = log2e * log(sum(exp(x)))

        // compute row sum of P
        auto rPv = Pmat[qi].format<half, 1, KV_PARTITION_STEP_NUM * REG_N>();
        cur_sum[qi] = cm_sum<float>(rPv[0]);
    }

    //if(kv_partition_idx==0 && kv_head_num_idx == 0) 
    //{
    //    printf("kv_partition_idx = %d, Pmat:\n", kv_partition_idx);
    //    show(Pmat);
    //}

    //# rO = P * V
    #if Q_RepeatCount != 1
    matrix <float, Q_RepeatCount, HEAD_SIZE/REG_N * REG_M*REG_N> Omat = 0;
    #else
    matrix <float, Q_SLICE_NUM, HEAD_SIZE/REG_N * REG_M*REG_N> Omat = 0;
    #endif
    {
        uint ki = 0;
        uint blk_indices = block_indices[start_block_idx];
        uint kv_base_offset = blk_indices * per_kv_block_element_num + kv_head_num_idx * (per_kv_block_element_num / KV_HEADS_NUM);
        uint kv_scale_zp_offset = kv_base_offset + KV_BLOCK_SIZE * HEAD_SIZE; // scale/zp offset
        if(kv_partition_idx % 2 == 1) // second half block
            kv_base_offset += KV_PARTITION_SIZE * HEAD_SIZE; // move to the 2nd half partition
            kv_scale_zp_offset += KV_PARTITION_SIZE * sizeof(half); // move to the 2nd half partition

    #if USE_LSC_BLOCK_2D_DESC
        //# vector load cannot be used for block_2d_desc
        //# note: candidate template ignored: deduced type 'details::Block2DRefTy<half, 1U, 16U, 1U, (LoadOp)0U>' (aka 'vector_ref<half,32>') of 1st parameter
        #if KV_CACHE_COMPRESSION
        lsc::block_2d_desc<uint8_t, 1, REG_K, REG_N> b2dV(value + kv_base_offset,  KV_PARTITION_SIZE - 1, HEAD_SIZE*sizeof(uint8_t) - 1, kv_pitch - 1, 0, 0);
        #else
        lsc::block_2d_desc<half, 1, REG_K, REG_N>   b2dV(value + kv_base_offset,  KV_PARTITION_SIZE - 1, HEAD_SIZE*sizeof(half) - 1, kv_pitch - 1, 0, 0);
        #endif
    #else
        uint kv_offset = kv_base_offset;
        uint kv_stride = HEAD_SIZE;
        uint kv_x0 = 0, kv_y0 = 0;
        uint kv_x1 = HEAD_SIZE*sizeof(half);
        uint kv_y1 = KV_BLOCK_SIZE;
    #endif
    
        //if(kv_partition_idx==kv_partition_num - 1 && head_num_idx == HEADS_NUM - 1) {
        //    printf("leftover_size = %d, leftover_aligned_size = %d, XE_ARCH = %d, KV_BLOCK_SIZE = %d\n", leftover_size, leftover_aligned_size, XE_ARCH, KV_BLOCK_SIZE);
        //} 
        uint kv_pos_end = KV_PARTITION_SIZE;
        if(leftover_size > 0) {
            kv_pos_end = leftover_size % KV_PARTITION_SIZE;
            if(kv_pos_end == 0) kv_pos_end = KV_PARTITION_SIZE;
        }

        #if KV_CACHE_COMPRESSION
            // load scale/zp
            vector<half, KV_PARTITION_SIZE> scale_vec;
            vector<half, KV_PARTITION_SIZE> zp_vec;
            cm_svm_block_read(reinterpret_cast<svmptr_t>(value + kv_scale_zp_offset), scale_vec);
            cm_svm_block_read(reinterpret_cast<svmptr_t>(value + kv_scale_zp_offset + KV_BLOCK_SIZE * sizeof(half)), zp_vec);
            if(kv_pos_end < KV_PARTITION_SIZE) {
                // fill leftover with last valid scale/zp
                for(int i = kv_pos_end; i < KV_PARTITION_SIZE; i++) {
                    scale_vec[i] = 0.0;
                    zp_vec[i] = 0.0;
                }
            }
        #endif
        #pragma unroll
        for(int kv_pos = 0; kv_pos < kv_pos_end; kv_pos += REG_K, ki++) {
            // uint kv_offset_y = kv_pos;
            #if KV_CACHE_COMPRESSION
            vector<half, REG_K> temp_scale = scale_vec.select<REG_K,1>(kv_pos);
            vector<half, REG_K> temp_zp = zp_vec.select<REG_K,1>(kv_pos);
            #endif
            #pragma unroll
            for(int k = 0, ri = 0; k < HEAD_SIZE; k += REG_N, ri ++ ) {
                // Load V into register & pack as VNNI(as dpas-B tile)
                matrix<half, REG_K, REG_N> VmatNormal;
                matrix<half, REG_M, REG_K*REG_N> Vmat;
            #if USE_LSC_BLOCK_2D_DESC
                b2dV.set_block_x(k);
                #if KV_CACHE_COMPRESSION
                    // dequantize
                    matrix<uint8_t, REG_K, REG_N> Vt_quant;
                    cm_load<lsc::Normal>(Vt_quant.format<uint8_t>(), b2dV.set_block_y(kv_pos));

                    #if 0
                    printf("Vt_quant: k = %d\n", k);
                    show_u8(Vt_quant.format<uint8_t, REG_K, REG_N>());
                    show(temp_scale);
                    show(temp_zp);
                    printf("\n");
                    #endif

                    #pragma unroll
                    for(int r = 0; r < REG_K; r++) {
                        VmatNormal[r] = Vt_quant[r] - temp_zp[r]; // vector - scalar
                        VmatNormal[r] = cm_mul<half>(VmatNormal[r], temp_scale[r]); // vector * scalar
                    }
                    // show(VmatNormal.format<half, REG_K, REG_N>());

                    if(kv_pos_end - kv_pos < KV_STEP) {
                        #pragma unroll
                        for(int r = kv_pos_end; r < KV_STEP; r++)  {
                            VmatNormal[r] = 0;
                        }
                    }
                    prepackAsVNNIWidth2(VmatNormal, Vmat.format<half, REG_K/2, REG_N*2>());
                #else
                    cm_load<lsc::VNNI>(Vmat[0].format<half>(), b2dV.set_block_y(kv_pos));
                    // somtimes KV cache would be filled with random Nan, so need to clean up the unused value data.
                    #if CLEAN_UNUSED_KVCACHE
                    if(kv_pos_end - kv_pos < KV_STEP) {
                        auto VmatRef = Vmat[0].format<half, REG_K/2, REG_N*2>();
                        uint valid_rows = kv_pos_end - kv_pos;
                        uint valid_rows_vnni = (valid_rows+1)/2;
                        //printf("V: kv_pos = %d, kv_pos_end = %d, leftover = %d\n", kv_pos, kv_pos_end, KV_STEP - valid_rows);
                        //show(VmatRef.format<half, REG_K/2, REG_N*2>());
                        for (int r = valid_rows_vnni; r < KV_STEP / 2; r++)
                            VmatRef.row(r) = 0.f;
                        if (valid_rows % 2 == 1)
                            VmatRef.row(valid_rows_vnni-1).select<REG_N,2>(1) = 0.f;
                        //printf("Kt: reset unused...\n");
                        //show(VmatRef.format<half, REG_K/2, REG_N*2>());
                        //printf("\n\n");
                    }
                    #endif
                #endif
                if(0) {
                    printf("Vmat: k = %d\n", k);
                    show(Vmat.format<half, REG_K, REG_N>());
                }
            #else
                matrix<half, REG_K, REG_N> temp;
                uint cur_kv_offset = kv_offset + kv_pos * kv_stride + k;
                #pragma unroll
                for(int kk = 0; kk < REG_K; kk++) {
                    cm_svm_block_read<half, REG_N>((svmptr_t)(value + cur_kv_offset + kk * kv_stride), temp[kk].format<half>());
                }
                auto Vref = Vmat[0].format<half, REG_K/2, 2*REG_N>();
                Vref.select<REG_K/2, 1, REG_N, 2>(0, 0) = temp.select<REG_K/2, 2, REG_N, 1>(0, 0);
                Vref.select<REG_K/2, 1, REG_N, 2>(0, 1) = temp.select<REG_K/2, 2, REG_N, 1>(1, 0);
            #endif
                #if Q_RepeatCount != 1
                matrix<half, Q_RepeatCount, REG_K> Pmat_data = Pmat.select<Q_RepeatCount,1,REG_K,1>(0, ki*REG_K);
                matrix<float, Q_RepeatCount, REG_N> Omat_data = 0;
                Omat_data = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, Q_RepeatCount>(
                            Omat_data.format<float>(),
                            Vmat[0].format<int32_t>(),
                            Pmat_data.format<int32_t>());
                Omat.select<Q_RepeatCount,1,REG_N,1>(0, ri*REG_N) += Omat_data;
                #else
                for(int qi = 0; qi < Q_SLICE_NUM; qi ++) {
                    auto Pmat_slice = Pmat[qi].format<half, KV_PARTITION_STEP_NUM * REG_M, REG_K>();
                    auto Omat_slice = Omat[qi].format<float, HEAD_SIZE/REG_N * REG_M, REG_N>();
                    Omat_slice[ri] = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, SystolicDepth, RepeatCount>(
                            Omat_slice[ri],
                            Vmat[0].format<int32_t>(),
                            Pmat_slice[ki].format<int32_t>());
                }
                #endif
                //if(kv_partition_idx==kv_partition_num - 1 && head_num_idx == 27) {
                //    printf("Omat[%d][%d]:\n",kv_pos, k);
                //    show(Omat);
                //}
            }
        }
    }

    //if(kv_partition_idx==5 && kv_head_num_idx == 5) {
    //    printf("Omat:\n");
    //    show(Omat);
    //}

    //# save Output
    #pragma unroll
    for (int qi = 0; qi < Q_SLICE_NUM; qi++) {
        matrix<float, REG_M, REG_N> cur_O_f32;
        uint o_offset = seq_idx * kv_partition_num * HEADS_NUM * HEAD_SIZE + kv_partition_num * (head_num_idx + qi) * HEAD_SIZE + wg_thread_id * HEAD_SIZE;
        float div_cur_sum = 1.0/cur_sum[qi];
        auto Omat_slice = Omat[qi].format<float, HEAD_SIZE/REG_N * REG_M, REG_N>();
        #pragma unroll
        for(int k = 0, ri=0; k < HEAD_SIZE; k += REG_N, ri++) {
            auto cO = Omat_slice[ri].format<float, REG_M, REG_N>();
            #if XE_ARCH==1
            cur_O_f32= cm_mul<float>(cO, div_cur_sum);
            #else
            cur_O_f32= cm_div_ieee(cO, cur_sum[qi]);
            #endif
            cm_svm_block_write<float, REG_N>((svmptr_t)(output + o_offset + k),cur_O_f32.format<float>());
        }
        uint lse_offset = seq_idx * HEADS_NUM * kv_partition_num + (head_num_idx + qi) * kv_partition_num + wg_thread_id;
        lse[lse_offset] = cur_lse[qi];
    }
}

extern "C" _GENX_MAIN_ void cm_sdpa_2nd_reduce(
    float* input [[type("svmptr_t")]], //
    half* output [[type("svmptr_t")]],
    float* lse [[type("svmptr_t")]],
    int kv_partition_num
    ) {
        auto batch = cm_global_id(0);
        auto head = cm_global_id(1);
        auto offset = cm_group_id(2) * REDUCE_SPLIT_SIZE;
        const int total_partition_num = (kv_partition_num * HEADS_NUM);

        // load lse
        float total_lse = 0.0;
        uint lse_offset = batch * total_partition_num + head * kv_partition_num;
        constexpr float log2e = 1.4426950408889634f;
        float* lse_vec = lse + lse_offset;
        float lse_max = lse_vec[0];
        for(int k = 1; k < kv_partition_num; k ++) {
            lse_max = cm_max<float>(lse_vec[k], lse_max);
        }

        int iter = kv_partition_num / 16;
        for(int k = 0; k < iter * 16; k += 16) {
            #pragma unroll
            for(int ki = k; ki < k + 16; ki ++) {
                float lse_value = cm_exp<float>((lse_vec[ki] - lse_max)*log2e);
                total_lse += lse_value;
            }
        }
        for(int k = iter * 16; k < kv_partition_num; k ++) {
            float lse_value = cm_exp<float>((lse_vec[k] - lse_max)*log2e);
            total_lse += lse_value;
        }

        // load input, total_partition_num = head_nums * kv_partition_num;
        matrix<half, 1, REDUCE_SPLIT_SIZE> out_mat = 0;
        matrix<float, 1, REDUCE_SPLIT_SIZE> out_mat_f32 = 0;
        matrix<float, 1, REDUCE_SPLIT_SIZE> data_mat;
        uint input_offset = batch * total_partition_num * HEAD_SIZE + head * kv_partition_num * HEAD_SIZE + offset;

        for(int k = 0; k < iter * 16; k += 16) {
            #pragma unroll
            for(int ki = k; ki < k + 16; ki ++) {
                cm_svm_block_read<float, REDUCE_SPLIT_SIZE>((svmptr_t)(input + input_offset), data_mat.format<float>());
                input_offset += HEAD_SIZE;
                float lse_value = cm_exp<float>((lse_vec[ki] - lse_max)*log2e);
                out_mat_f32 += cm_mul<float>(data_mat, (float)(lse_value/total_lse));
            }
        }
        for(int k = iter * 16; k < kv_partition_num; k ++) {
            cm_svm_block_read<float, REDUCE_SPLIT_SIZE>((svmptr_t)(input + input_offset), data_mat.format<float>());
            input_offset += HEAD_SIZE;
            float lse_value = cm_exp<float>((lse_vec[k] - lse_max)*log2e);
            out_mat_f32 += cm_mul<float>(data_mat, (float)(lse_value/total_lse));
        }

        //if(cm_global_id(1) == 28 && cm_global_id(2) == 0) {
        //    printf("out_mat_f32:\n");
        //    show(out_mat_f32);
        //}

        out_mat = cm_mul<half>(out_mat_f32, (float)1.0f);
        // write output
        uint output_offset = batch * HEADS_NUM * HEAD_SIZE + head * HEAD_SIZE + offset;
        cm_svm_block_write<half, REDUCE_SPLIT_SIZE>((svmptr_t)(output + output_offset), out_mat.format<half>());
    }
'''

cl.profiling(True)

t_q = cl.tensor(q.detach().numpy())
t_k = cl.tensor(k.contiguous().detach().numpy())
t_v = cl.tensor(v.contiguous().detach().numpy())
t_past_lens = cl.tensor(past_lens.detach().numpy())
t_block_indices = cl.tensor(block_indices.detach().numpy())
t_block_indices_begins = cl.tensor(block_indices_begins.detach().numpy())
t_subsequence_begins = cl.tensor(subsequence_begins.detach().numpy())
t_gws_subseq_mapping = cl.tensor(gws_subseq_mapping.detach().numpy())
t_out = cl.tensor([batch, num_heads, kv_partition_num, head_size], np.dtype(np.float32))
t_out_final = cl.tensor([batch, 1, num_heads, head_size], np.dtype(np.float16))
t_lse = cl.tensor([batch, num_heads, kv_partition_num], np.dtype(np.float32))


intermedia_mem_size=batch*num_heads*kv_partition_num*(head_size+1)*4
print("intermediate memory size = ", intermedia_mem_size/1024/1024, "MB")

if enable_kvcache_compression and False:
    print("k = ", k.shape, k.dtype)
    key_u8_data=k.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size * (head_size + 4))[:,:,:kv_block_size*head_size]
    print(key_u8_data.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size, head_size))

    print("v = ", v.shape, v.dtype)
    value_u8_data=v.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size * (head_size + 4))[:,:,:kv_block_size*head_size]
    print(value_u8_data.reshape(new_kv_len//kv_block_size, num_kv_heads, kv_block_size, head_size))


# f"-cmc -mdump_asm -g2 "
cwd = os.path.dirname(os.path.realpath(__file__))
print("compiling ...")
cm_kernels = cl.kernels(pyeval(src1), f"-cmc -Qxcm_register_file_size=256 -mCM_printregusage -I{cwd}")

#cm_kernels = cl.kernels(pyeval(src1), f"-cmc -mCM_printregusage -I{cwd}")
print("first call ...")
# cm_kernels.enqueue("cm_sdpa_2nd", GWS, LWS, q_len, kv_len, t_q, t_k, t_v, t_out, t_lse)

if kv_partition_size * 2 == kv_block_size:
    cm_kernels.enqueue("cm_sdpa_2nd_half_block", GWS, LWS, t_q, t_k, t_v,
                       t_past_lens, t_block_indices, t_block_indices_begins,
                       t_subsequence_begins,t_out, t_lse, t_gws_subseq_mapping, q_len)
else:
    cm_kernels.enqueue("cm_sdpa_2nd", GWS, LWS, t_q, t_k, t_v,
                       t_past_lens, t_block_indices, t_block_indices_begins,
                       t_subsequence_begins,t_out, t_lse, t_gws_subseq_mapping, q_len)

# np.save("bmg_out", t_out.numpy())

# f0 = torch.from_numpy(t_out.numpy())
# print("f0 = ", f0.shape, f0.dtype)
# for i in range(num_heads):
#     for j in range(kv_partition_num):
#         for k in range(0, head_size, kv_step):
#             stop = min(k + kv_step, head_size)
#             print(f"f0[{i}, {j}, {k}:{stop}] = ", f0[0, i, j, k:stop])

# lse0 = torch.from_numpy(t_lse.numpy())
# print("lse0 = ", lse0.shape, lse0.dtype, lse0)
# print("lse_sum = ", lse0.sum(2))

GWS_2 = [batch, num_heads, head_size//reduce_split_step]
LWS_2 = [1, 1, 1]
print("GWS_2=", GWS_2)
print("LWS_2=", LWS_2)
cm_kernels.enqueue("cm_sdpa_2nd_reduce", GWS_2, LWS_2, t_out, t_out_final, t_lse, kv_partition_num)
f1 = torch.from_numpy(t_out_final.numpy())

# lse_final = torch.from_numpy(t_lse.numpy())
# print("lse_final = ", lse_final.shape, lse_final.dtype)
# print("lse_final sum = ", lse_final)


print("ref = ", org.transpose(1,2)[0,0,-1,:])
print("res = ", f1[0,0,-1,:])
check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
#sys.exit(0)

loop_cnt = 100
all_layers = []
mem_size = 0
while len(all_layers) < loop_cnt and mem_size < 8e9:
    all_layers.append([
        cl.tensor(q.detach().numpy()),
        cl.tensor(k.detach().numpy()),
        cl.tensor(v.detach().numpy()),
        cl.tensor([batch, num_heads, kv_partition_num, head_size], np.dtype(np.float32)),
        cl.tensor([batch, 1, num_heads, head_size], np.dtype(np.float16)),
    ])
    mem_size += q.numel() * q.element_size()
    mem_size += k.numel() * k.element_size()
    mem_size += v.numel() * v.element_size()
    # print(f"nlayers={len(all_layers)} mem_size={mem_size*1e-9:.3f} GB")

for i in range(loop_cnt):
    j  = i % len(all_layers)
    if kv_partition_size * 2 == kv_block_size:
        cm_kernels.enqueue("cm_sdpa_2nd_half_block", GWS, LWS,
                           all_layers[j][0],
                           all_layers[j][1],
                           all_layers[j][2],
                           t_past_lens, t_block_indices, t_block_indices_begins,t_subsequence_begins,
                           all_layers[j][3],
                           t_lse, t_gws_subseq_mapping, q_len)
    else:
        cm_kernels.enqueue("cm_sdpa_2nd", GWS, LWS,
                           all_layers[j][0],
                           all_layers[j][1],
                           all_layers[j][2],
                           t_past_lens, t_block_indices, t_block_indices_begins,t_subsequence_begins,
                           all_layers[j][3],
                           t_lse, t_gws_subseq_mapping, q_len)

    cm_kernels.enqueue("cm_sdpa_2nd_reduce", GWS_2, LWS_2, all_layers[j][3],all_layers[j][4], t_lse, kv_partition_num)

latency = cl.finish()
first_kernel = 1
if enable_kvcache_compression:
    kvcache_size = new_kv_len * num_kv_heads * head_size * 1 * 2
else:
    kvcache_size = new_kv_len * num_kv_heads * head_size * 2 * 2
intermedia_size = batch * num_heads * kv_partition_num * (head_size + 1) * 4

kvcache_total_time=0
intermedia_total_time=0
num_runs = 0
for ns in latency:
    if first_kernel:
        kvcache_total_time += ns
        num_runs += 1
        print(f"  {ns*1e-6:.3f} ms,  Bandwidth = {kvcache_size/(ns):.3f} GB/s, kvcache_size = {kvcache_size*1e-6:.1f} MB")
        #print(f"  {ns*1e-6:.3f} ms,  Bandwidth = {kvcache_size/(ns):.3f} GB/s")
    else:
        intermedia_total_time += ns
        #print(f"  {ns*1e-6:.3f} ms,  Bandwidth = {intermedia_size/(ns):.3f} GB/s, intermedia = {intermedia_size*1e-6:.1f} MB")
    first_kernel =  1 - first_kernel


print()
print("intermedia_total_time = ", intermedia_total_time*1e-6, "ms")
print("kvcache_total_time = ", kvcache_total_time*1e-6, "ms")

print(f"num_runs = {num_runs}, avg kvcache = {kvcache_size*num_runs/kvcache_total_time:.3f} GB/s, avg intermedia = {intermedia_size*num_runs/(intermedia_total_time):.3f} GB/s")
print()

f1 = torch.from_numpy(all_layers[0][4].numpy())

check_close(org.transpose(1,2), f1, atol=1e-2, rtol=1e-3)
print(f"=========== cm_sdpa_2nd PASS GWS={GWS} LWS={LWS} GWS_2={GWS_2} LWS_2={LWS_2} ===========")

sys.exit(0)
