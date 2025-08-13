import torch
from einops import rearrange
from clops.utils import *

VERBOSE = -1
enable_vprint = False
def vprint(*all_args):
    global enable_vprint
    if enable_vprint:
        print(*all_args)

# blocking on kv-len dimension with online-softmax
def get_flash0(query, key, value, attention_mask, q_step = 16, kv_step = 16):
    # BLHS=>BHLS
    query, key, value = (rearrange(x, "b s h d -> b h s d")
                for x in [query, key, value])

    global enable_vprint
    B,H,q_len,hs = query.shape
    _,Hkv,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            hm = 0 if attention_mask.shape[1] == 1 else h
            hkv = h // (H//Hkv)
            Q = query[b, h, :, :]
            K = key[b, hkv, :, :]
            V = value[b, hkv, :, :]
            mask = attention_mask[b,hm,:,:]
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

                    if (VERBOSE >= 0):
                        print(f"======== i={i} j={j} ==========")

                    if (j == VERBOSE): enable_vprint = True
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
                    vprint("cur_O2=", cur_O)

                    cur_max = rowmax
                    # if (j == VERBOSE): assert 0

                cur_O_f16 = (cur_O/cur_sum.transpose(0, 1)).to(torch.float16)

                if (i == VERBOSE):
                    enable_vprint = True
                    vprint("cur_O_f16=", cur_O_f16.shape, cur_O_f16)
                    assert 0

                out[b, h, i:i1, :] = cur_O_f16
                
    out = rearrange(out, "b h s d -> b s h d ")
    return out

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=1024)

    # sparse to dense mask 
    def block_mask_to_attention_mask(block_mask: torch.Tensor, q_len: int, k_len: int, sparse_block_size: int) -> torch.Tensor:
        # block_mask shape [num_head, q_block_num, k_block_num] dtype bool ->
        # attention_mask shape [num_head, q_len, kv_len] dtype bool
        q_block_num = (q_len + sparse_block_size -1) // sparse_block_size
        k_block_num = (k_len + sparse_block_size -1) // sparse_block_size
        assert(block_mask.shape[-2] == q_block_num)
        assert(block_mask.shape[-1] == k_block_num)

        expanded = block_mask.repeat_interleave(sparse_block_size, dim=1)
        expanded = expanded.repeat_interleave(sparse_block_size, dim=2)
        return expanded[:, :q_len, :k_len]

    def flash0_ref(q, k, v, is_causal, approx_simple_mask, sparse_block_size):
        q_len, num_heads, head_size = q.shape
        kv_len, num_kv_heads, head_size = k.shape
        old_dtype = q.dtype
        
        # [num_head, q_len, kv_len] dtype bool ->
        # [1, num_head, q_len, kv_len] dtype float16
        if sparse_block_size > 1:
            attention_mask = block_mask_to_attention_mask(approx_simple_mask, q_len, kv_len, sparse_block_size)
        else:
            attention_mask = torch.full([num_heads, q_len, kv_len], True).to(dtype=torch.bool)
        if is_causal:
            causal_pattern = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device))
            causal_pattern = causal_pattern.unsqueeze(0).repeat_interleave(num_heads, dim=0)
            attention_mask = attention_mask & causal_pattern
        attention_mask = torch.where(attention_mask, 0, torch.finfo(q.dtype).min)
        attention_mask = attention_mask.unsqueeze(0)
        print(f"============flash0 {attention_mask.shape=} {attention_mask.is_contiguous()=}")
        print(f"============flash0 {attention_mask=}")

        attn_output = get_flash0(
                q.unsqueeze(0).to(torch.float16), k.unsqueeze(0).to(torch.float16), v.unsqueeze(0).to(torch.float16),
                attention_mask = attention_mask)
        attn_output = attn_output.squeeze(0)
        # print(f"============2 {attn_output.shape=} ")
        print(".")
        return attn_output.to(old_dtype)

    low = -1
    high = 2
    act_dtype = torch.float16
    
    seq_len, block_sz, blocks_per_trunk = 128*2, 256, 1
    num_heads, num_kv_heads, head_size = 1, 1, 16
    sparse_block_sz = 128

    q = torch.randint(low, high, [seq_len, num_heads, head_size]).to(dtype=act_dtype)
    k = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)
    v = torch.randint(low, high, [seq_len, num_kv_heads, head_size]).to(dtype=act_dtype)/high
    
    q_block_num = (seq_len + sparse_block_sz -1) // sparse_block_sz
    k_block_num = (seq_len + sparse_block_sz -1) // sparse_block_sz
    block_mask = torch.tensor([True, False, False, True], dtype=torch.bool).reshape(q_block_num, k_block_num).expand(num_heads, q_block_num, k_block_num)
    out = flash0_ref(q, k, v, True, block_mask, sparse_block_sz)
    print(f"============flash0 {out.shape=} {out.is_contiguous()=}")
    print(f"============flash0 {out=}")