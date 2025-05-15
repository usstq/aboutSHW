import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

batch = 1
num_heads = 1
q_len, q_step = 6, 2
kv_len, kv_step = 8, 4

#q_len, q_step = 160, 16
#kv_len, kv_step = 800, 16

head_size = 8
low = -7
high = 8
act_dtype = torch.float32
q = torch.randint(low, high, [batch, q_len, num_heads, head_size]).to(dtype=act_dtype)/high
k = torch.randint(low, high, [batch, kv_len, num_heads, head_size]).to(dtype=act_dtype)/high
v = torch.randint(low, high, [batch, kv_len, num_heads, head_size]).to(dtype=act_dtype)/high

# random attnmask
attention_mask = torch.full([batch, 1, q_len, kv_len], torch.finfo(act_dtype).min)
attention_mask[torch.rand(batch, 1, q_len, kv_len) > 0.5] = 0

# BLHS=>BHLS
q = q.transpose(1,2)
k = k.transpose(1,2)
v = v.transpose(1,2)

#q[:,:,:,:] = q[:,:,2,:]
#attention_mask[:,:,:,:] = attention_mask[:,:,2,:]

print("q", q.shape, q.dtype)
print("k", k.shape, k.dtype)
print("v", v.shape, v.dtype)
print("attention_mask", attention_mask.shape, attention_mask.dtype)


def get_org(Q, K, V, attention_mask):
    B,H,L,S = Q.shape
    out = torch.zeros([B,H,L,S], dtype=Q.dtype)
    scale_factor = S**(-0.5)
    for b in range(B):
        for h in range(H):
            attn_score = Q[b, h, :, :] @ (K[b, h, :,:].transpose(0,1))
            attn_score *= scale_factor
            attn_score += attention_mask[b,0,:,:]
            #print(attn_score.shape)
            attn_weights = F.softmax(attn_score, 1)
            out[b,h,:,:] = attn_weights @ V[b, h, :, :]
    return out


# blocking on kv-len dimension with online-softmax
def get_flash0(query, key, value, attention_mask):
    B,H,q_len,hs = query.shape
    _,_,kv_len,_ = key.shape
    out = torch.zeros([B,H,q_len,hs], dtype=value.dtype)
    scale_factor = hs**(-0.5)
    for b in range(B):
        for h in range(H):
            Q = query[b, h, :, :]
            K = key[b, h, :, :]
            V = value[b, h, :, :]
            mask = attention_mask[b,0,:,:]

            assert q_len % q_step == 0
            assert kv_len % kv_step == 0
            for i in range(0, q_len, q_step):
                i1 = min(i + q_step, q_len)
                # online softmax states:
                #     per-row max-value     : [q_step, 1]
                #     per-row sum           : [q_step, 1]
                #     current accumulated V : [q_step, S]
                cur_max = torch.full([i1-i, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
                cur_sum = torch.full([i1-i, 1], 0, dtype=torch.float32)
                cur_O = torch.full([i1-i, hs], 0, dtype=torch.float32)
                for j in range(0, kv_len, kv_step):
                    j1 = min(j + kv_step, kv_len)

                    # compute in local SRAM
                    # Step8: On chip, compute S(_𝑗)𝑖= Q𝑖K𝑇 𝑗∈ R𝐵𝑟 ×𝐵𝑐.
                    S = Q[i:i1, :] @ (K[j:j1,:].transpose(0,1))
                    S *= scale_factor
                    S += mask[i:i1, j:j1]

                    rowmax = S.max(-1, keepdim=True).values
                    new_max = torch.maximum(cur_max, rowmax)

                    # compute in local SRAM
                    P = torch.exp(S - new_max)
                    rowsumP = P.sum(-1, keepdim=True)

                    # corrected sum of previous block
                    prev_sum = cur_sum * torch.exp(cur_max - new_max)
                    new_sum = prev_sum + rowsumP

                    # softmax normalize is saved accoridng to flash-attn2 section 3.1.1
                    # We can instead maintain an “un-scaled” version of O(2) and keep around the statistics ℓ(2)
                    partial_attn_weight = P

                    if j > 0:
                        # correct last Output to current statistics
                        cur_O = (cur_O * torch.exp(cur_max - new_max))

                    cur_O += (partial_attn_weight @ (V[j:j1, :]).to(torch.float32))

                    cur_sum = new_sum
                    cur_max = new_max

                out[b, h, i:i1, :] = cur_O/new_sum
    return out



ref = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
res0 = get_org(q,k,v,attention_mask)
#print(ref)
#print(res0)

assert torch.allclose(res0, ref, atol=1e-3, rtol=1e-3)

# [batch, seq-len, heads, size] BLHS
print("ref:", ref.shape, ref.dtype)
#print(ref)

f0 = get_flash0(q,k,v,attention_mask)
#print(f0)
assert torch.allclose(f0, ref, atol=1e-3, rtol=1e-3)

