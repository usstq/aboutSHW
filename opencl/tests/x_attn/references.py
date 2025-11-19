import torch
import math
import torch.nn.functional as F

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

def get_partial_softmax_ref(A: torch.Tensor, block_size, S, wg_k=256, wg_q=128, valid_q=0):
    # [1, 32, 256, 8192]
    B, num_q_head, q_len_strided, k_len_strided = A.shape
    A_pad = torch.ones([B, num_q_head, rnd_up(q_len_strided, wg_q), rnd_up(k_len_strided, wg_k)], dtype=A.dtype) * -60000.0
    A_pad[:,:,:q_len_strided,:k_len_strided] = A
    q_len_strided, k_len_strided = rnd_up(q_len_strided, wg_q), rnd_up(k_len_strided, wg_k)

    # [1, 32, 256, 1]
    A_max = torch.max(A_pad, dim=-1, keepdim=True)[0]                           # real max needed for compensation [b, num_q_head, q_len_strided, 1]
    # [1, 32, 256, 64, 128]
    A_5d = A_pad.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_k, wg_k)
    # [1, 32, 256, 64, 1]
    A_5d_max = torch.max(A_5d, dim=-1, keepdim=True)[0]                         # local wg max needed for compensation [b, num_q_head, q_len_strided, k_len_strided // wg_n, 1]
    # [1, 32, 64, 256]
    A_5d_max_ret = torch.transpose(A_5d_max, -2, -3).squeeze(dim=-1)
    # [1, 32, 256, 64, 128]
    A_exp_partial = torch.exp(A_5d - A_5d_max) * (A_5d_max > -60000.0)
    # [1, 32, 256, 64, 16, 8]
    A_exp_partial = A_exp_partial.reshape(B, num_q_head, q_len_strided, k_len_strided // wg_k, wg_k // (block_size // S), block_size // S)
    # [1, 32, 256, 64, 16]
    A_exp_partial_sum = A_exp_partial.sum(dim=-1)                               # local wg sum
    # [1, 32, 256, 64 * 16]
    A_exp_partial_sum_ret = torch.reshape(A_exp_partial_sum, (B, num_q_head, q_len_strided, -1))

    # set invalid q to 0
    if valid_q:
        A_max[:,:,valid_q:,:] = 0
        A_5d_max_ret[:,:,:,valid_q:] = 0
        A_exp_partial_sum_ret[:,:,valid_q:,:] = 0

    # stage 2:
    # [1, 32, 256, 64, 16] * [1, 32, 256, 64, 1]
    A_exp_horz_sum = A_exp_partial_sum * torch.exp(A_5d_max - A_max.unsqueeze(dim=-1))
    # [1, 32, 256, 64 * 16]
    A_exp_horz_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided, -1)
    A_exp_horz_sum = A_exp_horz_sum / (A_exp_horz_sum.sum(dim=-1, keepdim=True)+1e-5)
    A_exp_vert_sum = A_exp_horz_sum.reshape(B, num_q_head, q_len_strided // (block_size // S), block_size // S, -1)
    A_exp_vert_sum = A_exp_vert_sum.sum(dim=-2)
    A_sum = F.softmax(A_pad, dim=-1) * (A_pad > -60000.0)
    if valid_q:
        A_sum[:,:,valid_q:,:] = 0
    A_sum = A_sum.reshape(B, num_q_head, q_len_strided // (block_size // S), (block_size // S), k_len_strided // (block_size // S), (block_size // S))

    A_sum = A_sum.sum(dim=-1).sum(dim=-2)
    # [1, 32, 32, 64 * 16]
    if not torch.allclose(A_sum, A_exp_vert_sum, atol=0.01, rtol=0.01):
        print(f'{A_sum=}\n{A_exp_vert_sum=}')
        assert 0
    ###################### fuse gemm+softmax end
    return A_max.squeeze(-1).contiguous(), A_5d_max_ret.contiguous(), A_exp_partial_sum_ret.contiguous(), A_sum.contiguous()

def get_gemm_ref(Q_org: torch.Tensor, K: torch.Tensor, q_start_strided, block_size, S, threshold=0.9, causal=True, wg_k=256, wg_q=128):
    # Q, K, V shape = (B, H, L, d)
    B, num_kv_head, k_len, d = K.shape
    # drop tails which are less than stride
    Q = Q_org[:,:,:Q_org.shape[2]//S*S,:]

    B, num_q_head, q_len, d = Q.shape
    assert num_q_head % num_kv_head == 0
    H = num_q_head
    K = K.repeat_interleave(num_q_head // num_kv_head, 1)
    
    Q_block = (q_len + block_size -1) // block_size
    K_block = (k_len + block_size -1) // block_size

    # Antidiagonal Reshaping (Stride-S)
    Q_resh = torch.cat([Q[:, :, S-1-i::S, :] for i in range(S)], dim=-1)
    K_resh = torch.cat([K[:, :, i::S, :] for i in range(S)], dim=-1)

    # Attention Estimation
    A = Q_resh @ K_resh.transpose(-1, -2)
    #print(f'{A.shape=} {Q_resh.shape=} {K_resh.shape=}')
    A = A / math.sqrt(d) / S
    if causal:
        causal_mask = torch.zeros(
            (
                B,
                num_q_head,
                q_len//S,
                k_len//S,
            ),
        )
        #causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
        chunk_start = q_start_strided
        chunk_end = chunk_start + q_len//S
        causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
            torch.ones(
                1,
                num_q_head,
                q_len//S,
                q_len//S,
            )
            * float("-60000.0"),
            diagonal=1,
        )
        A += causal_mask

    #A = F.softmax(A, dim=-1, dtype=torch.float32).to(Q.dtype)
    return get_partial_softmax_ref(A, block_size=block_size, S=S, wg_k=wg_k, wg_q=wg_q)

def find_blocks_ref(input_tensor_org, threshold, q_block, k_block, causal, current_index):
    input_tensor = input_tensor_org[...,:q_block,:k_block]
    B, HQ, Q_block, K_block = input_tensor.shape
    #input_tensor = input_tensor.to(float)
    input_tensor = input_tensor.to(torch.float32)
    total_sum = input_tensor.sum(dim=-1, keepdim=True)
    required_sum = total_sum * threshold
    input_tensor = input_tensor.to(torch.float16)
    #print(f'{threshold=}, {required_sum=}, {total_sum=}')
    if causal:
        chunk_num = Q_block
        block_num = K_block
        mask = torch.zeros_like(input_tensor, dtype=torch.bool)
        mask[:, :, :, current_index : current_index + chunk_num] = (
            torch.eye(chunk_num, device=mask.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, HQ, chunk_num, chunk_num)
        )
        mask[:, :, :, 0] = 1
        other_values = input_tensor.masked_fill(
            mask, 0
        )
        sorted_values, _ = torch.sort(
            other_values, dim=-1, descending=True, stable=True
        )
        sorted_values = sorted_values.to(input_tensor.device)

        sorted_values = torch.cat(
            [
                torch.zeros(
                    (B, HQ, chunk_num, 1), device=input_tensor.device
                ),
                torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                sorted_values[:, :, :, :-2],
            ],
            dim=-1,
        )

        _, org_index = torch.sort(
            torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
            dim=-1,
            descending=True, stable=True
        )
        cumulative_sum_without_self = torch.cat(
            [
                torch.zeros(
                    (B, HQ, chunk_num, 1), device=input_tensor.device
                ),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)

        index_mask = cumulative_sum_without_self < required_sum
        index = torch.where(index_mask,org_index,0)
        mask = mask.view(B,HQ*chunk_num,block_num)
        index = index.view(B,HQ*chunk_num,block_num)
        mask[:,torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),index] = True
        mask = mask.view(B,HQ,chunk_num,block_num)
        # copy from tail of xattn_estimate
        q_block_num = q_block
        mask[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool
                ),
                diagonal=0,
            ),
            mask[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    else:
        mask = torch.zeros_like(input_tensor, dtype=torch.int8)
        sorted_values, org_index = torch.sort(
            input_tensor, dim=-1, descending=True, stable=True
        )
        #print(f'{sorted_values=} {org_index=}')
        sorted_values = sorted_values.to(input_tensor.device)
        cumulative_sum_without_self = torch.cat(
            [
                torch.zeros(
                    (B, HQ, Q_block, 1), device=input_tensor.device
                ),
                sorted_values[:, :, :, 0:-1],
            ],
            dim=-1,
        ).cumsum(dim=-1)
        #print(f'{cumulative_sum_without_self=}')
        index_mask = cumulative_sum_without_self < required_sum
        #print(f'{index_mask=}')
        index = torch.where(index_mask, org_index, 0)
        mask = mask.view(B, HQ * Q_block, K_block)
        index = index.view(B, HQ * Q_block, K_block)
        mask[
            :,
            torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
            index,
        ] = 1
        mask = mask.view(B, HQ, Q_block, K_block)
    #print(f'ref_mask={mask}')
    return mask, sorted_values, org_index, cumulative_sum_without_self, required_sum
