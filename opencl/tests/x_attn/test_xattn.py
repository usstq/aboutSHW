#from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F

import numpy as np
import os

###################################################################################################
## from xattn/src/utils.py
def create_causal_mask(batch_size, head_num, block_size, block_num, divide_block_num):
    """
        Creates a causal attention mask used in transformer-based models.

        Parameters:
        - batch_size (int): The number of sequences in the batch.
        - head_num (int): The number of attention heads.
        - block_size (int): The size of each block in the sequence.
        - block_num (int): The total number of blocks in the sequence.
        - divide_block_num (int): The block index at which causality is applied.

        Returns:
        - torch.Tensor: A mask tensor of shape (batch_size, head_num, block_size, total_size)
        where total_size = block_size * block_num. The mask enforces causal attention by 
        setting certain positions to `-inf` to prevent information leakage from future tokens.
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask

def find_blocks_chunked(
    input_tensor, current_index, threshold, num_to_choose, decoding: bool, mode: str = "both", causal=True
):
    """
        Finds and selects relevant blocks of attention for transformer-based models based on a 
        threshold or a predefined number of blocks.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
        - current_index (int): The current index in the sequence processing.
        - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
        - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.
        - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
        - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
        - causal (bool): If True, applies causal masking to prevent future information leakage.

        Returns:
        - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
        indicating which blocks should be attended to.
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    # 0 -- -- -- -- current_index
    # 0 -- -- -- -- -- current_index+1
    # 0 -- -- -- -- -- ----------- current_index + chunk_num - 1
    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(1, head_num, chunk_num, chunk_num, device=input_tensor.device)
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[:, :, 0 : current_index + 1],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[:, :, current_index + 1 :],
                ],
                dim=-1,
            )
        else:
            return mask
    input_tensor = input_tensor.to(torch.float32)
    
    if threshold is not None:
        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(float)
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            ).expand((batch_size, head_num, chunk_num, 1)).to(input_tensor.device)
        else:
            required_sum = total_sum * threshold
        if causal:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            mask[:, :, :, 0] = 1
            other_values = input_tensor.masked_fill(
                mask, 0
            )
            sorted_values, _ = torch.sort(
                other_values, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)

            sorted_values = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask,index,0)
            mask = mask.view(batch_size,head_num*chunk_num,block_num)
            index = index.view(batch_size,head_num*chunk_num,block_num)
            mask[:,torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),index] = True
            mask = mask.view(batch_size,head_num,chunk_num,block_num)
            # assert(bool((torch.where(mask,input_tensor,0).sum(dim=-1,keepdim=True) >= required_sum*0.99).all()))
        else:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(
                input_tensor, dim=-1, descending=True
            )
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            print(f'{index.shape=}')
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            print(f'{torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1)}')
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        raise NotImplementedError("block num chunk prefill not impleted")
    
    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except:
        mask[:, :, :, current_index + chunk_num :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(input_tensor,dtype=bool,device=input_tensor.device)
            lambda_mask[:,:,:,0] = 1
            lambda_mask[:,:,:,current_index:current_index+chunk_num] = torch.eye(chunk_num, device=lambda_mask.device).unsqueeze(0).unsqueeze(0).expand(1,head_num,chunk_num,chunk_num)
            assert(torch.where(lambda_mask,mask,True).all())

    return mask

#####################################################################################################################################################################3

device = 'cpu'

def check_close(input, other, atol=1e-3, rtol=1e-3):
    print(f"[check_close] {input.shape}{input.dtype} vs {other.shape}{other.dtype}")
    # rtol_max = (((input - other).abs() - 1e-5)/other.abs())[other != 0].max()
    # atol_max = (((input - other).abs()) - 1e-5*other.abs()).max()
    # print(f"[check_close] rtol_max: {rtol_max}")
    # print(f"[check_close] atol_max: {atol_max}")
    if not torch.allclose(input, other, atol=atol, rtol=rtol):
        close_check = torch.isclose(input, other, atol=atol, rtol=rtol)
        not_close_indices = torch.where(~close_check) # Invert the close check to find failures
        # print(f"Not close indices: {not_close_indices}")
        print(f"    input_tensor: {input[not_close_indices]}")
        print(f"    other_tensor: {other[not_close_indices]}")
        # assert 0

def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=False,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    # Q, K, V shape = (B, H, L, d)
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    # Chunk and Pad (L → multiple of chunk_size)
    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size
    assert k_chunk_num >= q_chunk_num
    offset_token_chunk_num = k_chunk_num - q_chunk_num

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to(device)
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
            device
        )
    else:
        pad_query_states = query_states
        
    print(f'{pad_query_states.shape=} {pad_key_states.shape=}')
    print(f'{q_num_to_pad=} {k_num_to_pad=}')

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            for k in range(stride):
                a = (pad_key_states[:, :, k::stride, :])
                print(f'{k=} {a.shape=}')
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
            print(f'{reshaped_query.shape=} {reshaped_key.shape=}')
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len


    # Q_reshaped, K_reshaped → (B, H, L//S, S*d)

    for chunk_idx in range(q_chunk_num):
        if not use_triton:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size)
                // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                // kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,                  # chunked_query → (B, H, chunk_size//S, S*d)
                reshaped_key.transpose(2, 3),
            ).to(device)                        # attn_weights_slice -> (B, H, chunk_size//S, L//S)

            attn_weights_slice = (
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                if k_reshaped_num_to_pad:
                    causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = (chunk_idx + offset_token_chunk_num) * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

            # attn_weights_slice -> (B, H, chunk_size//S, L//S)
            # view -> (B, H, chunk_size//block_size, reshaped_block_size, L//block_size, reshaped_block_size)
            attn_weights_slice_view = attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                )

            attn_sum = (
                attn_weights_slice_view
                .sum(dim=-1)
                .sum(dim=-2)
                .to(device)
            )
            print(f'{chunk_idx=}: {attn_weights_slice.shape=} {attn_weights_slice_view.shape=} {attn_sum.shape=}')
            del chunked_query
        
        # Find blocks based on threshold
        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        print(f'{chunk_idx=}: {simple_mask.shape=}')
        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    # if not use_triton:
    #     del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=key_states.device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    return attn_sums, simple_masks, reshaped_query, reshaped_key

def importance_estimate(
    Q: torch.Tensor,
    K: torch.Tensor,
    block_size,
    S,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=False,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    # Q, K, V shape = (B, H, L, d)
    B, num_kv_head, k_len, d = K.shape
    B, num_q_head, q_len, d = Q.shape
    assert num_q_head == num_kv_head
    H = num_q_head
    
    Q_block = (q_len + block_size -1) // block_size
    K_block = (k_len + block_size -1) // block_size

    # Antidiagonal Reshaping (Stride-S)
    Q_resh = torch.cat([
        Q[:, :, S-1-i::S, :] for i in range(S)
    ], dim=-1)
    K_resh = torch.cat([
        K[:, :, i::S, :] for i in range(S)
    ], dim=-1)

    # Attention Estimation
    A = Q_resh @ K_resh.transpose(-1, -2)
    print(f'{A.shape=} {Q_resh.shape=} {K_resh.shape=}')
    A = F.softmax(A / math.sqrt(d) / S / norm, dim=-1, dtype=torch.float32
                ).to(Q.dtype)
    print(f'{A.shape=}')

    # Reshape softmax to blocks
    A_blocked = A.view(B, H, Q_block, block_size//S, K_block, block_size//S)
    attn_sums = A_blocked.sum(dim=-1).sum(dim=-2) # (B, H, Q_block, K_block)
    
    # Thresholded Block Selection
    def find_blocks(input_tensor):
            input_tensor = input_tensor.to(float)
            total_sum = input_tensor.sum(dim=-1, keepdim=True)
            required_sum = total_sum * threshold
            print(f'{threshold=}, {required_sum=}, {total_sum=}')

            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(
                input_tensor, dim=-1, descending=True
            )
            print(f'{sorted_values=} {index=}')
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (B, H, Q_block, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(B, H * Q_block, K_block)
            index = index.view(B, H * Q_block, K_block)
            print(f'------------------------------ {mask.shape=} {index.shape=} {torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1)}')
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(B, H, Q_block, K_block)
            return mask
    if True:
        keep_mask = find_blocks(attn_sums)
    else:
        # Find blocks based on threshold
        keep_mask = find_blocks_chunked(
            attn_sums,
            0,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )
    return attn_sums, keep_mask, Q_resh, K_resh


def Xattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
):
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )
    attn_sums, approx_simple_mask, reshaped_querry, reshaped_key = xattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=False,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )
    
    # (B, H, Q_blocks, K_blocks) 
    print(f'{attn_sums.shape=}, {approx_simple_mask.shape=}')
    print(f'{attn_sums=}, {approx_simple_mask=}')
    
    attn_sums_ov, approx_simple_mask_ov, reshaped_querry_ov, reshaped_key_ov = importance_estimate(
        query_states,
        key_states,
        block_size=block_size,
        S=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=False,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )
    # print(f'{attn_sums_ov=}, {approx_simple_mask_ov=}')

    if not torch.equal(approx_simple_mask, approx_simple_mask_ov):
        elementwise_match = (approx_simple_mask_ov == approx_simple_mask)
        accuracy = elementwise_match.sum() / elementwise_match.numel()
        mismatches = torch.where(approx_simple_mask_ov != approx_simple_mask)[0]
        print(f"Accuracy: {accuracy.item():.1%}")
        mismatch_indices = torch.nonzero(~elementwise_match)
        print("\nMismatch positions (row, col):")
        print(mismatch_indices)
        
        check_close(attn_sums, attn_sums_ov, atol=1e-2, rtol=1e-3)
        check_close(reshaped_querry, reshaped_querry_ov, atol=1e-2, rtol=1e-3)
        check_close(reshaped_key, reshaped_key_ov, atol=1e-2, rtol=1e-3)
        print("FAIL!")
    else:
        print("PASS!")
        
    return approx_simple_mask, approx_simple_mask_ov

def main(load_from_ov_dump: bool = False):
    seq_len = 1024*8
    
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    kv_block_sz = 256
    sparse_block_sz = 128
    
    if load_from_ov_dump:
        TENSORS_DIR='/home/ceciliapeng/openvino/tensors_bin_pr/'
        with open(os.path.join(TENSORS_DIR, 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_25973_src0__f16__8192_4096_1_1__bfyx.bin'), 'rb') as f:
            buffer = f.read()
            q = torch.frombuffer(buffer, dtype=torch.float16)
        with open(os.path.join(TENSORS_DIR, 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_25973_updated_src_3__f16__33_8_256_128__bfyx.bin'), 'rb') as f:
            buffer = f.read()
            kcache = torch.frombuffer(buffer, dtype=torch.float16)
        with open(os.path.join(TENSORS_DIR, 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_25973_updated_src_4__f16__33_8_256_128__bfyx.bin'), 'rb') as f:
            buffer = f.read()
            vcache = torch.frombuffer(buffer, dtype=torch.float16)
            
        q = q.reshape(seq_len, num_heads, head_size)
        print(f'=========================={q.shape =}')
        
        # It looks GENAI generates one more block than required, and blocks are in order starting from 0. (at least for 1 chunck case...) 
        valid_blk_num = (seq_len + kv_block_sz - 1) // kv_block_sz
        blk_num = valid_blk_num + 1
        kcache = kcache.reshape(blk_num, num_kv_heads, kv_block_sz, head_size)
        vcache = vcache.reshape(blk_num, num_kv_heads, kv_block_sz, head_size)
        print(f'=========================={kcache.shape =},{blk_num = }')
        
        k = kcache[:valid_blk_num, :].transpose(1, 2).reshape(valid_blk_num * kv_block_sz, num_kv_heads, head_size)
        v = vcache[:valid_blk_num, :].transpose(1, 2).reshape(valid_blk_num * kv_block_sz, num_kv_heads, head_size)
        print(f'=========================={k.shape =}')
        
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        
        k = k.repeat_interleave(num_heads//num_kv_heads, dim=1)
        v = v.repeat_interleave(num_heads//num_kv_heads, dim=1)
        print(f'================================================ {q.shape =} {k.shape =} {v.shape =}')

    else:
        bsz = 1
        if True:
            q = torch.randn((bsz,num_heads,seq_len,head_size),dtype=torch.float32)
            k = torch.randn((bsz,num_heads,seq_len,head_size),dtype=torch.float32)
            v = torch.randn((bsz,num_heads,seq_len,head_size),dtype=torch.float32)

            np.save("q.npy", q.numpy())
            np.save("k.npy", k.numpy())
            np.save("v.npy", v.numpy())
        else:
            q = torch.from_numpy(np.load("q.npy"))
            k = torch.from_numpy(np.load("k.npy"))
            v = torch.from_numpy(np.load("v.npy"))

    q = q.to(torch.bfloat16).to(device)
    k = k.to(torch.bfloat16).to(device)
    v = v.to(torch.bfloat16).to(device)

    # q = torch.arange(0, bsz*heads*seq_len*dim,dtype=torch.bfloat16).view(bsz,heads,seq_len,dim).to(device)
    # k = torch.randn((bsz,heads,seq_len,dim),dtype=torch.bfloat16).to(device)

    approx_simple_mask_ref, approx_simple_mask_ov = Xattention_prefill(query_states=q,key_states=k,value_states=v,stride=16,block_size=128,use_triton=False,causal=True,chunk_size=4096)
    
    if load_from_ov_dump:
        # compare
        with open(os.path.join(TENSORS_DIR, 'program1_network1_0_pagedattentionextension_PagedAttentionExtension_25973_intermediates_4__boolean__131072_1_1_1__bfyx.bin'), 'rb') as f:
            buffer = f.read()
            approx_simple_mask = torch.frombuffer(buffer, dtype=torch.bool)
            approx_simple_mask = approx_simple_mask.reshape(1, num_heads, seq_len//sparse_block_sz, seq_len//sparse_block_sz)
            print(f'=========================={approx_simple_mask.shape =}')
        
        print(f'FINAL CHECK {torch.equal(approx_simple_mask_ref, approx_simple_mask)} , {torch.equal(approx_simple_mask_ov, approx_simple_mask)}')

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    # requirements:
    # num_q_head == num_kv_head
    # chunk size alignment
    # causal_mask
    main(load_from_ov_dump = False)