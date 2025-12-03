
import torch

def count_false_percentage(mask):
    B, H, NQ, NL = mask.shape
    tril_mask = torch.tril(torch.ones((NQ, NL), dtype=torch.bool, device=mask.device))
    expanded_tril = tril_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    # Count elements in the tril region
    tril_elements = torch.sum(expanded_tril).item()
    # Count False elements in the tril region
    false_in_tril = torch.sum(~mask & expanded_tril).item()
    # Calculate percentage
    if tril_elements > 0:
        false_percentage = (false_in_tril / tril_elements) * 100
    else:
        false_percentage = 0.0
    return false_percentage

def adjust_true_ratio_tril(
    tensor: torch.Tensor,
    ratio: float,
    diagonal: int = 0,
    ratio_base: str = "all",  # "all" or "tril"
    seed: int = None
) -> torch.Tensor:
    """
    Adjust the number of True elements in a boolean tensor so that, after adjustment,
    all True positions lie within a lower-triangular (tril) region over the last two dims.

    Args:
        tensor (torch.Tensor): Boolean tensor of shape [num_head, q_block_num, k_block_num].
        ratio (float): Desired ratio of True elements (0..1).
        diagonal (int): Diagonal offset for the tril region (like torch.tril's `diagonal`).
                        0 = main diagonal; positive keeps more above; negative keeps fewer.
        ratio_base (str): 'all' -> ratio is w.r.t. total elements of the whole tensor;
                          'tril' -> ratio is w.r.t. the number of elements inside the tril mask.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: A new boolean tensor where all True elements lie within the specified tril region,
                      and the count of True elements is as close as possible to the requested ratio.
    """
    # --- validation ---
    if tensor.dtype != torch.bool:
        raise ValueError("Tensor must have dtype=torch.bool")
    if tensor.dim() != 3:
        raise ValueError("Tensor must be 3D with shape [num_head, q_block_num, k_block_num]")
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("ratio must be in [0, 1]")
    if ratio_base not in ("all", "tril"):
        raise ValueError("ratio_base must be 'all' or 'tril'")

    if seed is not None:
        g = torch.Generator(device=tensor.device)
        g.manual_seed(seed)
    else:
        g = None

    num_head, q_block_num, k_block_num = tensor.shape

    # --- build tril mask over the last two dims and expand over heads ---
    tril_2d = torch.tril(
        torch.ones((q_block_num, k_block_num), dtype=torch.bool, device=tensor.device),
        diagonal=diagonal
    )  # shape [q_block_num, k_block_num]
    tril_mask = tril_2d.unsqueeze(0).expand(num_head, -1, -1).contiguous()  # [num_head, q_block_num, k_block_num]

    # Clone so we don't mutate the input
    out = tensor.clone()

    # --- enforce: any True outside the tril region must be cleared ---
    out &= tril_mask  # zeroes out True values outside tril

    # --- compute target True count ---
    tril_capacity = tril_mask.sum().item()  # number of valid positions per all heads combined
    total_elements = tensor.numel()

    if ratio_base == "all":
        target_true = int(round(total_elements * ratio))
    else:  # 'tril'
        target_true = int(round(tril_capacity * ratio))

    # You cannot exceed the tril capacity
    target_true = min(target_true, tril_capacity)

    current_true = out.sum().item()

    # Fast path: already at target
    if current_true == target_true:
        return out

    # Flatten to 1D for easy indexing
    out_flat = out.view(-1)
    tril_flat = tril_mask.view(-1)

    if current_true < target_true:
        # Need to add True within tril where currently False
        candidates = (~out_flat) & tril_flat
        idx = torch.nonzero(candidates, as_tuple=False).squeeze(1)
        num_to_flip = min(target_true - current_true, idx.numel())
        if num_to_flip > 0:
            if g is None:
                perm = torch.randperm(idx.numel(), device=tensor.device)
            else:
                perm = torch.randperm(idx.numel(), generator=g, device=tensor.device)
            chosen = idx[perm[:num_to_flip]]
            out_flat[chosen] = True

    return out_flat.view_as(out)

def generate_block_mask_with_ratio(num_heads, seq_len, trunk_sz, sparse_block_sz, true_ratio, is_causal=True, device='cpu'):
    assert(sparse_block_sz > 1)

    trunk_num = (seq_len + trunk_sz -1) // trunk_sz
    q_block_num = (trunk_sz + sparse_block_sz -1) // sparse_block_sz
    k_block_num = (seq_len + sparse_block_sz -1) // sparse_block_sz

    x = torch.zeros((num_heads, trunk_num*q_block_num, k_block_num), dtype=torch.bool, device=device)
    x[:,:,0] = True  # the first column is always True
    x[:,-1,:] = True  # the last row is always True
    # Set diagonal elements to True for each head
    for h in range(num_heads):
        diag_len = min(q_block_num, k_block_num)
        for i in range(diag_len):
            x[h, i, i] = True

    y = adjust_true_ratio_tril(x, true_ratio, diagonal=0, ratio_base="tril" if is_causal else "all", seed=42)
    print("Original True count:", x.sum().item(), " / ", x.numel())
    print("Adjusted True count:", y.sum().item(), " / ", y.numel())        
    print("All True within tril:", torch.all(y <= torch.tril(torch.ones_like(y))).item())
    
    y = y.reshape(num_heads, trunk_num, q_block_num, k_block_num).transpose(0, 1).contiguous()    
    density = y.sum().item()/y.numel()
    if is_causal:
        density *= 2
    print(f'density {density}')
    
    return y, density

# --- Example usage ---
if __name__ == "__main__":
    num_heads, seq_len, trunk_sz, sparse_block_sz = 2, 4096*2, 4096, 128
    y = generate_block_mask_with_ratio(num_heads, seq_len, trunk_sz, sparse_block_sz, 0.25)
    # print(f'{y=}')

