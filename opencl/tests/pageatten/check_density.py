import torch
import numpy as np
import os
from clops.utils import Colors

torch.set_printoptions(linewidth=1024, precision=2)
np.set_printoptions(precision=2, suppress=True)

def get_tensor(name, dtype=np.float16):
    with open(name, 'rb') as f:
        data = f.read()
        np_data = np.frombuffer(data, dtype=dtype).copy()
        return torch.from_numpy(np_data)

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

def check_cuda_density(base, dump_name):
    base = '/home/ceciliapeng/OCL/xattn-cuda/'
    mask = torch.from_numpy(np.load(os.path.join(base, dump_name)))
    # print(f'{mask.shape=}')
    density = 100.0 - count_false_percentage(mask)

    B, H, NQ, NL = mask.shape
    densities = np.array([100.0 - count_false_percentage(mask[:, h, :, :].reshape(B, 1, NQ, NL)) for h in range (H)])
    print(f'densities over heads {densities}')
    densities = np.array([100.0 - count_false_percentage(mask[b, :, :, :].reshape(1, H, NQ, NL)) for b in range (B)])
    print(f'densities over layers {densities}')

    return density
    
def check_ov_density(base, prompt_len = 32*1024, pa_trunk_size = 4*1024):
    xattn_block_size = 128
    num_layers = 36
    num_heads = 32
    num_q_blocks = prompt_len // xattn_block_size
    num_k_blocks = prompt_len // xattn_block_size
    num_iters = (prompt_len + pa_trunk_size - 1)//pa_trunk_size
    # print(f'{num_iters=}, {num_q_blocks=}, {num_k_blocks=}')
    block_mask = torch.zeros([num_layers, num_heads, num_q_blocks, num_k_blocks], device='cpu', dtype=torch.bool)
    for i in range (num_iters):
        for l in range(num_layers):
            filename = 'PA' + str(i * num_layers + l) + '.bin'
            layer_mask = get_tensor(os.path.join(base, filename), np.uint8)
            cur_q_blocks = pa_trunk_size // xattn_block_size
            cur_k_blocks = (i + 1) * pa_trunk_size // xattn_block_size 
            # print(f'iter {i}, layer {l} {filename} {cur_q_blocks=}, {cur_k_blocks=}')
            layer_mask = layer_mask.reshape([num_heads, cur_q_blocks, cur_k_blocks])
            block_mask[l, :, i * cur_q_blocks : (i + 1) * cur_q_blocks, : cur_k_blocks] = layer_mask

    B, H, NQ, NL = block_mask.shape
    # densities = np.array([100.0 - count_false_percentage(block_mask[:, h, :, :].reshape(B, 1, NQ, NL)) for h in range (H)])
    # print(f'densities over heads {densities}')
    # densities = np.array([100.0 - count_false_percentage(block_mask[b, :, :, :].reshape(1, H, NQ, NL)) for b in range (B)])
    # print(f'densities over layers {densities}')
    return 100.0 - count_false_percentage(block_mask)

if __name__ == "__main__":
    # base = '/home/ceciliapeng/OCL/xattn-cuda/'
    # density = check_cuda_density(base, 'Qwen3-8B-demo-prompt-32k-16-0.9-int4lowbit.npy')
    # print(f'{Colors.GREEN}=============== density of CUDA with thresh 0.9: {density:.2f}%  ==============={Colors.END}\n')
    density = check_ov_density('/home/ceciliapeng/openvino/dump_thresh0.9.block_mask/')
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.9: {density:.2f}%  ==============={Colors.END}\n')

    # density = check_cuda_density(base, 'Qwen3-8B-demo-prompt-32k-16-0.6-int4lowbit.npy')
    # print(f'{Colors.GREEN}=============== density of CUDA with thresh 0.6: {density:.2f}%  ==============={Colors.END}\n')
    density = check_ov_density('/home/ceciliapeng/openvino/dump_thresh0.6.block_mask/')
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.6: {density:.2f}%  ==============={Colors.END}\n')
    
    density = check_ov_density('/home/ceciliapeng/openvino/dump_thresh0.1.block_mask/')
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.1: {density:.2f}%  ==============={Colors.END}\n')
    
    density = check_ov_density('/home/ceciliapeng/openvino/dump_thresh0.1.block_mask.1trunk/', pa_trunk_size = 32*1024)
    print(f'{Colors.BLUE}=============== density of OV with thresh 0.1 1trunk: {density:.2f}%  ==============={Colors.END}\n')

    def get_all_last_subdir_paths(base_path):
        last_subdirs = []
        for root, dirs, files in os.walk(base_path):
            if not dirs:  # Leaf directory
                last_subdirs.append(root)
        return last_subdirs

    # check density of all subdirs
    base_path = "/home/ceciliapeng/openvino/dump_block_mask"
    last_subdir_paths = get_all_last_subdir_paths(base_path)
    print("Full paths of last subdirectories:")
    for path in last_subdir_paths:
        print(f'{path}')
        if "32k" in path:
            density = check_ov_density(path, prompt_len = 32*1024, pa_trunk_size = 4*1024)
            print(f'{Colors.YELLOW}=============== density of OV with {path}: {density:.2f}%  ==============={Colors.END}\n')
        elif "64k" in path:
            density = check_ov_density(path, prompt_len = 64*1024, pa_trunk_size = 4*1024)
            print(f'{Colors.BLUE}=============== density of OV with {path}: {density:.2f}%  ==============={Colors.END}\n')