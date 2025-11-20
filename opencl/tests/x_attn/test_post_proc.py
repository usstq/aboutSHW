import os
import torch
import numpy as np

from clops import cl
from clops.utils import Colors

def div_up(a, b):
    return (a + b - 1) // b
def rnd_up(a, b):
    return (a + b - 1) // b * b

HQ = 32
HK = 8
HEAD_SIZE = 128

STRIDE = 16
BLOCK_SIZE = 128

MERGED_Q_NUM = 2

def create_kernels():
    # kernel
    src = r'''#include "xattn_post_proc.hpp"'''
    cwd = os.path.dirname(os.path.realpath(__file__))
    print(f"compiling {cwd} ...")

    jit_option = '-abortonspill -noschedule '
    kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}"
                        -mCM_printregusage -mdump_asm -g2
                        -Qxcm_register_file_size=128 -I{cwd}
                        -DSTRIDE={STRIDE} -DHQ={HQ} -DHK={HK}
                        -DBLOCK_SIZE={BLOCK_SIZE}
                        -DMERGED_Q_NUM={MERGED_Q_NUM}''')
    return kernels

def test_post_proc():
    kernels = create_kernels()
    sum_per_token_in_block = 8
    qk = [
        (4096, 4096),
        (4096+1, 4096)
    ]
    for q, k in qk:
        q_stride_pad = q // STRIDE
        q_block_valid = q_stride_pad // sum_per_token_in_block
        q_block_pad = div_up(q, BLOCK_SIZE)
        k_block_pad = div_up(k, BLOCK_SIZE)
        org = np.random.randint(low=0, high=2, size=[1, HQ, q_block_pad, k_block_pad], dtype=np.int8)
        if q_block_valid != q_block_pad:
            org[:,:,-1,:] = 1

        t_mask = cl.tensor(org)
        t_merged_mask = cl.tensor(np.ones([1, HQ, div_up(q_block_pad, MERGED_Q_NUM), k_block_pad], dtype=np.int8) * 100)
        # block_mask, merged_block_mask, q_stride_pad, q_block_pad, k_block_pad
        params = [t_mask, t_merged_mask, q_stride_pad, q_block_pad, k_block_pad]
        kernels.enqueue("post_proc_mask", [t_merged_mask.shape[2], HQ, 1], [1, 1, 1], *params)
        cl.finish()
        t_merged_mask_np = t_merged_mask.numpy()
        t_mask_np = t_mask.numpy()
        if q_block_valid != q_block_pad:
            assert np.all(t_mask_np[:,:,-(q_block_pad - q_block_valid):,:] == 1), f"new pad value must be one, {q=}, {k=}"
            org_pad = np.ones([1, HQ, q_block_pad + 1, k_block_pad], dtype=np.int8)
            org_pad[:,:,:-1,:] = org
            org = org_pad
        t_merged_mask_ref = org[:,:,0::2,:] | org[:,:,1::2,:]
        assert np.all(t_merged_mask_ref == t_merged_mask_np), f"merged mask not equal to ref, {q=} {k=}"
    print(f'{Colors.GREEN}test_post_proc done.{Colors.END}')

def main():
    test_post_proc()

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    cl.profiling(True)
    
    main()