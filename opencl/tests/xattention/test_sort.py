#from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F

import numpy as np

from clops import cl
import clops
import time
from clops import compare
import os
from clops.utils import Colors

cl.profiling(True)

# kernel
base_dir = os.path.dirname(os.path.abspath(__file__))
files = ['sort.hpp']
src = [open(f'{base_dir}/{file}').read() for file in files]
src.append(
r'''
#define ABS(x) (x) < 0 ? -(x) : (x)

_GENX_MAIN_ void sort(svmptr_t src ATTR, svmptr_t sorted_value ATTR, svmptr_t sorted_index ATTR, svmptr_t sort_tmp ATTR, uint n) {
    const uint slm_size = 32 * 16 * sizeof(ushort);
    cm_slm_init(slm_size);
    auto slm = cm_slm_alloc(slm_size);

    sort<half>(slm, src, sorted_value, sorted_index, sort_tmp, n);
}

''')
src = '\n'.join(src)

################# TODO: fuse gemm+softmax begin
#assert BLOCK_WG_N % block_size == 0, "a block must be in a workgroup"

jit_option = '-abortonspill -noschedule '
kernels = cl.kernels(src, f'''-cmc -Qxcm_jit_option="{jit_option}" -Qxcm_register_file_size=256 -mCM_printregusage -mdump_asm -g2
                    ''')

def test_sort(src:torch.Tensor):
    n = src.shape[0]
    tSrc = cl.tensor(src.detach().numpy())
    #tSorted_value = cl.tensor(np.zeros([n,], np.int16))
    tSorted_value = cl.tensor(np.zeros([n,], np.float16))
    tSorted_index = cl.tensor(np.zeros([n,], np.uint16))
    tSorted_tmp = cl.tensor(np.zeros([n,], np.uint16))

    # gemm
    for i in range(1):
        kernels.enqueue("sort", [1], [1], tSrc, tSorted_value, tSorted_index, tSorted_tmp, n)
    cl.finish()

    sorted_ref, index_ref = torch.sort(src, descending=True, stable=True)
    ref_np = sorted_ref.detach().numpy()
    cur_np = tSorted_value.numpy()
    compare(ref_np, cur_np)
    print(f'{Colors.GREEN}sort:value passed{Colors.END}')
    index_ref_np = index_ref.detach().numpy()
    sorted_np = tSorted_index.numpy()
    error_idx = np.where(index_ref_np != sorted_np)
    if error_idx[0].shape[0]:
        print(f'{error_idx=}\nidx_ref={index_ref_np[error_idx]}\nidx_cur={sorted_np[error_idx]}')
        print(f'ref_val={ref_np[error_idx]}\ncur_val={cur_np[error_idx]}')
        raise('error')
    compare(index_ref.detach().numpy(), tSorted_index.numpy())
    print(f'{Colors.GREEN}sort:index passed{Colors.END}')
    if 1:
        for i in range(0, 100):
            kernels.enqueue("sort", [1], [1], tSrc, tSorted_value, tSorted_index, tSorted_tmp, n)
            ns = cl.finish()
            for i, time_opt in enumerate(ns):
                print(f'(SORT)TPUT_{i}:[{n}] {time_opt*1e-3:,.0f} us')

def main():
    #data = torch.randint(-0, 30000, size=[32*200], dtype=torch.int16)
    data = torch.rand([32*2000], dtype=torch.float16)
    test_sort(data)

if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    main()