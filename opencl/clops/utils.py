from . import cl
import torch
import numpy as np

class KernelCache:
    def __init__(self):
        self.cache = {}
    
    def __call__(self, cl_kernel_sources, options="", dump=""):
        key = (cl_kernel_sources, options, dump)
        if key in self.cache:
            return self.cache[key]
        
        self.cache[key] = cl.kernels(cl_kernel_sources, options)
        return self.cache[key]

kernel_cache = KernelCache()

def to_cl(input):
    if isinstance(input, cl.tensor):
        return input
    if input is None:
        return None
    if isinstance(input, torch.nn.parameter.Parameter):
        return cl.tensor(input.detach().numpy())
    if isinstance(input, torch.Tensor):
        return cl.tensor(input.detach().numpy())
    assert("to_cl(): Unexpected input ", input)

def to_torch(input):
    return torch.from_numpy(input.numpy())

def compare(ref, opt, atol=0.01, rtol=0.01):
    if not np.allclose(ref, opt, atol=atol, rtol=rtol):
        pos = np.where(np.isnan(opt))
        if len(pos[0]) > 0:
            print(f'========================================================')
            print(f'pos nan = {len(pos)}x{len(pos[0])} {pos}')
            print(f'opt nan = {opt[pos]}')
            raise("failed.")

        pos = np.where(np.abs(ref - opt) > atol)
        print(f'========================================================')
        print(f'compare failed (ref={ref.dtype} {ref.shape}, opt={opt.dtype} {opt.shape}, atol={atol}, rtol={rtol})')
        print(f'pos = {len(pos)}x{len(pos[0])} {pos}')
        print(f'ref_val = {ref[pos]}')
        print(f'opt_val = {opt[pos]}')
        raise("failed.")
