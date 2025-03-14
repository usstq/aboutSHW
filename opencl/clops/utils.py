from clops import cl
import torch
import numpy as np

class KernelCache:
    def __init__(self):
        self.cache = {}
    
    def __call__(self, cl_kernel_sources, options="", dump=""):
        key = (cl_kernel_sources, options, dump)
        if key in self.cache:
            return self.cache[key]
        
        self.cache[key] = cl.kernels(cl_kernel_sources, options, dump)
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
            raise Exception("failed.")

        pos = np.where(np.abs(ref - opt) > atol)
        print(f'========================================================')
        print(f'compare failed (ref={ref.dtype} {ref.shape}, opt={opt.dtype} {opt.shape}, atol={atol}, rtol={rtol})')
        print(f'pos = {len(pos)}x{len(pos[0])} {pos}')
        print(f'ref_val = {ref[pos]}')
        print(f'opt_val = {opt[pos]}')
        raise Exception("failed.")

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"