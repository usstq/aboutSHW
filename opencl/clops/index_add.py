#!/usr/bin/python3
from . import cl
import numpy as np
from .utils import *
from . import compare
from . import utils

class index_add:
    def __init__(self, datatype: str = 'float'):       
        cl_source_file = "cl_kernels/index_add.cl"
        with open(cl_source_file, "r") as file:
            # Read the entire file content into a string
            cl_kernel_sources = file.read()
        self.cl_kernels = kernel_cache(cl_kernel_sources, options=f'-cl-mad-enable -D DATATYPE={datatype}')

    def __call__(self, dim, dst: torch.Tensor, src: torch.Tensor, index: torch.Tensor):
        # shape inference
        num_dims = len(dst.size())

        dst_sizes = dst.size()
        print(f"len(dst_sizes)={len(dst_sizes)}, dst_sizes={dst_sizes}")
        dst_sizes = to_cl(torch.tensor(dst_sizes).int())
        
        src_sizes = src.size()
        print(f"len(dst_sizes)={len(src_sizes)}, src_sizes={src_sizes}")
        src_sizes = to_cl(torch.tensor(src_sizes).int())
        
        dst_strides = [dst.size()[-1], 1]
        print(f"len(dst_sizes)={len(dst_strides)}, dst_strides={dst_strides}")
        dst_strides = to_cl(torch.tensor(dst_strides).int())
        
        src_strides = [src.size()[-1], 1]
        print(f"len(dst_sizes)={len(src_strides)}, src_strides={src_strides}")
        src_strides = to_cl(torch.tensor(src_strides).int())

        GWS = [src.numel()]
        LWS = [1]
        
        dst = to_cl(dst)

        print(f"GWS={GWS}, LWS={LWS}")
        self.cl_kernels.enqueue("index_add", GWS, LWS,
                            dst, to_cl(src), to_cl(index), dim, num_dims,
                            dst_sizes, dst_strides,
                            src_sizes, src_strides)

        return dst

if __name__ == "__main__":
    import sys
    cl.profiling(True)

    def cl_impl(dim, destination : torch.Tensor, source : torch.Tensor, index : torch.Tensor):
        assert destination.dtype==torch.float16 or destination.dtype==torch.float
        datatype = 'half' if destination.dtype==torch.float16 else 'float'
        impl = index_add(datatype)
        output = impl(dim, destination, source, index)

        duration = cl.finish()
        return output.numpy(), duration

    def test_acc(dim, x, source, index):        
        # cl impl
        opt, durs = cl_impl(dim, x.clone(), source, index)
        for ns in durs:
            print(f'{opt.shape=}, {ns*1e-6:.3f} ms')
        
        # ref impl
        ref = x.index_add_(dim, index, source)     
        print(f'{opt=}, {ref=}')

        try:
            if not np.allclose(ref, opt, atol=0.01, rtol=0.01, equal_nan=False):
                # print(f'{ref=}\n{opt=}')
                pos = np.where(np.abs(ref - opt) > 0.01)
                print(f'failed at shape = {opt.shape}')
                # print(f'pos = {pos}')
                # print(f'ref_val = {ref[pos]}\nopt_val={opt[pos]}')
                raise Exception("failed.")
            print('done.')
        except Exception as inst:
            print('failed.')

    test_acc(0, torch.ones([5, 3], dtype=torch.float), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float), torch.tensor([0, 4, 2], dtype=torch.int32))
    test_acc(0, torch.ones(10, 2048), torch.randn([6, 2048], dtype=torch.float), torch.tensor([0, 4, 2, 9, 1, 3], dtype=torch.int32))
    test_acc(0, torch.ones(10, 2048), torch.randn([6, 2048], dtype=torch.float), torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int32))
    
    test_acc(0, torch.ones([5, 3], dtype=torch.float16), torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float16), torch.tensor([0, 4, 2], dtype=torch.int32))
    test_acc(0, torch.ones([10, 2048], dtype=torch.float16), torch.randn([6, 2048], dtype=torch.float16), torch.tensor([0, 4, 2, 9, 1, 3], dtype=torch.int32))
