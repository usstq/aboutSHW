import argparse
import sys
import numpy as np
import time
import math

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from clops import cl
from clops.utils import *

import torch

from mmcv.ops.multi_scale_deform_attn import (
    multi_scale_deformable_attn_pytorch)


class MultiScaleDeformableAttnFunction_CL:
    def __init__(self, datatype: str = 'float'):
        self.dtype = datatype
        # print("compiling ...")
        with open("multi_scale_deformable_attn.cl", "r") as file:
            # Read the entire file content into a string
            src = file.read()
        # print(src)
        self.cl_kernels = kernel_cache(src, options=f'-cl-mad-enable -D scalar_t={datatype}')
        
    def ms_deformable_im2col(self, data_value, data_spatial_shapes,
                             data_level_start_index,
                             data_sampling_loc,
                             data_attn_weight,
                             batch_size, spatial_size,
                             num_heads, embed_dims,
                             num_levels, num_queries,
                             num_point):
        num_kernels = batch_size * num_queries * num_heads * embed_dims
        
        datatype = np.float16 if self.dtype == 'half' else np.float32
        t_data_col = cl.tensor([batch_size, num_queries, num_heads, embed_dims], np.dtype(datatype))

        print('============', type(data_value))

        GWS=[1, 1, num_kernels]
        LWS=[1, 1, 1]
        print("GWS=", GWS)
        print("LWS=", LWS)

        self.cl_kernels.enqueue("multi_scale_deformable_attn", GWS, LWS, num_kernels,
                                cl.tensor(data_value.detach().numpy()), cl.tensor(data_spatial_shapes.detach().numpy()),
                                cl.tensor(data_level_start_index.detach().numpy()), cl.tensor(data_sampling_loc.detach().numpy()),
                                cl.tensor(data_attn_weight.detach().numpy()),
                                batch_size, spatial_size, num_heads, embed_dims,
                                num_levels, num_queries, num_point,
                                t_data_col)
        return torch.from_numpy(t_data_col.numpy())


    def __call__(self, value, spatial_shapes, level_start_index,
                 sampling_locations,
                 attention_weights, im2col_step):
        '''
            Args:
                value (torch.Tensor): The value has shape
                    (bs, num_keys, num_heads, embed_dims)
                value_spatial_shapes (torch.Tensor): Spatial shape of
                    each feature map, has shape (num_levels, 2),
                    last dimension 2 represent (h, w)
                sampling_locations (torch.Tensor): The location of sampling points,
                    has shape
                    (bs, num_queries, num_heads, num_levels, num_points, 2),
                    the last dimension 2 represent (x, y).
                attention_weights (torch.Tensor): The weight of sampling points
                    used when calculate the attention, has shape
                    (bs, num_queries, num_heads, num_levels, num_points),
                im2col_step (torch.Tensor): The step used in image to column.
                level_start_index (torch.Tensor): The start index of each level.
                    A tensor has shape ``(num_levels, )`` and can be represented
                    as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

                ==> output (bs, num_queries, num_heads, embed_dims)
        '''
        assert value.is_contiguous()
        assert spatial_shapes.is_contiguous()
        assert level_start_index.is_contiguous()
        assert sampling_locations.is_contiguous()
        assert attention_weights.is_contiguous()

        batch, spatial_size, num_heads, embed_dims = value.shape

        num_levels = spatial_shapes.shape[0]

        num_queries = sampling_locations.size(1)
        num_point = sampling_locations.size(4)
        
        return self.ms_deformable_im2col(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            batch, spatial_size, num_heads, embed_dims, num_levels, num_queries,
            num_point).view(batch, num_queries, num_heads * embed_dims)

        im2col_step_ = min(batch, im2col_step)
        assert batch % im2col_step_ == 0

        output = torch.zeros([batch, num_queries, num_heads, embed_dims], device='cpu', dtype=torch.float32)

        batch_n = im2col_step_
        output_n = output.view(
            math.floor(batch / im2col_step_), batch_n, num_queries, num_heads, embed_dims)
        per_value_size = spatial_size * num_heads * embed_dims
        per_sample_loc_size = num_queries * num_heads * num_levels * num_point * 2
        per_attn_weight_size = num_queries * num_heads * num_levels * num_point
        for n in range(math.floor(batch / im2col_step_)):
            columns = output_n.select(0, n)
            
            assert columns.storage().data_ptr() == output.storage().data_ptr()

            ret_columns = self.ms_deformable_im2col(
                value[n * im2col_step_: (n + 1) * im2col_step_, : ], # + n * im2col_step_ * per_value_size
                spatial_shapes,
                level_start_index,
                sampling_locations[n * im2col_step_ : (n + 1) * im2col_step_, : ], # + n * im2col_step_ * per_sample_loc_size
                attention_weights[n * im2col_step_ : (n + 1) * im2col_step_, : ], # + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, embed_dims, num_levels, num_queries,
                num_point)
            print(f'{columns.storage().data_ptr()=}, {columns.is_contiguous()}, {ret_columns.storage().data_ptr()=}, {ret_columns.is_contiguous()}')
            columns.copy_(ret_columns)
            assert columns.storage().data_ptr() == output.storage().data_ptr()

        output = output.view(batch, num_queries, num_heads * embed_dims)

        return output

def test_forward_equal_with_pytorch(value_shape = [2, 2, 2], offsets_shape = [2, 2, 2], datatype = 'float'):
    assert datatype=='half' or datatype=='float'
    value_dtype = torch.float16 if datatype == 'half' else torch.float32

    N, H, S = value_shape
    Lq, L, P = offsets_shape
    
    # 5, 10, 20, 40, ...
    def generate_pairs(L: int) -> list[tuple[int, int]]:
        pairs = []
        for scale in range(L):
            pairs.append((5 * (2**scale), 5 * (2**scale) ))
        return pairs

    value_spatial_shapes = torch.as_tensor(generate_pairs(L), dtype=torch.int32)
    print(f'{value_spatial_shapes=}')

    level_start_index = torch.cat((value_spatial_shapes.new_zeros(
        (1, )), value_spatial_shapes.prod(1).cumsum(0)[:-1])).to(torch.int32)
    Lv = sum((H * W).item() for H, W in value_spatial_shapes)

    value = torch.rand(N, Lv, H, S, dtype=value_dtype)
    sampling_locations = torch.rand(N, Lq, H, L, P, 2, dtype=value_dtype)
    attention_weights = torch.rand(N, Lq, H, L, P) + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    attention_weights = attention_weights.to(value_dtype)
    im2col_step = 64
    
    print(f'{Lv=}')
    print(f'{value_spatial_shapes=}')         # (num_levels, 2) Spatial shape of each feature map, last dimension 2 represent (h, w)
    print(f'{level_start_index=}')  #(num_levels, ) start index of each level and can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    print(f'{value.shape=}')          # (bs, num_keys, num_heads, embed_dims)
    print(f'{sampling_locations.shape=}') # (bs ,num_queries, num_heads, num_levels, num_points, 2), the last dimension 2 represent (x, y).
    print(f'{attention_weights.shape=}')  # (bs ,num_queries, num_heads, num_levels, num_points), weight of sampling points

    output_pytorch = multi_scale_deformable_attn_pytorch(
        value, value_spatial_shapes, sampling_locations, attention_weights).detach().cpu()
    if datatype=='half':
        print('=============== to half')
        output_pytorch = output_pytorch.half()

    msda = MultiScaleDeformableAttnFunction_CL(datatype)
    output_device = msda(
        value, value_spatial_shapes, level_start_index,
        sampling_locations,
        attention_weights, im2col_step)
    duration = cl.finish()
    for ns in duration:
        print(f'{ns*1e-6:.3f} ms')

    print(f'{output_pytorch.shape=} {output_pytorch.dtype=} {output_pytorch=}')
    print(f'{output_device.shape=} {output_device.dtype=} {output_device=}')
    assert torch.allclose(output_device, output_pytorch, rtol=1e-2, atol=1e-3)
    # max_abs_err = (output_device - output_pytorch).abs().max()
    # max_rel_err = ((output_device - output_pytorch).abs() /
    #                output_pytorch.abs()).max()
    # print(f'{max_abs_err=}')
    # assert max_abs_err < 1e-9
    # assert max_rel_err < 1e-6
    
if __name__ == "__main__":
    torch.manual_seed(3)
    torch.set_printoptions(linewidth=1024)
    
    import sys
    cl.profiling(True)
    
    test_forward_equal_with_pytorch(datatype = 'float')
    test_forward_equal_with_pytorch(datatype = 'half')
    
    bs, num_heads, embed_dims = 1, 8, 32
    num_queries, num_levels, num_points = 2125, 4, 4
    test_forward_equal_with_pytorch([bs, num_heads, embed_dims], [num_queries, num_levels, num_points], datatype = 'float')
    test_forward_equal_with_pytorch([bs, num_heads, embed_dims], [num_queries, num_levels, num_points], datatype = 'half')
    