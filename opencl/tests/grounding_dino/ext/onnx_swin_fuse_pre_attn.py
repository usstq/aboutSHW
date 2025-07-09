import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import openvino as ov
import torch.nn.functional as F
import copy

ext_path = f"{os.path.dirname(os.path.abspath(__file__))}/build/libstub.so"
onnx_path = 'model.onnx'

# https://blog.openvino.ai/blog-posts/custom-pytorch-operations
class StubPadRollPermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, shift_size, window_size, embed_dims) -> torch.Tensor:
        B, H, W, C = query.shape
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size

        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))

        # cyclic shift
        if shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-shift_size, -shift_size),
                dims=(1, 2))
        else:
            shifted_query = query

        # nW*B, window_size, window_size, C
        # query_windows = self.window_partition(shifted_query)
        B, H, W, C = query.shape
        shifted_query = shifted_query.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = shifted_query.permute(0, 1, 3, 2, 4, 5).contiguous()
        query_windows = windows.view(-1, window_size, window_size, C)

        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, window_size**2, C)

        return query_windows

    @staticmethod
    def symbolic(g:torch.Graph, query: torch.Tensor, shift_size, window_size, embed_dims) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:[-1,{window_size*window_size},{embed_dims}]
                        type:PadRollPermute 
                        HEAD_DIMS:{embed_dims}
                        WINDOW_SIZE:{window_size}
                        SHIFT_SIZE:{shift_size}
                    ''',
                    'utf8'))))
        return g.op("Stub", query, attr)

class MyModelONNX(nn.Module):
    def __init__(self, shift_size, window_size, embed_dims):
        super(MyModelONNX, self).__init__()
        self.window_size = window_size
        self.embed_dims = embed_dims
        self.shift_size = shift_size

    def forward(self, query: torch.Tensor):
        return StubPadRollPermute.apply(query, self.shift_size, self.window_size, self.embed_dims)

def get_onnx_model(shift_size, window_size, embed_dims):
    model = MyModelONNX(shift_size, window_size, embed_dims)
    model.eval()
    return model

def export_onnx(onnx_model, query_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    query = Variable(torch.randn(query_shape))

    with torch.no_grad():
        torch.onnx.export(onnx_model, (query,), onnx_path,
                        input_names=['query', ],
                        output_names=['out'])

    print('done')

## Test
def test_ov(model_path, query, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    for _ in range(times):
        results = req.infer({
            "query": query,
        })

    values = list(results.values())[0]
    return values

HEAD_DIMS = [128, 256, 512, 1024]
WINDOW_SIZE = 12
SHIFT_SIZE = [0, 6]
HW = [(200,300), (100,150), (50, 75), (25,38)]
ov_test_times = 1

for shift_size in SHIFT_SIZE:
    print(f'{shift_size=}')
    for head_dims, hw in zip(HEAD_DIMS, HW):
        print(f'{head_dims=} {hw=}')

        query_shape=[1, hw[0], hw[1], head_dims]

        onnx_model = get_onnx_model(shift_size=shift_size, window_size=WINDOW_SIZE, embed_dims=head_dims)
        export_onnx(copy.deepcopy(onnx_model), query_shape=query_shape)

        query = np.random.random(query_shape).astype(dtype=np.float32)
        print(f'test "{onnx_path}"...', end='')
        cur = test_ov(onnx_path, query=query, times=ov_test_times)
        ref = onnx_model(torch.from_numpy(query)).detach().numpy()

        if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
            print(f'onnx failed:\n{ref=}\n{cur=}')
            print(np.where(np.abs(ref - cur) > 0.01))
        print('')
print('done.')