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
class StubAddInDeformable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, query_pos: torch.Tensor, head_dims) -> torch.Tensor:
        return query + query_pos

    @staticmethod
    def symbolic(g:torch.Graph, query: torch.Tensor, query_pos: torch.Tensor, head_dims) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:0
                        type:AddInDeformable
                        HEAD_DIMS:{head_dims}
                    ''',
                    'utf8'))))
        return g.op("Stub", query, query_pos, attr)

class MyModelONNX(nn.Module):
    def __init__(self, head_dims):
        super(MyModelONNX, self).__init__()
        self.head_dims = head_dims

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor):
        return StubAddInDeformable.apply(query, query_pos, self.head_dims)

def get_onnx_model(head_dims):
    model = MyModelONNX(head_dims)
    model.eval()
    return model

def export_onnx(onnx_model, query_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    query = Variable(torch.randn(query_shape))
    query_pos = Variable(torch.randn(query_shape))

    with torch.no_grad():
        torch.onnx.export(onnx_model, (query, query_pos), onnx_path,
                        input_names=['query', 'query_pos'],
                        output_names=['out'])

    print('done')

## Test
def test_ov(model_path, query, query_pos, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    for _ in range(times):
        results = req.infer({
            "query": query,
            "query_pos": query_pos
        })

    values = list(results.values())[0]
    return values

ov_test_times = 100
HEAD_DIMS = 256
print('testing...')
query_shape=[1, 22223, HEAD_DIMS]

onnx_model = get_onnx_model(HEAD_DIMS)
export_onnx(copy.deepcopy(onnx_model), query_shape=query_shape)

query = np.random.random(query_shape).astype(dtype=np.float32)
query_pos = np.random.random(query_shape).astype(dtype=np.float32)
print(f'test "{onnx_path}"...', end='')
cur = test_ov(onnx_path, query=query, query_pos=query_pos, times=ov_test_times)
ref = onnx_model(torch.from_numpy(query), torch.from_numpy(query_pos)).detach().numpy()

if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
    print(f'onnx failed:\n{ref=}\n{cur=}')
    print(np.where(np.abs(ref - cur) > 0.01))
print('done.')