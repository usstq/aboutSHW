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

MAX_CLAMP_VALUE = 50000
# https://blog.openvino.ai/blog-posts/custom-pytorch-operations
class StubMaxSubClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weights_T: torch.Tensor) -> torch.Tensor:
        attn_weights_l = (
            attn_weights_T -
            torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
        # Do not increase -50000, data type half has quite limited range
        attn_weights_l = torch.clamp(attn_weights_l, min=-MAX_CLAMP_VALUE)
        # Do not increase 50000, data type half has quite limited range
        attn_weights_l = torch.clamp(attn_weights_l, max=MAX_CLAMP_VALUE)

        return attn_weights_l

    @staticmethod
    def symbolic(g:torch.Graph, attn_weights: torch.Tensor) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:0
                        type:MaxSubClip
                        MAX_CLAMP_VALUE:{MAX_CLAMP_VALUE:.1f}
                    ''',
                    'utf8'))))
        return g.op("Stub", attn_weights, attr)

class MyModelONNX(nn.Module):
    def __init__(self):
        super(MyModelONNX, self).__init__()

    def forward(self, attn_weights: torch.Tensor):
        return StubMaxSubClip.apply(attn_weights)

def get_onnx_model():
    model = MyModelONNX()
    model.eval()
    return model

def export_onnx(onnx_model, attn_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    attn = Variable(torch.randn(attn_shape))

    with torch.no_grad():
        torch.onnx.export(onnx_model, (attn,), onnx_path,
                        input_names=['attn',],
                        output_names=['out'])

    print('done')

## Test
def test_ov(model_path, attn, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    for _ in range(times):
        results = req.infer({
            "attn": attn,
        })

    values = list(results.values())[0]
    return values

ov_test_times = 1
print('testing...')
attn_shape=[4, 18, 22223]

onnx_model = get_onnx_model()
export_onnx(copy.deepcopy(onnx_model), attn_shape=attn_shape)

attn = np.random.randint(-30, -2, attn_shape).astype(dtype=np.float32)
print(f'test "{onnx_path}"...', end='')
cur = test_ov(onnx_path, attn=attn, times=ov_test_times)
ref = onnx_model(torch.from_numpy(attn)).detach().numpy()

if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
    print(f'onnx failed:\n{ref=}\n{cur=}')
    print(np.where(np.abs(ref - cur) > 0.01))
    pass
print('done.')