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
class StubAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: torch.Tensor, mask: torch.Tensor, num_heads, scale, window_size, embed_dims) -> torch.Tensor:
        B, N, C = qkv.shape
        qkv = qkv.reshape(B, N, 3, num_heads,
                                C // num_heads // 3).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        return (attn @ v).transpose(1, 2).reshape(B, N, C // 3)

    @staticmethod
    def symbolic(g:torch.Graph, qkv: torch.Tensor, mask: torch.Tensor, num_heads, scale, window_size, embed_dims) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:[-1,{window_size*window_size},{embed_dims}]
                        type:WSAttention 
                        NUM_HEADS:{num_heads}
                        SCALE:{scale}
                        NUM_KV_HEADS:{num_heads}
                        HEAD_SIZE:{embed_dims//num_heads}
                    ''',
                    'utf8'))))
        return g.op("Stub", qkv, mask, attr)

class MyModelONNX(nn.Module):
    def __init__(self, num_heads, scale, window_size, embed_dims):
      super(MyModelONNX, self).__init__()
      self.num_heads = num_heads
      self.scale = scale
      self.window_size = window_size
      self.embed_dims = embed_dims

    def forward(self, qkv: torch.Tensor, mask: torch.Tensor):
      return StubAttention.apply(qkv, mask, self.num_heads, self.scale, self.window_size, self.embed_dims)

def get_onnx_model(num_heads, scale, window_size, embed_dims):
    model = MyModelONNX(num_heads, scale, window_size, embed_dims)
    model.eval()
    return model

def export_onnx(onnx_model, qkv_shape, mask_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    qkv = Variable(torch.randn(qkv_shape))
    mask = Variable(torch.randn(mask_shape))

    with torch.no_grad():
        torch.onnx.export(onnx_model, (qkv, mask), onnx_path,
                        input_names=['qkv', 'mask'],
                        output_names=['attn'])

    print('done')

## Test
def test_ov(model_path, qkv, mask, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    for _ in range(times):
        results = req.infer({
            "qkv": qkv,
            "mask": mask
        })

    values = list(results.values())[0]
    return values

HEAD_SIZE = 32
WIN_SIZE = 12
HEAD_NUM = [4, 8, 16, 32]
WIN_NO = [476, 126, 35, 12]
for head_num, win_no in zip(HEAD_NUM, WIN_NO):
    print(f'{head_num=} {win_no=}')
    HEAD_DIM = HEAD_SIZE * head_num

    qkv_shape=[win_no, WIN_SIZE*WIN_SIZE, HEAD_DIM * 3]
    mask_shape = [1, head_num, WIN_SIZE*WIN_SIZE, WIN_SIZE*WIN_SIZE]

    onnx_model = get_onnx_model(num_heads=head_num, scale=0.1767766952966369, window_size=WIN_SIZE, embed_dims=HEAD_DIM)
    export_onnx(copy.deepcopy(onnx_model), qkv_shape=qkv_shape, mask_shape=mask_shape)

    qkv = np.random.random(qkv_shape)
    mask = np.random.random(mask_shape)

    print(f'test "{onnx_path}"...', end='')
    cur = test_ov(onnx_path, qkv, mask)
    ref = onnx_model(torch.from_numpy(qkv), torch.from_numpy(mask)).detach().numpy()
    if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
        print(f'onnx failed:\n{ref=}\n{cur=}')
    print('')
print('done.')
