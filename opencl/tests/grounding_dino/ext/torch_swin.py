import os
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import openvino as ov
import torch.nn.functional as F
from openvino.frontend.pytorch import ModuleExtension
import copy

ext_path = f"{os.path.dirname(os.path.abspath(__file__))}/build/libstub.so"
torch_path = 'model.xml'

class MyModelTorch(nn.Module):
    def __init__(self, num_heads, scale, window_size, embed_dims):
       super(MyModelTorch, self).__init__()
       self.num_heads = num_heads
       self.scale = scale
       self.window_size = window_size
       self.embed_dims = embed_dims

    def forward(self, qkv: torch.Tensor, mask: torch.Tensor):
        B, N, C = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads,
                                C // self.num_heads // 3).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        return (attn @ v).transpose(1, 2).reshape(B, N, C // 3)

def get_torch_model(num_heads, scale, window_size, embed_dims):
    model = MyModelTorch(num_heads, scale, window_size, embed_dims)
    model.eval()
    return model

def export_torch(torch_model, qkv_shape, mask_shape):
    print(f'convert torch model to "{torch_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    qkv = Variable(torch.randn(qkv_shape))
    mask = Variable(torch.randn(mask_shape))

    # https://github.com/slyalin/vllm/blob/30605c85ea2792a9145aec034e3102644a204d6d/vllm/worker/model_runner.py#L262
    with torch.no_grad():
        ov_model = ov.convert_model(
            torch_model,
            example_input=(qkv, mask),
            extension=[
                ModuleExtension(
                    MyModelTorch,
                    target_op='Stub',
                    evaluate=lambda module, *args, **kwargs: torch.ones(
                        list(args[0].shape[:-1]) + [args[0].shape[-1] // 3],
                        dtype=torch.float32),
                    convert=lambda module, target_op, *args, **kwargs: target_op(
                        args[0], args[1], 
                        torch.ByteTensor(list(bytes(
                        f'''
                            out_dt:0 
                            out_shape:[-1,{module.window_size*module.window_size},{module.embed_dims}]
                            type:WSAttention 
                            NUM_HEADS:{module.num_heads}
                            SCALE:{module.scale}
                            NUM_KV_HEADS:{module.num_heads}
                            HEAD_SIZE:{module.embed_dims//module.num_heads}
                        ''', 
                        'utf8'))))
                ),
                ext_path
            ]
        )
        ov.serialize(ov_model, torch_path, f"{torch_path[:-3]}bin")

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
HEAD_NUM = 16
HEAD_DIM = HEAD_SIZE * HEAD_NUM
WIN_SIZE = 12
WIN_NO = 35

qkv_shape=[WIN_NO, WIN_SIZE*WIN_SIZE, HEAD_DIM * 3]
mask_shape = [1, HEAD_NUM, WIN_SIZE*WIN_SIZE, WIN_SIZE*WIN_SIZE]

torch_model = get_torch_model(num_heads=HEAD_NUM, scale=0.1767766952966369, window_size=WIN_SIZE, embed_dims=HEAD_DIM)
export_torch(copy.deepcopy(torch_model), qkv_shape=qkv_shape, mask_shape=mask_shape)

qkv = np.random.random(qkv_shape)
mask = np.random.random(mask_shape)

print(f'test "{torch_path}"...', end='')
cur = test_ov(torch_path, qkv, mask)
ref = torch_model(torch.from_numpy(qkv), torch.from_numpy(mask)).detach().numpy()
if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
    print(f'torch failed:\n{ref=}\n{cur=}')
print('done.')
