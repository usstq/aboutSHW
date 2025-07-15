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
class StubPermuteAddConcat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, head_dims, *mlvl_pos_embeds_bias) -> torch.Tensor:
        lvl_pos_embed_flatten = []
        num = len(mlvl_pos_embeds_bias) // 2
        for lvl in range(num):
            pos_embed = mlvl_pos_embeds_bias[lvl]
            #pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + mlvl_pos_embeds_bias[lvl + num].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        return lvl_pos_embed_flatten

    @staticmethod
    def symbolic(g:torch.Graph, head_dims, *mlvl_pos_embeds_bias) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:[-1,-1,{head_dims}]
                        HEAD_DIMS:{head_dims}
                        type:PermuteAddConcat
                        num:{len(mlvl_pos_embeds_bias)//2}
                    ''',
                    'utf8'))))
        return g.op("Stub", *mlvl_pos_embeds_bias, attr)

class MyModelONNX(nn.Module):
    def __init__(self, head_dims):
        super(MyModelONNX, self).__init__()
        self.head_dims=head_dims

    def forward(self, *mlvl_pos_embeds_bias):
        return StubPermuteAddConcat.apply(self.head_dims, *mlvl_pos_embeds_bias)

def get_onnx_model(head_dims):
    model = MyModelONNX(head_dims=head_dims)
    model.eval()
    return model

def export_onnx(onnx_model, spatial_shapes, bias_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    mlvl_pos_embeds = []
    level_embed = []
    names = [1] * len(spatial_shapes) * 2
    for i, shape in enumerate(spatial_shapes):
        mlvl_pos_embeds.append(Variable(torch.randn(shape)))
        level_embed.append(Variable(torch.randn(bias_shape)))
        names[i] = f"pos{i}"
        names[i + len(spatial_shapes)] = f"bias{i}"

    with torch.no_grad():
        torch.onnx.export(onnx_model, tuple(mlvl_pos_embeds + level_embed), onnx_path,
                        input_names=names,
                        output_names=['out'])

    print('done')

## Test
def test_ov(model_path, spatial, bias, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    params = {}
    for i in range(len(spatial)):
        params[f'pos{i}'] = spatial[i]
        params[f'bias{i}'] = bias[i]
    for _ in range(times):
        results = req.infer(params)

    values = list(results.values())[0]
    return values

ov_test_times = 100
HEAD_DIMS = 256
print('testing...')
hw_shapes = [100*150, 50 * 75, 25 * 38, 13 * 19]
spatial_shapes = [[1, ] + [hw, ] + [HEAD_DIMS, ] for hw in hw_shapes]

onnx_model = get_onnx_model(HEAD_DIMS)
export_onnx(copy.deepcopy(onnx_model), spatial_shapes=spatial_shapes, bias_shape=[HEAD_DIMS,])

mlvl_pos_embeds = []
level_embed = []
mlvl_pos_embeds_t = []
level_embed_t = []
for i, shape in enumerate(spatial_shapes):
    mlvl_pos_embeds.append(np.random.random(shape).astype(dtype=np.float32))
    level_embed.append(np.random.random([HEAD_DIMS,]).astype(dtype=np.float32))
    mlvl_pos_embeds_t.append(torch.from_numpy(mlvl_pos_embeds[i]))
    level_embed_t.append(torch.from_numpy(level_embed[i]))

print(f'test "{onnx_path}"...', end='')
cur = test_ov(onnx_path, mlvl_pos_embeds, level_embed, times=ov_test_times)
param_t = mlvl_pos_embeds_t + level_embed_t
ref = onnx_model(*param_t).detach().numpy()

if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
    print(f'onnx failed:\n{ref=}\n{cur=}')
    print(np.where(np.abs(ref - cur) > 0.01))
print('done.')