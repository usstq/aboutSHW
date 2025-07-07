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

class StubSWAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: torch.Tensor, mask: torch.Tensor, rotate_mask: torch.Tensor, HW_pad: torch.Tensor, num_heads, scale, window_size, embed_dims) -> torch.Tensor:
        B, N, C = qkv.shape
        qkv = qkv.reshape(B, N, 3, num_heads,
                                C // num_heads // 3).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + mask
        if rotate_mask is not None:
            nW = rotate_mask.shape[0]
            attn = attn.view(B // nW, nW, num_heads, N,
                            N) + rotate_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        return (attn @ v).transpose(1, 2).reshape(B, N, C // 3)

    @staticmethod
    def symbolic(g:torch.Graph, qkv: torch.Tensor, mask: torch.Tensor, rotate_mask: torch.Tensor, HW_pad: torch.Tensor, num_heads, scale, window_size, embed_dims) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:[-1,{window_size*window_size},{embed_dims}]
                        type:SWSAttention 
                        NUM_HEADS:{num_heads}
                        SCALE:{scale}
                        NUM_KV_HEADS:{num_heads}
                        HEAD_SIZE:{embed_dims//num_heads}
                        WINDOW_SIZE:{window_size}
                    ''',
                    'utf8'))))
        return g.op("Stub", qkv, mask, rotate_mask, HW_pad, attr)

class MyModelONNX(nn.Module):
    def __init__(self, num_heads, scale, window_size, embed_dims, is_swa=False):
        super(MyModelONNX, self).__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.window_size = window_size
        self.embed_dims = embed_dims
        self.is_swa = is_swa

    def forward(self, qkv: torch.Tensor, mask: torch.Tensor, rotate_mask: torch.Tensor=None, hw_pad: torch.Tensor=None):
        if self.is_swa:
            return StubSWAttention.apply(qkv, mask, rotate_mask, hw_pad, self.num_heads, self.scale, self.window_size, self.embed_dims)
        else:
            return StubAttention.apply(qkv, mask, self.num_heads, self.scale, self.window_size, self.embed_dims)

def get_onnx_model(num_heads, scale, window_size, embed_dims, is_swa=False):
    model = MyModelONNX(num_heads, scale, window_size, embed_dims, is_swa=is_swa)
    model.eval()
    return model

def export_onnx(onnx_model, qkv_shape, mask_shape, rotate_mask_shape=None):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    qkv = Variable(torch.randn(qkv_shape))
    mask = Variable(torch.randn(mask_shape))
    if rotate_mask_shape:
        rotate_mask = Variable(torch.randn(rotate_mask_shape))
        hw_pad = Variable(torch.tensor([1, 1], dtype=torch.int32))

    with torch.no_grad():
        if rotate_mask_shape:
            torch.onnx.export(onnx_model, (qkv, mask, rotate_mask, hw_pad), onnx_path,
                            input_names=['qkv', 'mask', 'rotate_mask', 'hw_pad'],
                            output_names=['attn'])
        else:
            torch.onnx.export(onnx_model, (qkv, mask), onnx_path,
                            input_names=['qkv', 'mask'],
                            output_names=['attn'])

    print('done')

## Test
def test_ov(model_path, qkv, mask, rotate_mask=None, hw_pad=None, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    for _ in range(times):
        if rotate_mask is not None:
            results = req.infer({
                "qkv": qkv,
                "mask": mask,
                "rotate_mask": rotate_mask,
                "hw_pad": hw_pad
            })
        else:
            results = req.infer({
                "qkv": qkv,
                "mask": mask
            })

    values = list(results.values())[0]
    return values

def gen_rotate_mask(H_pad, W_pad, window_size, shift_size):
    def window_partition(x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows
    
    img_mask = torch.zeros((1, H_pad, W_pad, 1))
    h_slices = (slice(0, -window_size),
                slice(-window_size,
                        -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size,
                        -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # nW, window_size, window_size, 1
    mask_windows = window_partition(img_mask)
    mask_windows = mask_windows.view(
        -1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                        float(-100.0)).masked_fill(
                                            attn_mask == 0, float(0.0))
    return attn_mask.detach().numpy()

HEAD_SIZE = 32
WIN_SIZE = 12
HEAD_NUM = [4, 8, 16, 32]
WIN_NO = [476, 126, 35, 12]
HW_PAD=[(17*12, 28*12), (9*12, 14*12), (5*12, 7*12), (3*12, 4*12)]
ov_test_times = 1
is_swa = True

for is_swa in (False, True):
    print(f'{is_swa=}')
    for head_num, win_no, hw_pad in zip(HEAD_NUM, WIN_NO, HW_PAD):
        print(f'{head_num=} {win_no=}')
        HEAD_DIM = HEAD_SIZE * head_num

        qkv_shape=[win_no, WIN_SIZE*WIN_SIZE, HEAD_DIM * 3]
        mask_shape = [1, head_num, WIN_SIZE*WIN_SIZE, WIN_SIZE*WIN_SIZE]
        if is_swa:
            rotate_mask_shape = [win_no, WIN_SIZE*WIN_SIZE, WIN_SIZE*WIN_SIZE]
        else:
            rotate_mask_shape = None

        onnx_model = get_onnx_model(num_heads=head_num, scale=0.1767766952966369, window_size=WIN_SIZE, embed_dims=HEAD_DIM, is_swa=is_swa)
        export_onnx(copy.deepcopy(onnx_model), qkv_shape=qkv_shape, mask_shape=mask_shape, rotate_mask_shape=rotate_mask_shape)

        qkv = np.random.random(qkv_shape)
        mask = np.random.random(mask_shape)
        print(f'test "{onnx_path}"...', end='')
        if is_swa:
            rotate_mask = gen_rotate_mask(hw_pad[0], hw_pad[1], WIN_SIZE, WIN_SIZE // 2)
            hw_pad_t = np.array(hw_pad, dtype=np.int32)
            cur = test_ov(onnx_path, qkv, mask, rotate_mask, hw_pad_t, times=ov_test_times)
            ref = onnx_model(torch.from_numpy(qkv), torch.from_numpy(mask), torch.from_numpy(rotate_mask), torch.from_numpy(hw_pad_t)).detach().numpy()
        else:
            cur = test_ov(onnx_path, qkv, mask, times=ov_test_times)
            ref = onnx_model(torch.from_numpy(qkv), torch.from_numpy(mask)).detach().numpy()

        if not np.allclose(ref, cur, atol=1e-3, rtol=1e-3):
            print(f'onnx failed:\n{ref=}\n{cur=}')
        print('')
print('done.')
