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
class StubBiAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_states: torch.Tensor, proj_key: torch.Tensor, proj_vision_values: torch.Tensor, proj_lang_values: torch.Tensor, attention_mask_l: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
        def _shape(tensor: torch.Tensor, seq_len: int, bsz: int):
            return tensor.view(bsz, seq_len, num_heads,
                            head_dim).transpose(1, 2).contiguous()
        bsz, tgt_len, _ = query_states.shape
        key_states = _shape(proj_key, -1, bsz)
        value_v_states = _shape(proj_vision_values, -1, bsz)
        value_l_states = _shape(proj_lang_values, -1, bsz)

        proj_shape = (bsz * num_heads, -1, head_dim)
        query_states = _shape(query_states, tgt_len,
                                   bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # if self.stable_softmax_2d:
        #     attn_weights = attn_weights - attn_weights.max()

        if True:
            # Do not increase -50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, min=-MAX_CLAMP_VALUE)
        if True:
            # Do not increase 50000, data type half has quite limited range
            attn_weights = torch.clamp(attn_weights, max=MAX_CLAMP_VALUE)

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (
            attn_weights_T -
            torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
        if True:
            # Do not increase -50000, data type half has quite limited range
            attn_weights_l = torch.clamp(attn_weights_l, min=-MAX_CLAMP_VALUE)
        if True:
            # Do not increase 50000, data type half has quite limited range
            attn_weights_l = torch.clamp(attn_weights_l, max=MAX_CLAMP_VALUE)

        # if attention_mask_v is not None:
        #     attention_mask_v = (
        #         attention_mask_v[:, None,
        #                          None, :].repeat(1, self.num_heads, 1,
        #                                          1).flatten(0, 1))
        #     attn_weights_l.masked_fill_(attention_mask_v, float('-inf'))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            # assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            # attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            # attention_mask = attention_mask.masked_fill(
            #     attention_mask == 0, -9e15)

            # if attention_mask_l.size() != (bsz, 1, tgt_len, src_len):
            #     raise ValueError('Attention mask should be of '
            #                      f'size {(bsz, 1, tgt_len, src_len)}')
            attn_weights = attn_weights.view(bsz, num_heads, tgt_len,
                                             src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * num_heads, tgt_len,
                                             src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        # attn_probs_v = F.dropout(
        #     attn_weights_v, p=self.dropout, training=self.training)
        # attn_probs_l = F.dropout(
        #     attn_weights_l, p=self.dropout, training=self.training)
        attn_probs_v = attn_weights_v
        attn_probs_l = attn_weights_l

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        attn_output_v = attn_output_v.view(bsz, num_heads, tgt_len,
                                           head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        #attn_output_v = attn_output_v.reshape(bsz, tgt_len, embed_dim)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, -1)

        attn_output_l = attn_output_l.view(bsz, num_heads, src_len,
                                           head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        # attn_output_l = attn_output_l.reshape(bsz, src_len, embed_dim)
        attn_output_l = attn_output_l.reshape(bsz, src_len, -1)
        return attn_output_v, attn_output_l

    @staticmethod
    def symbolic(g:torch.Graph, query_states: torch.Tensor, proj_key: torch.Tensor, proj_vision_values: torch.Tensor, proj_lang_values: torch.Tensor, attention_mask_l: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
        attr = g.op("Constant", value_t=torch.ByteTensor(list(bytes(
                    f'''
                        out_dt:0 
                        out_shape:0
                        out_dt1:1
                        out_shape1:1
                        type:BiAttention
                        NUM_HEADS:{num_heads}
                        NUM_KV_HEADS:{num_heads}
                        HEAD_SIZE:{head_dim}
                        SCALE:1
                        MAX_CLAMP_VALUE:{MAX_CLAMP_VALUE:.1f}
                    ''',
                    'utf8'))))
        attn_output_v, attn_output_l = g.op("Stub", query_states, proj_key, proj_vision_values, proj_lang_values, attention_mask_l, attr, outputs=2)
        return attn_output_v, attn_output_l

class MyModelONNX(nn.Module):
    def __init__(self, num_heads, head_dim):
        super(MyModelONNX, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, query_states: torch.Tensor, proj_key: torch.Tensor, proj_vision_values: torch.Tensor, proj_lang_values: torch.Tensor, attention_mask_l: torch.Tensor):
        v, l = StubBiAttention.apply(query_states, proj_key, proj_vision_values, proj_lang_values, attention_mask_l, self.num_heads, self.head_dim)
        return v + torch.ones([1], dtype=torch.float32), l + torch.ones([1], dtype=torch.float32)

def get_onnx_model(num_heads, head_dim):
    model = MyModelONNX(num_heads, head_dim)
    model.eval()
    return model

def export_onnx(onnx_model, query_shape, proj_key_shape, proj_vision_values_shape, proj_lang_values_shape, attention_mask_l_shape):
    print(f'convert onnx model to "{onnx_path}"...', end='')
    np.random.seed(324)
    torch.manual_seed(32)

    query = Variable(torch.randn(query_shape))
    key = Variable(torch.randn(proj_key_shape))
    vison_values = Variable(torch.randn(proj_vision_values_shape))
    lang_values = Variable(torch.randn(proj_lang_values_shape))
    attn_mask_l = Variable(torch.randn(attention_mask_l_shape))
    dynamic_axes={
        "query": {0: "batch_size", 1: "seq_len"},
        "key": {0: "batch_size", 1: "seq_len"},
    }
    with torch.no_grad():
        torch.onnx.export(onnx_model, (query, key, vison_values, lang_values, attn_mask_l), onnx_path,
                        input_names=['query', 'key', 'vison_values', 'lang_values', 'attn_mask_l'],
                        output_names=['attn_output_v', 'attn_output_l'],
                        dynamic_axes=dynamic_axes)

    print('done')

## Test
def test_ov(model_path, query, key, vison_values, lang_values, attn_mask_l, times=100):
    core = ov.Core()
    core.add_extension(ext_path)
    net = core.read_model(model_path)
    # ov.serialize(net, 'result.xml', 'result.bin')
    exec_net = core.compile_model(net, 'GPU')
    req = exec_net.create_infer_request()
    for _ in range(times):
        results = req.infer({
            'query': query,
            'key': key,
            'vison_values': vison_values,
            'lang_values': lang_values,
            'attn_mask_l': attn_mask_l
        })
    return results['attn_output_v'], results['attn_output_l']

ov_test_times = 1
print('testing...')

HEAD_NUM = 4
HEAD_DIM = 256
V_SEQ_LEN = 22223
L_SEQ_LEN = 18
query_shape = [1, V_SEQ_LEN, HEAD_NUM * HEAD_DIM]
proj_key_shape = [1, L_SEQ_LEN, HEAD_NUM * HEAD_DIM]
proj_vision_values_shape = [1, V_SEQ_LEN, HEAD_NUM * HEAD_DIM]
proj_lang_values_shape = [1, L_SEQ_LEN, HEAD_NUM * HEAD_DIM]
attention_mask_l_shape = [1, L_SEQ_LEN]

onnx_model = get_onnx_model(HEAD_NUM, HEAD_DIM)
export_onnx(copy.deepcopy(onnx_model), query_shape, proj_key_shape, proj_vision_values_shape, proj_lang_values_shape, attention_mask_l_shape)

query = np.random.random(query_shape).astype(np.float32)
key = np.random.random(proj_key_shape).astype(np.float32)
vison_values = np.random.random(proj_vision_values_shape).astype(np.float32)
lang_values = np.random.random(proj_lang_values_shape).astype(np.float32)
attn_mask_l = np.random.random(attention_mask_l_shape).astype(np.float32)
print(f'test "{onnx_path}"...', end='')
curs = test_ov(onnx_path, query, key, vison_values, lang_values, attn_mask_l, times=ov_test_times)
refs = onnx_model(torch.from_numpy(query),
                 torch.from_numpy(key),
                 torch.from_numpy(vison_values),
                 torch.from_numpy(lang_values),
                 torch.from_numpy(attn_mask_l)
                 )
refs = [x.detach().numpy() for x in refs]

assert(len(curs) == len(refs))

for idx, (ref, cur) in enumerate(zip(refs, curs)):
    if not np.allclose(ref, cur, atol=0.01, rtol=0.01):
        print(f'onnx failed at idx {idx}:\n{ref=}\n{cur=}')
        print(np.where(np.abs(ref - cur) > 0.01))
        break
print('done.')