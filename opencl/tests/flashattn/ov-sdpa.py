import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(0)

device = "cpu"

class SdpaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads) -> None:
        super().__init__()
        head_size = hidden_size // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.enable_gqa = self.num_heads > self.num_kv_heads
        self.q_proj = nn.Linear(hidden_size, head_size * num_heads, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_size * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_size * num_kv_heads, bias=False)
        with torch.no_grad():
            self.k_proj.weight[1:, :] = 0
            self.v_proj.weight[1:, :] = 0

    def forward(self, hidden_states, attention_mask):
        batch, seq_length, n_states = hidden_states.shape
        
        #q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self.q_proj(hidden_states).reshape(batch, seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(batch, seq_length, self.num_kv_heads, -1)
        v = self.v_proj(hidden_states).reshape(batch, seq_length, self.num_kv_heads, -1)

        q = q.transpose(2, 1)
        k = k.transpose(2, 1)
        v = v.transpose(2, 1)

        print("  [q]:", q.shape)
        print("  [k]:", k.shape)
        print("  [v]:", v.shape)
        print("  [attention_mask]:", attention_mask.shape)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attention_mask, dropout_p=0.0
        )

        attn_output = attn_output.transpose(2, 1).reshape(batch, seq_length, n_states)

        print("  [attn_output]:", attn_output.shape)
        return attn_output

class MLSdpaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SdpaAttention(hidden_size, num_heads, num_kv_heads) for _ in range(4)])

    def forward(self, hidden_states, attention_mask):
        for blk in self.blocks:
            hidden_states = blk.forward(hidden_states, attention_mask)
        return hidden_states

num_heads = 28
num_kv_heads = 28
head_size = 128
hidden_size = num_heads * head_size

model = MLSdpaAttention(hidden_size, num_heads, num_kv_heads)

seq_length = 1024
attention_mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.float32) + torch.finfo(torch.float16).min
hidden_states = torch.ones([1, seq_length, hidden_size], device=device, dtype=torch.float32)

print("[hidden_states]:", hidden_states.shape)
print("[attention_mask]:", attention_mask.shape)
ref = model(hidden_states, attention_mask)
ref = ref.detach().numpy()


import openvino
print("openvino.convert_model ...")
ovm = openvino.convert_model(model,
                             input=[("hidden_states",[-1, -1, hidden_size]),("attention_mask",[-1, -1, -1])],
                             example_input=[hidden_states, attention_mask],
                             verbose=True)
#openvino.serialize(ovm, "ovm.xml")
print("openvino.compile_model ...")
ov_model = openvino.compile_model(ovm, device_name="GPU")

seq_length = 8192
attention_mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.float32) + torch.finfo(torch.float16).min
hidden_states = torch.ones([1, seq_length, hidden_size], device=device, dtype=torch.float32)

print("openvino.infer ...")
for i in range(10):
    output = ov_model([hidden_states.numpy(), attention_mask.numpy()])

res = output[0]

np.set_printoptions(linewidth=1024)
if not np.allclose(ref, res, atol=1e-3, rtol=1e-3):
    #print(ref - res)
    print(ref[0,0:2,:])
    print(res[0,0:2,:])
else:
    print(ref[0,0:2,:8])
    print(res[0,0:2,:8])
    print("PASS")