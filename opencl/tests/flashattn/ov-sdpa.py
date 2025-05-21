import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(0)

device = "cpu"

class SdpaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.k_proj.weight[1:, :] = 0
            self.v_proj.weight[1:, :] = 0

    def forward(self, hidden_states, attention_mask):
        seq_length = hidden_states.shape[0]
        
        #q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        with_sdpa_fix = True
        if with_sdpa_fix:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        print("[q]:", q.shape)
        print("[k]:", k.shape)
        print("[v]:", v.shape)
        print("[attention_mask]:", attention_mask.shape)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attention_mask, dropout_p=0.0
        )
        if with_sdpa_fix:
            attn_output = attn_output.squeeze(0)
        #attn_output = attn_output.transpose(0, 1)

        print("===", attn_output.shape)
        return attn_output


seq_length = 8192
attention_mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.float32) + torch.finfo(torch.float16).min

print("[attention_mask]:", attention_mask.shape)

num_heads = 16
head_size = 80
hidden_size = num_heads * head_size
model = SdpaAttention(hidden_size, num_heads)
hidden_states = torch.ones([seq_length, hidden_size], device=device, dtype=torch.float32)
print("[hidden_states]:", hidden_states.shape)
ref = model(hidden_states, attention_mask)
ref = ref.detach().numpy()


import openvino

ovm = openvino.convert_model(model, example_input=[hidden_states, attention_mask], verbose=True)
# openvino.serialize(ovm, "ovm.xml")
ov_model = openvino.compile_model(ovm, device_name="GPU")

for i in range(30):
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