import torch
import torch.nn.functional as F
torch.manual_seed(0)
torch.set_printoptions(linewidth=1024)

batch = 1
q_len = 2
kv_len = 80
low = -8
high = 8
input = torch.randint(low, high, [batch, q_len, kv_len]).to(dtype=torch.float32)/high
ref = F.softmax(input, -1)

print(input, input.shape)
print(ref, ref.shape, ref.sum(-1))


kv_block = 4

res = torch.full([batch, q_len, kv_len], 0, dtype=torch.float32)
cur_max = torch.full([batch, q_len, 1], torch.finfo(torch.float32).min, dtype=torch.float32)
cur_sum = torch.full([batch, q_len, 1], 0, dtype=torch.float32)
for i0 in range(0, kv_len, kv_block):
    i1 = i0 + kv_block
    x2 = input[:,:,i0:i1]
    max2 = x2.max(-1, True).values
    # print(blk_max.shape)
    all_max = torch.maximum(cur_max, max2)

    # this step undate all existing attention-weights
    res[:, :, :i0] *= torch.exp(cur_max - all_max)
    res[:, :, i0:i1] = torch.exp(x2 - all_max)
    cur_sum += res[:,:,i0:i1].sum(-1, keepdim=True)
    cur_max = all_max

res = res / cur_sum

print(torch.allclose(ref, res))