#!/usr/bin/python3
import cl
import numpy as np
import sys

print(dir(cl))

# Open the file in read mode
with open("sdpa.cl", "r") as file:
    # Read the entire file content into a string
    sdpa_src = file.read()

print(sdpa_src[:1000])

# [kernel] sdpa_opt_multi_tokens_sa args 10 :
#         0 int* shape_info
#         1 half* query_input
#         2 half* key_input
#         3 half* value_input
#         4 half* attn_mask
#         5 half* scale
#         6 half* output
#         7 float* exp_sums
#         8 float* max_logits
#         9 half* tmp_out

sdpa_kernel = cl.kernels(sdpa_src, "")

SUBGROUP_SIZE=16
GQA_SIZE=4
B=1
H=28
Lq=8416
Lk=8416
S=128

shape_info = [
    # // input0 query
	B, H, 1, 1, 1, 1, Lq, S,
    # // input1 key
    B, int(H/GQA_SIZE), 1, 1, 1, 1, Lk, S, 0, 0,
    # // input2 value
    B, int(H/GQA_SIZE), 1, 1, 1, 1, Lk, S, 0, 0,
    #  input3 attn_mask
    1, 1, 1, 1, 1, 1, Lq, Lk,
    #  input4 scale
    #  output
    B, H, 1, 1, 1, 1, Lq, S
]
print(f"len(shape_info)={len(shape_info)}, shape_info={shape_info}")
shape_info_input = cl.tensor(np.array(shape_info).astype(np.int32))

query = np.random.rand(B, H, Lq, S).astype(np.float16)
key = np.ones([B, H, Lk, S], dtype=np.float16)
value = np.ones([B, H, Lk, S], dtype=np.float16)
query_input = cl.tensor(query)
key_input = cl.tensor(key)
value_input = cl.tensor(value)

scale_input = cl.tensor(np.ones([1], dtype=np.float16))
attn_mask = cl.tensor(np.zeros([B, H, Lq, Lk], dtype=np.float16))

exp_sums = cl.tensor(np.zeros([4], dtype=np.float32))
max_logits = cl.tensor(np.zeros([4], dtype=np.float32))
tmp_out = cl.tensor(np.zeros([2], dtype=np.float16))

output = cl.tensor(np.zeros([B, H, Lq, S], dtype=np.float16))

GWS = [B*H, int(Lq/SUBGROUP_SIZE), S]
LWS = [1, 1, S]

print(f"query_input.shape={query_input.shape}, key_input.numpy()={query_input.numpy()[0,0,0,:]}")
print(f"GWS={GWS}, LWS={LWS}")
sdpa_kernel.enqueue("sdpa_opt_multi_tokens_6761455398808095608_0_0__sa", GWS, LWS,
                    shape_info_input, query_input, key_input, value_input, attn_mask, scale_input, output,
                    exp_sums, max_logits, tmp_out)
print(f"output.shape={output.shape}, output.numpy()={output.numpy()[0,0,0,:]}")

