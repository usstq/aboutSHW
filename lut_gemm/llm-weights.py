
num_hidden_layers = 26
hidden_size = 3200
intermediate_size = 8640

qkvo_params = hidden_size * hidden_size * 4      # qkvo
mlp_params = hidden_size * intermediate_size * 3 # gate/up/down

def num_param_to_bytes(num_params):
    return num_params/3 * 5 / 8 # TL2 kernel
    #return num_params * 2 / 8 # I2S kernel

param_bytes = num_param_to_bytes(num_hidden_layers * (qkvo_params + mlp_params))

mem_bandwidth_bytes_per_second = 88e9 # 8 Pcores
mem_bandwidth_bytes_per_second = 70e9 # 4 Pcores
mem_bandwidth_bytes_per_second = 35e9 # 1 Pcores
print(f"assuming DDR bandwidth : {mem_bandwidth_bytes_per_second*1e-9:.1f} GB/s")
print(f"TL2 compressed weight size: {param_bytes*1e-9:.2f} GB = {param_bytes:.1f} bytes, minimal latency bound: {param_bytes/mem_bandwidth_bytes_per_second*1e3:.3f} ms")

q_time = num_param_to_bytes(hidden_size * hidden_size)/mem_bandwidth_bytes_per_second
g_time = num_param_to_bytes(hidden_size * intermediate_size)/mem_bandwidth_bytes_per_second

print(f"q_time = {q_time*1e3:.3f} ms")
print(f"g_time = {g_time*1e3:.3f} ms")
