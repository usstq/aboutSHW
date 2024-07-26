import torch
import torchPerfProfiler as tp
import time

logits = torch.zeros(1280, 32000, dtype=torch.float32)


p0 = tp.PerfData("all", False)

for i in range(10):
    # support context manager style
    with tp.PerfData(f"assign{i}", True) as p:
        logits = 1000
    
    time.sleep(0.01)

p0.finish()

print(dir(tp))