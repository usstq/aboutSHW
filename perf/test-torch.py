import torch
import torchPerfProfiler as tp

logits = torch.zeros(1280, 32000, dtype=torch.float32)


p0 = tp.PerfData("all")

for i in range(10):
    # support context manager style
    with tp.PerfData("assign") as p:
        logits = 1000

p0.finish()

print(dir(tp))