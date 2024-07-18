import torch
import torchPerfProfiler as tp

logits = torch.zeros(1280, 32000, dtype=torch.float32)

for i in range(10):
    tp.start("assign", 0)
    logits = 1000
    tp.finish()

print(dir(tp))