from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime.op import util as op_util
from openvino.runtime import opset8 as opset
from openvino.runtime.passes import Manager
import numpy as np
import sys, os

import torch
import torchPerfProfiler as tp
import time
 
IC=4096
 
Parameter = opset.parameter([-1,-1,IC], Type.f32, name = 'Parameter')
 
act = Parameter
weight = np.random.rand(IC, IC).astype(np.float32)
for i in range(4):
    fc = opset.matmul(act, weight.copy(), transpose_a=False,transpose_b=True, name = f'FC{i}')
    act = opset.relu(fc)
    print(f"created layer {i}...")

Result = opset.result(act.output(0), name='Result')
small_model = Model([Result], [Parameter], 'Model')

core = Core()
cmodel = core.compile_model(small_model, "CPU")

x = np.random.rand(1, 100, IC).astype(np.float32)
# warmup
r = cmodel(x)

# pure ov
with tp.PerfData("pure-ov", True) as p0:
    pure = []
    for i in range(10):
        beg = time.time()
        r = cmodel(x)
        end = time.time()
        pure.append(end - beg)

# mix ov and torch
mix = []
for i in range(10):
    with tp.PerfData("ovmodel", False) as p1:
        beg = time.time()
        r = cmodel(x)
        end = time.time()
        mix.append(end - beg)

    with tp.PerfData("softmax", True) as p2:
        for k in r:
            v = torch.from_numpy(r[k])
            v2 = torch.nn.functional.softmax(v, -1)
 
print('pure(ms):')
for i in pure:
    print(f'{i*1000:.2f} ', end='')
print('\nmix(ms):')
for i in mix:
    print(f'{i*1000:.2f} ', end='')
print('\ndone')