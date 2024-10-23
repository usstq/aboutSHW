import cl
import torch

def to_cl(input):
    if isinstance(input, cl.tensor):
        return input
    if input is None:
        return None
    if isinstance(input, torch.nn.parameter.Parameter):
        return cl.tensor(input.detach().numpy())
    if isinstance(input, torch.Tensor):
        return cl.tensor(input.detach().numpy())
    assert("to_cl(): Unexpected input ", input)

def to_torch(input):
    return torch.from_numpy(input.numpy())

