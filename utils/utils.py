import numpy as np
import torch
from torch import nn

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def to_numpy(ft):
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        return None

def norm2d(type):
    if type == 'batch':
        return nn.BatchNorm2d
    elif type == 'instance':
        return nn.InstanceNorm2d
    elif type == 'none':
        return nn.Identity
    else:
        raise ValueError("Invalid normalization type: ", type)
