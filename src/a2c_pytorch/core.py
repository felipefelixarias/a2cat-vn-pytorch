import torch
import numpy as np
from collections import namedtuple

RolloutBatch = namedtuple('RolloutBatch', ['observations', 'returns','actions', 'masks', 'states'])

def to_tensor(value, device):
    if isinstance(value, RolloutBatch):
        return RolloutBatch(*to_tensor(list(value), device))

    elif isinstance(value, list):
        return [to_tensor(x, device) for x in value]

    elif isinstance(value, tuple):
        return tuple(to_tensor(list(value), device))

    elif isinstance(value, dict):
        return {key: to_tensor(val) for key, val in value.items()}

    elif isinstance(value, np.ndarray):
        if value.dtype == np.bool:
            value = value.astype(np.float32)

        return torch.from_numpy(value).to(device)
    elif torch.is_tensor(value):
        return value.to(device)
    else:
        raise Exception('%s Not supported'% type(value))

def to_numpy(tensor):
    if isinstance(tensor, tuple):
        return tuple(to_numpy(list(tensor)))
    elif isinstance(tensor, list):
        return [to_numpy(x) for x in tensor]
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif isinstance(tensor, float) or isinstance(tensor, int):
        return tensor
    else:
        raise Exception('Not supported type %s' % type(tensor))

def pytorch_call(device):
    def wrap(function):
        def call(*args, **kwargs): 
            results = function(*to_tensor(args, device), **to_tensor(kwargs, device))
            return to_numpy(results)
        return call
    return wrap