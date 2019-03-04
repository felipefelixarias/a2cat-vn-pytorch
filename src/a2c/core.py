import torch
import numpy as np
from collections import namedtuple
from math import ceil

RolloutBatch = namedtuple('RolloutBatch', ['observations', 'returns','actions', 'masks', 'states'])
KeepTensor = namedtuple('KeepTensor', ['data'])

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
    if isinstance(tensor, KeepTensor):
        return tensor.data
    elif isinstance(tensor, tuple):
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

def detach_all(data):
    if isinstance(data, list):
        return [detach_all(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(detach_all(list(data)))
    else:
        return data.detach()

def forward_masked_rnn(inputs, masks, states, forward_rnn):
    def mask_states(states, mask):
        if isinstance(states, tuple):
            return tuple(mask_states(list(states), mask))
        elif isinstance(states, list):
            return [mask_states(x, mask) for x in states]
        else:
            return states * mask.view(1, -1, 1)

    has_zeros = ((masks[:, 1:] == 0.0) \
        .any(dim=0)
        .nonzero()
        .squeeze()
        .cpu())

    T = masks.size()[1]

    # +1 to correct the masks[1:]
    if has_zeros.dim() == 0:
        # Deal with scalar
        has_zeros = [has_zeros.item() + 1]
    else:
        has_zeros = (has_zeros + 1).numpy().tolist()

    # add t=0 and t=T to the list
    has_zeros = [0] + has_zeros + [T]
    outputs = []

    for i in range(len(has_zeros) - 1):
        # We can now process steps that don't have any zeros in masks together!
        # This is much faster
        start_idx = has_zeros[i]
        end_idx = has_zeros[i + 1]
        
        rnn_scores, states = forward_rnn(
            inputs[:, start_idx:end_idx],
            mask_states(states, masks[:, start_idx])
        )

        outputs.append(rnn_scores)

    # assert len(outputs) == T
    # x is a (N, T, -1) tensor
    outputs = torch.cat(outputs, dim=1)
    
    # flatten
    return outputs, states


def minibatch_gradient_update(inputs, compute_loss_fn, zero_grad_fn, optimize_fn, chunks = 1):
    def split_inputs(inputs, chunks, axis):
        if isinstance(inputs, list):
            return list(map(list, split_inputs(tuple(inputs), chunks, axis)))
        elif isinstance(inputs, tuple):
            return list(zip(*[split_inputs(x, chunks, axis) for x in inputs]))
        else:
            return torch.chunk(inputs, chunks, axis)

    # Split inputs to chunks
    if chunks == 1:
        zero_grad_fn()
        losses = compute_loss_fn(*inputs)
        losses[0].backward()
        optimize_fn()
        return [x.item() for x in losses]


    main_inputs = split_inputs(inputs[:-1], chunks, 0)
    states_inputs = split_inputs(inputs[-1:], chunks, 1)
    inputs = [x + y for x, y in zip(main_inputs, states_inputs)]

    # Zero gradients
    zero_grad_fn()
    total_results = None
    for minibatch in inputs:
        results = compute_loss_fn(*minibatch)
        results = list(map(lambda x: x / minibatch[1].size(0), results))
        loss = results[0]
        loss.backward()

        if total_results is None:
            total_results = results
        else:
            total_results = list(map(lambda x,y: x + y, total_results, results))

    # Optimize
    optimize_fn()
    return [x.item() for x in total_results]

class AutoBatchSizeOptimizer:
    def __init__(self, zero_grad_fn, compute_loss_fn, apply_gradients_fn):
        self._chunks = 1
        self.zero_grad_fn = zero_grad_fn
        self.compute_loss_fn = compute_loss_fn
        self.apply_gradients_fn = apply_gradients_fn

    def optimize(self, inputs):
        batch_size = inputs[1].size()[0]
        results = None
        while results is None:
            try:
                results = minibatch_gradient_update(inputs, self.compute_loss_fn, self.zero_grad_fn, self.apply_gradients_fn, self._chunks)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    # We will try to recover from this error
                    torch.cuda.empty_cache()
                    print('Training failed with mini-batch size %s' % ceil(float(batch_size) / float(self._chunks)))
                    print('Trying to split the minibatch (%s -> %s)' % (self._chunks, self._chunks + 1))
                    print('Resuming training')
                    self._chunks += 1
                else:
                    raise e

        return results