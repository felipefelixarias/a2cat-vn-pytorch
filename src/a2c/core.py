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


def minibatch_gradient_update(self, inputs, compute_loss_fn, zero_grad_fn, optimize_fn, minibatch_size = None):
    def split_inputs(inputs, chunks, axis):
        if isinstance(inputs, list):
            return list(map(list, split_inputs(tuple(inputs), chunks, axis)))
        elif isinstance(inputs, tuple):
            return list(zip(*[split_inputs(x, chunks, axis) for x in inputs]))
        else:
            return torch.chunk(inputs, chunks, axis)                

    batch_size = inputs[1].size()[0]
    if minibatch_size is None:
        minibatch_size = batch_size

    # Compute chunks
    chunks = int(ceil(float(batch_size) / float(minibatch_size)))

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


    minibatch_size = int(ceil(float(batch_size) / float(chunks)))
    return minibatch_size, [x.item() for x in total_results]
