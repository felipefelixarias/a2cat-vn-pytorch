from collections import namedtuple
import numpy as np

RecurrentModel = namedtuple('RecurrentModel', ['model', 'inputs','outputs', 'output_names', 'states_in', 'states_out', 'mask'])

def expand_recurrent_model(model):
    states_in = []
    pure_inputs = [x for x in model.inputs]
    
    mask = None
    for x in model.inputs:
        if 'rnn_state' in x.name:
            states_in.append(x)
            pure_inputs.remove(x)
        if 'rnn_mask' in x.name:
            mask = x
            pure_inputs.remove(x)

    pure_outputs = model.outputs[:-len(states_in)] if len(states_in) > 0 else model.outputs
    pure_output_names = model.output_names[:-len(states_in)] if len(states_in) > 0 else model.output_names
    states_out = model.outputs[-len(states_in):] if len(states_in) > 0 else []

    assert len(states_in) == 0 or mask is not None
    return RecurrentModel(model, pure_inputs, pure_outputs, pure_output_names, states_in, states_out, mask)

def create_initial_state(n_envs, state_in):
    return [np.zeros((n_envs,) + tuple(x.shape[1:])) for x in state_in]