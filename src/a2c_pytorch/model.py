import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class TimeDistributed(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        batch_shape = args[0].size()[:2]
        args = [x.contiguous().view(-1, *x.size()[2:]) for x in args]
        results = self.inner.forward(*args)
        def reshape_res(x):
            return x.view(*(batch_shape + x.size()[1:]))

        if isinstance(results, list):
            return [reshape_res(x) for x in results]
        elif isinstance(results, tuple):
            return tuple([reshape_res(x) for x in results])
        else:
            return reshape_res(results)

class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        def init_layer(layer, activation = None, gain = None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain = gain)
            nn.init.zeros_(layer.bias.data)
            output = [layer]
            if activation is not None:
                output.append(getattr(nn, activation)())
            return output

        layers = []
        layers.extend(init_layer(nn.Conv2d(num_inputs, 32, 8, stride = 4), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(32, 64, 4, stride = 2), activation='ReLU'))
        layers.extend(init_layer(nn.Conv2d(64, 32, 3, stride = 1), activation='ReLU'))
        layers.append(Flatten())
        layers.extend(init_layer(nn.Linear(32 * 7 * 7, 512), activation='ReLU'))
        
        self.main = nn.Sequential(*layers)
        self.critic = init_layer(nn.Linear(512, 1))[0]
        self.policy_logits = init_layer(nn.Linear(512, num_outputs), gain = 0.01)[0]

    def forward(self, inputs):
        main_features = self.main.forward(inputs)
        policy_logits = self.policy_logits.forward(main_features)
        critic = self.critic.forward(main_features)
        return [policy_logits, critic]

    @property
    def output_names(self):
        return ['policy_logits', 'value']


def TimeDistributedCNN(num_inputs, num_outputs):
    inner = CNN(num_inputs, num_outputs)
    model = TimeDistributed(inner)
    model.output_names = property(lambda self: inner.output_names)
    _forward = model.forward
    model.forward = lambda inputs, masks, states: _forward(inputs)
    return model