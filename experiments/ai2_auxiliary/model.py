import torch
import torch.nn as nn
import math

from deep_rl.model import TimeDistributed, Flatten, MaskedRNN
from models import GoalUnrealModel

class Unflatten(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(-1, *self.shape)

class GoalModelWithAuxiliary(GoalUnrealModel):
    def __init__(self, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)
        self._create_deconv_networks()
        self.deconv_cell_size = 4

    def _create_deconv_networks(self):
        self.deconv_depth = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 32 * 9 * 9),
            nn.ReLU(),
            Unflatten(32, 9, 9),
            nn.ConvTranspose2d(32, 1, kernel_size = 4, stride = 2),
        ))

        self.deconv_mask = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64 * 9 * 9),
            nn.ReLU(),
            Unflatten(64, 9, 9),
            nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2)
        ))

        self.deconv_mask_goal = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64 * 9 * 9),
            nn.ReLU(),
            Unflatten(64, 9, 9),
            nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2)
        ))


        self.deconv_depth.apply(self.init_weights)
        self.deconv_mask_goal.apply(self.init_weights)
        self.deconv_mask.apply(self.init_weights)
    
    def forward_deconv(self, *inputs):
        features, states = self._forward_base(*inputs)
        depth = self.deconv_depth(features)
        mask = self.deconv_mask(features)
        mask_goal = self.deconv_mask_goal(features)
        return (depth, mask, mask_goal), states