from common.train_wrappers import wrap
import os
import gym
from functools import reduce
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c import A2CAgent, A2CTrainerDynamicBatch as A2CTrainer
import numpy as np

import torch
from torch import nn

from graph.env import OrientedGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
from common.env_wrappers import FrameStack
from a2c.model import TimeDistributedModel, TimeDistributed, Flatten
from model.resnet import resnet50, resnet18


def create_model(num_steps):
    class _Model(TimeDistributedModel):
        def __init__(self):
            super().__init__()
            self.resnet = TimeDistributed(resnet18(pretrained = True))

            self.main_merged = nn.Sequential(*
                self.init_layer(nn.Conv2d(num_steps * 512, 64, 1), activation = 'ReLU') + \
                [TimeDistributed(Flatten())] + \
                self.init_layer(nn.Linear(64 * 7 * 7, 256), activation = 'ReLU')
            )

            self.actor = nn.Sequential(*
                self.init_layer(nn.Linear(256, 256), activation = 'ReLU') + \
                self.init_layer(nn.Linear(256, 4), gain = 0.01)
            )

            self.critic = nn.Sequential(*
                self.init_layer(nn.Linear(256, 256), activation = 'ReLU') + \
                self.init_layer(nn.Linear(256, 1), gain = 1.0)
            )
        
        def forward(self, inputs, masks, states):
            streams = torch.split(inputs, 3, dim = 2)
            streams = [self.resnet(x) for x in streams]
            features = torch.cat(streams, dim = 2)
            features = self.main_merged(features)
            return self.actor(features), self.critic(features), states
    return _Model()


@register_trainer(max_time_steps = 1000000, validation_period = 100,  episode_log_interval = 10, saving_period = 10000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 5
        self.total_timesteps = 1000000
        self.gamma = 1.0

        self._last_figure_draw = 0

    def create_model(self, num_frame_stack, **kwargs):
        return create_model(num_frame_stack)

def default_args():
    with open('./scenes/kitchen-224.pkl', 'rb') as f:
        graph = load_graph(f)

    env = lambda: TimeLimit(OrientedGraphEnv(graph, (0,4), rewards=[0.0, -1.0, -1.0]), max_episode_steps = 100)
    return dict(
        env_kwargs = dict(
            id = env,
            num_frame_stack = 4
        ),
        model_kwargs = dict(num_frame_stack = 4)
    )