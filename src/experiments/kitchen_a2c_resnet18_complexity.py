import os
import gym
from functools import reduce
import numpy as np
from gym.wrappers import TimeLimit

import torch
from torch import nn

from a2c.model import TimeDistributedModel, TimeDistributed, Flatten
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c import A2CAgent, A2CTrainerDynamicBatch as A2CTrainer

from model.resnet import resnet50, resnet18
from graph.env import OrientedGraphEnv
from graph.util import load_graph

from common.schedules import LinearSchedule

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


@register_trainer(max_time_steps = 1000000, validation_period = 100,  episode_log_interval = 10, validation_episodes = 10, saving_period = 100000)
class Trainer(A2CTrainer):
    def __init__(self, max_time_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.num_steps = 5
        self.gamma = 0.99

        self._last_figure_draw = 0
        self.scene_complexity = LinearSchedule(0.1, 1.0, max_time_steps)
        self._last_complexity_update = -100000

    def process(self, *args, **kwargs):
        if self._global_t - self._last_complexity_update > 1000: 
            self.env.unwrapped.rpc_unwrapped('set_complexity', self.scene_complexity)
            self._last_complexity_update = self._global_t
        return super().process(*args, **kwargs)

    def create_env(self, env):
        graph = env
        factory = lambda: TimeLimit(OrientedGraphEnv(graph, graph.goal), max_episode_steps = 100)

        return super().create_env(dict(
            id = factory,
            num_frame_stack = 4
        ))

    def create_model(self, num_frame_stack, **kwargs):
        return create_model(num_frame_stack)

def default_args():
    with open('./scenes/kitchen-224.pkl', 'rb') as f:
        graph = load_graph(f)
    
    graph.goal = (0, 4)
    return dict(
        env_kwargs = graph,
        model_kwargs = dict(num_frame_stack = 4)
    )