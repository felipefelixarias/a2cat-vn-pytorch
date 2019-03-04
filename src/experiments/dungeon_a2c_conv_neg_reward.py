import os
import gym
from functools import reduce
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c import A2CTrainer, A2CAgent
from a2c.model import TimeDistributedModel, TimeDistributed, Flatten
import numpy as np
from torch import nn
import torch

from graph.env import SimpleGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
from experiments.util import display_q, display_policy_value
import matplotlib.pyplot as plt
import matplotlib

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])

def create_model():
    class _Model(TimeDistributedModel):
        def __init__(self):
            super().__init__()
            self.base = nn.Sequential(* \
                self.init_layer(nn.Conv2d(3, 32, 8, stride=4), activation = 'ReLU') + \
                [TimeDistributed(Flatten())]
            )

            self.actor = nn.Sequential(* \
                self.init_layer(nn.Linear(4 * 4 * 32, 64), activation = 'ReLU') + \
                self.init_layer(nn.Linear(64, 4), activation = None, gain = 0.01)
            )

            self.critic = nn.Sequential(* \
                self.init_layer(nn.Linear(4 * 4 * 32, 64), activation = 'ReLU') + \
                self.init_layer(nn.Linear(64, 1), activation = None, gain = 1.0)
            )

        def forward(self, inputs, masks, states):
            features = self.base(inputs)
            return self.actor(features), self.critic(features), states

    return _Model()

@register_agent('dungeon-a2c-conv-neg-reward')
class Agent(A2CAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_model(self):
        return create_model()

@register_trainer('dungeon-a2c-conv-neg-reward', max_time_steps = 100000000, validation_period = 1000,  episode_log_interval = 100, saving_period = 500000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = 32
        self.n_steps = 5
        self.total_timesteps = 100000000
        self.gamma = 1.0
        self.allow_gpu = False

        self._last_figure_draw = 0

    def create_model(self):
        return create_model()

    def _initialize(self):
        model = super()._initialize()
        def eval_pv(observation):
            # This function takes single batch of observations
            # Returns also single batch of returns
            observations = torch.from_numpy(observation).unsqueeze(1)
            mask = torch.ones([1, 1], dtype = torch.float32)
            policy_logits, value, _ = model.forward(observations, mask, [])
            action = policy_logits.argmax(dim = -1).squeeze(0).squeeze(0).detach().item()
            value = value.squeeze(0).squeeze(0).detach().item()
            return [action, value]
        pass

        self._eval_pv = eval_pv

    def save(self, path):
        super().save(path)
        plt.figure(self._figure.number)
        plt.savefig(os.path.join(path, 'policy_value.pdf'), format = 'pdf')
        plt.savefig(os.path.join(path, 'policy_value.eps'), format = 'eps')

    def run(self, *args, **kwargs):
        self._figure = plt.figure()
        self._figure_window = None
        return super().run(*args, **kwargs)

    def process(self, mode = 'train', context = dict(), **kwargs):
        res = super().process(mode = mode, context = context, **kwargs)
        if mode == 'train' and (self._global_t - self._last_figure_draw > 100000 or self._last_figure_draw ==0):
            self._figure.clf()
            display_policy_value(self, self._figure)
            self._figure.canvas.flush_events()
            if 'visdom' in context:
                viz = context.get('visdom')
                self._figure_window = viz.matplot(plt, win = self._figure_window, opts = dict(
                    title = 'policy, value'
                ))
            self._last_figure_draw = self._global_t

        return res

def default_args():
    size = (20, 20)
    with open('./scenes/dungeon-20-1.pkl', 'rb') as f:  #dungeon-%s-1.pkl' % size[0]
        graph = load_graph(f)

    env = lambda: TimeLimit(SimpleGraphEnv(graph, rewards=[0.0, -1.0, -1.0]), max_episode_steps = 50)
    #env.unwrapped.set_complexity(0.1)
    return dict(
        env_kwargs = dict(
            id = env,
            num_frame_stack = 1,
            allow_early_resets = True
        ),
        model_kwargs = dict()
    )