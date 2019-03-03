if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0,parentdir)

from common import register_trainer, make_trainer
from a2c import Trainer
from a2c.model import LSTMMultiLayerPerceptron
from common.env import make_vec_envs
import gym
import random
import numpy as np

import torch
import torch.nn as nn
from a2c.model import TimeDistributed
from a2c.core import forward_masked_rnn

class Model(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        
        self.rnn = nn.LSTM(action_space_size, 
            hidden_size = 16, 
            num_layers = 1,
            batch_first = True)

        self.actor = TimeDistributed(nn.Linear(16, action_space_size))
        self.critic = TimeDistributed(nn.Linear(16, 1))

    def initial_states(self, batch_size):
        return tuple([torch.zeros([1, batch_size, 16], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = forward_masked_rnn(inputs, masks, states, self.rnn.forward)
        return self.actor(features), self.critic(features), states

class TestLstm(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4,))
        self.random = random.Random()
        self.length = 2

    def seed(self, seed = None):
        self.random.seed(seed)

    def reset(self):
        self.time = 0
        self.chosen = self.random.randrange(self.action_space.n)
        return self.observe()

    def step(self, action):
        self.time += 1
        if self.time != self.length:
            return self.observe(), 0.0, False, dict()
        else:
            if action == self.chosen:
                return self.observe(), 1.0, True, dict()
            else:
                return self.observe(), 0.0, True, dict()

    def observe(self):
        r = np.zeros((self.action_space.n,), dtype = np.float32)
        if self.time == 0:
            r[self.chosen] = 1.0
        return r

if __name__ == '__main__':
    gym.register(
        id = 'lstm-v1',
        entry_point = 'experiments.tests.a2c_lstm:TestLstm'
    )

@register_trainer('test-a2c', episode_log_interval = 100, save = False)
class SomeTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = 'lstm-v1', model_kwargs = dict(), **kwargs)
        self.num_steps = 10
        self.allow_gpu = False

    def create_model(self, **model_kwargs):
        observation_space = self.env.observation_space
        action_space_size = self.env.action_space.n
        return Model(action_space_size)

if __name__ == '__main__':
    t = make_trainer('test-a2c')
    t.run()

