from common.train_wrappers import wrap
from common.env_wrappers import ColorObservationWrapper
import os
import gym
from functools import reduce
from math import sqrt
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c.a2c_keras2 import A2CTrainer
import numpy as np
from gym.wrappers import TimeLimit
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])


@register_trainer('breakout-a2c-keras', max_time_steps = 10e6, validation_period = None,  episode_log_interval = 10, save = False)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = 16
        self.n_steps = 5
        self.total_timesteps = 10e6
        self.gamma = .99

def default_args():
    return dict(
        env_kwargs = 'BreakoutNoFrameskip-v4',
        model_kwargs = dict()
    )