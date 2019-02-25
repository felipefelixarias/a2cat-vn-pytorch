from common.train_wrappers import wrap

import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU, Flatten, Conv2D, TimeDistributed
from keras.models import Model
from keras import initializers
import keras.backend as K
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c.a2c import A2CTrainer, A2CAgent
import numpy as np

from graph.env import SimpleGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
from experiments.util import display_q, display_policy_value
import matplotlib.pyplot as plt
import matplotlib

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])


register_agent('dungeon-a2c-conv-neg-reward')(A2CAgent)
@register_trainer('dungeon-a2c-conv-neg-reward', max_time_steps = 1000000, validation_period = 1000,  episode_log_interval = 100, saving_period = 100000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.n_envs = 32
        self.n_steps = 5
        self.total_timesteps = 1000000
        self.gamma = 1.0

    def create_model(self, action_space_size, **kwargs):
        inputs = [Input(batch_shape = (None, None, 20,20,3))]
        model = inputs[0]
        model = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation = 'relu'))(model)
        model = TimeDistributed(Flatten())(model)
        policy = TimeDistributed(Dense(64, activation = 'relu'))(model)
        policy = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01), activation = 'sigmoid'))(policy)
        value = TimeDistributed(Dense(64, activation = 'relu'))(model)
        value = TimeDistributed(Dense(1, activation = None, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01)))(value)

        model = Model(inputs = inputs, outputs = [policy, value])
        return model

    def run(self, *args, **kwargs):
        plt.ion()

        self._figure = plt.figure()
        plt.show()

        return super().run(*args, **kwargs)

    def process(self, mode = 'train', **kwargs):
        res = super().process(mode = mode, **kwargs)
        if mode == 'train':
            display_policy_value(self._figure, self.)

        return res

def default_args():
    size = (20, 20)
    with open('./scenes/dungeon-20-1.pkl', 'rb') as f:  #dungeon-%s-1.pkl' % size[0]
        graph = load_graph(f)

    env = lambda: TimeLimit(SimpleGraphEnv(graph, graph.goal, rewards=[0.0, -1.0, -1.0]), max_episode_steps = 50)
    #env.unwrapped.set_complexity(0.1)
    return dict(
        env_kwargs = env,
        model_kwargs = dict(action_space_size = 4)
    )