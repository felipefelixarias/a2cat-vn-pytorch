from common.train_wrappers import wrap
import os
import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU, Flatten, Conv2D, TimeDistributed
from keras.models import Model
from keras import initializers
from keras.applications.resnet50 import ResNet50
import keras.backend as K
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c.a2c import A2CTrainer, A2CAgent
import numpy as np

from graph.env import OrientedGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])


register_agent('kitchen-a2c')(A2CAgent)
@register_trainer('kitchen-a2c', max_time_steps = 10000000, validation_period = 1000,  episode_log_interval = 100, saving_period = 100000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = 32
        self.n_steps = 5
        self.total_timesteps = 10000000
        self.gamma = 0.95

        self._last_figure_draw = 0

    def create_model(self, action_space_size, **kwargs):
        inputs = [Input(batch_shape = (None, None, 224,224,3))]

        resnet = ResNet50(include_top=False, weights='imagenet', pooling=None)
        model = TimeDistributed(resnet)(inputs[0])
        model = TimeDistributed(Flatten())(model)
        policy = TimeDistributed(Dense(256, activation = 'relu'))(model)
        policy = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01), activation = 'sigmoid'))(policy)
        value = TimeDistributed(Dense(256, activation = 'relu'))(model)
        value = TimeDistributed(Dense(1, activation = None, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01)))(value)
        model = Model(inputs = inputs, outputs = [policy, value])
        return model

def default_args():
    with open('./scenes/kitchen-224.pkl', 'rb') as f:
        graph = load_graph(f)

    env = lambda: TimeLimit(OrientedGraphEnv(graph, (0,4)), max_episode_steps = 100)
    #env.unwrapped.set_complexity(0.1)
    return dict(
        env_kwargs = env,
        model_kwargs = dict(action_space_size = 4)
    )