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
from a2c.a2c_gradient_accumulation import A2CTrainer
from a2c.a2c import A2CAgent
import numpy as np

from graph.env import OrientedGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
from common.env_wrappers import FrameStack





register_agent('kitchen-a2c-last-frames')(A2CAgent)
@register_trainer('kitchen-a2c-last-frames', max_time_steps = 1000000, validation_period = 100,  episode_log_interval = 10, saving_period = 10000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = 8
        self.n_steps = 5
        self.n_timeframes = 4
        self.total_timesteps = 1000000
        self.gamma = 1.0

        self._last_figure_draw = 0

    def create_model(self, action_space_size, **kwargs):
        inputs = [Input(batch_shape = (None, None, 224,224,3 * self.n_timeframes))]
        resnet = TimeDistributed(ResNet50(include_top=False, weights='imagenet', pooling=None))

        resnet_outputs = []
        for i in range(self.n_timeframes):
            single_input = TimeDistributed(Lambda(lambda x: x[..., i * 3:(i+1) * 3], output_shape = (224,224,3,)))(inputs[0])
            single_output = resnet(single_input)
            resnet_outputs.append(single_output)

        model = Concatenate()(resnet_outputs)
        model = TimeDistributed(Conv2D(64, 1, activation = 'relu'))(model)
        model = TimeDistributed(Flatten())(model)
        model = TimeDistributed(Dense(256, activation = 'relu'))(model)
        policy = TimeDistributed(Dense(256, activation = 'relu'))(model)
        policy = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01), activation = 'sigmoid'))(policy)
        value = TimeDistributed(Dense(256, activation = 'relu'))(model)
        value = TimeDistributed(Dense(1, activation = None, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01)))(value)
        model = Model(inputs = inputs, outputs = [policy, value])
        return model

def default_args():
    with open('./scenes/kitchen-224.pkl', 'rb') as f:
        graph = load_graph(f)

    env = lambda: FrameStack(TimeLimit(OrientedGraphEnv(graph, (0,4)), max_episode_steps = 100), 4)
    return dict(
        env_kwargs = env,
        model_kwargs = dict(action_space_size = 4)
    )