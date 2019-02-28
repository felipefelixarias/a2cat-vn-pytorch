from common.train_wrappers import wrap
from common.env_wrappers import ColorObservationWrapper
import os
import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU, Flatten, Conv2D, TimeDistributed
from keras.models import Model
from keras import initializers
import keras.backend as K
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c.a2c import A2CTrainer, A2CAgent
import numpy as np
from gym.wrappers import TimeLimit
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])


register_agent('pong-a2c')(A2CAgent)
@register_trainer('pong-a2c', max_time_steps = 10e6, validation_period = 100,  episode_log_interval = 10, saving_period = 500000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = 16
        self.n_steps = 5
        self.total_timesteps = 10e6
        self.gamma = .99

    def create_model(self, action_space_size, **kwargs):
        inputs = [Input(batch_shape = (None, None, 84,84,1))]
        model = inputs[0]
        model = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation = 'relu'))(model)
        model = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation = 'relu'))(model)
        model = TimeDistributed(Conv2D(32, 3, strides=1, activation = 'relu'))(model)
        model = TimeDistributed(Flatten())(model)
        model = TimeDistributed(Dense(512, kernel_initializer = initializers.Orthogonal(), bias_initializer = 'zeros', activation = 'relu'))(model)
        policy = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01), activation = 'sigmoid'))(model)
        value = TimeDistributed(Dense(1, activation = None, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01)))(model)

        model = Model(inputs = inputs, outputs = [policy, value])
        model.output_names = ['policy', 'value']
        return model

def default_args():
    return dict(
        env_kwargs = lambda: wrap_deepmind(make_atari('PongNoFrameskip-v4'), scale=True),
        model_kwargs = dict(action_space_size = 4)
    )