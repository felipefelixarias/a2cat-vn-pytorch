if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    sys.path.insert(0,parentdir)

import tensorflow as tf
from keras.layers import Dense, Input, TimeDistributed
from keras import layers
from keras.models import Model, Sequential
from keras import initializers
from common import register_trainer, make_trainer
from a2c.a2c import A2CTrainer
import gym
import random
import numpy as np

class TestLstm(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(4,))
        self.random = random.Random()
        self.length = 5

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
        r = np.zeros((self.action_space.n,))
        if self.time == 0:
            r[self.chosen] = 1.0
        return r

if __name__ == '__main__':
    gym.register(
        id = 'lstm-v1',
        entry_point = 'experiments.tests.a2c_lstm:TestLstm'
    )

@register_trainer('test-a2c', episode_log_interval = 100, save = False)
class SomeTrainer(A2CTrainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = dict(id = 'lstm-v1'), model_kwargs = dict(), **kwargs)
        self.n_steps = 10

    def create_model(self, **model_kwargs):
        observation_space = self.env.observation_space
        action_space_size = self.env.action_space.n
        n_envs = self.n_envs

        mask = Input(batch_shape=(n_envs, None), name = 'rnn_mask')
        state_placeholders = []
        output_states = []

        def LSTM(number_of_units, **kwargs):
            layer = layers.LSTM(number_of_units,
                return_sequences = True, 
                return_state = True,
                **kwargs)

            def call(model):
                n_plh = len(state_placeholders)
                states = [Input((number_of_units,), name = 'rnn_state_%s' % (n_plh + i)) for i in range(2)]
                model, hidden, cell = layer(model, states, mask = mask)
                output_states.extend([hidden, cell])
                state_placeholders.extend(states)
                return model
            return call

        input_placeholder = Input(batch_shape=(n_envs, None) + observation_space.shape)
        policy_latent = LSTM(32)(input_placeholder)
        value_latent = LSTM(32)(input_placeholder)
    
        policy_probs = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', activation='softmax', kernel_initializer = initializers.Orthogonal(gain=0.01)))(policy_latent)
        value = TimeDistributed(Dense(1, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain = 1.0)))(value_latent)

        model = Model(inputs = [input_placeholder, mask] + state_placeholders, outputs = [policy_probs, value] + output_states)
        return model

if __name__ == '__main__':
    t = make_trainer('test-a2c')
    t.run()

