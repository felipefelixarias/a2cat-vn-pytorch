from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, Input, Lambda
from keras.initializers import Orthogonal
from math import sqrt
import numpy as np
import tensorflow as tf

from tensorflow.distributions import Categorical

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init

class Policy:
    def __init__(self, obs_shape, action_space):
        self.model, self.distribution = CNNBase(action_space.n, obs_shape)

    def act(self, deterministic=False):
        if deterministic:
            action = self.distribution.mode()
        else:
            action = self.distribution.sample()

        action_log_probs = self.distribution.log_probs(action)
        return value, action, action_log_probs

    @property
    def value(self):
        return self.model.outputs[1]

    @property
    def entropy(self):
        return tf.reduce_mean(self.distribution.entropy())

    @property
    def inputs(self):
        return self.model.inputs

    def action_log_progs(self, action):
        return self.distribution.log_probs(action)


def CNNBase(num_actions, input_shape):
        initializer = Orthogonal(gain = sqrt(2))

        inputs = [Input(input_shape)]
        model = inputs[0]
        model = Lambda(lambda x: x / 255.0)(model)
        model = Conv2D(32, 8, strides = 4, kernel_initializer = initializer, bias_initialized = 'zeros', activation = 'relu')(model)
        model = Conv2D(64, 4, strides = 6, kernel_initializer = initializer, bias_initialized = 'zeros', activation = 'relu')(model)
        model = Conv2D(32, 3, strides = 1, kernel_initializer = initializer, bias_initialized = 'zeros', activation = 'relu')(model)
        model = Flatten()(model)
        model = Dense(512, kernel_initializer = initializer, bias_initialized = 'zeros', activation = 'relu')(model)

        critic_linear = Dense(1, kernel_initializer = Orthogonal(1.0), bias_initializer = 'zeros')(model)
        policy = Dense(num_actions, kernel_initializer = Orthogonal(0.01), bias_initializer = 'zeros')(model)
        model = Model(inputs = inputs, outputs = [policy, critic_linear])
        model.output_names = ['policy_logits', 'critic']
        distribution = Categorical(logits = self.model.outputs[0])
        return model, distribution