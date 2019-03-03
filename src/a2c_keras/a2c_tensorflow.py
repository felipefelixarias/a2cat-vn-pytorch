from abc import abstractclassmethod
from collections import namedtuple
import numpy as np
from math import sqrt
from common.train import AbstractTrainer, SingleTrainer
from common import MetricContext

import gym
from common.vec_env import SubprocVecEnv
import tempfile










import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
from keras.layers import Input, TimeDistributed, Conv2D, Flatten, Dense
from keras import initializers
from keras import backend as K
from keras.models import Model

import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.train import RMSPropOptimizer as RMSprop


from a2c.storage import RolloutStorage
from common.env import VecTransposeImage, make_vec_envs

device = 'cpu:0'

def time_distributed(inner, inputs):
    batch_shape = list(inputs.get_shape()[:2])
    inputs = inputs.reshape([-1,] + inputs.get_shape()[2:])

    res = inner(inputs)
    return [x.reshape(batch_shape + x.get_shape()[1:] for x in res]

def create_model(action_space_size, **kwargs):
    inputs = [tf.placeholder(tf.float32, (None, None, 84,84, 4), 'observations')]
    layer_kwargs = dict(activation = 'relu', kernel_initializer = initializers.Orthogonal(gain = sqrt(2)), bias_initializer = 'zeros')

    model = inputs[0]
    def inner(model):
        model = tf.layers.conv2d(model, 32, 8, strides = 4, kernel_initializer=tf.initiali)

    outputs = time_distributed(lambda model:)
    model = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), **layer_kwargs))(model)
    model = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), **layer_kwargs))(model)
    model = TimeDistributed(Conv2D(32, 3, strides=1, **layer_kwargs))(model)
    model = TimeDistributed(Flatten())(model)
    model = TimeDistributed(Dense(512, **layer_kwargs))(model)
    policy = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01), activation = None))(model)
    value = TimeDistributed(Dense(1, activation = None, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=1.0)))(model)

    model = Model(inputs = inputs, outputs = [policy, value])
    model.output_names = ['policy', 'value']
    return model


class A2CModel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5

        def not_initialized(*args, **kwargs):
            raise Exception('Not initialized')
        self._train = self._step = self._value = not_initialized

    def build_model(self):
        return create_model(self.env.action_space.n)

    @property
    def learning_rate(self):
        return 7e-4

    def _build_graph(self):
        sess = tf.Session()
        K.set_session(sess)

        model = self.build_model()
        optimizer = RMSprop(learning_rate = self.learning_rate, epsilon=self.rms_epsilon, decay=self.rms_alpha)

        returns_placeholder = tf.placeholder(tf.float32, (None, None), 'returns')
        actions_placeholder = tf.placeholder(tf.int32, (None, None), name = 'actions')
        masks_placeholder = tf.placeholder(tf.float32, (None, None), 'masks')

        policy_logits, value = model.outputs
        dist = tf.distributions.Categorical(logits = policy_logits)
        action_log_probs = dist.log_prob(actions_placeholder)
        dist_entropy = tf.reduce_mean(dist.entropy())

        sampled_action = dist.sample()
        sampled_action_log_probs = dist.log_prob(sampled_action)

        # Compute losses
        advantages = returns_placeholder - tf.squeeze(value, -1)
        value_loss = tf.reduce_mean(tf.pow(advantages, 2))
        action_loss = -tf.reduce_mean(tf.stop_gradient(advantages) * action_log_probs)
        loss = value_loss * self.value_coefficient + \
            action_loss - \
            dist_entropy * self.entropy_coefficient   

        # Optimize
        params = tf.trainable_variables()
        grads = tf.gradients(loss, params, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        if self.max_gradient_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, self.max_gradient_norm)

        grads = list(zip(grads, params))
        optimize_op = optimizer.apply_gradients(grads)

        sess.run(tf.global_variables_initializer())

        # Build train and act functions
        def train(observations, returns, actions, masks, states = []):
            ret = sess.run([loss, action_loss, value_loss, dist_entropy, optimize_op], feed_dict = {
                model.inputs[0]: observations,
                actions_placeholder: actions,
                returns_placeholder: returns,
                masks_placeholder: masks
            })

            return tuple(ret[:4])

        def step_fn(observations, masks, states):
            batch_size = observations.shape[0]
            observations = np.reshape(observations, [batch_size, 1] + list(observations.shape[1:]))
            masks = masks.reshape([batch_size, 1])

            action_value, value_value, action_log_probs_value = sess.run([sampled_action, value, sampled_action_log_probs], feed_dict = {
                model.inputs[0]: observations,
                masks_placeholder: masks
            })

            return action_value.squeeze(1), value_value.squeeze(1).squeeze(-1), action_log_probs_value.squeeze(1)

        def value_fn(observations, masks, states):
            batch_size = observations.shape[0]
            observations = np.reshape(observations, [batch_size, 1] + list(observations.shape[1:]))
            masks = masks.reshape([batch_size, 1])

            action_value = sess.run(sampled_action, feed_dict = {
                model.inputs[0]: observations,
                masks_placeholder: masks
            })

            return action_value.squeeze(1)

        self._step = step_fn
        self._value = value_fn
        self._train = train
        return model

class A2CTrainer(SingleTrainer, A2CModel):
    def __init__(self, name, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self.name = name
        self.num_steps = 5
        self.num_processes = 16
        self.num_env_steps = int(10e6)
        self.gamma = 0.99

        self.log_dir = None
        self.win = None

    def _initialize(self):
        super()._build_graph()
        self._tstart = time.time()
        self.rollouts = RolloutStorage(self.env.reset())  

    def _finalize(self):
        if self.log_dir is not None:
            self.log_dir.cleanup()

    def create_env(self, env):
        self.log_dir = tempfile.TemporaryDirectory()

        seed = 1
        self.validation_env = make_vec_envs(env, seed, 1, self.gamma, self.log_dir.name, None, device, False)
        self.validation_env = VecTransposeImage(self.validation_env)

        envs = make_vec_envs(env, seed + 1, self.num_processes,
                        self.gamma, self.log_dir.name, None, device, False)
        return envs
        

    def process(self, context, mode = 'train', **kwargs):
        metric_context = MetricContext()
        if mode == 'train':
            return self._process_train(context, metric_context)
        else:
            raise Exception('Mode not supported')


    def _sample_experience_batch(self):
        finished_episodes = ([], [])
        for _ in range(self.num_steps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, action_log_prob = self._step(self.rollouts.observations, self.rollouts.terminals, self.rollouts.states)

            # Take actions in env and look the results
            observations, rewards, terminals, infos = self.env.step(actions)

            # Collect true rewards

            for info in infos:
                if 'episode' in info.keys():
                    finished_episodes[0].append(info['episode']['l'])
                    finished_episodes[1].append(info['episode']['r'])
            
            self.rollouts.insert(np.copy(observations), actions, rewards, terminals, values)

        last_values = self._value(self.rollouts.observations, self.rollouts.terminals, self.rollouts.states)
        batched = self.rollouts.batch(last_values, self.gamma)

        # Prepare next batch starting point
        return batched, (len(finished_episodes[0]),) + finished_episodes


    def _process_train(self, context, metric_context):
        batch, report = self._sample_experience_batch()
        loss, value_loss, action_loss, dist_entropy = self._train(*batch)

        fps = int(self._global_t/ (time.time() - self._tstart))
        metric_context.add_cummulative('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('action_loss', action_loss)
        metric_context.add_scalar('entropy', dist_entropy)
        metric_context.add_last_value_scalar('fps', fps)
        return self.num_steps * self.num_processes, report, metric_context