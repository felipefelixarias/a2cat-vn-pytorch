if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)
from abc import abstractclassmethod
from functools import reduce

from keras.layers import Dense, Input, TimeDistributed
import keras.backend as K
from keras import initializers

from common import register_trainer, make_trainer, MetricContext
import gym
import environment.qmaze

import gym
import time
from common.vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import numpy as np

import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util


from baselines.a2c.utils import Scheduler, find_trainable_variables

from tensorflow import losses
from keras.layers import Input

from baselines.common.models import get_network_builder
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.distributions import make_pdtype
from baselines.common.tf_util import adjust_shape
from baselines.a2c.utils import fc
from baselines.common import tf_util
from baselines.common.mpi_running_mean_std import RunningMeanStd

from keras.layers import Dense, TimeDistributed, Input
from keras.models import Sequential
def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False):
    return Sequential(layers = [
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
    ])
    
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h

    return network_fn

from baselines.a2c.utils import fc
from keras.initializers import Orthogonal
from keras.models import Model as MD

def get_model(n_envs, observation_space, action_space):
    input_placeholder = Input(batch_shape=(n_envs, None) + observation_space.shape)
    policy_latent = TimeDistributed(mlp())(input_placeholder)
    value_latent =TimeDistributed(mlp())(input_placeholder)
    
    policy_probs = TimeDistributed(Dense(action_space.n, bias_initializer = 'zeros', activation='softmax', kernel_initializer = Orthogonal(gain=0.01)))(policy_latent)
    value = TimeDistributed(Dense(1, bias_initializer = 'zeros', kernel_initializer = Orthogonal(gain = 1.0)))(value_latent)
    return MD(inputs = [input_placeholder], outputs = [policy_probs, value])

class ActorCriticModelBase:
    def __init__(self):
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.alpha = 0.99
        self.epsilon = 1e-5

        self.n_envs = None
        self._initial_state = None

    @abstractclassmethod
    def create_model(self):
        pass

    @property
    def learning_rate(self):
        return 7e-4

    def _build_graph(self, sess, **model_kwargs):
        # Keras will use this session
        K.set_session(sess)

        # Create model and outputs
        model = self.create_model(**model_kwargs)
        policy, values = model.outputs
        values = tf.squeeze(values, axis = 2)
        
        policy_distribution = tf.distributions.Categorical(probs = policy)

        # Action to take
        action = policy_distribution.sample()

        # Create loss placeholders
        actions = tf.placeholder(tf.int32, [self.n_envs, None], name = 'actions')
        adventages = tf.placeholder(tf.float32, [self.n_envs, None], name = 'adventages')
        returns = tf.placeholder(tf.float32, [self.n_envs, None], name = 'returns')
        learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')

        # Policy gradient loss
        selected_negative_log_prob = -policy_distribution.log_prob(actions)
        policy_gradient_loss = tf.reduce_mean(adventages * selected_negative_log_prob)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy_distribution.entropy())

        # Value loss
        value_loss = losses.mean_squared_error(values, returns)

        # Total loss
        loss = policy_gradient_loss \
            + value_loss * self.value_coefficient \
            - entropy * self.entropy_coefficient

        # Optimize step
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=self.alpha, epsilon=self.epsilon)
        params = model.trainable_weights
        grads = tf.gradients(loss, params)
        if self.max_gradient_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, self.max_gradient_norm)
        grads = list(zip(grads, params))
        optimize_op = optimizer.apply_gradients(grads)

        # Initialize variables
        tf.global_variables_initializer().run(session = sess)


        # Create train fn
        def train(b_obs, b_returns, b_masks, b_actions, b_values, b_states = None):
            b_adventages = b_returns - b_values
            feed_dict = {
                model.inputs[0]:b_obs, 
                actions:b_actions, 
                adventages:b_adventages, 
                returns:b_returns, 
                learning_rate:self.learning_rate
            }

            loss_v, policy_loss_v, value_loss_v, policy_entropy_v, _ = sess.run(
                [loss, policy_gradient_loss, value_loss, entropy, optimize_op],
                feed_dict=feed_dict
            )
            return loss_v, policy_loss_v, value_loss_v, policy_entropy_v

        # Create step fn
        def step(observation, S = None, M = None):
            # This function takes single batch of observations
            # Returns also single batch of returns
            observation = observation.reshape([self.n_envs, -1] + list(observation.shape[1:]))
            action_v, value_v = sess.run([action, values], feed_dict={
                model.inputs[0]: observation
            })

            action_v = action_v.squeeze(1)
            value_v = value_v.squeeze(1)        
            return [action_v, value_v, None, None]

        # Create value fn
        def value(observation, S = None, M = None):
            # This function takes single batch of observations
            # Returns also single batch of returns
            observation = observation.reshape([self.n_envs, -1] + list(observation.shape[1:]))
            values_v = sess.run(values, feed_dict={
                model.inputs[0]: observation
            }).reshape([-1])

            return values_v

        self._step = step
        self._train = train
        self._value = value
        return model



from common.train import SingleTrainer

from collections import namedtuple

Experience = namedtuple('Experience', ['observations', 'returns', 'masks', 'actions', 'values'])
def batch_experience(batch, last_values, previous_batch_terminals, gamma):
    # Batch in time dimension
    b_observations, b_actions, b_values, b_rewards, b_terminals = list(map(lambda *x: np.stack(x, axis = 1), *batch))

    # Compute cummulative returns
    last_returns = (1.0 - b_terminals[:, -1]) * last_values
    b_returns = np.concatenate([np.zeros_like(b_rewards), np.expand_dims(last_returns, 1)], axis = 1)
    for n in reversed(range(len(batch))):
        b_returns[:, n] = b_rewards[:, n] + \
            gamma * (1.0 - b_terminals[:, n]) * b_returns[:, n + 1]

    # Compute RNN reset masks
    b_masks = np.concatenate([np.expand_dims(previous_batch_terminals, 1), b_terminals[:,:-1]], axis = 1)
    return Experience(b_observations, b_returns[:, :-1], b_masks, b_actions, b_values)


class Trainer(SingleTrainer, ActorCriticModelBase):
    def __init__(self, name, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)
        self.name = name
        self.n_steps = 5
        self.n_envs = 16
        self.total_timesteps = 1000000
        self.gamma = 0.99

        self._last_terminals = None
        self._last_observations = None
        self._last_states = None

    @abstractclassmethod
    def create_model(self, **model_kwargs):
        pass

    def wrap_env(self, env):
        return DummyVecEnv([lambda: gym.make(**self._env_kwargs) for _ in range(self.n_envs)])

    def _initialize(self, **model_kwargs):
        self.nenv = nenv = self.env.num_envs if hasattr(self.env, 'num_envs') else 1

        sess = K.get_session()
        model = self._build_graph(sess, **model_kwargs)

        self._last_terminals = np.zeros(shape = (nenv,), dtype = np.bool)
        self._last_states = self._initial_state
        self._last_observations = self.env.reset()

        self._initialize_stats()
        return model

    def _initialize_stats(self):
        self._reports = [(0, 0.0) for _ in range(self.n_envs)]
        self._cum_reports = [(0, [], []) for _ in range(self.n_envs)]

        self._global_t = 0
        self._lastlog = 0
        self._tstart = time.time()

    def _update_report(self, rewards, terminals):
        self._reports = [(x + (1 - a), y + b) for (x, y), a, b in zip(self._reports, terminals, rewards)]
        for i, terminal in enumerate(terminals):
            if terminal:
                (ep, leng, rew) = self._cum_reports[i]
                leng.append(self._reports[i][0])
                rew.append(self._reports[i][1])
                self._cum_reports[i] = (ep + 1, leng, rew)
                self._reports[i] = (0, 0.0)

    def _collect_report(self):
        output = list(map(lambda *x: reduce(lambda a,b:a+b, x), *self._cum_reports))
        self._cum_reports = [(0, [], []) for _ in range(self.n_envs)]
        return output

    def _sample_experience_batch(self):
        batch = []

        terminals = self._last_terminals
        observations = self._last_observations
        states = self._last_states
        for _ in range(self.n_steps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self._step(observations, S=states, M=terminals)  

            # Take actions in env and look the results
            next_observations, rewards, terminals, _ = self.env.step(actions)
            self._update_report(rewards, terminals)

            batch.append((np.copy(observations), actions, values, rewards, terminals))
            observations = next_observations

        last_values = self._value(observations, S=states, M=terminals)

        batched = batch_experience(batch, last_values, self._last_terminals, self.gamma)

        # Prepare next batch starting point
        self._last_terminals = terminals
        self._last_states = states
        self._last_observation = observations
        return batched, self._collect_report()

    def _get_end_stats(self):
        return None

    def process(self):
        metric_context = MetricContext()

        batch, experience_stats = self._sample_experience_batch()
        loss, policy_loss, value_loss, policy_entropy = self._train(*batch)

        fps = int(self._global_t/ (time.time() - self._tstart))
        metric_context.add_last_value_scalar('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('policy_loss', policy_loss)
        metric_context.add_scalar('entropy', policy_entropy)
        metric_context.add_last_value_scalar('fps', fps)
        self._global_t += self.n_steps * self.n_envs
        return (self.n_steps * self.n_envs, experience_stats, metric_context)


@register_trainer('test-a2c', episode_log_interval = 100, save = False)
class SomeTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = dict(id = 'CartPole-v0'), model_kwargs = dict(), **kwargs)

    def create_model(self, **model_kwargs):
        return get_model(self.n_envs, self.env.observation_space, self.env.action_space)


if __name__ == '__main__':
    t = make_trainer('test-a2c')
    t.run()




