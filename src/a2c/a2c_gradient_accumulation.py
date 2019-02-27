from abc import abstractclassmethod
from collections import namedtuple
from functools import reduce
import functools
import time
import gym
from math import ceil

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, TimeDistributed
import keras.backend as K
from keras.models import Model, Sequential
from keras.initializers import Orthogonal

from common import register_trainer, make_trainer, MetricContext, AbstractAgent
from common.train import SingleTrainer
from common.vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


RecurrentModel = namedtuple('RecurrentModel', ['model', 'inputs','outputs', 'states_in', 'states_out', 'mask'])

def expand_recurrent_model(model):
    states_in = []
    pure_inputs = [x for x in model.inputs]
    
    mask = None
    for x in model.inputs:
        if 'rnn_state' in x.name:
            states_in.append(x)
            pure_inputs.remove(x)
        if 'rnn_mask' in x.name:
            mask = x
            pure_inputs.remove(x)

    pure_outputs = model.outputs[:-len(states_in)] if len(states_in) > 0 else model.outputs
    states_out = model.outputs[-len(states_in):] if len(states_in) > 0 else []

    assert len(states_in) == 0 or mask is not None
    return RecurrentModel(model, pure_inputs, pure_outputs, states_in, states_out, mask)


def create_initial_state(n_envs, state_in):
    return [np.zeros((n_envs,) + tuple(x.shape[1:])) for x in state_in]

class A2CModelBase:
    def __init__(self):
        self.entropy_coefficient = 0.01
        self.value_coefficient = 0.5
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5

        self.n_envs = None
        self.minibatch_size = 1
        self._initial_state = None

    @abstractclassmethod
    def create_model(self):
        pass

    @property
    def learning_rate(self):
        return 7e-4

    def on_graph_built(self, **kwargs):
        pass

    def _build_graph(self, sess, **model_kwargs):
        # Keras will use this session
        K.set_session(sess)

        # Create model and outputs
        model = self.create_model(**model_kwargs)
        rnn_model = expand_recurrent_model(model)

        self._initial_state = create_initial_state(self.n_envs, rnn_model.states_in)
        self._validation_initial_state = create_initial_state(1, rnn_model.states_in)
        policy, values = rnn_model.outputs
        values = tf.squeeze(values, axis = 2)
        
        policy_distribution = tf.distributions.Categorical(probs = policy)

        # Action to take
        action = policy_distribution.sample()

        # Create loss placeholders
        actions = tf.placeholder(tf.int32, [None, None], name = 'actions')
        adventages = tf.placeholder(tf.float32, [None, None], name = 'adventages')
        returns = tf.placeholder(tf.float32, [None, None], name = 'returns')
        learning_rate = tf.placeholder(tf.float32, [], name = 'learning_rate')

        # Policy gradient loss
        selected_negative_log_prob = -policy_distribution.log_prob(actions)
        policy_gradient_loss = tf.reduce_mean(adventages * selected_negative_log_prob)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(policy_distribution.entropy())

        # Value loss
        value_loss = tf.losses.mean_squared_error(values, returns)

        # Total loss
        loss = policy_gradient_loss \
            + value_loss * self.value_coefficient \
            - entropy * self.entropy_coefficient

        # Optimize step
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=self.rms_alpha, epsilon=self.rms_epsilon)
        params = model.trainable_weights
        grads = tf.gradients(loss, params)
        
        accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()),trainable=False) for tv in params]                                        
        metrics = ['loss', 'policy_gradient_loss', 'value_loss', 'entropy']
        metrics_tvars = [tf.Variable(0, dtype = tf.float32, trainable = False, name = x + '_acc') for x in metrics]
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars] + [tv.assign(tf.zeros_like(tv)) for tv in metrics_tvars]
        accum_ops = [accum_tvars[i].assign_add(batch_grad_var) for i, batch_grad_var in enumerate(grads)]

        # Accumulate metrics
        loc = locals()
        minibatch_real_size = tf.to_float(tf.shape(returns)[0])
        accum_ops = accum_ops + [
            metrics_tvars[i].assign_add(tf.to_float(loc[x]) / minibatch_real_size) for i, x in enumerate(metrics)
        ]

        if self.max_gradient_norm is not None:
            # Clip the gradients (normalize)
            accum_tvars, grad_norm = tf.clip_by_global_norm(accum_tvars, self.max_gradient_norm)

        batch_grads = [(accum_tvars[i], param) for i, param in enumerate(params)]
        optimize_op = optimizer.apply_gradients(batch_grads)

        # Initialize variables
        tf.global_variables_initializer().run(session = sess)


        # Create train fn
        def train(b_obs, b_returns, b_masks, b_actions, b_values, states = []):
            b_adventages = b_returns - b_values

            sess.run(zero_ops)
            for i in range(ceil(float(self.n_envs) / self.minibatch_size)):
                start = i * self.minibatch_size
                end = min((i + 1) * self.minibatch_size, b_actions.shape[0])
                feed_dict = {
                    K.learning_phase(): 1,
                    rnn_model.inputs[0]:b_obs[start:end], 
                    actions:b_actions[start:end], 
                    adventages:b_adventages[start:end], 
                    returns:b_returns[start:end], 
                    learning_rate:self.learning_rate,
                    **{state: value[start:end] for state, value in zip(rnn_model.states_in, states)}
                }
                if rnn_model.mask is not None:
                    feed_dict[rnn_model.mask] = b_masks[start:end]


                sess.run(accum_ops,
                    feed_dict=feed_dict
                )

            loss_v, policy_loss_v, value_loss_v, policy_entropy_v, _ = sess.run(metrics_tvars + [optimize_op], feed_dict={
                learning_rate: self.learning_rate,
                K.learning_phase(): 1
            })
            return loss_v, policy_loss_v, value_loss_v, policy_entropy_v

        # Create step fn
        def step(observation, mask, states = [], mode = 'train'):
            # This function takes single batch of observations
            # Returns also single batch of returns
            observation = observation.reshape([-1, 1] + list(observation.shape[1:]))
            mask = mask.reshape([-1, 1])
            state_out_v = [list() for _ in rnn_model.states_out]
            action_v = []
            value_v = []
            for i in range(ceil(float(self.n_envs) / self.minibatch_size)):
                start = i * self.minibatch_size
                end = min((i + 1) * self.minibatch_size, observation.shape[0])
                feed_dict = {
                    K.learning_phase(): 1 if mode == 'train' else 0,
                    model.inputs[0]: observation[start:end],
                    **{state: value[start:end] for state, value in zip(rnn_model.states_in, states)}
                }

                if rnn_model.mask is not None:
                    feed_dict[rnn_model.mask] = mask[start:end]

                action_vs, value_vs, state_out_vs = sess.run([action, values, rnn_model.states_out], feed_dict=feed_dict)
                action_v.append(action_vs)
                value_v.append(value_vs)
                for i, s in enumerate(state_out_vs):
                    state_out_v[i].append(s)

            action_v = np.concatenate(action_v, 0)
            value_v = np.concatenate(value_v, 0)
            state_out_v = [np.concatenate(x, 0) for x in state_out_v]
            action_v = action_v.squeeze(1)
            value_v = value_v.squeeze(1)
            return [action_v, value_v, state_out_v, None]

        # Create value fn
        def value(observation, mask, states = [], mode = 'train'):
            # This function takes single batch of observations
            # Returns also single batch of returns
            observation = observation.reshape([self.n_envs, -1] + list(observation.shape[1:]))
            mask = mask.reshape([self.n_envs, -1])
            value_v = []
            for i in range(ceil(float(self.n_envs) / self.minibatch_size)):
                start = i * self.minibatch_size
                end = min((i + 1) * self.minibatch_size, observation.shape[0])
                feed_dict = {
                    K.learning_phase(): 1 if mode == 'train' else 0,
                    model.inputs[0]: observation[start:end],
                    **{state: value[start:end] for state, value in zip(rnn_model.states_in, states)}
                }
                if rnn_model.mask is not None:
                    feed_dict[rnn_model.mask] = mask

                value_vs = sess.run(values, feed_dict=feed_dict)
                value_v.append(value_vs)

            value_v = np.concatenate(value_v, 0).squeeze(1)
            return value_v

        self._step = step
        self._train = train
        self._value = value

        feed_dict = locals()
        feed_dict.pop('self')
        self.on_graph_built(**feed_dict)
        return model


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


class A2CTrainer(SingleTrainer, A2CModelBase):
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

    def create_env(self, env):
        def create_single_factory(env, use_singleton = True):
            if isinstance(env, dict):
                return lambda: gym.make(**env)
            elif callable(env):
                return env
            elif use_singleton:
                return lambda: env
            else:
                raise Exception('Environment not supported')

        if isinstance(env, list):
            if len(env) != self.n_envs + 1:
                raise Exception('Unsupported number of environments. Must be number of environments + 1')
            envs = [create_single_factory(e, True) for e in env]
        else:
            envs = [create_single_factory(env, False) for _ in range(self.n_envs + 1)]

        # Create validation environment
        self.validation_env = envs[0]()
        return DummyVecEnv(envs[1:])

    def _initialize(self, **model_kwargs):
        self.nenv = nenv = self.env.num_envs if hasattr(self.env, 'num_envs') else 1

        sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True,
            gpu_options = tf.GPUOptions(
                allow_growth = True
            )))
        K.set_session(sess)
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
            actions, values, states, _ = self._step(observations, terminals, states)  

            # Take actions in env and look the results
            next_observations, rewards, terminals, _ = self.env.step(actions)
            self._update_report(rewards, terminals)

            batch.append((np.copy(observations), actions, values, rewards, terminals))
            observations = next_observations

        last_values = self._value(observations, terminals, states)

        batched = batch_experience(batch, last_values, self._last_terminals, self.gamma)
        batched = batched + (self._last_states,)

        # Prepare next batch starting point
        self._last_terminals = terminals
        self._last_states = states
        self._last_observation = observations
        return batched, self._collect_report()

    def _get_end_stats(self):
        return None

    def _process_train(self, metric_context):
        batch, experience_stats = self._sample_experience_batch()
        loss, policy_loss, value_loss, policy_entropy = self._train(*batch)

        fps = int(self._global_t/ (time.time() - self._tstart))
        metric_context.add_cummulative('updates', 1)
        metric_context.add_scalar('loss', loss)
        metric_context.add_scalar('value_loss', value_loss)
        metric_context.add_scalar('policy_loss', policy_loss)
        metric_context.add_scalar('entropy', policy_entropy)
        metric_context.add_last_value_scalar('fps', fps)
        self._global_t += self.n_steps * self.n_envs
        return (self.n_steps * self.n_envs, experience_stats, metric_context)

    def _process_validation(self, metric_context):
        done = False
        states = self._validation_initial_state
        ep_reward = 0.0
        ep_length = 0
        obs = self.validation_env.reset()
        while not done:
            observations = np.expand_dims(obs, 0)
            action, values, states, _ = self._step(observations, np.zeros((1,)), states)
            action = action[0]
            obs, reward, done, _ = self.validation_env.step(action)
            
            ep_reward += reward
            ep_length += 1

        return (ep_length, (ep_length, ep_reward), dict())

    def process(self, mode = 'train', **kwargs):
        metric_context = MetricContext()

        if mode == 'train':
            return self._process_train(metric_context)
        elif mode == 'validation':
            return self._process_validation(metric_context)
        else:
            raise Exception('Mode not supported')

class A2CAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

        model = self._load(self.name)
        self.state = None
        self.rnn_model = expand_recurrent_model(model)
        # TODO: implement loading and act


    def reset_state(self):
        self.state = create_initial_state(1, self.rnn_model.states_in)

    def _build_graph(self):
        policy, values = rnn_model.outputs
        values = tf.squeeze(values, axis = 2)
        
        policy_distribution = tf.distributions.Categorical(probs = policy)

    def act(state):
        pass