from abc import abstractclassmethod
from collections import namedtuple
from functools import reduce
import functools
import time
import gym

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

class MonitorWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, stats = super().step(action)
        if stats is None:
            stats = dict()

        stats['reward'] = reward
        return obs, reward, done, stats

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

class A2CTrainer(SingleTrainer):
    def __init__(self, env_kwargs, model_kwargs):
        super().__init__(env_kwargs = env_kwargs, model_kwargs = model_kwargs)

    @abstractclassmethod
    def create_model(self, **model_kwargs):
        pass

    def create_env(self, env):
        frame_stack_size = 4
        env = make_vec_env('', 'atari', nenv, seed, gamestate=None, reward_scale=1.0)
        env = VecFrameStack(env, frame_stack_size)
        return env

    def _initialize(self, **model_kwargs):

        nenvs = env.num_envs
        policy = build_policy(self.env, 'cnn', **network_kwargs)

        # Instantiate the model object (that creates step_model and train_model)
        model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)

        # Instantiate the runner object
        runner = Runner(env, model, nsteps=nsteps, gamma=gamma)

        # Calculate the batch_size
        nbatch = nenvs*nsteps

        # Start total timer
        tstart = time.time()

        for update in range(1, total_timesteps//nbatch+1):
            # Get mini batch of experiences
            obs, states, rewards, masks, actions, values = runner.run()

            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
            nseconds = time.time()-tstart

            # Calculate the fps (frame per second)
            fps = int((update*nbatch)/nseconds)
            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.dump_tabular()
        return model


        return None

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
            next_observations, rewards, terminals, stats = self.env.step(actions)

            # Collect true rewards
            
            true_rewards = [x['reward'] for x in stats] if 'reward' in stats[0] else rewards
            self._update_report(true_rewards, terminals)

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

    def process(self, mode = 'train', **kwargs):
        metric_context = MetricContext()

        if mode == 'train':
            return self._process_train(metric_context)
        elif mode == 'validation':
            return self._process_validation(metric_context)
        else:
            raise Exception('Mode not supported')


@register_trainer('test-a2c', saving_period = None, validation_period = None)(A2CTrainer)