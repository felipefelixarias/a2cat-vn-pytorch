from abc import abstractclassmethod
from collections import namedtuple
import numpy as np
from common.train import AbstractTrainer, SingleTrainer
from common import MetricContext
import torch

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c.model import CNNBase, TimeDistributed

from a2c.storage import RolloutStorage
from a2c_pytorch.core import pytorch_call, to_tensor, to_numpy
from common.env import VecTransposeImage, make_vec_envs

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
        return TimeDistributed(CNNBase(self.env.observation_space.shape[0], self.env.action_space.n))

    @property
    def learning_rate(self):
        return 7e-4

    def _build_graph(self, devices = []):
        if len(devices) == 0:
            devices = ['cpu']

        model = self.build_model()
        cuda_devices = torch.cuda.device_count()
        if cuda_devices == 0:
            print('Using CPU only')
        elif cuda_devices > 1:
            print('Using %s GPUs' % cuda_devices)
            main_device = torch.device('cpu')
            model = nn.DataParallel(model, output_device=main_device)
        else:
            print('Using single GPU')
            main_device = torch.device('cuda:0')
            model = model.to(main_device)

        optimizer = optim.RMSprop(model.parameters(), self.learning_rate, eps=self.rms_epsilon, alpha=self.rms_alpha)

        # Build train and act functions
        def train(observations, returns, actions, masks, states = []):
            policy_logits, value = model.forward(observations, masks)

            dist = torch.distributions.Categorical(logits = policy_logits)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()
            
            # Compute losses
            advantages = returns - value.squeeze(-1)
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
            loss = value_loss * self.value_coefficient + \
                action_loss - \
                dist_entropy * self.entropy_coefficient   

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm)
            optimizer.step()

            return loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item()

        def step(observations, masks, states):
            with torch.no_grad():
                batch_size = observations.size()[0]
                observations = observations.view(batch_size, 1, *observations.size()[1:])
                masks = masks.view(batch_size, 1)

                policy_logits, value = model.forward(observations, masks)
                dist = torch.distributions.Categorical(logits = policy_logits)
                action = dist.sample()
                action_log_probs = dist.log_prob(action)
                return action.squeeze(1).detach(), value.squeeze(1).squeeze(-1).detach(), action_log_probs.squeeze(1).detach()

        def value(observations, masks, states):
            with torch.no_grad():
                batch_size = observations.size()[0]
                observations = observations.view(batch_size, 1, *observations.size()[1:])
                masks = masks.view(batch_size, 1)

                _, value = model.forward(observations, masks)
                return value.squeeze(1).squeeze(-1).detach()

        self._step = pytorch_call(main_device)(step)
        self._value = pytorch_call(main_device)(value)
        self._train = pytorch_call(main_device)(train)
        return model

class A2CTrainer(SingleTrainer, A2CModel):
    def __init__(self, name, env_kwargs, model_kwargs, devices = []):
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
        self.episode_rewards = deque(maxlen=10)

        self._tstart = time.time()

        self.rollouts = RolloutStorage(self.env.reset())  

    def _finalize(self):
        if self.log_dir is not None:
            self.log_dir.cleanup()

    def create_env(self, env):
        self.log_dir = tempfile.TemporaryDirectory()

        seed = 1
        self.validation_env = make_vec_envs(env, seed, 1, self.gamma, self.log_dir.name, None, False)
        self.validation_env = VecTransposeImage(self.validation_env)

        envs = make_vec_envs(env, seed + 1, self.num_processes,
                        self.gamma, self.log_dir.name, None, False)
        return VecTransposeImage(envs)
        

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
                    self.episode_rewards.append(info['episode']['r'])
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