from abc import abstractclassmethod
from collections import namedtuple
import numpy as np
from common.train import AbstractTrainer, SingleTrainer
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

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c.model import Policy
from a2c_pytorch.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from a2c.model import CNNBase, TimeDistributed

device = 'cpu:0'

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
        print(self.__dict__.keys())
        return TimeDistributed(CNNBase(self.env.observation_space.shape[0], self.env.action_space.n))

    @property
    def learning_rate(self):
        return 7e-4

    def _build_graph(self):
        model = self.build_model()
        optimizer = optim.RMSprop(model.parameters(), self.learning_rate, eps=self.rms_epsilon, alpha=self.rms_alpha)

        # Build train and act functions
        def train(observations, returns, actions, masks, states = []):
            policy_logits, value = model(observations, masks)

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

        def step(observations, masks):
            with torch.no_grad():
                batch_size = observations.size()[0]
                observations = observations.view(batch_size, 1, *observations.size()[1:])
                masks = masks.view(1, *masks.size())

                policy_logits, value = model(observations, masks)
                dist = torch.distributions.Categorical(logits = policy_logits)
                action = dist.sample()
                action_log_probs = dist.log_prob(action)
                return action.detach(), value.squeeze(-1).detach(), action_log_probs.detach()

        def value(observations, masks):
            with torch.no_grad():
                batch_size = observations.size()[0]
                observations = observations.view(batch_size, 1, *observations.size()[1:])
                masks = masks.view(1, *masks.size())

                _, value = model(observations, masks)
                return value.squeeze(-1).detach()

        self._step = step
        self._value = value
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
        
        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                        self.env.observation_space.shape, self.env.action_space, 1)
        self._obs = self.env.reset()

        self.rollouts.obs[:, 0].copy_(self._obs)
        self.rollouts.to(device)

        self.episode_rewards = deque(maxlen=10)
        self.start = time.time()
  

    def _finalize(self):
        if self.log_dir is not None:
            self.log_dir.cleanup()

    def create_env(self, env):
        self.log_dir = tempfile.TemporaryDirectory()
        return make_vec_envs(env, 1, self.num_processes,
                        self.gamma, self.log_dir.name, None, device, False)

    def process(self, context, mode = 'train', **kwargs):
        if mode == 'train':
            return self._process_train(context)


    def _process_train(self, context):
        viz = context.get('visdom')

        finished_episodes = ([], [])
        for step in range(self.num_steps):
            # Sample actions
            action, value, action_log_prob = self._step(self.rollouts.obs[:,step], self.rollouts.masks[:,step])

            # Obser reward and next obs
            obs, reward, done, infos = self.env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    finished_episodes[0].append(info['episode']['l'])
                    finished_episodes[1].append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([0.0 if done_ else 1.0
                                       for done_ in done])

            self.rollouts.insert(obs, action.squeeze(-1), action_log_prob.squeeze(-1), value.squeeze(-1), reward.squeeze(-1), masks)

        next_value = self._value(self.rollouts.obs[:,-1],
                                            self.rollouts.masks[:,-1])

        batch = self.rollouts.sample(next_value, self.gamma)
        loss, value_loss, action_loss, dist_entropy = self._train(*batch)

        self.rollouts.after_update()

        if self._global_t % (10 * self.num_processes * self.num_steps) == 0 and len(self.episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(self._global_t // (self.num_processes * self.num_steps), self._global_t,
                       int(self._global_t / (end - self.start)),
                       len(self.episode_rewards),
                       np.mean(self.episode_rewards),
                       np.median(self.episode_rewards),
                       np.min(self.episode_rewards),
                       np.max(self.episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if self._global_t % (100  * self.num_processes * self.num_steps) == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                self.win = visdom_plot(viz, self.win, self.log_dir.name, 'Breakout-v2', 'a2c', self.num_env_steps)
            except IOError:
                pass


        return self.num_steps * self.num_processes, (len(finished_episodes[0]), finished_episodes[0], finished_episodes[1]), dict()