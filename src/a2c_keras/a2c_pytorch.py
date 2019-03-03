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
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
from a2c_ppo_acktr.visualize import visdom_plot
from a2c_pytorch.storage import RolloutStorage

device = 'cpu:0'



class A2CTrainer(SingleTrainer):
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
        num_updates = int(self.num_env_steps) // self.num_steps // self.num_processes
        self.actor_critic = Policy(self.env.observation_space.shape, self.env.action_space,
            base_kwargs={'recurrent': False})
        self.actor_critic.to(device)

        self.agent = algo.A2C_ACKTR(self.actor_critic, 0.5, 0.01, lr=7e-4, eps=1e-5, alpha=0.99,max_grad_norm=0.5)

        self.rollouts = RolloutStorage(self.num_steps, self.num_processes,
                        self.env.observation_space.shape, self.env.action_space,
                        self.actor_critic.recurrent_hidden_state_size)
        self._obs = self.env.reset()

        self.rollouts.obs[0].copy_(self._obs)
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
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.rollouts.obs[step],
                        self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = self.env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    finished_episodes[0].append(info['episode']['l'])
                    finished_episodes[1].append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            self.rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(self.rollouts.obs[-1],
                                                self.rollouts.recurrent_hidden_states[-1],
                                                self.rollouts.masks[-1]).detach()

        self.rollouts.compute_returns(next_value, False, self.gamma, None)

        value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

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