# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import gym
import random

from environment.environment import Environment
from model.model_keras import ActorCriticModel
from train.experience import Experience, ExperienceFrame, ExperienceReplay


class Trainer:
    def __init__(self):
        self._replay_size = 50000
        self._stop_signal = False
        self._episode_length = 50
        self._model : ActorCriticModel = None
        self._env = None
        pass

    def _initialize(self):
        self.experience_replay = ExperienceReplay(self._replay_size)

    def _act(self, state):
        if random.random() < epsilon:
            return random.randint(0, self._env.action_space.n - 1)

        else:
            p = self._model.evaluate_policy(state)
            return np.random.choice(self._env.action_space.n, p = p)

    def _train(self, experience):
        (state, action, reward, new_state) = experience
        pass

    def optimize(self):
        pass

    def _run_episode(self):
        steps = 0
        state = self._env.reset()
        while True:         
            time.sleep(THREAD_DELAY) # yield 

            action = self._act(state)
            new_state, reward, done, info = self._env.step(action)

            if done:
                new_state = None

            self._train((state, action, reward, new_state,))

            state = new_state

            if done or self._stop_signal or (steps >= self._episode_length and self._episode_length != -1):
                break
    
    def run(self):
        while not self._stop_signal:
            self._run_episode()
        pass

class TrainingSession:
    def __init__(self):
        pass

    def 