import gym
import gym.spaces
import numpy as np
import ai2thor.controller
import cv2
import random

from .env import EnvBase
from .goal import GoalEnvBase

ACTIONS = [
    dict(action='MoveAhead', magnitude = 0.6, snapToGrid = False),
    dict(action='MoveBack', magnitude = 0.6, snapToGrid = False),
    dict(action='MoveLeft', magnitude = 0.25, snapToGrid = False),
    dict(action='MoveRight', magnitude = 0.25, snapToGrid = False),
    dict(action='Rotate', angle = 30),
    dict(action='Rotate', angle = -30),
    dict(action='LookUp'),
    dict(action='LookDown')
]

    
class ContinuousEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.initialize_kwargs['continuous'] = True
    
    def step(self, action):
        event = self._controller_step(action)
        return self._finish_step(event)


    def _controller_step(self, action):
        action = ACTIONS[action]
        if action['action'] == 'Rotate':
            deltaangle = action['angle']
            angle = (self.state[1]['y'] + deltaangle) % 360
            return self.controller.step(dict(action = 'Rotate', rotation = angle))
        else:
            return self.controller.step(action)


class GoalContinuousEnv(GoalEnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.initialize_kwargs['continuous'] = True
    
    def step(self, action):
        event = self._controller_step(action)
        return self._finish_step(event)


    def _controller_step(self, action):
        action = ACTIONS[action]
        if action['action'] == 'Rotate':
            deltaangle = action['angle']
            angle = (self.state[1]['y'] + deltaangle) % 360
            return self.controller.step(dict(action = 'Rotate', rotation = angle))
        else:
            return self.controller.step(action)