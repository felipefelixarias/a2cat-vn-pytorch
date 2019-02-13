# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from datetime import datetime

def wrap_environment(env, action_space_size, use_goal_input):
  import gym.spaces
  class EnvWrap:
    def __init__(self, env, action_space_size, use_goal_input):
        self.env = env
        self.action_space_size = action_space_size
        self.use_goal_input = use_goal_input

    def reset(self):
        self.env.reset()
        return self._transform_observation(self.env.last_state)

    @property
    def observation_space(self):
      if self.use_goal_input:
        return gym.spaces.Dict({
          'image': gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3)),
          'goal': gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3))
        })
      else:
        return gym.spaces.Dict({
          'image': gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3))
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.action_space_size)

    def step(self, action):
        new_obs, rew, done, _ = self.env.process(action)


        return (self._transform_observation(new_obs), rew, done, None)

    def stop(self):
        self.env.stop()

    def _transform_observation(self, state):
        return state

  return EnvWrap(env, action_space_size, use_goal_input)



def wrap_environment_keras(env, action_space_size):
  import gym.spaces
  class EnvWrap:
    def __init__(self, env, action_space_size):
        self.env = env
        self.action_space_size = action_space_size

    def reset(self):
        self.env.reset()
        return self._transform_observation(self.env.last_state)

    @property
    def observation_space(self):
      return gym.spaces.Tuple((
        gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3)),
        gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3)),
      ))

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.action_space_size)

    def step(self, action):
        new_obs, rew, done, _ = self.env.process(action)


        return (self._transform_observation(new_obs), rew, done, None)

    def stop(self):
        self.env.stop()

    def _transform_observation(self, state):
        return [state['image'], state['goal']]

  return EnvWrap(env, action_space_size)


class Environment(object):
  # cached action size
  action_size = -1

  LOG_DIR = None

  def __init__(self, env_type, env_name):
    super().__init__()
    self.env_type = env_type
    self.env_name = env_name

  @staticmethod
  def get_log_dir():
    if Environment.LOG_DIR is None:
      timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
      Environment.LOG_DIR = './logs/' + timestamp
    return Environment.LOG_DIR

  @staticmethod
  def set_log_dir(dir):
    Environment.LOG_DIR = dir

  @staticmethod
  def create_environment(env_type, env_name, env_args=None, thread_index=0):
    if env_type == 'maze':
      from . import maze_environment
      return maze_environment.MazeEnvironment(env_name)
    elif env_type == 'lab':
      from . import lab_environment
      return lab_environment.LabEnvironment(env_name)
    elif env_type == 'indoor':
      from . import indoor_environment
      return indoor_environment.IndoorEnvironment(env_name, env_args, thread_index)
    elif env_type == 'thor_cached':
      from . import thor_cached_environment
      return thor_cached_environment.THORDiscreteCachedEnvironment(env_name)
    else:
      from . import gym_environment
      return gym_environment.GymEnvironment(env_name)
  
  def get_action_size(self, env_name = None):
    if isinstance(self, Environment):
      return Environment.can_use_goal(self.env_type, self.env_name)

    if Environment.action_size >= 0:
      return Environment.action_size

    if self == 'maze':
      from . import maze_environment
      Environment.action_size = \
        maze_environment.MazeEnvironment.get_action_size()
    elif self == "lab":
      from . import lab_environment
      Environment.action_size = \
        lab_environment.LabEnvironment.get_action_size(env_name)
    elif self == "indoor":
      from . import indoor_environment
      Environment.action_size = \
        indoor_environment.IndoorEnvironment.get_action_size(env_name)

    elif self == 'thor_cached':
      from . import thor_cached_environment
      return thor_cached_environment.THORDiscreteCachedEnvironment.get_action_size(env_name)
    else:
      from . import gym_environment
      Environment.action_size = \
        gym_environment.GymEnvironment.get_action_size(env_name)
    return Environment.action_size

  def can_use_goal(self, env_name = None):
    if isinstance(self, Environment):
      return Environment.can_use_goal(self.env_type, self.env_name)

    if self == 'thor_cached':
      return True
    elif self == 'maze' and env_name.startswith('g'):
      return True
    else:
      return False

  def get_objective_size(self, env_name = None):
    if isinstance(self, Environment):
      return Environment.get_objective_size(self.env_type, self.env_name)

    if self == "indoor":
      from . import indoor_environment
      return indoor_environment.IndoorEnvironment.get_objective_size(env_name)
    return 0

  def process(self, action):
    pass

  def reset(self):
    pass

  def stop(self):
    pass

  def get_keyboard_map(self):
    return dict(down=3, up=0, left=2, right=1)

  def is_all_scheduled_episodes_done(self):
    return False

  def _subsample(self, a, average_width):
    s = a.shape
    sh = s[0]//average_width, average_width, s[1]//average_width, average_width
    return a.reshape(sh).mean(-1).mean(1)  

  def _calc_pixel_change(self, state, last_state):
    d = np.absolute(state[2:-2,2:-2,:] - last_state[2:-2,2:-2,:])
    # (80,80,3)
    m = np.mean(d, 2)
    c = self._subsample(m, 4)
    return c


  def get_env(self):
    return wrap_environment(self, self.get_action_size(), self.can_use_goal())

def create_env(*arg, **kwargs):
  env = Environment.create_environment(*arg, **kwargs)
  return wrap_environment_keras(env, env.get_action_size())

def get_action_space_size(env_type, env_name, **kwargs):
  return Environment.get_action_size(env_type, env_name)