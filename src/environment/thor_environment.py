# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from environment import environment
import ai2thor.controller
import random




# Kitchens:       FloorPlan1 - FloorPlan30
# Living rooms:   FloorPlan201 - FloorPlan230
# Bedrooms:       FloorPlan301 - FloorPlan330
# Bathrooms:      FloorPLan401 - FloorPlan430
controller.reset('FloorPlan28')

# gridSize specifies the coarseness of the grid that the agent navigates on
controller.step(dict(action='Initialize', gridSize=0.25))
event = controller.step(dict(action='MoveAhead'))


class ThorEnvironment(environment.Environment):

  ACTION_LIST = [
    [1,0,0],
    [0,1,0],
    [0,0,1]
  ]

  @staticmethod
  def get_action_size(env_name):
    return len(IndoorEnvironment.ACTION_LIST)

  @staticmethod
  def get_objective_size(env_name):
    simargs = sim_config.get(env_name)
    return simargs.get('objective_size', 0)

  def __init__(self, env_name, thread_index, is_continuous = True, noise = 0.01,  random_seed = None):
    environment.Environment.__init__(self)

    self.controller = ai2thor.controller.Controller()
    self.env_name = env_name
    self.is_continuous = is_continuous
    self.random = random.Random()
    if random_seed is not None:
        self.random.seed(random_seed)

# this will try to move 0.33 but if you hit a wall, you will go wherever the Unity physics engine puts you
controller.step(dict(action='MoveAhead', moveMagnitude=0.33))

# you can turn in any direction by using the following command
# keep in mind that this is the absolution rotation of the agent in world space, so running the command a second time will not rotate the agent, it will just set the rotation to 275 degrees about the Y axis
controller.step(dict(action='Rotate', rotation=275.0)) 

# you can also controller where the agent looks in any amount, as with Rotate, this is an absolute value 
controller.step(dict(action='Look', horizon=37.0))
    
    self.last_state = None
    self.last_action = 0
    self.last_reward = 0

    simargs = sim_config.get(env_name)
    simargs['id'] = 'sim%02d' % thread_index
    simargs['logdir'] = os.path.join(IndoorEnvironment.get_log_dir(), simargs['id'])

    # Merge in extra env args
    if env_args is not None:
      simargs.update(env_args)

    self._sim = RoomSimulator(simargs)
    self._sim_obs_space = self._sim.get_observation_space(simargs['outputs'])
    self.reset()

  def reset(self):
    controller.reset(self.env_name)
    event = controller.step(dict(action='Initialize', continuous=self.is_continuous))
    controller.random_initialize(
        random_seed=None,
        randomize_open=False,
        unique_object_types=False,
        exclude_receptacle_object_pairs=[],
        max_num_repeats=1,
        remove_prob=0.5)
        
    self._episode_info = result.get('episode_info')
    self._last_full_state = result.get('observation')
    obs = self._last_full_state['observation']['sensors']['color']['data']
    objective = self._last_full_state.get('measurements')
    state = { 'image': self._preprocess_frame(obs), 'objective': objective }
    self.last_state = state
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    if self._sim is not None:
        self._sim.close_game()

  def _preprocess_frame(self, image):
    if len(image.shape) == 2:  # assume gray
        image = np.dstack([image, image, image])
    else:  # assume rgba
        image = image[:, :, :-1]
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = IndoorEnvironment.ACTION_LIST[action]

    full_state = self._sim.step(real_action)
    self._last_full_state = full_state  # Last observed state
    obs = full_state['observation']['sensors']['color']['data']
    reward = full_state['rewards']
    terminal = full_state['terminals']
    objective = full_state.get('measurements')

    if not terminal:
      state = { 'image': self._preprocess_frame(obs), 'objective': objective }
    else:
      state = self.last_state

    pixel_change = self._calc_pixel_change(state['image'], self.last_state['image'])
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change

  def is_all_scheduled_episodes_done(self):
    return self._sim.is_all_scheduled_episodes_done()