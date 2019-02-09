# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from environment import environment

class MazeEnvironment(environment.Environment):
  @staticmethod
  def get_action_size():
    return 4
  
  def __init__(self, name):
    '''
    args:
      name: Examples 'gr' - goal, random goal
    '''
    assert name and len(name) >= 2
    super().__init__('maze', name)
    
    self._maze_size = 7
    self._map_data = \
                     "--+---G" \
                     "--+-+++" \
                     "S-+---+" \
                     "--+++--" \
                     "--+-+--" \
                     "--+----" \
                     "-----++" 
    
    self._setup()
    self.reset()

  def _setup(self):
    image = np.zeros( (84, 84, 3), dtype=float )

    start_pos = (-1, -1)
    goal_pos  = (-1, -1)
  
    for y in range(7):
      for x in range(7):
        p = self._get_pixel(x,y)
        if p == '+':
          self._put_pixel(image, x, y, 0)
        elif p == 'S':
          start_pos = (x, y)
        elif p == 'G':
          goal_pos = (x, y)

    self._maze_image = image
    self._start_pos = goal_pos if self.env_name[1] != 'r' else self._get_random_position(['-', 'S'])
    self._goal_pos = goal_pos if self.env_name[1] != 'r' else self._get_random_position()

  def _iter_pos(self, types):
    for y in range(self._maze_size):
      for x in range(self._maze_size):
        p = self._get_pixel(x,y)
        if p in types:
          yield (x, y)
        else:
          pass

  def _get_random_position(self, types = ['-', 'G']):
    goal_potentials = list(self._iter_pos(types))
    return random.choice(goal_potentials)
    
  def reset(self):
    self.x = self._start_pos[0]
    self.y = self._start_pos[1]
    self.last_state = { 
      'image': self._get_current_image((self.x, self.y,)),
      'goal': self._get_current_image(self._goal_pos)
    }
    self.last_action = 0
    self.last_reward = 0    
    
  def _put_pixel(self, image, x, y, channel):
    for i in range(12):
      for j in range(12):
        image[12*y + j, 12*x + i, channel] = 1.0
        
  def _get_pixel(self, x, y):
    data_pos = y * 7 + x
    return self._map_data[data_pos]

  def _is_wall(self, x, y):
    return self._get_pixel(x, y) == '+'

  def _clamp(self, n, minn, maxn):
    if n < minn:
      return minn, True
    elif n > maxn:
      return maxn, True
    return n, False
  
  def _move(self, dx, dy):
    new_x = self.x + dx
    new_y = self.y + dy

    new_x, clamped_x = self._clamp(new_x, 0, 6)
    new_y, clamped_y = self._clamp(new_y, 0, 6)

    hit_wall = False

    if self._is_wall(new_x, new_y):
      new_x = self.x
      new_y = self.y
      hit_wall = True

    hit = clamped_x or clamped_y or hit_wall
    return new_x, new_y, hit

  def _get_current_image(self, position):
    image = np.array(self._maze_image)
    self._put_pixel(image, position[0], position[1], 1)
    return image

  def get_keyboard_map(self):
    return dict(up=0, down=1, left=2, right=3)

  def process(self, action):
    dx = 0
    dy = 0
    if action == 0: # UP
      dy = -1
    if action == 1: # DOWN
      dy = 1
    if action == 2: # LEFT
      dx = -1
    if action == 3: # RIGHT
      dx = 1

    self.x, self.y, hit = self._move(dx, dy)

    image = self._get_current_image((self.x, self.y,))
    
    terminal = (self.x == self._goal_pos[0] and
                self.y == self._goal_pos[1])

    if terminal:
      reward = 1
    elif hit:
      reward = -1
    else:
      reward = 0

    pixel_change = self._calc_pixel_change(image, self.last_state['image'])
    self.last_state = {'image': image, 'goal': self.last_state['goal']}
    self.last_action = action
    self.last_reward = reward
    return self.last_state, reward, terminal, pixel_change
