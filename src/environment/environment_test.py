# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from environment.environment import Environment


class TestEnvironment(unittest.TestCase):
  def test_lab(self):
    has_lab = True
    try:
      import deepmind_lab
    except ImportError:
      has_lab = False
      print("Failed to import lab. Skipping lab environment testing.")

    if has_lab:
      env_type = "lab"
      env_name = "nav_maze_static_01"
      self.check_environment(env_type, env_name)

  def test_gym(self):
    env_type = "gym"
    env_name = "MontezumaRevenge-v0"
    self.check_environment(env_type, env_name)

  def test_maze(self):
    env_type = "maze"
    env_name = ""
    self.check_environment(env_type, env_name)

  def test_thor_cached(self):
    env_type = "maze"
    env_name = "bedroom_04"
    self.check_environment(env_type, env_name)

  def check_environment(self, env_type, env_name):
    environment = Environment.create_environment(env_type, env_name)
    # action_size = Environment.get_action_size(env_type, env_name) # Not used

    for i in range(3):
      state, reward, terminal, pixel_change = environment.process(0)

      # Check shape
      self.assertTrue( state['image'].shape == (84,84,3) )
      self.assertTrue( environment.last_state['image'].shape == (84,84,3) )
      if 'goal' in state:
        self.assertTrue( state['goal'].shape == (84,84,3) )
        self.assertTrue( environment.last_state['goal'].shape == (84,84,3) )
      
      self.assertTrue( pixel_change.shape == (20,20) )

      # state and pixel_change value range should be [0,1]
      self.assertTrue( np.amax(state['image']) <= 1.0 )
      self.assertTrue( np.amin(state['image']) >= 0.0 )
      if 'goal' in state:
        self.assertTrue( np.amax(state['goal']) <= 1.0 )
        self.assertTrue( np.amin(state['goal']) >= 0.0 )
      self.assertTrue( np.amax(pixel_change) <= 1.0 )
      self.assertTrue( np.amin(pixel_change) >= 0.0 )

    environment.stop()    
                       


if __name__ == '__main__':
  unittest.main()
