# -*- coding: utf-8 -*-
import sys
import os
import h5py
import json
import numpy as np
import random
import skimage.io
from skimage.transform import resize
from environment import environment

class THORDiscreteCachedEnvironment(environment.Environment):
    @staticmethod
    def _get_h5_file_path(scene_name):
        path = '/media/data/datasets/visual_navigation_precomputed'
        if 'THOR_DATASET_PATH' in os.environ:
            path = os.environ['THOR_DATASET_PATH']

        return "%s/%s.h5" % (path, scene_name)

    def __init__(self, env_name = 'bedroom_04', rand_seed = None, image_size = (84,84), **kwargs):
        super(THORDiscreteCachedEnvironment, self).__init__('thor_cached', env_name)
        h5_file_path = THORDiscreteCachedEnvironment._get_h5_file_path(env_name)

        self._random = random.Random(x = rand_seed)
        
        self._h5_file = h5py.File(h5_file_path, 'r')
        self._observations = self._h5_file['observation'][()]

        # pylint: disable=no-member
        self._n_locations = self._h5_file['location'][()].shape[0]
        self._transition_graph = self._h5_file['graph'][()]
        self._shortest_path_distances = self._h5_file['shortest_path_distance'][()]

        (self._current_state_idx, self._current_goal_idx) = (None, None)
        self.image_size = image_size
        self.reset()

    def _get_random_start_goal_tuple(self):
        goal = self._random.randrange(self._n_locations)
        current_state = None
        while True:
            current_state = random.randrange(self._n_locations)
            if self._shortest_path_distances[current_state][goal] > 0:
                break
        return (current_state, goal)

    def reset(self, initial_state_id = None):
        # randomize initial state and goal
        (self._current_state_idx, self._current_goal_idx) = self._get_random_start_goal_tuple()
        obs = self._render_observation(self._current_state_idx)
        goal = self._render_observation(self._current_goal_idx)

        self.last_state = { 
            'image': self._preprocess_frame(obs),
            'goal': self._preprocess_frame(goal)
        }

        self.last_action = 0
        self.last_reward = 0

    def _render_observation(self, idx):
        return self._observations[idx, :, :, :]

    def _preprocess_frame(self, image):        
        image = image.astype(np.float32)
        image = image / 255.0
        image = resize(image, self.image_size, anti_aliasing=True)
        return image

    @staticmethod
    def get_action_size(env_name):
        return 4

    @property
    def reward_configuration(self):
        return (1.0, 0.0, 0.0)

    def process(self, action):
        collided = False
        if self._transition_graph[self._current_state_idx][action] != -1:
            self._current_state_idx = self._transition_graph[self._current_state_idx][action]
        else:
            collided = True

        obs = self._render_observation(self._current_state_idx)
        goal = self._render_observation(self._current_goal_idx)
        terminal = self._current_goal_idx == self._current_state_idx
        reward = -self.reward_configuration[1]
        if terminal:
            reward = self.reward_configuration[0]
        if collided:
            reward = self.reward_configuration[2]

        if not terminal:
            state = { 
                'image': self._preprocess_frame(obs),
                'goal': self._preprocess_frame(goal)
            }
        else:
            state = self.last_state
        
        pixel_change = self._calc_pixel_change(state['image'], self.last_state['image'])
        self.last_state = state
        self.last_action = action
        self.last_reward = reward
        return state, reward, terminal, pixel_change