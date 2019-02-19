import gym
import gym.spaces
from math import floor
import numpy as np
import random
import cv2

_objects = [
    dict(reward = -1.0, color = (255, 0, 0), type='pickable', count = 0.1),
    dict(reward = 1.0, color = (0, 255, 0), type='pickable', count = 0.05),
    dict(reward = -0.5, color = (0, 0, 0), type='wall', count = 0.1)
]

actions = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1)
]

class MushroomEnv(gym.Env):
    def __init__(self, size = (7, 7), window_size = (5,5), observation_size = (84,84)):
        space_counts = np.product(list(size))
        self.total_counts = [floor(space_counts * x['count']) for x in _objects]
        self.maze = None
        self.observation_space = gym.spaces.Box(0, 255, shape = observation_size + (3,), dtype = np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.window_size = (3, 3)
        self.size = size

        self.position = None
        self.random = random.Random()

    def seed(self, seed = None):
        self.random.seed(seed)

    def reset(self):
        self.maze = np.zeros(shape = self.size, dtype = np.uint8)
        self.free_cells = [(y, x) for y in range(self.size[0]) for x in range(self.size[1])]
        self.random.shuffle(self.free_cells)
        self._generate_items()
        self.position = self.free_cells[0]
        return self.observe()

    def _generate_items(self):
        for i, (tp, count) in enumerate(zip(_objects, self.total_counts)):
            position = self.free_cells.pop()
            for _ in range(count):
                self.maze[position[0], position[1]] = i + 1

    def _in_maze(self, position):
        y, x = position
        return x >= 0 and y >= 0 and y < self.size[0] and x < self.size[1]

    def _render_window(self, position):
        window = np.ndarray(self.window_size + (3,), dtype = np.uint8)
        window_position = (self.position[0] - self.window_size[0] // 2, self.position[1] - self.window_size[1] // 2)
        for y in range(window_position[0], window_position[0] + self.window_size[0]):
            for x in range(window_position[1], window_position[1] + self.window_size[1]):
                (wy, wx) = (y - window_position[0], x - window_position[1])
                if not self._in_maze((y,x)):
                    window[wy, wx] = [0, 0, 0]
                elif self.maze[y, x] == 0:
                    window[wy, wx] = [255, 255, 255]
                else:
                    window[wy, wx] = _objects[self.maze[y, x] - 1]['color']
        return window
                    
    def step(self, action):
        dif = actions[action]
        npos = (self.position[0] + dif[0], self.position[1] + dif[1])
        done = False
        reward = 0.0
        stats = dict()
        if not self._in_maze(npos):
            reward = -0.5
        elif self.maze[npos[0], npos[1]] == 0:
            self.position = npos
        else:
            obj_type = self.maze[npos[0], npos[1]] - 1
            obj = _objects[obj_type]
            reward = obj['reward']
            if obj['type'] == 'wall':
                pass
            elif obj['type'] == 'pickable':
                self.maze[npos[0], npos[1]] = 0
                nfree = self.free_cells.pop()
                self.free_cells.insert(0, npos)
                self.maze[nfree[0], nfree[1]] = obj_type + 1
                self.position = npos
        return self.observe(), reward, done, stats

    def observe(self):
        window = self._render_window(self.position)
        if (self.window_size + (3,)) != self.observation_space.shape:
            window = cv2.resize(window, tuple(list(self.observation_space.shape)[:-1]), interpolation = cv2.INTER_NEAREST)

        return window