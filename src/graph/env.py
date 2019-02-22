import gym
import gym.spaces
from graph.util import step, is_valid_state, load_graph, enumerate_positions, sample_initial_state, sample_initial_position, direction_to_change
import numpy as np
from operator import add
import random

class OrientedGraphEnv(gym.Env):
    def __init__(self, graph, goal):
        self.goal = goal
        
        if isinstance(graph, str):
            with open(graph, 'rb') as f:
                self.graph = load_graph(f)
        else:
            self.graph = graph

        if graph.dtype == np.float32:
            self.observation_space = gym.spaces.Box(0.0, 1.0, graph.observation_shape, graph.dtype)
        elif graph.dtype == np.uint8:
            self.observation_space = gym.spaces.Box(0, 255, graph.observation_shape, graph.dtype)
        else:
            raise Exception('Unsupported observation type')

        self.action_space = gym.spaces.Discrete(4)
        self.state = None
        self.largest_distance = np.max(self.graph.graph)
        self.complexity = None

    @property
    def unwrapped(self):
        return self

    def set_complexity(self, complexity = None):
        self.complexity = complexity

    def reset(self):
        optimal_distance = None
        if self.complexity is not None:
            optimal_distance = self.complexity * (self.largest_distance - 1) + 1
        state = sample_initial_state(self.graph, self.goal, optimal_distance = optimal_distance)
        self.state = state
        return self.observe(self.state)

    def observe(self, state):
        return self.graph.render(state[:2], state[2])

    def step(self, action):
        nstate = step(self.state, action)
        if not is_valid_state(self.graph.maze, nstate):
            # We can either end the episode with failure
            # Or continue with negative reward
            return self.observe(self.state), 0.0, False, dict(state = self.state)

        else:
            self.state = nstate
            if self.state[:2] == self.goal:
                return self.observe(self.state), 1.0, True, dict(state = self.state, win = True)
            else:
                return self.observe(self.state), 0.0, False, dict(state = self.state)

    def render(self, mode = 'human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(self.observe(self.state))
            plt.show()


class SimpleGraphEnv(gym.Env):
    def __init__(self, graph, goal):
        self.goal = goal
        
        if isinstance(graph, str):
            with open(graph, 'rb') as f:
                self.graph = load_graph(f)
        else:
            self.graph = graph

        if graph.dtype == np.float32:
            self.observation_space = gym.spaces.Box(0.0, 1.0, graph.observation_shape, graph.dtype)
        elif graph.dtype == np.uint8:
            self.observation_space = gym.spaces.Box(0, 255, graph.observation_shape, graph.dtype)
        else:
            raise Exception('Unsupported observation type')

        self.action_space = gym.spaces.Discrete(4)
        self.state = None
        self.largest_distance = np.max(self.graph.graph)
        self.complexity = None

    @property
    def unwrapped(self):
        return self

    def set_complexity(self, complexity = None):
        self.complexity = complexity

    def reset(self):
        optimal_distance = None
        if self.complexity is not None:
            optimal_distance = self.complexity * (self.largest_distance - 1) + 1
        state = sample_initial_position(self.graph, self.goal, optimal_distance = optimal_distance)
        self.state = state
        return self.observe(self.state)

    def observe(self, state):
        return np.array(self.graph.render(state))

    def step(self, action):
        nstate = tuple(map(add, self.state, direction_to_change(action)))
        if not is_valid_state(self.graph.maze, nstate):
            # We can either end the episode with failure
            # Or continue with negative reward
            return self.observe(self.state), -0.25, False, dict(state = self.state)

        else:
            self.state = nstate
            if self.state[:2] == self.goal:
                return self.observe(self.state), 1.0, True, dict(state = self.state, win = True)
            else:
                return self.observe(self.state), -0.04, False, dict(state = self.state)

    def render(self, mode = 'human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(self.observe(self.state))
            plt.show()
        elif mode == 'rgbarray':
            array = (self.observe(self.state) * 255).astype(np.uint8)
            import cv2
            return cv2.resize(array, (300, 300), interpolation = cv2.INTER_NEAREST)