import gym
import gym.spaces
from graph.util import step, is_valid_state, load_graph, enumerate_positions, sample_initial_state, sample_initial_position, direction_to_change
import numpy as np
from operator import add
import random

class OrientedGraphEnv(gym.Env):
    def __init__(self, graph, goal, rewards = [1.0, 0.0, 0.0]):
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
        self.rewards = rewards

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
            return self.observe(self.state), self.rewards[2], False, dict(state = self.state)

        else:
            self.state = nstate
            if self.state[:2] == self.goal:
                return self.observe(self.state), self.rewards[0], True, dict(state = self.state, win = True)
            else:
                return self.observe(self.state), self.rewards[1], False, dict(state = self.state)

    def render(self, mode = 'human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(self.observe(self.state))
            plt.show()


class SimpleGraphEnv(gym.Env):
    def __init__(self, graph, rewards = [1.0, 0.0, 0.0]):
        if isinstance(graph, str):
            with open(graph, 'rb') as f:
                self.graph = load_graph(f)
        else:
            self.graph = graph

        self.goal = self.graph.goal            

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
        self._rewards = rewards

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
        if action is None:
            # Return the latest observation
            return self.observe(self.state), 0.0, False, dict()

        nstate = tuple(map(add, self.state, direction_to_change(action)))
        if not is_valid_state(self.graph.maze, nstate):
            # We can either end the episode with failure
            # Or continue with negative reward
            return self.observe(self.state), self._rewards[2], False, dict(state = self.state)

        else:
            self.state = nstate
            if self.state[:2] == self.goal:
                return self.observe(self.state), self._rewards[0], True, dict(state = self.state, win = True)
            else:
                return self.observe(self.state), self._rewards[1], False, dict(state = self.state)

    def render(self, mode = 'human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(self.observe(self.state))
            plt.show()
        elif mode == 'rgbarray':
            array = (self.observe(self.state) * 255).astype(np.uint8)
            import cv2
            return cv2.resize(array, (300, 300), interpolation = cv2.INTER_NEAREST)






class MultipleGraphEnv(gym.Env):
    def __init__(self, graphs, rewards = [1.0, 0.0, 0.0]):        
        if isinstance(graphs[0], str):
            self.graphs = []
            for g in graphs:
                with open(g, 'rb') as f:
                    self.graphs.append(load_graph(f))
        else:
            self.graphs = graphs

        if self.graphs[0].dtype == np.float32:
            self.observation_space = gym.spaces.Box(0.0, 1.0, self.graphs[0].observation_shape, self.graphs[0].dtype)
        elif self.graphs[0].dtype == np.uint8:
            self.observation_space = gym.spaces.Box(0, 255, self.graphs[0].observation_shape, self.graphs[0].dtype)
        else:
            raise Exception('Unsupported observation type')

        self.action_space = gym.spaces.Discrete(4)
        self.state = None
        self.largest_distances = [np.max(x.graph) for x in self.graphs]
        self.graph_number = None
        self.complexity = None
        self._rewards = rewards

    @property
    def unwrapped(self):
        return self

    def set_complexity(self, complexity = None):
        self.complexity = complexity

    def reset(self):
        self.graph_number = random.randrange(len(self.graphs))
        optimal_distance = None
        if self.complexity is not None:
            optimal_distance = self.complexity * (self.largest_distances[self.graph_number] - 1) + 1
        state = sample_initial_position(self.graphs[self.graph_number], self.graphs[self.graph_number].goal, optimal_distance = optimal_distance)
        self.state = state
        return self.observe(self.state)

    def observe(self, state):
        return np.array(self.graphs[self.graph_number].render(state))

    def step(self, action):
        if action is None:
            # Return the latest observation
            return self.observe(self.state), 0.0, False, dict()

        nstate = tuple(map(add, self.state, direction_to_change(action)))
        if not is_valid_state(self.graphs[self.graph_number].maze, nstate):
            # We can either end the episode with failure
            # Or continue with negative reward
            return self.observe(self.state), self._rewards[2], False, dict(state = self.state)

        else:
            self.state = nstate
            if self.state[:2] == self.graphs[self.graph_number].goal:
                return self.observe(self.state), self._rewards[0], True, dict(state = self.state, win = True)
            else:
                return self.observe(self.state), self._rewards[1], False, dict(state = self.state)

    def render(self, mode = 'human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(self.observe(self.state))
            plt.show()
        elif mode == 'rgbarray':
            array = (self.observe(self.state) * 255).astype(np.uint8)
            import cv2
            return cv2.resize(array, (300, 300), interpolation = cv2.INTER_NEAREST)