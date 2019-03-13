import gym
import numpy as np
import gym.spaces
from graph.util import load_graph, step, sample_initial_state, is_valid_state
from .download import get_graph

class OrientedGraphEnv(gym.Env):
    def __init__(self, graph_name, rewards = [1.0, 0.0, 0.0]):
        self.graph = get_graph(graph_name)
        self.goal = self.graph.goal

        
        if self.graph.dtype == np.float32:
            self.observation_space = gym.spaces.Box(0.0, 1.0, self.graph.observation_shape, np.float32)
        elif self.graph.dtype == np.uint8:
            self.observation_space = gym.spaces.Box(0, 255, self.graph.observation_shape, np.uint8)
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
            optimal_distance = self.complexity * (self.largest_distance + 4 - 1) + 1
        state = sample_initial_state(self.graph, self.goal, optimal_distance = optimal_distance)
        self.state = state
        return self.observe(self.state)

    def observe(self, state):
        observation = self.graph.render(state[:2], state[2])
        return observation

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
        img = self.observe(self.state)
        if self.observation_space.dtype == np.float32:
            img = (255 * img).astype(np.uint8)

        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        elif mode == 'rgbarray':
            return img
        else:
            raise Exception(f"Render mode '{mode}' is not supported")
