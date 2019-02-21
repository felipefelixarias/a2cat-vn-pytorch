import gym
import gym.envs
from graph.util import step, is_valid_state, find_state, enumerate_states


class GraphEnv(gym.Env):
    def __init__(self, graph):
        self.observation_space = 