if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import gym
import gym.spaces
import environment.qmaze
import numpy as np
import gym_2048

from a2c.a2c import learn
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv


np.warnings.filterwarnings('ignore')
n_envs = 4


class EnvWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (16,), dtype = np.float32)

    def reset(self):
        self.env.reset()
        return self.observation(self.unwrapped.board)

    def step(self, action):
        state, reward, done, k = self.env.step(action)
        reward = float(reward)

        win = None
        if done:
            if np.max(state) >= 2048:
                k.update(dict(win = True))
            else:
                k.update(dict(win = False))

        return self.observation(state), np.clip(reward, -1, 1), done, k

    def observation(self, observation):
        return np.reshape(np.log(1.0 + observation) / np.log(2048), (-1,))

env = DummyVecEnv([lambda: EnvWrapper(gym.make('2048-v0')) for _ in range(n_envs)])


learn('mlp', env)