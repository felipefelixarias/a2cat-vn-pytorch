if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import gym
import environment.qmaze

from baselines.a2c.a2c import learn
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

n_envs = 8
env = DummyVecEnv([lambda: gym.make('QMaze-v0') for _ in range(n_envs)])


learn('mlp', env)