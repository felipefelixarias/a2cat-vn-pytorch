if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import deepq.dqn
from common.train_wrappers import wrap
import gym
import gym_2048
from keras.layers import Input, Dense, Concatenate, Lambda
import keras.backend as K
import numpy as np
from deepq.models import mlp
from common import register_trainer, make_trainer, register_agent

class EnvWrapper(gym.Wrapper):
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

@register_trainer('deepq-2048', max_time_steps=1000000, episode_log_interval=10, saving_period = 100000)
class Trainer(deepq.dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
            env_kwargs = dict(id='2048-v0'), 
            model_kwargs = dict(action_space_size = 4), 
            **kwargs)

        self.gamma = 0.99
        self.annealing_steps = 100000
        self.preprocess_steps = 10000
        self.learning_rate = 0.001
        self.replay_size = 50000
        self.max_episode_steps = None

    def wrap_env(self, env):
        env.reset()
        return EnvWrapper(env)

    def create_inputs(self, name, **kwargs):
        return [Input(shape = (16,), name = name + '_input')]

    def create_model(self, inputs, action_space_size, **kwargs):
        return mlp(inputs, action_space_size)

register_agent('deepq-2048')(deepq.dqn.DeepQAgent)

if __name__ == '__main__':
    np.warnings.filterwarnings('ignore')
    trainer = make_trainer('deepq-2048')
    trainer.run()