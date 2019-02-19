if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

from deepq import dqn as dqn
from common.train_wrappers import wrap

import gym
import environment.qmaze
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU
from keras.models import Model
from deepq.models import atari
from gym.wrappers import TimeLimit
import keras.backend as K
from common.env_wrappers import ColorObservationWrapper
from common import register_trainer, make_trainer, register_agent, make_agent

@register_trainer('deepq-maze', max_time_steps = 1000000, episode_log_interval = 10)
class Trainer(dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.annealing_steps = 100000
        self.preprocess_steps = 1000
        self.learning_rate = 5e-4
        self.replay_size = 100000
        self.minibatch_size = 32
        self.gamma = 0.99
        self.max_episode_steps = None


    def create_inputs(self, name, maze_size, **kwargs):
        return [Input(shape = maze_size + (3,), name = name + '_input')]

    def create_model(self, inputs, action_space_size, **kwargs):
        return atari(inputs, action_space_size)

    def wrap_env(self, env):
        return env

register_agent('deepq-maze')(dqn.DeepQAgent)

if __name__ == '__main__':
    trainer = make_trainer(
        id = 'deepq-maze',
        env_kwargs = dict(id='Maze-v0'), 
        model_kwargs = dict(action_space_size = 4, maze_size = (84,84))
    )

    trainer.run()