if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

from deepq import dqn as dqn
from common.train_wrappers import wrap

import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU
from keras.models import Model
import keras.backend as K
from common import register_trainer, make_trainer, register_agent, make_agent
from deepq.models import atari

from graph.env import SimpleGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
import numpy as np

register_agent('deepq-dungeon-dynamic')(dqn.DeepQAgent)
@register_trainer('deepq-dungeon-dynamic', max_time_steps = 100000, validation_period = 100, episode_log_interval = 10)
class Trainer(dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.annealing_steps = 10000
        self.preprocess_steps = 1000
        self.replay_size = 50000
        self.minibatch_size = 32
        self.gamma = .95
        self.max_episode_steps = None
        self.rewards = [0.0]


    def create_inputs(self, name, **kwargs):
        return [Input(shape = size + (3,), name = name + '_input')]

    def create_model(self, inputs, **kwargs):
        return atari(inputs, 4)

    def wrap_env(self, env):
        self.env = env
        return env

    def process(self, **kwargs):
        a, episode_end, c = super().process(**kwargs)
        if episode_end:
            length, reward = episode_end
            self.rewards.append(reward)
            complexity = np.mean(self.rewards[-100:])
            self.env.unwrapped.set_complexity(complexity)
        return a, episode_end, c

if __name__ == '__main__':
    size = (20, 20)

    with open('./scenes/dungeon-%s-1.pkl' % size[0], 'rb') as f:
        graph = load_graph(f)

    env = TimeLimit(SimpleGraphEnv(graph, graph.goal), max_episode_steps = 100)
    env.unwrapped.set_complexity(None)
    trainer = make_trainer(
        id = 'deepq-dungeon-dynamic',
        env_kwargs = env,
        model_kwargs = dict(action_space_size = 4)
    )

    trainer.run()