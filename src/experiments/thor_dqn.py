if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

from deepq import dqn as dqn
from common.train_wrappers import wrap

import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU, Flatten, Conv2D
from keras.models import Model
import keras.backend as K
from common import register_trainer, make_trainer, register_agent, make_agent
from deepq.models import mlp
import numpy as np

from graph.env import SimpleGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
from experiments.util import display_q
import matplotlib.pyplot as plt
import matplotlib

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])


register_agent('thor-dqn')(dqn.DeepQAgent)
@register_trainer('thor-dqn', max_time_steps = 1000000, validation_period = 100,  episode_log_interval = 10)
class Trainer(dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.annealing_steps = 100000
        self.preprocess_steps = 1000
        self.replay_size = 50000
        self.minibatch_size = 64
        self.learning_rate = 0.001
        self.gamma = 1.0
        self.max_episode_steps = None
        self.q_figure = None

    def process(self, **kwargs):
        ret = super().process(**kwargs)
        if self._global_t % 10000 == 0 or self._global_t == 1:
            self._q_figure.clf()
            display_q(self, self._q_figure)
            self._q_figure.canvas.flush_events()

        return ret

    def create_inputs(self, name, **kwargs):
        return [Input(shape = (20,20,3), name = name + '_input')] # size + (3,)

    def create_model(self, inputs, action_space_size, **kwargs):
        model = inputs[0]
        model = Conv2D(16, 8, strides=4, activation = 'relu')(model)
        model = Flatten()(model)
        model = Conv2D(32, 4, strides = 2, activation = 'relu')(model)
        action = Dense(256, activation = 'relu')(model)
        action = Dense(action_space_size, activation = None)(action)
        state = Dense(256, activation = 'relu')(model)
        state = Dense(1, activation = None)(state)
        model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)))([state, action])
        return Model(inputs = inputs, outputs = [model])

    def wrap_env(self, env):
        return env

    def run(self, *args, **kwargs):
        plt.ion()

        self._q_figure = plt.figure()
        plt.show()

        return super().run(*args, **kwargs)

def default_args():
    size = (20, 20)
    with open('./scenes/kitchen-84.pkl', 'rb') as f:
        graph = load_graph(f)

    env = TimeLimit(SimpleGraphEnv(graph, graph.goal, rewards=[0.0, -1.0, -1.0]), max_episode_steps = 100)
    #env.unwrapped.set_complexity(0.1)
    return dict(
        env_kwargs = env,
        model_kwargs = dict(action_space_size = 4)
    )

    
if __name__ == '__main__':
    trainer = make_trainer(
        id = 'thor-dqn',
        **default_args()
    )
    
    trainer.run()