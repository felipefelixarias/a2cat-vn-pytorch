if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

from deepq import dqn as dqn
from common.train_wrappers import wrap

import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU, Flatten
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


register_agent('deepq-dungeon')(dqn.DeepQAgent)
@register_trainer('deepq-dungeon', max_time_steps = 1000000, validation_period = 100,  episode_log_interval = 10)
class Trainer(dqn.DeepQTrainer):
    def __init__(self, *args, q_figure = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.annealing_steps = 100000
        self.preprocess_steps = 1000
        self.replay_size = 50000
        self.minibatch_size = 64
        self.learning_rate = 0.001
        self.gamma = .90
        self.max_episode_steps = None
        self.q_figure = q_figure

    def process(self, **kwargs):
        ret = super().process(**kwargs)
        if self._global_t % 10000 == 0 or self._global_t == 1:
            q_figure.clf()
            display_q(self, q_figure)
            q_figure.canvas.flush_events()

        return ret

    def create_inputs(self, name, **kwargs):
        return [Input(shape = (20,20,3), name = name + '_input')] # size + (3,)

    def create_model(self, inputs, action_space_size, **kwargs):
        model = inputs[0]
        #model = Dense(64, activation = 'tanh')(inputs[0])
        #model = Dense(64, activation = 'tanh')(model)
        model = Lambda(output_shape = (20, 20, 1,),
            function = lambda model: model[:,:,:,:1] - model[:,:,:,2:])(model)
        model = Flatten()(model)
        action_stream = model
        # state_stream = inputs[0]
        #action_stream = Dense(256, activation = 'relu')(model)
        action_stream = Dense(action_space_size, activation = None)(action_stream)
        # state_stream = Dense(256, activation = 'relu')(model)
        # state_stream = Dense(1, activation = None)(state_stream)
        # model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)))([state_stream, action_stream])
        model = action_stream
        return Model(inputs = inputs, outputs = [model])

    def wrap_env(self, env):
        return env
    
if __name__ == '__main__':
    size = (20, 20)
    plt.ion()

    q_figure = plt.figure()
    plt.show()

    with open('./scenes/dungeon-20-1.pkl', 'rb') as f:  #dungeon-%s-1.pkl' % size[0]
        graph = load_graph(f)

    env = TimeLimit(SimpleGraphEnv(graph), max_episode_steps = 100)
    #env.unwrapped.set_complexity(0.1)
    trainer = make_trainer(
        id = 'deepq-dungeon',
        env_kwargs = env,
        model_kwargs = dict(action_space_size = 4)
    )

    trainer.run()