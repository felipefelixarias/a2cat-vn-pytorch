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
import deepq.catch_experiment
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU
from keras.models import Model
import keras.backend as K

class QMazeModel(Model):
    def __init__(self, maze_size, action_space_size, **kwargs):
        super().__init__(self, **kwargs)
        self.dense1 = Dense(maze_size, input_shape=(maze_size,))
        self.prelu1 = PReLU()
        self.dense2 = Dense(maze_size)
        self.prelu2 = PReLU()
        self.dense3 = Dense(action_space_size)

        self._maze_size = maze_size
        pass

    def create_inputs(self, name = 'main'):
        return [Input(shape = (self._maze_size,), name = name + '_input')]

    def call(self, inputs):
        model = inputs
        model = self.dense1(model)
        model = self.prelu1(model)
        model = self.dense2(model)
        model = self.prelu2(model)
        model = self.dense3(model)
        return model


class Trainer(dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'deepq-cartpole'

        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.annealing_steps = 10000
        self.preprocess_steps = 1000
        self.replay_size = 50000
        self.minibatch_size = 32
        self.gamma = 1.0
        self.max_episode_steps = None


    def create_inputs(self, name, maze_size, **kwargs):
        return [Input(shape = (maze_size,), name = name + '_input')]

    def create_model(self, inputs, action_space_size, maze_size, **kwargs):
        model = Dense(64, activation = 'tanh')(inputs[0])
        model = Dense(64, activation = 'tanh')(model)
        action_stream = Dense(256, activation = 'relu')(model)
        action_stream = Dense(4, activation = None)(action_stream)
        state_stream = Dense(256, activation = 'relu')(model)
        state_stream = Dense(1, activation = None)(state_stream)
        model = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)))([state_stream, action_stream])
        return Model(inputs = inputs, outputs = [model])

    def _wrap_env(self, env):
        return env

if __name__ == '__main__':
    trainer = Trainer(
        env_kwargs = dict(id='QMaze-v0'), 
        model_kwargs = dict(action_space_size = 4, maze_size = 49))

    trainer = wrap(trainer, max_time_steps=100000, episode_log_interval=10, save = False).compile()
    trainer.run()

else:
    raise('This script cannot be imported')