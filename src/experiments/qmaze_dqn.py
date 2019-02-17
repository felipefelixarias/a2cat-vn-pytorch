if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import deepq.dqn
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


class Trainer(deepq.dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'deepq-cartpole'

        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.annealing_steps = 15000
        self.preprocess_steps = 1000
        self.replay_size = 10000
        self.minibatch_size = 32
        self.gamma = 0.95
        self.max_episode_steps = None


    def create_inputs(self, name, maze_size, **kwargs):
        return [Input(shape = (maze_size,), name = name + '_input')]

    def _optimize(self):
        state, action, reward, done, next_state = self._replay.sample(self.minibatch_size)
        for _ in range(7):
            self.model.train_on_batch([state, action, reward, done, next_state])
        return self.model.train_on_batch([state, action, reward, done, next_state])

    def create_backbone(self, action_space_size, maze_size, **kwargs):
        layers = []
        layers.append(Dense(maze_size, input_shape=(maze_size,)))
        layers.append(PReLU())
        layers.append(Dense(maze_size))
        layers.append(PReLU())
        layers.append(Dense(action_space_size))
        def call(inputs):
            return reduce(lambda res, layer: layer(res), layers, inputs[0])

        return call

    def _wrap_env(self, env):
        return env

if __name__ == '__main__':
    trainer = Trainer(
        env_kwargs = dict(id='QMaze-v0'), 
        model_kwargs = dict(action_space_size = 4, maze_size = 49))

    trainer = wrap(trainer, max_number_of_episodes=1000, episode_log_interval=16).compile()
    trainer.run()

else:
    raise('This script cannot be imported')