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
import keras.backend as K

class Trainer(deepq.dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'deepq-cartpole'

        self.epsilon_start = 0.1
        self.epsilon_end = 0.1
        self.preprocess_steps = 1000
        self.replay_size = 49 * 8
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