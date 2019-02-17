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

class Trainer(deepq.dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'deepq-2048'

        self.gamma = 0.99
        self.annealing_steps = 100000
        self.preprocess_steps = 100000
        self.replay_size = 100000
        self.max_episode_steps = None

    def _wrap_env(self, env):
        env.reset()
        return EnvWrapper(env)

    def create_inputs(self, name, **kwargs):
        return [Input(shape = (16,), name = name + '_input')]

    def create_backbone(self, action_space_size, **kwargs):
        layer1 = Dense(
            units=64,
            activation="relu",
            name="fc1")

        layer2 = Dense(
            units=256,
            activation="relu",
            name="fc2")


        adventage = Dense(
            units=action_space_size,
            activation=None,
            name="policy_fc"
        )

        value = Dense(
            units=1,
            name="value_fc"
        )

        final_merge = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)),name="final_out")

        def call(inputs):
            model = layer2(layer1(inputs[0]))
            model = final_merge([value(model), adventage(model)])
            return model

        return call


if __name__ == '__main__':
    trainer = Trainer(
        env_kwargs = dict(id='2048-v0'), 
        model_kwargs = dict(action_space_size = 4))

    trainer = wrap(trainer, max_time_steps=1000000, episode_log_interval=10).compile()
    trainer.run()

else:
    raise('This script cannot be imported')