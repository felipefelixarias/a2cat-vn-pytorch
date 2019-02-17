if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

import deepq.dqn
from common.train_wrappers import wrap
import gym
import gym_maze
import deepq.catch_experiment
from keras.layers import Input, Dense, Concatenate, Lambda
import keras.backend as K

class Trainer(deepq.dqn.DeepQTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'deepq-cartpole'

        self.annealing_steps = 10000
        self.preprocess_steps = 1000
        self.replay_size = 50000
        self.max_episode_steps = None


    def create_inputs(self, name):
        return [Input(shape = (4,), name = name + '_input')]

    def create_backbone(self, action_space_size, **kwargs):
        
        layer1 = Dense(
            units=256,
            activation="relu",
            name="fc3")

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
            model = layer1(inputs[0])
            model = final_merge([value(model), adventage(model)])
            return model

        return call

    def _wrap_env(self, env):
        return env

if __name__ == '__main__':
    trainer = Trainer(
        env_kwargs = dict(id='CartPole-v0'), 
        model_kwargs = dict(action_space_size = 2))

    trainer = wrap(trainer, max_time_steps=100000, episode_log_interval=10).compile()
    trainer.run()

else:
    raise('This script cannot be imported')