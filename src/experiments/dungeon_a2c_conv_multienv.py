from common.train_wrappers import wrap
import os
import gym
from functools import reduce
from keras.layers import Input, Dense, Concatenate, Lambda, PReLU, Flatten, Conv2D, TimeDistributed
from keras.models import Model
from keras import initializers
import keras.backend as K
from common import register_trainer, make_trainer, register_agent, make_agent
from a2c.a2c import A2CTrainer, A2CAgent
import numpy as np

from graph.env import SimpleGraphEnv, MultipleGraphEnv
from graph.util import load_graph
from gym.wrappers import TimeLimit
from experiments.util import display_q, display_policy_value
import matplotlib.pyplot as plt
import matplotlib

class FlatWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return np.reshape(observation, [-1])


register_agent('dungeon-a2c-conv-multienv')(A2CAgent)
@register_trainer('dungeon-a2c-conv-multienv', max_time_steps = 100000000, validation_period = 1000,  episode_log_interval = 100, saving_period = 500000)
class Trainer(A2CTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_envs = 32
        self.n_steps = 5
        self.total_timesteps = 100000000
        self.gamma = 1.0

        self._last_figure_draw = 0

    def create_model(self, action_space_size, **kwargs):
        inputs = [Input(batch_shape = (None, None, 20,20,3))]
        model = inputs[0]
        model = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation = 'relu'))(model)
        model = TimeDistributed(Flatten())(model)
        policy = TimeDistributed(Dense(64, activation = 'relu'))(model)
        policy = TimeDistributed(Dense(action_space_size, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01), activation = 'sigmoid'))(policy)
        value = TimeDistributed(Dense(64, activation = 'relu'))(model)
        value = TimeDistributed(Dense(1, activation = None, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain=0.01)))(value)

        model = Model(inputs = inputs, outputs = [policy, value])
        return model

    def on_graph_built(self, sess, policy_distribution, rnn_model, model, values, **kwargs):
        action = policy_distribution.mode()
        
        # Create step fn
        def eval_pv(observation):
            # This function takes single batch of observations
            # Returns also single batch of returns
            observation = np.expand_dims(np.expand_dims(observation, 0), 0)
            mask = np.zeros((1, 1))
            feed_dict = {
                model.inputs[0]: observation,
                **{state: value for state, value in zip(rnn_model.states_in, self._validation_initial_state)}
            }

            if rnn_model.mask is not None:
                feed_dict[rnn_model.mask] = mask

            action_v, value_v = sess.run([action, values], feed_dict=feed_dict)

            action_v = action_v.squeeze(0).squeeze(0)
            value_v = value_v.squeeze(0).squeeze(0)
            return [action_v, value_v]
        pass

        self._eval_pv = eval_pv

    def save(self, path):
        super().save(path)
        plt.figure(self._figure.number)
        plt.savefig(os.path.join(path, 'policy_value.pdf'), format = 'pdf')
        plt.savefig(os.path.join(path, 'policy_value.eps'), format = 'eps')

    def run(self, *args, **kwargs):
        self._figure = plt.figure()
        self._figure_window = None
        return super().run(*args, **kwargs)

    def process(self, mode = 'train', context = dict(), **kwargs):
        res = super().process(mode = mode, context = context, **kwargs)
        if mode == 'train' and (self._global_t - self._last_figure_draw > 100000 or self._last_figure_draw ==0):
            self._figure.clf()
            display_policy_value(self, self._figure)
            self._figure.canvas.flush_events()
            if 'visdom' in context:
                viz = context.get('visdom')
                self._figure_window = viz.matplot(plt, win = self._figure_window, opts = dict(
                    title = 'policy, value'
                ))
            self._last_figure_draw = self._global_t

        return res

def default_args():
    valid_env = lambda: TimeLimit(SimpleGraphEnv('./scenes/dungeon-20-1.pkl', rewards = [0.0, -1.0, -1.0]), max_episode_steps = 50)
    env = lambda: TimeLimit(MultipleGraphEnv(['./scenes/dungeon-%s-%s.pkl' % (20, i) for i in range(2, 32)], rewards=[0.0, -1.0, -1.0]), max_episode_steps = 50)
    return dict(
        env_kwargs = [valid_env] + [env for _ in range(32)],
        model_kwargs = dict(action_space_size = 4)
    )