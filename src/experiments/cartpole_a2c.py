if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

from keras.layers import Dense, Input, TimeDistributed
from keras.models import Model, Sequential
from keras import initializers
from common import register_trainer, make_trainer
from a2c.a2c import A2CTrainer
import gym
import environment.qmaze


def mlp():
    return Sequential(layers = [
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
    ])

def get_model(n_envs, observation_space, action_space):
    input_placeholder = Input(batch_shape=(n_envs, None) + observation_space.shape)
    policy_latent = TimeDistributed(mlp())(input_placeholder)
    value_latent =TimeDistributed(mlp())(input_placeholder)
    
    policy_probs = TimeDistributed(Dense(action_space.n, bias_initializer = 'zeros', activation='softmax', kernel_initializer = initializers.Orthogonal(gain=0.01)))(policy_latent)
    value = TimeDistributed(Dense(1, bias_initializer = 'zeros', kernel_initializer = initializers.Orthogonal(gain = 1.0)))(value_latent)
    return Model(inputs = [input_placeholder], outputs = [policy_probs, value])


@register_trainer('test-a2c', episode_log_interval = 100, save = False)
class SomeTrainer(A2CTrainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = dict(id = 'CartPole-v0'), model_kwargs = dict(), **kwargs)

    def create_model(self, **model_kwargs):
        return get_model(self.n_envs, self.env.observation_space, self.env.action_space)


if __name__ == '__main__':
    t = make_trainer('test-a2c')
    t.run()

