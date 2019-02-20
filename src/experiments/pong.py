if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

from a2c.a2c import Trainer
from keras.layers import Dense, Input, TimeDistributed, Conv2D, Flatten
from keras import initializers
from common import register_trainer, make_trainer


@register_trainer('test-a2c', episode_log_interval = 100, max_time_steps = 10e6, save = False)
class SomeTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(env_kwargs = dict(id = 'PongNoFrameskip-v4'), model_kwargs = dict(), **kwargs)

    def create_inputs(self, name):
        return Input(batch_shape=(self.n_env, None, 49))

    def create_model(self, inputs):
        from keras.models import Model
        from math import sqrt

        layer_initializer = initializers.Orthogonal(gain=sqrt(2))

        model = TimeDistributed(Conv2D(
            filters = 32,
            kernel_size = 8,
            strides = 4,
            activation = 'relu',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(inputs)

        model = TimeDistributed(Conv2D(
            filters = 64,
            kernel_size = 4,
            strides = 2,
            activation = 'relu',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(model)

        model = TimeDistributed(Conv2D(
            filters = 32,
            kernel_size = 3,
            strides = 1,
            activation = 'relu',
            bias_initializer = 'zeros',
            kernel_initializer = layer_initializer))(model)

        model = TimeDistributed(Flatten())(model)

        model = TimeDistributed(Dense(512, 
            activation = 'relu',
            bias_initializer='zeros',
            kernel_initializer=layer_initializer))(model)

        actor = TimeDistributed(Dense(
            units = 4, 
            activation= 'softmax',
            bias_initializer='zeros',
            kernel_initializer = initializers.Orthogonal(gain=0.01)))(model)


        critic = TimeDistributed(Dense(
            units = 1,
            activation = None,
            bias_initializer='zeros',
            kernel_initializer=layer_initializer
        ))(model)

        return Model(inputs = [inputs], outputs = [actor, critic])


if __name__ == '__main__':
    t = make_trainer('test-a2c')
    t.run()