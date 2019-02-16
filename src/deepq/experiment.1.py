from keras.layers import Dense, Conv2D, Input, Flatten, Concatenate, Lambda
from keras.models import Model
from keras import optimizers
from common.train import SingleTrainer, AbstractTrainer
from common.env_wrappers import ColorObservationWrapper
from trfl import qlearning
import keras.backend as K
import tensorflow as tf
import numpy as np
import random

def create_inputs(name = 'main'):
    return [Input(shape=list((84,84,)) + [3], name="%s_observation" % name)]

def create_model(actions):
    block1 = Conv2D(
        filters=32,
        kernel_size=[8,8],
        strides=[4,4],
        activation="relu",
        padding="valid",
        name="conv1")

    block2 = Conv2D(
        filters=32, #TODO: test 64
        kernel_size=[4,4],
        strides=[2,2],
        activation="relu",
        padding="valid",
        name="conv2")

    concatenate3 = Concatenate(3)

    layer3 = Conv2D(
        filters = 32,
        kernel_size =(1,1),
        strides = (1,1),
        activation = "relu",
        name = "merge"
    )

    flatten3 = Flatten()
    layer4 = Dense(
        units=256,
        activation="relu",
        name="fc3")

    adventage = Dense(
        units=actions,
        activation=None,
        name="policy_fc"
    )

    value = Dense(
        units=1,
        name="value_fc"
    )

    final_merge = Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1],axis=1,keepdims=True)),name="final_out")

    def call(inputs):
        streams = list(map(lambda x: block2(block1(x)), inputs))
        if len(streams) > 1:
            model = concatenate3(streams)
            model = layer3(model)
        else:
            model = streams[0]

        model = flatten3(model)
        model = layer4(model)
        model = final_merge([value(model), adventage(model)])
        return model

    return call

def build_loss(q, actions, rewards, pcontinues, qtarget):
    return qlearning(q, actions, rewards, pcontinues, qtarget)

def build_model_for_training(action_space_size, create_model = create_model):
    actions = tf.placeholder(tf.uint8, (None,))
    rewards = tf.placeholder(tf.float32, (None,))
    terminals = tf.placeholder(tf.bool, (None,))
    gamma = tf.placeholder_with_default(0.99, tuple())

    inputs = create_inputs()
    model_stream = create_model(action_space_size)
    q = model_stream(inputs)
    model = Model(inputs = inputs, outputs = [q])

    # Create predict function
    model.predict_on_batch = K.function(inputs = inputs, outputs = [K.argmax(q, axis = 1)])
    
    # Next input targets
    next_step_inputs = create_inputs('next')
    next_q = K.stop_gradient(model_stream(next_step_inputs))
    
    # Build loss
    pcontinues = (1.0 - tf.to_float(terminals)) * gamma
    loss, _ = build_loss(q, actions, rewards, pcontinues, next_q)
    loss = K.mean(loss)


    # Build optimize
    optimizer = tf.train.AdamOptimizer(0.001)
    update = optimizer.minimize(loss)
    train_on_batch = K.Function(model.inputs + [actions, rewards, terminals] + next_step_inputs, [loss], updates = [update])
    model.train_on_batch = train_on_batch

    # Return model for evaluation and training
    return model

class Replay:
    def __init__(self, size):
        self.buffer = []
        self.size = size
        self._last_idx = 0

    def add(self, item):
        if len(self.buffer) >= self.size:
            self.buffer[self._last_idx] = item
            self._last_idx = (self._last_idx + 1) % self.size
        else:
            self.buffer.append(item)

    def sample(self, n):
        batch = random.sample(self.buffer, min(n, len(self.buffer)))
        batch = list(map(lambda *x:np.stack(x, axis = 0), *batch))
        return batch

class DeepQTrainer(SingleTrainer):
    def __init__(self, env_kwargs, model_kwargs, annealing_steps, preprocess_steps = 10000, max_episode_steps = None):
        super().__init__(env_kwargs, model_kwargs)

        self._state = None
        self._episode_length = 0
        self._episode_reward = 0.0
        self._local_timestep = 0
        self._minibatch_size = 32
        self._global_t = 0
        self._annealing_steps = annealing_steps
        self._preprocess_steps = preprocess_steps
        self._replay = Replay(50000)
        self.model_kwargs = model_kwargs
        self._max_episode_steps = max_episode_steps

    def _wrap_env(self, env):
        return ColorObservationWrapper(env)

    def _create_model(self, **model_kwargs):
        model = build_model_for_training(**model_kwargs)
        model.summary()
        return model

    @property
    def _epsilon(self):
        start_eps = 1.0
        end_eps = 0.01
        if self._global_t < self._preprocess_steps:
            return start_eps

        return max(start_eps - (start_eps - end_eps) * ((self._global_t - self._preprocess_steps) / self._annealing_steps), end_eps)

    def act(self, state):
        if random.random() < self._epsilon:
            return random.randrange(self.model_kwargs.get('action_space_size'))

        return self.model.predict_on_batch([[state]])[0][0]


    def process(self):
        episode_end = None

        if self._state is None:
            self._state = self.env.reset()

        old_state = self._state
        action = self.act(self._state)

        self._state, reward, done, _ = self.env.step(action)
        self._replay.add((old_state, action, reward, done, self._state))

        self._episode_length += 1
        self._episode_reward += reward

        if done or (self._max_episode_steps is not None and self._episode_length >= self._max_episode_steps):
            episode_end = (self._episode_length, self._episode_reward)
            self._episode_length = 0
            self._episode_reward = 0.0
            self._state = self.env.reset()

        stats = None
        if self._global_t >= self._preprocess_steps:
            state, action, reward, done, next_state = self._replay.sample(self._minibatch_size)
            loss = self.model.train_on_batch([state, action, reward, done, next_state])
            stats = dict(loss = loss, epsilon = self._epsilon)
        
        self._global_t += 1
        return (1, episode_end, stats)