from keras.layers import Dense, Conv2D, Input, Flatten, Concatenate, Lambda
from keras.models import Model
from keras import optimizers
from common.train import SingleTrainer, AbstractTrainer
from common.env_wrappers import ColorObservationWrapper
from trfl import double_qlearning
import keras.backend as K
from functools import partial
import tensorflow as tf
import numpy as np
import random
import abc

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

def create_model(action_space_size):
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

def update_target_graph(main_graph, target_graph, tau):
    updated_weights = (np.array(main_graph.get_weights()) * tau) + \
        (np.array(target_graph.get_weights()) * (1 - tau))
    target_graph.set_weights(updated_weights)

class DeepQTrainer(SingleTrainer):
    def __init__(self, env_kwargs, model_kwargs):
        self.name = 'doubledeepq'        
        self.minibatch_size = 32
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        
        self.annealing_steps = 100000
        self.preprocess_steps = 100000
        self.replay_size = 50000
        self.model_kwargs = model_kwargs
        self.max_episode_steps = None
        self.sync_steps = 10000  
        
        # Initialize model and everything
        super().__init__(env_kwargs, model_kwargs)

        self._global_t = 0
        self._state = None
        self._episode_length = 0
        self._episode_reward = 0.0
        self._local_timestep = 0
        self._replay = None

    def _wrap_env(self, env):
        return ColorObservationWrapper(env)

    @abc.abstractclassmethod
    def create_inputs(self, name = 'main'):
        pass

    def _build_loss(self, q, actions, rewards, pcontinues, qtarget, qselector):
        return double_qlearning(q, actions, rewards, pcontinues, qtarget, qselector)

    @abc.abstractclassmethod
    def create_backbone(self, *args, **kwargs):
        pass

    def _build_model_for_training(self, **kwargs):
        actions = tf.placeholder(tf.uint8, (None,))
        rewards = tf.placeholder(tf.float32, (None,))
        terminals = tf.placeholder(tf.bool, (None,))

        inputs = self.create_inputs('main')
        model_stream = self.create_backbone()
        target_model_stream = self.create_backbone()

        q = model_stream(inputs)
        q_selector = target_model_stream(inputs)
        model = Model(inputs = inputs, outputs = [q])
        target_model = Model(inputs = inputs, outputs = [q_selector])

        # Create predict function
        model.predict_on_batch = K.function(inputs = inputs, outputs = [K.argmax(q, axis = 1)])
        
        # Next input targets
        next_step_inputs = self.create_inputs('next')
        next_q = K.stop_gradient(model_stream(next_step_inputs))
        
        # Build loss
        pcontinues = (1.0 - tf.to_float(terminals)) * self.gamma
        loss, _ = self._build_loss(q, actions, rewards, pcontinues, next_q, q_selector)
        loss = K.mean(loss)


        # Build optimize
        optimizer = tf.train.AdamOptimizer(0.001)
        update = optimizer.minimize(loss)
        train_on_batch = K.Function(inputs + [actions, rewards, terminals] + next_step_inputs, [loss], updates = [update])
        model.train_on_batch = train_on_batch

        # Return model for evaluation and training
        return model, partial(update_target_graph, model, target_model)

    def _create_model(self, **model_kwargs):
        model, self._update_weights = self._build_model_for_training(**model_kwargs)
        self._update_weights(1.0)
        model.summary()
        return model

    @property
    def epsilon(self):
        start_eps = self.epsilon_start
        end_eps = self.epsilon_end
        if self._global_t < self.preprocess_steps:
            return start_eps

        return max(start_eps - (start_eps - end_eps) * ((self._global_t - self.preprocess_steps) / self.annealing_steps), end_eps)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.model_kwargs.get('action_space_size'))

        return self.model.predict_on_batch([[state]])[0][0]


    def process(self):
        episode_end = None

        if self._state is None:
            self._state = self.env.reset()
        if self._replay is None:
            self._replay = Replay(self.replay_size)

        old_state = self._state
        action = self.act(self._state)

        self._state, reward, done, _ = self.env.step(action)
        self._replay.add((old_state, action, reward, done, self._state))

        self._episode_length += 1
        self._episode_reward += reward

        if done or (self.max_episode_steps is not None and self._episode_length >= self.max_episode_steps):
            episode_end = (self._episode_length, self._episode_reward)
            self._episode_length = 0
            self._episode_reward = 0.0
            self._state = self.env.reset()

        stats = None
        if self._global_t >= self.preprocess_steps:
            state, action, reward, done, next_state = self._replay.sample(self.minibatch_size)
            loss = self.model.train_on_batch([state, action, reward, done, next_state])
            stats = dict(loss = loss, epsilon = self.epsilon)

            if self._global_t % self.sync_steps == 0:
                self._update_weights(1.0)
        
        self._global_t += 1
        return (1, episode_end, stats)