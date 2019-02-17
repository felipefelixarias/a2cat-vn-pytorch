from threading import Thread
import tensorflow as tf
import random
import numpy as np

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

class Trainer(Thread):
    def __init__(self, env_fn, model_fn, global_net, optimizer):
        self._model_fn = model_fn
        self._env_fn = env_fn
        self._global_net = global_net
        self._optimizer = optimizer

        self._local_t = 0
        self._global_t = 0
        self._state = None
        self.local_max_t = 20

        self._local_experience = []
        pass

    def _initialize(self):
        self.model = self._model_fn()
        self.env = self._env_fn()

        main_inputs = self.model.create_inputs('main')
        self.model = self._model_fn(main_inputs)

    def _build_optimize(self):


    def act(self, state):
        pass

    def train_on_batch(self):
        pass

    def process(self, global_t, **kwargs):
        tdiff = 0

        if self._state is None:
            self._state = self.env.reset()
        
        action = self.act(self._state)
        next_state, reward, done, stats = self.env.step(action)

        self._local_experience.append((self._state, action, reward, done, next_state))
        self._state = next_state


        if done or self._local_t >= self.local_max_t:
            tdiff = self._local_t
            self._local_t = 0
            self._local_experience = []

            



        return (tdiff, )
        pass