import io
import os.path as path
import os
import json

from model.model_keras import ActorCriticModel

class Agent:
    def __init__(self, checkpoint_dir, model_fn, device, learning_rate, beta, name = 'net', model_kwargs = None):
        self._checkpoint_dir = checkpoint_dir
        self._model_fn = model_fn
        self._model_kwargs = model_kwargs
        self._name = name
        self._device = device
        self._model = None
        self.learning_rate = learning_rate[0]
        self.beta = beta[0]

    def _load(self, checkpoint_dir):
        print('Restoring model')
        model_kwargs = None
        with io.open(path.join(checkpoint_dir, 'config.json'), 'r') as f:
            model_kwargs = json.load(f)

        print('Using provided checkpoint config')
        #if self._model_kwargs is not None:
        #    model_kwargs.update(self._model_kwargs)
        model = self._model_fn(name = self._name, device = self._device, **model_kwargs)
        print("Loading weights")
        model.load_weights(path.join(checkpoint_dir, '%s.h5' % self._name))
        return (model, model_kwargs)

    def save(self):
        print('Saving model')
        os.makedirs(self._checkpoint_dir)
        with io.open(path.join(self._checkpoint_dir, 'config.json'), 'w+') as f:
            json.dump(self._model_kwargs, f)

        weights_file = path.join(self._checkpoint_dir, '%s.h5' % self._name)
        self._model.save_weights(weights_file)
        print('Saving model weights to "%s"' % weights_file)

    def initialize(self):
        if self._checkpoint_dir is not None:
            (self._model, self._model_kwargs) = self._load(self._checkpoint_dir)
        else:
            self._model = self._model_fn(name = self._name, device = self._device, **self._model_kwargs)

    def _wrap_state(self, state, last_action_reward):
        return [[state['image']], [state['goal']], [last_action_reward]]

    def act(self, state, last_action_reward):
        state_wrap = self._wrap_state(state, last_action_reward)
        return self._model.predict(state_wrap)[0]

class ActorCriticTrainingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_kwargs = kwargs.get('model_kwargs')
        if model_kwargs.get('head') != 'ac':
            raise Exception('Cannot use non-actor-critic model with actor-critic head.')

class DeepQTrainingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_kwargs = kwargs.get('model_kwargs')
        if model_kwargs.get('head') != 'dqn':
            raise Exception('Cannot use non-DQN model with DQN agent.')
