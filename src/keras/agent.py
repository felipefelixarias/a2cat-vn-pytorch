import io
import os.path as path
import json

from model.model_keras import ActorCriticModel

class Agent:
    def __init__(self, checkpoint_dir, model_fn, model_kwargs = None):
        self._checkpoint_dir = None
        self._model_fn = model_fn
        self._model_kwargs = model_kwargs
        self._model = None

    def _load(self, checkpoint_dir):
        print('Restoring model')
        model_kwargs = None
        with io.open(path.join(checkpoint_dir, 'config.json'), 'r') as f:
            model_kwargs = json.load(f)

        print('Using provided checkpoint config')
        #if self._model_kwargs is not None:
        #    model_kwargs.update(self._model_kwargs)

        return (self._model_fn(**model_kwargs), model_kwargs)

    def save(self):
        print('Saving model')
        with io.open(path.join(self._checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(self._model_kwargs, f)

    def initialize(self):
        if self._checkpoint_dir is not None:
            (self._model, self._model_kwargs) = self._load(self._checkpoint_dir)
        else:
            self._model = self._model_fn(**self._model_kwargs)

    def _wrap_state(self, state, last_action_reward):
        return [[state['image']], [state['goal']], [last_action_reward]]

    def act(self, state, last_action_reward):
        state_wrap = self._wrap_state(state, last_action_reward)
        return self._model.predict(state_wrap)[0]