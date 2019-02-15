import random
import abc
import gym

class AbstractAgent:
    def __init__(self, name):
        self.name = name

    @abc.abstractclassmethod
    def act(self, state):
        pass

    def wrap_env(self, env):
        return env

    def reset_state(self):
        pass

class RandomAgent(AbstractAgent):
    def __init__(self, action_space_size, seed = None):
        super().__init__('random')
        self._action_space_size = action_space_size
        self._random = random.Random(x = seed)

    def act(self, state):
        return self._random.randrange(0, self._action_space_size)

class LambdaAgent(AbstractAgent):
    def __init__(self, name, act_fn, **kwargs):
        super().__init__(name)
        self.act = lambda state: act_fn(state, **kwargs)

class AbstractTrainer:
    def __init__(self, env_kwargs, model_kwargs):
        self.env = self._wrap_env(gym.make(**env_kwargs))
        self.model = self._create_model(**model_kwargs)
        pass

    def _wrap_env(self, env):
        return env

    @abc.abstractclassmethod
    def _create_model(self, model_kwargs):
        pass

    @abc.abstractclassmethod
    def process(self, **kwargs):
        pass

class AbstractTrainerWrapper(AbstractTrainer):
    def __init__(self, trainer, *args, **kwargs):
        self.trainer = trainer

    def _wrap_env(self, env):
        return self.trainer._wrap_env(env)

    def _create_model(self, model_kwargs):
        return self.trainer._create_model(model_kwargs)

    def process(self, **kwargs):
        return self.trainer.process(**kwargs)