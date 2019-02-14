import random
import abc

class AbstractAgent:
    def __init__(self, name):
        self.name = name

    @abc.abstractclassmethod
    def act(self, state, **kwargs):
        pass

class RandomAgent(AbstractAgent):
    def __init__(self, action_space_size, seed = None):
        super().__init__('random')
        self._action_space_size = action_space_size
        self._random = random.Random(x = seed)

    def act(self, state, **kwargs):
        return self._random.randrange(0, self._action_space_size)

class LambdaAgent(AbstractAgent):
    def __init__(self, name, act_fn, **kwargs):
        super().__init__(name)
        self.act = lambda state, **kwargs: act_fn(state, **kwargs)