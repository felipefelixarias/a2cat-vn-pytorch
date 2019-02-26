from abc import ABC, abstractclassmethod
class AuxiliaryTask(ABC):
    def __init__(self, name):
        self.name = name

    @abstractclassmethod
    def build_heads(self, model):
        pass

    @abstractclassmethod
    def build_optimization_placeholders(self):
        pass

    @abstractclassmethod
    def build_loss(self, heads, placeholders):
        pass

    @abstractclassmethod
    def bind_input(self, observation):
        pass

    def wrap_env(self, env):
        return env


task_registry = dict()
def register_auxiliary_task(name):
    def reg(task):
        task_registry[name] = dict(
            task = task
        )

    return reg