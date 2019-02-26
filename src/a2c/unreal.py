from a2c.auxiliary import register_auxiliary_task, AuxiliaryTask
from gym.wrappers import ObservationWrapper
from trfl import pixel_control_ops

class UnrealEnvWrapper(ObservationWrapper):
    def observe(self, observation):
        pass


class UnrealAuxiliaryTask(AuxiliaryTask):
    def wrap_env(self, env):
        if not isinstance(env, UnrealEnvWrapper):
            return UnrealEnvWrapper(env)
        return env

@register_auxiliary_task('pixel-change')
class PixelChangeTask(UnrealAuxiliaryTask):
    def build_heads(self, model):
        pass

    def build_optimization_placeholders(self):
        pass

    def build_loss(self, heads, placeholders):
        pass

    def bind_input(self, observation):
        pixel_control_ops.pixel_control_loss()
        observation['pixel_change']