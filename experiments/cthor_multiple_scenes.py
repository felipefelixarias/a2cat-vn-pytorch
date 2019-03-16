import environments
import numpy as np
import gym

from deep_rl import register_trainer
from deep_rl.a2c_unreal import UnrealTrainer
from deep_rl.a2c_unreal.util import UnrealEnvBaseWrapper
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.schedules import LinearSchedule

from deep_rl.common.vec_env import SubprocVecEnv, DummyVecEnv
from deep_rl.common.env import RewardCollector, TransposeImage, ScaledFloatFrame

@register_trainer(max_time_steps = 40e6, validation_period = 200, validation_episodes = 20,  episode_log_interval = 10, saving_period = 100000, save = True)
class Trainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 8
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.num_steps = 20
        self.gamma = .99
        self.allow_gpu = True
        self.learning_rate = LinearSchedule(7e-4, 0, self.max_time_steps)

        self.rp_weight = 1.0
        self.pc_weight = 0.05
        self.vr_weight = 1.0

    def _create_env(self, scenes, validation_scenes, **kwargs):
        def _create_thunk(scenes):
            def _thunk():
                env = gym.make(scenes = scenes, **kwargs)
                env = RewardCollector(env)
                env = TransposeImage(env)
                env = ScaledFloatFrame(env)
                env = UnrealEnvBaseWrapper(env)
                return env
            return _thunk

        self.validation_env = DummyVecEnv([_create_thunk(validation_scenes)])
        return SubprocVecEnv([_create_thunk(scenes) for _ in range(self.num_processes)])

    def create_env(self, env):
        return self._create_env(**env)


def default_args():
    return dict(
        env_kwargs = dict(
            id = 'ContinuousThor-v0', 
            goals = ['laptop'], 
            cameraY = 0.2, 
            screen_size=(84,84),
            scenes = list(range(201, 230)),
            validation_scenes = [230]
        ),
        model_kwargs = dict()
    )