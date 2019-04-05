from experiments.data import TRAIN2, VALIDATION2
from environments.gym_house.multi import create_multiscene
from deep_rl.common.env import RewardCollector, TransposeImage, ScaledFloatFrame
from deep_rl.common.vec_env import DummyVecEnv, SubprocVecEnv
from deep_rl.a2c_unreal.util import UnrealEnvBaseWrapper
from deep_rl.common.schedules import LinearSchedule, MultistepSchedule

import deep_rl
import environments
import numpy as np

from deep_rl import register_trainer
from deep_rl.a2c_unreal import UnrealTrainer
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.schedules import LinearSchedule

from torch import nn
from deep_rl.model import TimeDistributed, Flatten, MaskedRNN
from models import GoalUnrealModel
import math

VALIDATION_PROCESSES = 1 # note: single environment is supported at the moment


@register_trainer(max_time_steps = 40e6, validation_period = None, validation_episodes = None,  episode_log_interval = 10, saving_period = 100000, save = True)
class Trainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
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
        #self.pc_cell_size = 

        self.scene_complexity = MultistepSchedule(0.3, [
            (5000000, LinearSchedule(0.3, 1.0, 5000000)),
            (10000000, 1.0)
        ])

    def process(self, *args, **kwargs):
        a, b, metric_context = super().process(*args, **kwargs)
        self.env.set_hardness(self.scene_complexity)
        metric_context.add_last_value_scalar('scene_complexity', self.scene_complexity)
        return a, b, metric_context

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0][0]

    def create_env(self, kwargs):
        env, self.validation_env = create_envs(self.num_processes, kwargs)
        return env

    def create_model(self):
        return GoalUnrealModel(6, self.env.action_space.n)


def create_envs(num_training_processes, env_kwargs):
    def wrap(env):
        env = RewardCollector(env)
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        env = UnrealEnvBaseWrapper(env)
        return env

    env = create_multiscene(num_training_processes, TRAIN2, wrap = wrap, **env_kwargs)
    env.set_hardness = lambda hardness: env.call_unwrapped('set_hardness', hardness)
    val_env = create_multiscene(VALIDATION_PROCESSES, VALIDATION2, wrap = wrap, **env_kwargs)
    val_env.set_hardness = lambda hardness: val_env.call_unwrapped('set_hardness', hardness)
    val_env.set_hardness(0.6)
    return env, val_env


def default_args():
    return dict(
        env_kwargs = dict(
            id = 'GoalHouse-v1', 
            screen_size=(84,84),
            enable_noise = True,
            hardness = 0.3,
            configuration=deep_rl.configuration.get('house3d').as_dict()),
        model_kwargs = dict()
    )