from environments.gym_house.multi import create_multiscene
from deep_rl.common.env import RewardCollector, TransposeImage, ScaledFloatFrame
from deep_rl.common.vec_env import DummyVecEnv, SubprocVecEnv
from deep_rl.a2c_unreal.util import UnrealEnvBaseWrapper
from deep_rl.configuration import configuration
import deep_rl
import environments
import os
import numpy as np
import torch

from deep_rl import register_trainer
from experiments.ai2_auxiliary.trainer import AuxiliaryTrainer
from models import AuxiliaryBigGoalHouseModel5 as Model
from deep_rl.common.schedules import LinearSchedule, MultistepSchedule
from torch import nn
from deep_rl.model import TimeDistributed, Flatten, MaskedRNN
from deep_rl.common.tester import TestingEnv, TestingVecEnv
import math

VALIDATION_PROCESSES = 1 # note: single environment is supported at the moment

TestingEnv.set_hardness = lambda _, hardness: print('Hardnes was set to %s' % hardness)
TestingVecEnv.set_hardness = lambda _, hardness: print('Hardnes was set to %s' % hardness)

@register_trainer(max_time_steps = 40e6, validation_period = 200, validation_episodes = 20,  episode_log_interval = 10, saving_period = 100000, save = True)
class Trainer(AuxiliaryTrainer):
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
        self.auxiliary_weight = 0.1
        #self.pc_cell_size = 

        self.scene_complexity = MultistepSchedule(0.3, [
            (5000000, LinearSchedule(0.3, 1.0, 5000000)),
            (10000000, 1.0)
        ])

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0][0]

    def create_env(self, kwargs):
        env = create_envs(self.num_processes, **kwargs)
        return env

    def create_model(self):
        model = Model(self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)
        model_path = os.path.join(configuration.get('models_path'),'chouse-auxiliary4-supervised', 'weights.pth')
        print('Loading weights from %s' % model_path)
        model.load_state_dict(torch.load(model_path))
        return model

    def process(self, *args, **kwargs):
        a, b, metric_context = super().process(*args, **kwargs)
        self.env.set_hardness(self.scene_complexity)
        metric_context.add_last_value_scalar('scene_complexity', self.scene_complexity)
        return a, b, metric_context

def create_envs(num_training_processes, tasks, **env_kwargs):
    def wrap(env):
        env = RewardCollector(env)
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        env = UnrealEnvBaseWrapper(env)
        return env

    env_fns = [lambda: environments.make(graph_name = scene, goals = goal, **env_kwargs) for (scene, goals) in tasks for goal in goals]
    env = SubprocVecEnv(env_fns)
    env.set_hardness = lambda hardness: env.call_unwrapped('set_complexity', hardness)
    env.set_hardness(0.3)
    return env

def default_args():
    return dict(
        env_kwargs = dict(
            id = 'AuxiliaryGraph-v0',
            tasks = [('thor-cached-212', [(3, 1, 2), (13, 21, 3), (10, 2, 1), (10, 14, 0)]),
                ('thor-cached-208', [(6, 3, 1), (13, 3, 0), (7, 18, 2), (6, 25, 1)]),
                ('thor-cached-218', [(6, 22, 1), (7, 0, 0), (18, 18, 3), (13, 31, 3)]),
                ('thor-cached-225', [(3, 17, 2), (12, 17, 3), (15, 10, 0), (14, 8, 3)])
            ],
            screen_size=(172,172),),
        model_kwargs = dict()
    )