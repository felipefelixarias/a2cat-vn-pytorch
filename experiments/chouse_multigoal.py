import environments
import numpy as np

import deep_rl
from deep_rl import register_trainer, register_agent
from deep_rl.a2c_unreal.unreal import UnrealTrainer, UnrealAgent
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.schedules import LinearSchedule, MultistepSchedule
from torch import nn
from deep_rl.model import TimeDistributed, Flatten, MaskedRNN
from deep_rl.common.tester import TestingEnv, TestingVecEnv
from models import GoalUnrealModel
import math

TestingEnv.set_hardness = lambda _, hardness: print('Hardnes was set to %s' % hardness)
TestingVecEnv.set_hardness = lambda _, hardness: print('Hardnes was set to %s' % hardness)

@register_agent()
class Agent(UnrealAgent):
    def create_model(self):
        return GoalUnrealModel(3, 6)


@register_trainer(max_time_steps = 40e6, validation_period = None, validation_episodes = None,  episode_log_interval = 10, saving_period = 100000, save = True)
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
        #self.pc_cell_size = 
        
        self.scene_complexity = MultistepSchedule(0.3, [
            (5000000, LinearSchedule(0.3, 1.0, 5000000)),
            (10000000, 1.0)
        ])

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0][0]
    
    def create_env(self, envkwargs):
        env = super().create_env(envkwargs)
        env.set_hardness = lambda hardness: env.call_unwrapped('set_hardness', hardness)
        if hasattr(self, 'validation_env') and self.validation_env is not None:
            valid_env = self.validation_env
            valid_env.set_hardness = lambda hardness: valid_env.call_unwrapped('set_hardness', hardness)
        return env

    def create_model(self):
        return GoalUnrealModel(self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)

    def process(self, *args, **kwargs):
        a, b, metric_context = super().process(*args, **kwargs)
        self.env.set_hardness(self.scene_complexity)
        metric_context.add_last_value_scalar('scene_complexity', self.scene_complexity)
        return a, b, metric_context

def default_args():
    return dict(
        env_kwargs = dict(
            id = 'GoalHouse-v1', 
            screen_size=(84,84), 
            scene = '05cac5f7fdd5f8138234164e76a97383', 
            hardness = 0.3,
            configuration=deep_rl.configuration.get('house3d').as_dict()),
        model_kwargs = dict()
    )