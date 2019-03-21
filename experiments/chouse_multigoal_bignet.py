import environments
import numpy as np

import deep_rl
from deep_rl import register_trainer
from deep_rl.a2c_unreal import UnrealTrainer
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.schedules import LinearSchedule

from torch import nn
from deep_rl.model import TimeDistributed, Flatten, MaskedRNN
from models import BigGoalHouseModel
import math

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

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0][0]

    def create_model(self):
        return BigGoalHouseModel(self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)


def default_args():
    return dict(
        env_kwargs = dict(
            id = 'GoalHouse-v1', 
            screen_size=(224,224), 
            scene = '05cac5f7fdd5f8138234164e76a97383', 
            hardness = 0.3,
            configuration=deep_rl.configuration.get('house3d').as_dict()),
        model_kwargs = dict()
    )