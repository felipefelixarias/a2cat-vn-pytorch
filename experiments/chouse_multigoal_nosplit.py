import environments
import numpy as np

import deep_rl
from deep_rl import register_trainer
from deep_rl.a2c_unreal import UnrealTrainer
from deep_rl.a2c_unreal.model import UnrealModel
from deep_rl.common.schedules import LinearSchedule

import torch
from torch import nn
from deep_rl.model import TimeDistributed, Flatten, MaskedRNN
import math

class GoalUnrealModel(nn.Module):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        elif type(module) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.zeros_(module.bias.data)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight.data)        
            d = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(module.weight.data, -d, d)

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.conv_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(num_inputs * 2, 32, 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride = 2),
            nn.ReLU(),
        ))

        self.conv_merge = TimeDistributed(nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * 32, 256),
            nn.ReLU()
        ))

        self.main_output_size = 256
        
        self.critic = TimeDistributed(nn.Linear(self.main_output_size, 1))
        self.policy_logits = TimeDistributed(nn.Linear(self.main_output_size, num_outputs))

        self.lstm_layers = 1
        self.lstm_hidden_size = 256
        self.rnn = MaskedRNN(nn.LSTM(256 + num_outputs + 1, # Conv outputs + last action, reward
            hidden_size = self.lstm_hidden_size, 
            num_layers = self.lstm_layers,
            batch_first = True))

        self._create_pixel_control_network(num_outputs)
        self._create_rp_network()

        self.apply(self.init_weights)
        self.pc_cell_size = 4

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype = torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, inputs, masks, states):
        observations, last_reward_action = inputs
        image, goal = observations
        features = self.conv_base(torch.cat((image, goal), 2))
        features = self.conv_merge(features)
        features = torch.cat((features, last_reward_action,), dim = 2)
        return self.rnn(features, masks, states)

    def _create_pixel_control_network(self, num_outputs):
        self.pc_base = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 32 * 9 * 9),
            nn.ReLU()
        ))

        self.pc_action = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size = 4, stride = 2),
            nn.ReLU()
        ))

        self.pc_value = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, num_outputs, kernel_size = 4, stride=2),
            nn.ReLU()
        ))

    def _create_rp_network(self):
        self.rp = nn.Linear(9 ** 2 * 32 * 3, 3)

    def reward_prediction(self, inputs):
        observations, _ = inputs
        image, goal = observations
        features = self.conv_base(torch.cat((image, goal), 2))
        features = self.conv_base(features)
        features = features.view(features.size()[0], -1)
        features = self.rp(features)
        return features

    def pixel_control(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        features = self.pc_base(features)
        features = features.view(*(features.size()[:2] + (32, 9, 9)))
        action_features = self.pc_action(features)
        features = self.pc_value(features) + action_features - action_features.mean(2, keepdim=True)
        return features, states

    def value_prediction(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        critic = self.critic(features)
        return critic, states



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
        return GoalUnrealModel(self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)


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