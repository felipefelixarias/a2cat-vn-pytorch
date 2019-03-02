import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a2c_ppo_acktr.utils import init

class FixedCategorical(torch.distributions.Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self):
        return super().sample().unsqueeze(-1)

    def mode(self):
        return super().probs.argmax(dim= -1, keepdim = True)

    def log_prob(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(Policy, self).__init__()
        num_outputs = action_space.n

        self.base = CNNBase(obs_shape[0], num_outputs)        
        self.dist = lambda policy_logits: FixedCategorical(logits = policy_logits)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        policy_logits, value = self.base(inputs, masks)
        dist = self.dist(policy_logits)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        _, value = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, masks, action):
        policy_logits, value = self.base(inputs, masks)
        dist = self.dist(policy_logits)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
    

class CNNBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        hidden_size = 512
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0), 0.01)

        self.policy_logits = init_(nn.Linear(hidden_size, num_outputs))
        self.train()

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, masks):
        x = self.main(inputs / 255.0)
        return self.policy_logits(x), self.critic_linear(x)