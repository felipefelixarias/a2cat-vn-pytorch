import torch
from collections import namedtuple

ExperienceSample = namedtuple('Experience', ['observations', 'returns', 'actions', 'masks'])

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        self.obs = torch.zeros(num_processes, num_steps + 1, *obs_shape)
        self.rewards = torch.zeros(num_processes, num_steps)
        self.value_preds = torch.zeros(num_processes, num_steps + 1)
        self.returns = torch.zeros(num_processes, num_steps + 1)
        self.action_log_probs = torch.zeros(num_processes, num_steps)
        self.actions = torch.zeros(num_processes, num_steps).long()
        self.masks = torch.ones(num_processes, num_steps + 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[:,self.step + 1].copy_(obs)
        self.actions[:,self.step].copy_(actions)
        self.action_log_probs[:,self.step].copy_(action_log_probs)
        self.value_preds[:,self.step].copy_(value_preds)
        self.rewards[:,self.step].copy_(rewards)
        self.masks[:,self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[:,0].copy_(self.obs[:,-1])
        self.masks[:,0].copy_(self.masks[:,-1])

    def compute_returns(self, next_value, gamma):
        self.returns[:,-1] = next_value
        for step in reversed(range(self.rewards.size(1))):
            self.returns[:,step] = self.returns[:,step + 1] * \
                gamma * self.masks[:,step + 1] + self.rewards[:,step]

    def sample(self, next_value, gamma):
        self.compute_returns(next_value, gamma)
        return ExperienceSample(self.obs[:, :-1], self.returns[:, :-1], self.actions, self.masks)