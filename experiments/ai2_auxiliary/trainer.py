from deep_rl.a2c_unreal import UnrealTrainer
from deep_rl.a2c_unreal.unreal import without_last_item
from deep_rl.a2c_unreal.util import autocrop_observations
from deep_rl.common.pytorch import to_tensor
from torch.nn import functional as F
import torch


def compute_auxiliary_target(observations, cell_size = 4, output_size = None):
    with torch.no_grad():
        observations = autocrop_observations(observations, cell_size, output_size = output_size).contiguous()
        obs_shape = observations.size()
        abs_diff = observations.view(-1, *obs_shape[2:])        
        avg_abs_diff = F.avg_pool2d(abs_diff, cell_size, stride=cell_size)
        return avg_abs_diff.view(*obs_shape[:2] + avg_abs_diff.size()[1:])

def compute_auxiliary_targets(observations, cell_size, output_size):
    observations = observations[0]
    return tuple(map(lambda x: compute_auxiliary_target(x, cell_size, output_size), observations[2:]))

class AuxiliaryTrainer(UnrealTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auxiliary_weight = 0.05

    def sample_training_batch(self):
        values, report = super().sample_training_batch()
        aux_batch = self.replay.sample_sequence() if self.auxiliary_weight > 0.0 else None
        values['auxiliary_batch'] = aux_batch
        return values, report

    def compute_auxiliary_loss(self, model, batch, main_device):
        loss, losses = super().compute_auxiliary_loss(model, batch, main_device)
        auxiliary_batch = batch.get('auxiliary_batch')

        # Compute pixel change gradients
        if not auxiliary_batch is None:
            devconv_loss = self._deconv_loss(model, auxiliary_batch, main_device)
            loss += (devconv_loss * self.auxiliary_weight)
            losses['aux_loss'] = devconv_loss.item()

        return loss, losses

    def _deconv_loss(self, model, batch, device):
        observations, _, rewards, _ = batch
        observations = without_last_item(observations)
        masks = torch.ones(rewards.size(), dtype = torch.float32, device = device)
        initial_states = to_tensor(self._initial_states(masks.size()[0]), device)
        predictions, _ = model.forward_deconv(observations, masks, initial_states)
        targets = compute_auxiliary_targets(observations, model.deconv_cell_size, predictions[0].size()[3:])
        loss = 0
        for prediction, target in zip(predictions, targets):
            loss += F.mse_loss(prediction, target)

        return loss