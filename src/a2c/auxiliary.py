from abc import ABC, abstractclassmethod, abstractproperty
from collections import defaultdict

class AuxiliaryTask(ABC):
    def __init__(self, name, *args, **kwargs):
        self.name = name

    @abstractproperty
    def head_names(self):
        pass

    @abstractclassmethod
    def build_loss(self, heads, feed_builder):
        pass


task_registry = dict()
def register_auxiliary_task(name):
    def reg(task):
        task_registry[name] = dict(
            task = task
        )

    return reg

def create_auxiliary_task(name, **kwargs):
    return task_registry[name]['task'](**kwargs)


def build_auxiliary_losses(tasks, heads, names, feed_builder, **kwargs):
    heads = dict(zip(heads, names))
    losses = dict()
    for aux_task in tasks:
        task_heads = []
        for head_name in aux_task.head_names:
            hname = aux_task.name + '_' + head_name
            if not hname in heads:
                raise Exception('Missing head %s' % hname)

            task_heads.append(heads[hname])

        loss = aux_task.build_loss(task_heads, feed_builder)
        losses[aux_task.name] = loss

    return losses

def merge_auxiliary_losses(losses, weights = defaultdict(lambda: 1.0)):
    total_loss = 0.0
    for key, loss in losses.items():
        total_loss += weights[key] * loss
    return total_loss

import trfl
import keras.backend as K
import tensorflow as tf

@register_auxiliary_task('pixel_control')
class PixelControl(AuxiliaryTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_size = (4, 4)

    @property
    def head_names(self):
        return ['main']

    def build_loss(self, heads, feed_builder):
        head = heads[0][:, -1]
        actions = feed_builder.get('actions')
        observations = feed_builder.get('observations')
        terminals = feed_builder.get('terminals')
        number_of_actions = int(head.get_shape()[-1])
        actions_expanded = K.one_hot(K.expand_dims(K.expand_dims(actions, 2), 3), number_of_actions)

        q_prediction = head * actions_expanded
        q_true = tf.transpose(
            trfl.pixel_control_rewards(
                tf.transpose(observations, [1, 0, 2, 3, 4]), 
                self.cell_size
            ),
            [1, 0, 2, 3]
        )

        loss = tf.nn.l2_loss((q_true - q_prediction) * (1.0 - tf.to_float(terminals)))
        return tf.reduce_mean(loss)

@register_auxiliary_task('reward_prediction')
class RewardPrediction(AuxiliaryTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def head_names(self):
        return ['main']

    def build_loss(self, heads, feed_builder):
        head = heads[0][:, -1]
        rewards = feed_builder.get('rewards')
        terminals = feed_builder.get('terminals')
        r_true = tf.stack([rewards == 0, rewards > 0, rewards < 0], axis = -1)
        loss = tf.nn.softmax_cross_entropy_with_logits(head, r_true)
        loss = loss * (1.0 - tf.to_float(terminals))
        return tf.reduce_mean(loss)

@register_auxiliary_task('value_prediction')
class ValuePrediction(AuxiliaryTask):
    def __init__(self, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma

    @property
    def head_names(self):
        return ['main', 'next_value']

    def build_loss(self, heads, feed_builder):
        head = heads[0][:, -1]
        rewards = feed_builder.get('rewards')
        terminals = feed_builder.get('terminals')
        base_value = feed_builder.value[:, -1]
        pcontinues = self.gamma * (1.0 - tf.to_float(terminals))
        cummulative_rewards = trfl.scan_discounted_sum(
            tf.transpose(rewards),
            tf.transpose(pcontinues),
            base_value, 
            reverse=True,
            name='cummulative_reward')
        cummulative_rewards = tf.transpose(cummulative_rewards)
        return tf.nn.l2_loss(head - cummulative_rewards)