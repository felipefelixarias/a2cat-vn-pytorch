# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import deque
import random
from util.segment_tree import MinSegmentTree, SumSegmentTree


class ExperienceFrame(object):
  def __init__(self, state, reward, action, terminal, pixel_change, last_action, last_reward):
    self.state = state
    self.action = action # (Taken action with the 'state')
    self.reward = np.clip(reward, -1, 1) # Reward with the 'state'. (Clipped)
    self.terminal = terminal # (Whether terminated when 'state' was inputted)
    self.pixel_change = pixel_change
    self.last_action = last_action # (After this last action was taken, agent move to the 'state')
    self.last_reward = np.clip(last_reward, -1, 1) # (After this last reward was received, agent move to the 'state') (Clipped)

  def get_last_action_reward(self, action_size):
    """
    Return one hot vectored last action + last reward.
    """
    return ExperienceFrame.concat_action_and_reward(self.last_action, action_size,
                                                    self.last_reward, self.state)

  def get_action_reward(self, action_size):
    """
    Return one hot vectored action + reward.
    """
    return ExperienceFrame.concat_action_and_reward(self.action, action_size,
                                                    self.reward, self.state)

  @staticmethod
  def concat_action_and_reward(action, action_size, reward, state):
    """
    Return one hot vectored action and reward.
    """
    action_reward = np.zeros([action_size+1])
    if action >= 0:
      action_reward[action] = 1.0

    action_reward[-1] = float(reward)
    objective = state.get('objective')
    if objective is not None:
      return np.concatenate((action_reward, objective))
    else:
      return action_reward

class Experience(object):
  def __init__(self, history_size):
    self._history_size = history_size
    self._frames = deque(maxlen=history_size)
    # frame indices for zero rewards
    self._zero_reward_indices = deque()
    # frame indices for non zero rewards
    self._non_zero_reward_indices = deque()
    self._top_frame_index = 0

  def get_debug_string(self):
    return "{} frames, {} zero rewards, {} non zero rewards".format(
      len(self._frames), len(self._zero_reward_indices), len(self._non_zero_reward_indices))

  def add_frame(self, frame):
    if frame.terminal and len(self._frames) > 0 and self._frames[-1].terminal:
      # Discard if terminal frame continues
      print("Terminal frames continued.")
      return

    frame_index = self._top_frame_index + len(self._frames)
    was_full = self.is_full()

    # append frame
    self._frames.append(frame)

    # append index
    if frame_index >= 3:
      if frame.reward == 0:
        self._zero_reward_indices.append(frame_index)
      else:
        self._non_zero_reward_indices.append(frame_index)
    
    if was_full:
      self._top_frame_index += 1

      cut_frame_index = self._top_frame_index + 3
      # Cut frame if its index is lower than cut_frame_index.
      if len(self._zero_reward_indices) > 0 and \
         self._zero_reward_indices[0] < cut_frame_index:
        self._zero_reward_indices.popleft()
        
      if len(self._non_zero_reward_indices) > 0 and \
         self._non_zero_reward_indices[0] < cut_frame_index:
        self._non_zero_reward_indices.popleft()


  def is_full(self):
    return len(self._frames) >= self._history_size


  def sample_sequence(self, sequence_size):
    # -1 for the case if start pos is the terminated frame.
    # (Then +1 not to start from terminated frame.)
    start_pos = np.random.randint(0, self._history_size - sequence_size -1)

    if self._frames[start_pos].terminal:
      start_pos += 1
      # Assuming that there are no successive terminal frames.

    sampled_frames = []
    
    for i in range(sequence_size):
      frame = self._frames[start_pos+i]
      sampled_frames.append(frame)
      if frame.terminal:
        break
    
    return sampled_frames

  
  def sample_rp_sequence(self):
    """
    Sample 4 successive frames for reward prediction.
    """
    if np.random.randint(2) == 0:
      from_zero = True
    else:
      from_zero = False
    
    if len(self._zero_reward_indices) == 0:
      # zero rewards container was empty
      from_zero = False
    elif len(self._non_zero_reward_indices) == 0:
      # non zero rewards container was empty
      from_zero = True

    if from_zero:
      index = np.random.randint(len(self._zero_reward_indices))
      end_frame_index = self._zero_reward_indices[index]
    else:
      index = np.random.randint(len(self._non_zero_reward_indices))
      end_frame_index = self._non_zero_reward_indices[index]

    start_frame_index = end_frame_index-3
    raw_start_frame_index = start_frame_index - self._top_frame_index

    sampled_frames = []
    
    for i in range(4):
      frame = self._frames[raw_start_frame_index+i]
      sampled_frames.append(frame)

    return sampled_frames


class ExperienceReplay():
  def __init__(self, size):
    self._storage = []
    self._maxsize = size
    self._next_idx = 0
  
  def add(self,experience):
    data = experience
    if self._next_idx >= len(self._storage):
      self._storage.append(data)
    else:
      self._storage[self._next_idx] = data

    self._next_idx = (self._next_idx + 1) % self._maxsize

  def extend(self, experience):
    for e in experience:
      self.add(e)

  @staticmethod
  def create_batch(items):
    STATE_INDICES = [0,3]
    def convert_dict(i):
      if len(items) == 0:
        return {}
      else:
        return {key:np.stack([y[i][key] for y in items], 0) for key in items[0][i].keys()}

    def convert(i):
      if i in STATE_INDICES:
        return convert_dict(i)
      return np.array([x[i] for x in items])

    return tuple([convert(i) for i in range(5)]) 

  def _encode_sample(self, idxes):
    items = [self._storage[i] for i in idxes]
    return ExperienceReplay.create_batch(items)
          
  def sample(self, batch_size, **kwargs):
    idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    return self._encode_sample(idxes) + (None, idxes)


  def update_priorities(self, batch_idxes, priorities):
      pass

class PrioritizedExperienceReplay(ExperienceReplay):
  def __init__(self, size, alpha = 0.6):
    super().__init__(size)
    self._alpha = alpha
    it_capacity = 1
    while it_capacity < size:
        it_capacity *= 2

    self._it_sum = SumSegmentTree(it_capacity)
    self._it_min = MinSegmentTree(it_capacity)
    self._max_priority = 1.0

  def add(self, *args, **kwargs):
    idx = self._next_idx
    super().add(*args, **kwargs)
    self._it_sum[idx] = self._max_priority ** self._alpha
    self._it_min[idx] = self._max_priority ** self._alpha

  def _sample_proportional(self, batch_size):
    res = []
    p_total = self._it_sum.sum(0, len(self._storage) - 1)
    every_range_len = p_total / batch_size
    for i in range(batch_size):
      mass = random.random() * every_range_len + i * every_range_len
      idx = self._it_sum.find_prefixsum_idx(mass)
      res.append(idx)
    return res

  def sample(self, batch_size, beta = 0.4, **kwargs):
    assert beta > 0

    idxes = self._sample_proportional(batch_size)

    weights = []
    p_min = self._it_min.min() / self._it_sum.sum()
    max_weight = (p_min * len(self._storage)) ** (-beta)

    for idx in idxes:
        p_sample = self._it_sum[idx] / self._it_sum.sum()
        weight = (p_sample * len(self._storage)) ** (-beta)
        weights.append(weight / max_weight)
    weights = np.array(weights)
    encoded_sample = self._encode_sample(idxes)
    return tuple(list(encoded_sample) + [weights, idxes])

  def update_priorities(self, idxes, priorities):
    priorities = priorities()
    assert len(idxes) == len(priorities)
    for idx, priority in zip(idxes, priorities):
      assert priority > 0
      assert 0 <= idx < len(self._storage)
      self._it_sum[idx] = priority ** self._alpha
      self._it_min[idx] = priority ** self._alpha

      self._max_priority = max(self._max_priority, priority)