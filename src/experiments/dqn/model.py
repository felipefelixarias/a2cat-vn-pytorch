# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class Model(object):
  """
  UNREAL algorithm network model.
  """
  def __init__(self,
               action_size,
               objective_size,
               use_goal_input,
               device,
               name = 'net',
               reuse = False,
               **net_kwargs):
    self._action_size = action_size
    self._objective_size = objective_size
    self._use_goal_input = use_goal_input
    self._device = device
    self._image_shape = [84,84] # Note much of network parameters are hard coded so if we change image shape, other parameters will need to change
    self._create_network(name, reuse, **net_kwargs)
    
  def _create_network(self, scope_name, reuse, **net_kwargs):
    with tf.variable_scope(scope_name, reuse) as scope:
      # lstm
      self._create_base_network(reuse, **net_kwargs)
      self.reset_state()
      self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


  def _create_base_network(self, reuse, base_input = None, goal_input = None):
    # State (Base image input)
    self.base_input = base_input if base_input is not None else tf.placeholder("float", [None, self._image_shape[0], self._image_shape[1], 3], name='base_input')

    if self._use_goal_input:
      self.goal_input = goal_input if goal_input is not None else tf.placeholder("float", [None, self._image_shape[0], self._image_shape[1], 3], name='goal_input')

    # Conv layers
    base_conv_output = self._base_conv_layers(self.base_input, reuse = reuse)

    if self._use_goal_input:
      # Shared convolution for goal and base input
      shared_base_input = self._base_conv_layers(self.base_input, name = "shared_conv", reuse = reuse)
      goal_base_input = self._base_conv_layers(self.goal_input, name = "shared_conv", reuse = True)
      base_conv_output = tf.concat((base_conv_output, shared_base_input, goal_base_input,), 3)
      with tf.variable_scope("merge_conv", reuse = reuse) as scope:
        W_conv, b_conv = self._conv_variable([1, 1, 3 * 32, 32],  "merge_conv") # => 9x9x32
        base_conv_output = tf.nn.relu(self._conv2d(base_conv_output, W_conv, 1) + b_conv) # => 9x9x32

    self.base_fcn_outputs = self._base_fcn_layer(base_conv_output, reuse = reuse)
    self.base_pi = self._base_policy_layer(self.base_fcn_outputs, reuse = reuse) # policy output

    
  def _base_conv_layers(self, state_input, reuse=False, name = "base_conv"):
    with tf.variable_scope(name, reuse = reuse) as scope:
      # Weights
      W_conv1, b_conv1 = self._conv_variable([8, 8, 3, 16],  "base_conv1") # 16 8x8 filters
      W_conv2, b_conv2 = self._conv_variable([4, 4, 16, 32], "base_conv2") # 32 4x4 filters

      # Nodes
      h_conv1 = tf.nn.relu(self._conv2d(state_input, W_conv1, 4) + b_conv1) # stride=4 => 19x19x16
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1,     W_conv2, 2) + b_conv2) # stride=2 => 9x9x32
      return h_conv2


  def _base_fcn_layer(self, conv_output,
                      reuse=False):
    with tf.variable_scope("base_fcn", reuse=reuse) as scope:
      # Weights (9x9x32=2592)
      W_fc1, b_fc1 = self._fc_variable([2592, 256], "base_fc1")

      # Nodes
      conv_output_flat = tf.reshape(conv_output, [-1, 2592])
      # (-1,9,9,32) -> (-1,2592)
      conv_output_fc = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
      # (unroll_step, 256)
      return conv_output_fc


  def _base_lstm_layer(self, conv_output, last_action_reward_objective_input, initial_state_input,
                       reuse=False):
    with tf.variable_scope("base_lstm", reuse=reuse) as scope:
      # Weights (9x9x32=2592)
      W_fc1, b_fc1 = self._fc_variable([2592, 256], "base_fc1")

      # Nodes
      conv_output_flat = tf.reshape(conv_output, [-1, 2592])
      # (-1,9,9,32) -> (-1,2592)
      conv_output_fc = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
      # (unroll_step, 256)

      step_size = tf.shape(conv_output_fc)[:1]

      lstm_input = tf.concat([conv_output_fc, last_action_reward_objective_input], 1)

      # (unroll_step, 256+action_size+1+objective_size)

      lstm_input_reshaped = tf.reshape(lstm_input, [1, -1, 256+self._action_size+1+self._objective_size])
      # (1, unroll_step, 256+action_size+1+objective_size)

      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                   lstm_input_reshaped,
                                                   initial_state = initial_state_input,
                                                   sequence_length = step_size,
                                                   time_major = False,
                                                   scope = scope)
      
      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
      #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
      return lstm_outputs, lstm_state


  def _base_policy_layer(self, lstm_outputs, reuse=False):
    with tf.device(self._device), tf.variable_scope("base_policy", reuse=reuse) as scope:
      input_size = lstm_outputs.get_shape().as_list()[1]
      # Weight for policy output layer
      W_fc_p, b_fc_p = self._fc_variable([input_size, self._action_size], "base_fc_p")
      # Policy (output)
      base_pi = tf.matmul(lstm_outputs, W_fc_p) + b_fc_p
      return base_pi

  
  def reset_state(self):
    # Clear buffers
    # And LSTM state
    pass
  
  def get_vars(self):
    return self.variables
  

  def _fc_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias

  
  def _conv_variable(self, weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias

  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")


  def _get2d_deconv_output_size(self,
                                input_height, input_width,
                                filter_height, filter_width,
                                stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * stride + filter_height
      out_width  = (input_width  - 1) * stride + filter_width
      
    elif padding_type == 'SAME':
      out_height = input_height * stride
      out_width  = input_width  * stride
    
    return out_height, out_width


  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           'VALID')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID')
