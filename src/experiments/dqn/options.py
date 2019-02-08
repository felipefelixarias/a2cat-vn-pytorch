# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_options(option_type):
    tf.app.flags.DEFINE_string("env_type", "thor_cached", "environment type (lab or gym or maze or indoor or thor_cached)")
    tf.app.flags.DEFINE_string("env_name", "bedroom_04",  "environment name (nav_maze_static_01)")

    if option_type == 'training':
        tf.app.flags.DEFINE_string("log_dir", "./logs", "log file directory")
        return tf.app.flags.FLAGS
