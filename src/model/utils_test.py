# -*- coding: utf-8 -*-
import numpy as np
import math
import tensorflow as tf
from utils import CircularBuffer

class TestUtils(tf.test.TestCase):
  def test_circular_buffer_size(self):
    """ Check size of circular buffer"""
    buffer = CircularBuffer(4, (2,3))
    self.assertEqual(4, len(buffer))
    self.assertEqual(tuple(buffer().shape), (4, 2, 3))

  def test_circular_buffer_fill(self):
    """ Test fill circular buffer"""
    buffer = CircularBuffer(4, (2,3))
    
    data = np.random.uniform(size = (2,3))
    buffer.fill(data)
    self.assertAllEqual(np.repeat(data[np.newaxis, :, :], 4, axis = 0), buffer())

  def test_circular_buffer_append(self):
    """ Test append to circular buffer"""
    buffer = CircularBuffer(4, (2,3))
    
    data = np.random.uniform(size = (2,3))
    buffer.fill(np.zeros((2,3,)))
    buffer.append(data)

    result = np.zeros((4,2,3,))
    result[3, :, :] = data
    self.assertAllEqual(result, buffer())

if __name__ == "__main__":
  tf.test.main()