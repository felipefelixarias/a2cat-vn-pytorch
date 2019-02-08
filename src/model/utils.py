import numpy as np

class CircularBuffer:
    def __init__(self, length, size = [1], type = np.float):
        self.length = length
        self.data = np.ndarray((length,) + size, dtype = np.float)

    def __len__(self):
        return self.length

    def __call__(self):
        return self.data

    def append(self, data):
        self.data = np.concatenate((self.data.take(axis = 0, indices = range(1, self.length)), np.expand_dims(data, axis = 0),), axis = 0)

    def fill(self, data):
        self.data  = np.repeat(np.expand_dims(data, axis = 0), self.length, axis = 0)