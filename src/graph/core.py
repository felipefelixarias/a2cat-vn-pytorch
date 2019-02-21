from abc import ABC, abstractproperty, abstractclassmethod
import numpy as np
class GridWorldScene:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.graph = None
        self.optimal_distances = None

    @abstractproperty
    def maze(self):
        pass

    @abstractclassmethod
    def render(self, position, rotation):
        pass

    @property
    def observation_shape(self):
        return (81, 81, 3)

    @property
    def dtype(self):
        return np.uint8