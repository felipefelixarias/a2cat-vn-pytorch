from abc import ABC, abstractproperty, abstractclassmethod
class GridWorldScene:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractproperty
    def maze(self):
        pass

    @abstractclassmethod
    def render(self, position, rotation):
        pass