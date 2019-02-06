class Environment:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode = 'image'):
        pass

    @property
    def actions(self):
        return ['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveBackward']