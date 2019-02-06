import malmoenv
from agent.environment import Environment

class MalmoEnvironment(Environment):
    def __init__(self, **kwargs):
        self.env = malmoenv.Env()
        

    def start(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        self.env.step(action)
        pass

    def render(self, mode = 'image'):
        pass

    @property
    def actions(self):
        return ['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveBackward']


    def 
        self.env.render()