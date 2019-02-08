from agent.environment import Environment
from ai2thor.controller import Controller
import os

class AI2THOREnvironment(Environment):
    def __init__(self, executable_path = None, **kwargs):
        super(AI2THOREnvironment, self).__init__(**kwargs)
        self.executable_path = executable_path
        if executable_path is None:
            if 'AI2THOR_PATH' in os.environ:
                self.executable_path = os.environ['AI2THOR_PATH']
                
        self.config = kwargs
        self.controller = Controller()
        self.controller.local_executable_path = self.executable_path

    def start(self):
        self.controller.start()
        self.controller.reset('FloorPlan29')
        event = self.controller.step(dict(action='Initialize', gridSize=0.25))
        self._set_state(event)

    def _set_state(self, event):
        self.frame = event.frame


    def reset(self):
        self.controller.reset('FloorPlan29')
        event = self.controller.step(dict(action='Initialize', gridSize=0.25))
        self._set_state(event)

    def step(self, action = 'MoveAhead'):
        event = self.controller.step(dict(action=action))
        self._set_state(event)

    def fork(self):
        # TODO: implement efficient fork
        env = AI2THOREnvironment(self.executable_path, **self.config)
        return env

    def render(self, mode):
        if mode == 'image':
            return self.frame
        else:
            assert False

    @property
    def actions(self):
        return ['MoveAhead', 'RotateLeft', 'RotateRight', 'MoveBackward']