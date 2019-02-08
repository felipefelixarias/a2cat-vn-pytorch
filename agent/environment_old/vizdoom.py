import vizdoom
import os
import itertools as it
from agent.environment import Environment

class VizdoomDiscreteEnvironment(Environment):
    def __init__(self, **kwargs):
        self.game = vizdoom.DoomGame()
        self.config_name = 'my_way_home.cfg'

        self.game.load_config(self._find_local_config(self.config_name))
        self.game.set_window_visible(False)
        self.game.set_mode(vizdoom.Mode.PLAYER)
        self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)

        self._init()
        pass

    def _find_local_config(self, path):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../resources/{path}')

    def _init(self):
        self.game.clear_available_buttons()
        self.game.add_available_button(vizdoom.Button.MOVE_FORWARD, 400)
        self.game.add_available_button(vizdoom.Button.TURN_LEFT_RIGHT_DELTA)

    def _fix_angle(self):
        angle = self.game.get_game_variable(vizdoom.GameVariable.ANGLE)
        angle_r = round(angle / 90) * 90
        self.game.make_action([0, angle - angle_r])
        print(angle)
        print(angle_r)
        print(self.game.get_game_variable(vizdoom.GameVariable.ANGLE))

    def start(self):
        self.game.init()
        self._fix_angle()

        self.game.new_episode()

    def reset(self):
        self.game.new_episode()
        pass

    def step(self, action):
        step = None
        if action == 'MoveAhead':
            step = [400,0]
        if action == 'RotateLeft':
            step = [0, -90]
        elif action == 'RotateRight':
            step = [0, 90]

        self.game.make_action(step, 1)

    def render(self, mode = 'image'):
        return self.game.get_state().screen_buffer

    @property
    def actions(self):
        return ['MoveAhead', 'RotateLeft', 'RotateRight']