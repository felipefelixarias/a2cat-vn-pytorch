import gym
import gym.spaces
import numpy as np
import ai2thor.controller

ACTIONS = [
    dict(action='MoveAhead', moveMagnitude=0.33),
    dict(action='MoveAhead', moveMagnitude=-0.33)
]

class ContinuousEnv(gym.Env):
    def __init__(self, scene_id, screen_size = (224, 224)):
        self.controller = ai2thor.controller.Controller()
        self.scene_id = scene_id

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 255, shape = screen_size + (3,), dtype = np.uint8)
        
        self._was_started = False

    def reset(self):
        if not self._was_started:
            self.controller.start()
            self._was_started = True

        self.controller.reset('FloorPlan%s' % self.scene_id)
        self.controller.step(dict(action='Initialize', continuous=True))

        for _ in range(10):
            env.step({'action' : 'MoveAhead'})
            env.step({'action' : 'RotateRight'})
        total_time = time.time() - t_start_total
        print('total time', total_time, 20 / total_time, 'fps')

    def _has_finished(self):

    def step(self, action):
        event = self.controller.step(ACTIONS[action])

    def stop(self):
        if self._was_started:
            self.controller.stop()

    
