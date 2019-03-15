import gym
import gym.spaces
import numpy as np
import ai2thor.controller
import cv2
import random


class EnvBase(gym.Env):
    def __init__(self, scene_id, screen_size = (224, 224), goals = ['Mug'], cameraY = 0.675):
        self.screen_size = screen_size
        self.controller = ai2thor.controller.Controller(quality='Very Low')
        self.scene_id = scene_id
        self.goals = goals
        
        self.observation_space = gym.spaces.Box(0, 255, shape = screen_size + (3,), dtype = np.uint8)
        
        self._was_started = False
        self.treshold_distance = 1.5
        self._last_event = None

        self.random = random.Random()
        self.initialize_kwargs = dict(cameraY = cameraY)
        self.state = None

    def reset(self):
        if not self._was_started:
            self.controller.start()
            self._was_started = True

        self.controller.reset('FloorPlan%s' % self.scene_id)
        event = self.controller.step(dict(action='Initialize', **self.initialize_kwargs))
        event = self._pick_goal(event)        

        num_trials = 0
        while self._has_finished(event):
            event = self._sample_start_position(event)
            num_trials += 1
            print('WARNING: Reset invoked to sample nonterminal state')

        
        return self.observe(event)

    def _reset_objects(self):
        seed = self.random.randint(1, 1000000)
        return self.controller.step(dict(action = 'InitialRandomSpawn', randomSeed = seed, forceVisible = True, maxNumRepeats = 5))

    def render(self, mode = 'human'):
        return self.observe() 

    def observe(self, event = None):
        if event is None:
            event = self._last_event
        self._last_event = event
        self.state = (event.metadata['agent']['position'], event.metadata['agent']['rotation'])
        return cv2.resize(event.frame, self.screen_size, interpolation = cv2.INTER_CUBIC)

    def _sample_start_position(self, event):
        event = self.controller.step(dict(action='GetReachablePositions'))
        position = self.random.choice(event.metadata['actionReturn'])
        rotation = self.random.random() * 360.0
        event = self.controller.step(dict(action='Teleport', horizon=0.0, rotation=rotation, **position))
        return event

    def _get_object_type(self, object):
        s = object['objectId']
        return s[:s.index('|')].lower()

    def _has_finished(self, event):
        for o in event.metadata['objects']:
            tp = self._get_object_type(o)
            if tp in self.goals and o['distance'] < self.treshold_distance:
                return True
        
        return False

    def _pick_goal(self, event):
        # No goal env
        if len(self.goals) == 0:
            return event

        hasgoal = False
        numtrials = 0
        while not hasgoal:
            event = self._reset_objects()
            for o in event.metadata['objects']:
                tp = self._get_object_type(o)
                if tp in self.goals:
                    hasgoal = True

            if numtrials > 0:
                print('WARNING: (%s) Reset invoked to sample scene with goals' % numtrials)

            numtrials += 1            

        return event

    def _finish_step(self, event):
        done = self._has_finished(event)
        reward = 0 if not done else 1.0
        return self.observe(event), reward, done, dict()

    def stop(self):
        if self._was_started:
            self.controller.stop()