from .env import EnvBase
from download import resource as fetch_resource
import os
import cv2
import random
import numpy as np

DEFAULT_GOALS = [
    "ottoman", "laptop", "vase", "sofa", "plunger", "soapbottle", "apple", "knife", "ladle", "towel", "kettle", "bowl", "watch", "chair", "window", "potato", "safe", "spatula", "bottle", "boots", "cabinet", "handtowel", "laundryhamper", "tissuebox", "microwave", "painting", "pillow", "toiletpaperroll", "candle", "box", "bread", "cup", "egg", "toiletpaper", "lettuce", "television", "wateringcan", "spoon", "toaster", "plate", "winebottle", "cloth", "dresser", "stove burner", "televisionarmchair", "toilet", "drawer", "teddybear", "statue", "fridge", "pan", "alarmclock", "dishsponge", "shelf", "baseballbat", "stove knob", "sink", "coffeemachine", "garbagecan", "pot", "desklamp", "book", "scrubbrush", "houseplant", "poster", "pillowarmchair", "tennisracket", "towelholder", "mug"
]


class GoalImageCache:
    def __init__(self, image_size):
        self.scenes = dict()
        self.cache = dict()
        self.image_size = image_size
        self.random = random.Random()
        pass

    def sample_image(self, collection):
        return self.random.choice(collection)

    def fetch_scene(self, scene):
        if not scene in self.scenes:
            self.scenes[scene] = sceneobj = dict(
                path = fetch_resource('thor-scene-images-%s' % scene),
                resources = dict()
            )

            sceneobj['available_goals'] = os.listdir(sceneobj['path'])

        return self.scenes[scene]

    def all_goals(self, scene):
        return self.fetch_scene(scene)['available_goals']

    def fetch_random(self, scene, resource):
        self.fetch_scene(scene)

        if not resource in self.scenes[scene]['resources']:
            root = os.path.join(self.scenes[scene]['path'], resource)
            images = os.listdir(root)
            self.scenes[scene]['resources'][resource] = dict(
                root = root,
                images = images
            )
        else:
            row = self.scenes[scene]['resources'][resource]
            root, images = row['root'], row['images']

        sampled_image = self.sample_image(images)
        if (scene, sampled_image) in self.cache:
            return self.cache[(scene, sampled_image)]

        impath = os.path.join(root, sampled_image)
        image = cv2.imread(impath)
        print(self.image_size)
        image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
        self.cache[(scene, sampled_image)] = image
        return image
        



class GoalEnvBase(EnvBase):
    def __init__(self, scenes, screen_size = (224, 224), goals = [], **kwargs):
        if len(goals) == 0:
            goals = list(DEFAULT_GOALS)

        self.goal_source = GoalImageCache(screen_size)
        super().__init__(scenes, screen_size=screen_size, goals=goals, **kwargs)        
        pass
    
    def _has_finished(self, event):
        for o in event.metadata['objects']:
            tp = self._get_object_type(o)
            if tp == self.goal and o['distance'] < self.treshold_distance:
                return True
        
        return False

    def _pick_goal(self, event, scene):
        allgoals = set(self.goal_source.all_goals(scene))
        allgoals.intersection_update(set(self.goals))            

        # Resamples if no goals are available
        event = super()._pick_goal(event, scene)

        goals = set()
        for o in event.metadata['objects']:
            tp = self._get_object_type(o)
            if tp in allgoals:
                goals.add(tp)

        self.goal = self.random.choice(list(goals))
        self.goal_observation = self.goal_source.fetch_random(scene, self.goal)
        return event

    def observe(self, event = None):
        main_observation = super().observe(event)
        return (main_observation, np.copy(self.goal_observation))

    def browse(self):
        from .browser import GoalKeyboardAgent
        return GoalKeyboardAgent(self)