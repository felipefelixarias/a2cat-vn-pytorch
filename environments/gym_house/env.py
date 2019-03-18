import gym
from deep_rl.configuration import configuration
from House3D import objrender, Environment
from House3D.roomnav import RoomNavTask
import numpy as np
import random
import os
import cv2

def create_configuration():
    path = configuration.get('house3d.framework_path')
    return {
        "colorFile": os.path.join(path, "House3D/metadata/colormap_coarse.csv"),
        "roomTargetFile": os.path.join(path,"House3D/metadata/room_target_object_map.csv"),
        "modelCategoryFile": os.path.join(path,"House3D/metadata/ModelCategoryMapping.csv"),
        "prefix": os.path.join(configuration.get('house3d.dataset_path'), 'house')
    }

class GymHouseWrapper(gym.ObservationWrapper):
    def __init__(self, env, room_types = None, screen_size = None):
        super().__init__(env)
        self.screen_size = screen_size
        self.room_types = room_types

    def observation(self, observation):
        return observation

    def step(self, action):
        r = super().step(action)
        print(r[1])
        return r

    def reset(self):
        goals = set(self.env.house.all_desired_roomTypes)
        if self.room_types is not None:
            goals.intersection_update(set(self.room_types))

        target = random.choice(list(goals))

        return self.observation(self.env.reset(target))

def GymHouseEnv(scene = '05cac5f7fdd5f8138234164e76a97383', screen_size = (84,84), goals = ['bedroom']):
    h, w = screen_size
    api = objrender.RenderAPI(w = w, h = h, device = 0)
    env = Environment(api, scene, create_configuration())
    env.reset()
    env = RoomNavTask(env, discrete_action = True, depth_signal = False, segment_input = False, reward_type=None)
    env.observation_space.dtype = np.uint8
    return GymHouseWrapper(env, room_types=goals, screen_size = screen_size)