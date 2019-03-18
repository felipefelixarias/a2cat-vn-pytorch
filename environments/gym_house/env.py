import gym
from deep_rl.configuration import configuration
from House3D import objrender, Environment
from House3D.roomnav import RoomNavTask
import numpy as np
import os

def create_configuration():
    path = configuration.get('house3d.framework_path')
    return {
        "colorFile": os.path.join(path, "House3D/metadata/colormap_coarse.csv"),
        "roomTargetFile": os.path.join(path,"House3D/metadata/room_target_object_map.csv"),
        "modelCategoryFile": os.path.join(path,"House3D/metadata/ModelCategoryMapping.csv"),
        "prefix": os.path.join(configuration.get('house3d.dataset_path'), 'house')
    }

class GymHouseWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        print(observation.shape)
        return observation

def GymHouseEnv(scene = '05cac5f7fdd5f8138234164e76a97383', screen_size = (84,84)):
    h, w = screen_size
    api = objrender.RenderAPI(w = 300, h = 300, device = 0)
    env = Environment(api, scene, create_configuration())
    env.reset()
    env = RoomNavTask(env, discrete_action = True, depth_signal = False, segment_input = False)
    env.observation_space.dtype = np.uint8
    return GymHouseWrapper(env)