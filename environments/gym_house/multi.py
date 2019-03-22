from deep_rl.common.env import SubprocVecEnv, DummyVecEnv
from House3D.core import Environment, local_create_house
import random
import time
import gym

def create_multiscene(num_processes, scenes, wrap = lambda e: e, **kwargs):
    assert len(scenes) % num_processes == 0, "The number of processes %s must devide the number of scenes %s" % (num_processes, len(scenes))
    scenes_per_process = len(scenes) // num_processes

    funcs = []
    for i in range(num_processes):
        funcs.append(lambda: wrap(gym.make(**kwargs, scene = scenes[i * scenes_per_process:(i + 1) * scenes_per_process])))

    if num_processes == 1:
        return DummyVecEnv(funcs)

    else:
        return SubprocVecEnv(funcs)