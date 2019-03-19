# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys, os
import numpy as np
import random
import csv
import copy
import cv2
import pickle

import gym
from gym import spaces
from House3D.house import House
from House3D.core import Environment, MultiHouseEnv
from House3D import objrender

###############################################
# Task related definitions and configurations
###############################################
flag_print_debug_info = False  # flag for printing debug info

success_distance_range = 1.0
success_stay_time_steps = 5
success_see_target_time_steps = 2   # time steps required for success under the "see" criteria

# sensitivity setting
rotation_sensitivity = 30  # 45   # maximum rotation per time step
default_move_sensitivity = 0.5  # 1.0   # maximum movement per time step

# discrete action space actions, totally <13> actions
# Fwd, L, R, Lrot, Rrot, back, s-Fwd, s-L, s-R, s-Lrot, s-Rrot
discrete_actions=[(1.,0.,0.), (0.,0.7,0.), (0.,-0.7,0.),
                  (0.,0.,1.), (0.,0.,-1.),
                  (-0.4, 0., 0.)]
                  #,
                  #(0.4,0.,0.), (0.,0.,0.4), (0.,0.,-0.4)]
n_discrete_actions = len(discrete_actions)


#################
# Util Functions
#################

def reset_see_criteria(resolution):
    total_pixel = resolution[0] * resolution[1]
    global n_pixel_for_object_see, n_pixel_for_object_sense, L_pixel_reward_range
    n_pixel_for_object_see = max(int(total_pixel * 0.045), 5)
    n_pixel_for_object_sense = max(int(total_pixel * 0.005), 1)
    L_pixel_reward_range = n_pixel_for_object_see - n_pixel_for_object_sense


class RoomNavTask(gym.Env):
    def __init__(self, env,
                 seed=None,
                 reward_type='delta',
                 hardness=None,
                 move_sensitivity=None,
                 segment_input=True,
                 joint_visual_signal=False,
                 depth_signal=True,
                 max_steps=-1,
                 success_measure='see',
                 discrete_action=False):
        """RoomNav task wrapper with gym api
        Note:
            all the settings are the default setting to run a task
            only the <env> argument is necessary to launch the task

        Args:
            env: an instance of environment (multi-house or single-house)
            seed: if not None, set the random seed
            reward_type (str, optional): reward shaping, currently available: none, linear, indicator, delta and speed
            hardness (double, optional): if not None, must be a real number between 0 and 1, indicating the hardness
                                         namely the distance from birthplace to target (1 is the hardest)
                                         None means 1.0
            move_sensitivity (double, optional): if not None, set the maximum movement per time step (generally should not be changed)
            segment_input (bool, optional): whether to use semantic segmentation mask for observation
            joint_visual_signal (bool, optional): when true, use both visual signal and segmentation mask as observation
                                                  when true, segment_input will be set true accordingly
            depth_signal (bool, optional): whether to include depth signal in observation
            max_steps (int, optional): when max_steps > 0, the task will be cut after <max_steps> steps
            success_measure (str, optional): criteria for success, currently support 'see' and 'stay'
            discrete_action (bool, optional):  when true, use discrete actions; otherwise use continuous actions
        """
        self.env = env
        assert isinstance(env, Environment), '[RoomNavTask] env must be an instance of Environment!'
        if env.resolution != (120, 90): reset_see_criteria(env.resolution)
        self.resolution = resolution = env.resolution
        assert reward_type in [None, 'none', 'linear', 'indicator', 'delta', 'speed']
        self.reward_type = reward_type
        self.colorDataFile = self.env.config['colorFile']
        self.segment_input = segment_input
        self.joint_visual_signal = joint_visual_signal
        self.depth_signal = depth_signal
        n_channel = 3
        if segment_input:
            self.env.set_render_mode('semantic')
        else:
            self.env.set_render_mode('rgb')
        if joint_visual_signal: n_channel += 3
        if depth_signal: n_channel += 1
        self._observation_shape = (resolution[0], resolution[1], n_channel)
        self._observation_space = spaces.Box(0, 255, shape=self._observation_shape)

        self.max_steps = max_steps

        self.discrete_action = discrete_action
        if discrete_action:
            self._action_space = spaces.Discrete(n_discrete_actions)
        else:
            self._action_space = spaces.Tuple([spaces.Box(0, 1, shape=(4,)), spaces.Box(0, 1, shape=(2,))])

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # configs
        self.move_sensitivity = (move_sensitivity or default_move_sensitivity)  # at most * meters per frame
        self.rot_sensitivity = rotation_sensitivity
        self.successRew = 1.0
        self.colideRew = 0
        self.goodMoveRew = 0  # reward when moving closer to target

        self.last_obs = None
        self.last_info = None
        self._object_cnt = 0

        # config hardness
        self.hardness = None
        self.availCoors = None
        self._availCoorsDict = None
        self.reset_hardness(hardness)

        # temp storage
        self.collision_flag = False

        # episode counter
        self.current_episode_step = 0

        # config success measure
        assert success_measure in ['stay', 'see']
        self.success_measure = success_measure
        print('[RoomNavTask] >> Success Measure = <{}>'.format(success_measure))
        self.success_stay_cnt = 0
        if success_measure == 'see':
            self.room_target_object = dict()
            self._load_target_object_data(self.env.config['roomTargetFile'])

    def _load_target_object_data(self, roomTargetFile):
        with open(roomTargetFile) as csvFile:
            reader = csv.DictReader(csvFile)
            for row in reader:
                c = np.array((row['r'],row['g'],row['b']), dtype=np.uint8)
                room = row['target_room']
                if room not in self.room_target_object:
                    self.room_target_object[room] = []
                self.room_target_object[room].append(c)

    """
    reset the target room type to navigate to
    when target is None, a valid target will be randomly selected
    """
    def reset_target(self, target=None):
        if target is None:
            target = random.choice(self.house.all_desired_roomTypes)
        else:
            assert target in self.house.all_desired_roomTypes, '[RoomNavTask] desired target <{}> does not exist in the current house!'.format(target)
        if self.house.setTargetRoom(target):  # target room changed!!!
            _id = self.house._id
            if self.house.targetRoomTp not in self._availCoorsDict[_id]:
                if self.hardness is None:
                    self.availCoors = self.house.connectedCoors
                else:
                    allowed_dist = self.house.maxConnDist * self.hardness
                    self.availCoors = [c for c in self.house.connectedCoors
                                       if self.house.connMap[c[0], c[1]] <= allowed_dist]
                self._availCoorsDict[_id][self.house.targetRoomTp] = self.availCoors
            else:
                self.availCoors = self._availCoorsDict[_id][self.house.targetRoomTp]

    @property
    def house(self):
        return self.env.house

    """
    gym api: reset function
    when target is not None, we will set the target room type to navigate to
    """
    def reset(self, target=None):
        # clear episode steps
        self.current_episode_step = 0
        self.success_stay_cnt = 0
        self._object_cnt = 0

        # reset house
        self.env.reset_house()

        self.house.targetRoomTp = None  # [NOTE] IMPORTANT! clear this!!!!!

        # reset target room
        self.reset_target(target=target)  # randomly reset

        # general birth place
        gx, gy = random.choice(self.availCoors)
        self.collision_flag = False
        # generate state
        x, y = self.house.to_coor(gx, gy, True)
        self.env.reset(x=x, y=y)
        self.last_obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = np.concatenate([self.env.render(mode='rgb'), self.last_obs], axis=-1)
        ret_obs = self.last_obs
        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            ret_obs = np.concatenate([ret_obs, dep_sig], axis=-1)
        self.last_info = self.info
        return ret_obs

    def _apply_action(self, action):
        if self.discrete_action:
            return discrete_actions[action]
        else:
            rot = action[1][0] - action[1][1]
            act = action[0]
            return (act[0] - act[1]), (act[2] - act[3]), rot

    def _is_success(self, raw_dist):
        if raw_dist > 0:
            self.success_stay_cnt = 0
            return False
        if self.success_measure == 'stay':
            self.success_stay_cnt += 1
            return self.success_stay_cnt >= success_stay_time_steps
        # self.success_measure == 'see'
        flag_see_target_objects = False
        object_color_list = self.room_target_object[self.house.targetRoomTp]
        if (self.last_obs is not None) and self.segment_input:
            seg_obs = self.last_obs if not self.joint_visual_signal else self.last_obs[:,:,3:6]
        else:
            seg_obs = self.env.render(mode='semantic')
        self._object_cnt = 0
        for c in object_color_list:
            cur_n = np.sum(np.all(seg_obs == c, axis=2))
            self._object_cnt += cur_n
            if self._object_cnt >= n_pixel_for_object_see:
                flag_see_target_objects = True
                break
        if flag_see_target_objects:
            self.success_stay_cnt += 1
        else:
            self.success_stay_cnt = 0  # did not see any target objects!
        return self.success_stay_cnt >= success_see_target_time_steps

    """
    gym api: step function
    return: obs, reward, done, info (a dictionary containing some state information)
    """
    def step(self, action):
        reward = 0
        det_fwd, det_hor, det_rot = self._apply_action(action)
        move_fwd = det_fwd * self.move_sensitivity
        move_hor = det_hor * self.move_sensitivity
        rotation = det_rot * self.rot_sensitivity

        self.env.rotate(rotation)
        if not self.env.move_forward(move_fwd, move_hor):
            if flag_print_debug_info:
                print('Collision! No Movement Performed!')
            #reward -= self.colideRew
            self.collision_flag = True
        else:
            self.collision_flag = False
            if flag_print_debug_info:
                print('Move Successfully!')

        self.last_obs = obs = self.env.render()
        if self.joint_visual_signal:
            self.last_obs = obs = np.concatenate([self.env.render(mode='rgb'), obs], axis=-1)
        cur_info = self.info
        raw_dist = cur_info['dist']

        done = False
        if self._is_success(raw_dist):
            reward += self.successRew
            done = True
        # accumulate episode step
        self.current_episode_step += 1
        if (self.max_steps > 0) and (self.current_episode_step >= self.max_steps): done = True

        if self.depth_signal:
            dep_sig = self.env.render(mode='depth')
            if dep_sig.shape[-1] > 1:
                dep_sig = dep_sig[..., 0:1]
            obs = np.concatenate([obs, dep_sig], axis=-1)
        self.last_info = cur_info
        return obs, reward, done, cur_info

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def info(self):
        ret = self.env.info
        gx, gy = ret['grid']
        ret['dist'] = dist = self.house.connMap[gx, gy]
        ret['scaled_dist'] = self.house.getScaledDist(gx, gy)
        ret['optsteps'] = int(dist / (self.move_sensitivity / self.house.grid_det) + 0.5)
        ret['collision'] = int(self.collision_flag)
        ret['target_room'] = self.house.targetRoomTp
        return ret

    """
    return all the available target room types of the current house
    """
    def get_avail_targets(self):
        return self.house.all_desired_roomTypes

    """
    reset the hardness of the task
    """
    def reset_hardness(self, hardness=None):
        self.hardness = hardness
        if hardness is None:
            self.availCoors = self.house.connectedCoors
        else:
            allowed_dist = self.house.maxConnDist * hardness
            self.availCoors = [c for c in self.house.connectedCoors
                               if self.house.connMap[c[0], c[1]] <= allowed_dist]
        n_house = self.env.num_house
        self._availCoorsDict = [dict() for i in range(n_house)]
        self._availCoorsDict[self.house._id][self.house.targetRoomTp] = self.availCoors


from .env import create_configuration

class GymHouseEnv(gym.Env):
    def __init__(self, scene = '2364b7dcc432c6d6dcc59dba617b5f4b', screen_size = (84,84), goals = ['kitchen'], hardness=0.3, configuration = None):
        super().__init__()

        self.screen_size = screen_size
        self.room_types = goals
        self.scene = scene
        self.configuration = create_configuration(configuration)
        self._env = None

        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        self.observation_space = gym.spaces.Box(0, 255, screen_size + (3,), dtype = np.uint8)

    def _initialize(self):
        h, w = self.screen_size
        api = objrender.RenderAPI(w = w, h = h, device = 0)
        env = Environment(api, self.scene, self.configuration)
        env.reset()
        env = RoomNavTask(env, discrete_action = True, depth_signal = False, segment_input = False, reward_type=None)
        self._env = env

    def observation(self, observation):
        return observation

    def _ensure_env_ready(self):
        if self._env is None:
            self._initialize()

    def reset(self):
        self._ensure_env_ready()

        goals = set(self._env.house.all_desired_roomTypes)
        if self.room_types is not None:
            goals.intersection_update(set(self.room_types))

        target = random.choice(list(goals))

        return self.observation(self._env.reset(target))

    @property
    def info(self):
        self._ensure_env_ready()
        return self._env.info

    def step(self, action):
        self._ensure_env_ready()
        obs, reward, done, info = self._env.step(action)
        return self.observation(obs), reward, done, info