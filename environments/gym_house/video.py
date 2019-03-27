import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from House3D.core import Environment
from House3D.objrender import RenderAPI
from deep_rl.configuration import configuration
from .cenv import GymHouseState
from cv2 import VideoWriter, VideoWriter_fourcc
import os


class RenderVideoWrapper(gym.Wrapper):
    def __init__(self, env, path, action_frames = 20, width = 500, height = 500, renderer_config = None):
        super().__init__(env)
        self.path = path
        self.last_video_id = 1
        self.action_frames = action_frames
        self.size = (height, width)
        self.api = RenderAPI(w = width, h = height, device = 0)
        self.renderer_config = renderer_config if renderer_config is not None else configuration.get('house3d').as_dict()
        
        self.ep_states = []
        pass

    def reset(self):
        if len(self.ep_states) > 0:
            self.render_video(self.ep_states)
            self.ep_states = []

        observation = self.env.reset()
        self.ep_states = [self.unwrapped.state]
        return observation

    def step(self, action):
        obs, reward, done, stats = self.env.step(action)
        self.ep_states.append(self.unwrapped.state)
        if done:
            self.render_video(self.ep_states)
            self.ep_states = []

        return obs, reward, done, stats

    def render_video(self, states):
        renderer = Environment(self.api, states[0].house_id, self.renderer_config)
        output_filename = "vid-%s.avi" % self.last_video_id
        self.last_video_id += 1
        writer = VideoWriter(os.path.join(self.path, output_filename), VideoWriter_fourcc(*"MJPG"), 30, self.size)

        def render_single(position):
            renderer.reset(*position)
            frame = renderer.render(mode = 'rgb', copy = True)
            writer.write(frame)

        state = states[0]
        position = position = state.x, state.y, state.rotation
        for state in states[1:]:
            old_position = position           
            position = position = state.x, state.y, state.rotation

            for j in range(self.action_frames):
                interpolated = tuple(map(lambda a, b: a + (b - a) * j / self.action_frames, old_position, position))
                render_single(interpolated)

        for _ in self.action_frames:
            render_single(position)

        render_single(position)        
        writer.release()

        

    