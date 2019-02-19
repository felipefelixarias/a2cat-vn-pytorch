#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from options import get_options
from common.env_wrappers import ColorObservationWrapper
import gym
import environment.qmaze
import environment

flags = get_options('keyboard')

class Explorer:
    def __init__(self):
        self.env = gym.make('Mushroom-v0') #ColorObservationWrapper(gym.make('Maze-v0'))
        self.is_goal = isinstance(self.env.observation_space, gym.spaces.Dict)

        self.keyboard_map = {key: i for (i, key) in enumerate(['up', 'down', 'left', 'right']) }

    def show(self):
        (ax1, ax2) = (None, None)
        if self.is_goal:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            fig = plt.figure()
        
        def redraw(state):
            if self.is_goal:
                ax1.imshow(state['observation'])
                ax2.imshow(state['desired_goal'])
            else:
                plt.imshow(state)
            fig.canvas.draw()
        redraw(self.env.reset())

        def press(event):
            if event.key == 's':
                mpimg.imsave("output.png", state['observation'])
            elif event.key in ['up', 'down', 'right', 'left']:
                action = self.keyboard_map[event.key]
                state, _, done, _ = self.env.step(action)
                if done:
                    state = self.env.reset()
                redraw(state)

            pass

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    Explorer().show()