#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import environments

import deep_rl
from configuration import configuration
deep_rl.configure(**configuration)

class KeyboardAgent:
    def __init__(self, **kwargs):
        self.config = kwargs
        #self.env = environments.make('ContinuousThor-v0', goals = ['laptop'], scenes = list(range(201, 230)))
        self.env = environments.make('House-v0', scene = '00cfe094634578865b4384f3adef49e6', goals=['kitchen'])#, goals = ['living_room'])
        self.obs = self.env.reset()

    def show(self):
        fig = plt.figure()
        imgplot = plt.imshow(self.obs)
        def press(event):
            def redraw():
                plt.imshow(self.obs)
                fig.canvas.draw()
                
            done = False
            if event.key == 's':
                mpimg.imsave("output.png",self.env.render(mode = 'rgbarray'))
            elif event.key == 'up':
                self.obs, _, done, _ = self.env.step(0)
                redraw()
            elif event.key == 'right':
                self.obs, _, done, _ = self.env.step(4)
                redraw()
            elif event.key == 'left':
                self.obs, _, done, _ = self.env.step(5)
                redraw()

            print(self.env.unwrapped.info['target_room'])
            print(self.env.unwrapped._env.house.all_desired_roomTypes)

            if hasattr(self.env.unwrapped, 'state'):
                print(self.env.unwrapped.state)

            if done:
                print('Goal reached')
                self.obs = self.env.reset()

                redraw()

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()


class GoalKeyboardAgent:
    def __init__(self, env, actions = [0,1,2,3]):
        self.env = env
        self.actions = actions

    def show(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        self.o = self.env.reset()
        def redraw():
            a = self.o[0]
            b = self.o[1]
            ax1.imshow(a)
            ax2.imshow(b)
            fig.canvas.draw()

        def press(event):                
            done = False
            if event.key == 's':
                mpimg.imsave("output.png",self.env.render(mode = 'rgbarray'))
            elif event.key == 'up':
                self.o, _, done, _ = self.env.step(self.actions[0])
                redraw()
            elif event.key == 'right':
                self.o, _, done, _ = self.env.step(self.actions[2])
                redraw()
            elif event.key == 'left':
                self.o, _, done, _ = self.env.step(self.actions[3])
                redraw()

            elif event.key == 'r':
                self.o = self.env.reset()
                redraw()

            if hasattr(self.env.unwrapped, 'state'):
                print(self.env.unwrapped.state)

            if done:
                print('Goal reached')
                self.o = self.env.reset()
                redraw()

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

        redraw()

if __name__ == '__main__':
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent scene explorer.')
    parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}.h5')
    parser.add_argument('--unity_path', type=str)

    parser.add_argument('--scene', help='Scene to run the explorer on', default='bedroom_04', type = str)

    args = vars(parser.parse_args())

    from experiments.data import TRAIN, VALIDATION
    # env = environments.make('GoalHouse-v1',screen_size=(500,500), scene =  ['0b6d4fe900eaddd80aecf4bc79248dd9']) #['b814705bc93d428507a516b866efda28','e3ae3f7b32cf99b29d3c8681ec3be321','5f3f959c7b3e6f091898caa8e828f110'])
   
    #from environments.gym_house.video import RenderVideoWrapper
    #env = RenderVideoWrapper(env, '')   
    '''
    208,
    212
    '''

    env = environments.make('AuxiliaryGraph-v0', goals = (5, 6, 2), graph_name = 'thor-cached-225') #  graph_file = 'kitchen.pkl')
    GoalKeyboardAgent(env).show()