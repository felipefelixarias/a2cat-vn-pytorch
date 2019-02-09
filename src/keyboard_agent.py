#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from options import get_options
from environment.environment import Environment

flags = get_options('keyboard')

class Explorer:
    def __init__(self):
        self.env = Environment.create_environment(flags.env_type, flags.env_name)

    def show(self):
        (ax1, ax2) = (None, None)
        if self.env.can_use_goal():
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            fig = plt.figure()
        
        def redraw():
            if self.env.can_use_goal():
                ax1.imshow(self.env.last_state['image'])
                ax2.imshow(self.env.last_state['goal'])
            else:
                plt.imshow(self.env.last_state['image'])
            fig.canvas.draw()
        redraw()

        def press(event):
            

            if event.key == 's':
                mpimg.imsave("output.png", self.env.last_state['image'])
            elif event.key in ['up', 'down', 'right', 'left']:
                self.env.process(self.env.get_keyboard_map()[event.key])
                redraw()

            pass

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    Explorer().show()