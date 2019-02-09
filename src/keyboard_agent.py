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
        fig = plt.figure()
        imgplot = plt.imshow(self.env.last_state['image'])
        def press(event):
            def redraw():
                plt.imshow(self.env.last_state['image'])
                fig.canvas.draw()

            if event.key == 's':
                mpimg.imsave("output.png", self.env.last_state['image'])
            elif event.key == 'up':
                self.env.process(0)
                redraw()
            elif event.key == 'right':
                self.env.process(1)
                redraw()
            elif event.key == 'left':
                self.env.process(2)
                redraw()

            pass

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    Explorer().show()