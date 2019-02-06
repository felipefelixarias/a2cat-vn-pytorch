#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from agent.environment.vizdoom import VizdoomDiscreteEnvironment
from agent.environment.ai2thor import AI2THOREnvironment

class KeyboardAgent:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.env = VizdoomDiscreteEnvironment(**kwargs)
        self.env.start()

    def show(self):
        fig = plt.figure()
        imgplot = plt.imshow(self.env.render(mode = 'image'))
        def press(event):
            def redraw():
                plt.imshow(self.env.render(mode = 'image'))
                fig.canvas.draw()

            if event.key == 's':
                mpimg.imsave("output.png",self.env.render(mode = 'image'))
            elif event.key == 'up':
                self.env.step('MoveAhead')
                redraw()
            elif event.key == 'right':
                self.env.step('RotateRight')
                redraw()
            elif event.key == 'left':
                self.env.step('RotateLeft')
                redraw()

            pass

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent scene explorer.')
    parser.add_argument('--h5_file_path', type = str, default='/app/data/{scene}.h5')
    parser.add_argument('--unity_path', type=str)

    parser.add_argument('--scene', help='Scene to run the explorer on', default='bedroom_04', type = str)

    args = vars(parser.parse_args())

    KeyboardAgent(**args).show()