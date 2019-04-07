#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from environment.thor_cached_environment import THORDiscreteCachedEnvironment

class KeyboardAgent:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.env = THORDiscreteCachedEnvironment(**kwargs)

    def show(self):
        fig = plt.figure()
        imgplot = plt.imshow(self.env.last_state['image'])
        def press(event):
            def redraw():
                plt.imshow(self.env.last_state['image'])
                fig.canvas.draw()

            if event.key == 's':
                mpimg.imsave("output.png",self.env.render(mode = 'image'))
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
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Deep reactive agent scene explorer.')
    parser.add_argument('--env_name', help='Scene to run the explorer on', default='bedroom_04', type = str)

    args = vars(parser.parse_args())

    KeyboardAgent(**args).show()