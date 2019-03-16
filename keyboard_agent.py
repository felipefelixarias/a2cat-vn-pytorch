#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import environments

class KeyboardAgent:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.env = environments.make('ContinuousThor-v0', goals = ['laptop'], scenes = list(range(201, 230)))
        self.env.reset()

    def show(self):
        fig = plt.figure()
        imgplot = plt.imshow(self.env.render(mode = 'rgbarray'))
        def press(event):
            def redraw():
                plt.imshow(self.env.render(mode = 'rgbarray'))
                fig.canvas.draw()
                
            done = False
            if event.key == 's':
                mpimg.imsave("output.png",self.env.render(mode = 'rgbarray'))
            elif event.key == 'up':
                _, _, done, _ = self.env.step(0)
                redraw()
            elif event.key == 'right':
                _, _, done, _ = self.env.step(4)
                redraw()
            elif event.key == 'left':
                _, _, done, _ = self.env.step(5)
                redraw()

            if hasattr(self.env.unwrapped, 'state'):
                print(self.env.unwrapped.state)

            if done:
                print('Goal reached')
                self.env.reset()

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