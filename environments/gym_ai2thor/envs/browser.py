import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import environments

class KeyboardAgent:
    def __init__(self, env):
        self.env = env
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


class GoalKeyboardAgent:
    def __init__(self, env):
        self.env = env

    def show(self):
        fig, (ax1, ax2) = plt.subplots(1,2)
        self.o = self.env.reset()
        def redraw():
            a, b = self.o
            ax1.imshow(a)
            ax2.imshow(b)
            fig.canvas.draw()

        def press(event):                
            done = False
            if event.key == 's':
                mpimg.imsave("output.png",self.env.render(mode = 'rgbarray'))
            elif event.key == 'up':
                self.o, _, done, _ = self.env.step(0)
                redraw()
            elif event.key == 'right':
                self.o, _, done, _ = self.env.step(4)
                redraw()
            elif event.key == 'left':
                self.o, _, done, _ = self.env.step(5)
                redraw()

            if hasattr(self.env.unwrapped, 'state'):
                print(self.env.unwrapped.state)

            if done:
                print('Goal reached')
                self.o = self.env.reset()
                print(self.env.goal)
                redraw()

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()