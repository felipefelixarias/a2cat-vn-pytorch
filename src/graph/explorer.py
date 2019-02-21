#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def step(state, action):
    def direction_to_change(direction):
        if direction == 0:
            return (1, 0)
        elif direction == 1:
            return (0, 1)
        elif direction == 2:
            return (-1, 0)
        elif direction == 3:
            return (0, -1)
        return None

    if action == 0:
        change = direction_to_change(state[2])
        return (state[0] + change[0], state[1] + change[1], state[2])
    elif action == 2:
        change = direction_to_change(state[2])
        return (state[0] - change[0], state[1] - change[1], state[2])
    elif action == 1:
        return (state[0], state[1], (state[2] + 1) % 4)
    elif action == 3:
        return (state[0], state[1], (state[2] + 3) % 4)

def find_state(maze):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y]:
                return (x, y, 0)

def is_valid_state(maze, state):
    return state[0] >= 0 and state[1] >= 0 and state[0] < maze.shape[0] and state[1] < maze.shape[1] and maze[state[0], state[1]]

class Explorer:
    def __init__(self, graph):
        self.graph = graph
        self.keyboard_map = {key: i for (i, key) in enumerate(['up', 'right', 'down', 'left']) }
        self.state = find_state(self.graph.maze)

    def show(self):
        fig = plt.figure()
        
        def redraw():
            image = self.graph.render(tuple(self.state[:2]), self.state[2])
            plt.imshow(image)
            fig.canvas.draw()


        def press(event):
            if event.key in ['up', 'right', 'down', 'left']:
                action = self.keyboard_map[event.key]
                nstate = step(self.state, action)
                if is_valid_state(self.graph.maze, nstate):
                    self.state = nstate
                    redraw()

            pass

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    Explorer().show()