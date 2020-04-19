#Graph for maze environment? Not AI2THOR

from .core import GridWorldScene
import numpy as np
from .util import compute_shortest_path_data, enumerate_positions

class MazeGraph(GridWorldScene):
    def __init__(self, maze, goal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._maze = maze
        self.graph, self.optimal_actions = compute_shortest_path_data(maze)
        self.goal = goal

    @property
    def maze(self):
        return self._maze

    @property
    def observation_shape(self):
        return self._maze.shape + (3,)

    def render(self, state):
        render = np.tile(np.expand_dims(self._maze, 2), [1, 1, 3]).astype(np.float32)
        render[state[0], state[1]] = np.array([1.0, 0., 0.])
        render[self.goal[0], self.goal[1]] = np.array([0.0, 1.0, 0.])
        return render