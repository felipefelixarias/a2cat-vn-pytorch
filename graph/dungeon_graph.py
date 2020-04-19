#Seems like some vizdoom stuff

from .core import GridWorldScene
from environment.util.dungeon import Generator
from .maze_graph import MazeGraph
from .util import enumerate_positions
import numpy as np

def generate_maze(size):
    gen = Generator(width=size[1], height=size[0], tiles = dict(
        stone = 0,
        floor = 1,
        wall = 0
    ))
    gen.gen_level()
    gen.gen_tiles_level()
    return np.array(gen.tiles_level)

class DungeonGraph(MazeGraph):
    def __init__(self, size):
        maze = generate_maze(size)
        goal = next(enumerate_positions(maze))
        super().__init__(maze, goal)
        