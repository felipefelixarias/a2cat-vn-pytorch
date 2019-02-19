import numpy as np
import random
import gym
import gym.spaces

rat_mark = 0.5

maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]
])

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

class Qmaze(gym.Env):
    def __init__(self, rat=(0,0), maze = maze, rewards = [1.0, -0.75, -0.04, -0.25]):
        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell where the "cheese" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        
        self.original_rat = rat
        self.rewards = rewards

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, (49,), np.float32)

    def reset(self, rat_pos = None):
        self.rat = rat_pos if rat_pos is not None else random.choice(self.free_cells)
        self.maze = np.copy(self._maze)        
        row, col = self.rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.maze.size
        self.total_reward = 0
        self.visited = set()
        return self.observe()

    @property
    def position(self):
        y, x, _ = self.state
        return (x, y)

    def _change_to_action(self, d):
        dy, dx = d
        if dx == -1:
            return LEFT
        elif dx == 1:
            return RIGHT
        elif dy == -1:
            return UP
        else:
            return DOWN

    def update_state(self, action):
        nrow, ncol, nmode = rat_row, rat_col, _ = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return self.rewards[0]
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return self.rewards[3]
        if mode == 'invalid':
            return self.rewards[1]
        if mode == 'valid':
            return self.rewards[2]

    def step(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()

        stats = dict()
        if status != 'not_over':
            stats['win'] = status == 'win'
        return envstate, reward, status != 'not_over', stats

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((-1,))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions

    def render(self, mode = 'human'):
        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.grid('on')
            nrows, ncols = self.maze.shape
            ax = plt.gca()
            ax.set_xticks(np.arange(0.5, nrows, 1))
            ax.set_yticks(np.arange(0.5, ncols, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            canvas = np.copy(self.maze)
            for row,col in self.visited:
                canvas[row,col] = 0.6
            rat_row, rat_col, _ = self.state
            canvas[rat_row, rat_col] = 0.3   # rat cell
            canvas[nrows-1, ncols-1] = 0.9 # cheese cell
            img = plt.imshow(canvas, interpolation='none', cmap='gray')
            plt.show()
            return img


class Maze(Qmaze):
    def observe(self):
        canvas = np.expand_dims(self.maze, 2).repeat(3, axis = 2)
        rat_row, rat_col, _ = self.state
        canvas[rat_row, rat_col, :] = [1.0, 0, 0]
        envstate = canvas.repeat(12, axis = 1).repeat(12, axis = 0)
        return envstate



gym.register(
    id = 'QMaze-v0', 
    entry_point='environment.qmaze:Qmaze'
)

gym.register(
    id = 'Maze-v0', 
    entry_point='environment.qmaze:Maze'
)