import numpy as np
from math import ceil
from functools import partial
import random

def build_graph(maze, goal, change_to_action):
    shape, is_empty = maze
    distances = np.ndarray(shape, dtype=np.int32)
    actions = np.ndarray(shape + (4,), dtype=np.bool)
    actions.fill(0)
    distances.fill(np.iinfo(np.int32).max)
    def fill_distance(pos, cal_pos, dist):
        x, y = pos
        if x < 0 or y < 0 or x >= shape[0] or y >= shape[1]:
            return

        if not is_empty(pos):
            return

        if distances[pos] < dist:
            return

        run_neighboors = distances[pos] != dist
        d = (cal_pos[0] - pos[0], cal_pos[1] - pos[1])
        actions[pos[0], pos[1], change_to_action(d)] = 1 
        distances[pos] = dist
      
        if run_neighboors:
            fill_distance((x + 1, y), pos, dist + 1)
            fill_distance((x - 1, y), pos, dist + 1)
            fill_distance((x, y + 1), pos, dist + 1)
            fill_distance((x, y - 1), pos, dist + 1)

    actions[goal[0], goal[1], :] = 1
    fill_distance(goal, goal, 0)
    return (actions, distances,) 

def build_graph_from_env(env):
    is_empty = lambda point: env._get_pixel(point[0], point[1]) != '+'
    return build_graph(((env._maze_size, env._maze_size,), is_empty,), env._goal_pos, env._change_to_action)

def iterate_env_positions(goalenv):
    return list(goalenv._iter_pos())

def render_maze(env, sample):
    (position, goal) = sample
    env._pos = position
    env._goal_pos = goal
    state = env._get_current_state()
    if type(state) == dict:
        return [state['observation'] / 255.0, state['desired_goal'] / 255.0]
    else:
        return [state / 255.0]

def build_single_goal_dataset(deterministic = False):
    import gym
    import gym_maze

    goalenv = gym_maze.GoalMazeEnv(fixed_goal = True)
    goalenv.reset()

    actions, _ = build_graph_from_env(goalenv)

    if deterministic:
        shape = actions.shape
        actions = actions.reshape((-1, 4))
        actions_data = np.zeros(actions.shape, dtype = np.float32)
        actions_data[np.arange(actions.shape[0]), np.argmax(actions == 1, 1)] = 1
        actions_data = actions_data.reshape(shape)
    else:
        actions_data = actions.astype(np.float)

    actions = lambda pos: actions_data[pos[0], pos[1], :]

    positions = [(x, goalenv._goal_pos) for x in goalenv._iter_pos()]
    return (positions, partial(render_maze, goalenv), actions)

def build_multiple_goal_dataset():
    import gym
    import gym_maze

    goalenv = gym_maze.GoalMazeEnv()
    positions = [(x, y) for x in goalenv._iter_pos() for y in goalenv._iter_pos() if x != y]
    return (positions, partial(render_maze, goalenv),)

class Dataset:
    def __init__(self, dataset, batchsize):
        (self.data, self.render_maze, self.actions) = dataset
        self.data = list(self.data)
        self.batchsize = batchsize
        self._pos = 0

    def shuffle(self):
        random.shuffle(self.data)
        self._pos = 0

    def iter(self):
        for i in range(ceil(len(self.data) / self.batchsize)):
            from_pos = i * self.batchsize
            to_pos = min(len(self.data) + 1, from_pos + self.batchsize)

            batch = self.data[from_pos:to_pos]
            
            batch_render = [self.render_maze(x) for x in batch]
            data = list(map(lambda *x: np.array(x), *batch_render))
            labels = np.array([self.actions(x[0]) for x in batch])
            return (data, labels)


    def numpy(self):
        batch_render = [self.render_maze(x) for x in self.data]
        data = list(map(lambda *x: np.array(x), *batch_render))
        labels = np.array([self.actions(x[0]) for x in self.data])
        return (data, labels)
