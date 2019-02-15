from gym.envs.registration import register
from gym_maze.env import MazeEnv, GoalMazeEnv

register(
    id='Maze-v0',
    entry_point='gym_maze:MazeEnv',
    max_episode_steps = 100,
)

register(
    id='GoalMaze-v0',
    entry_point='gym_maze:GoalMazeEnv',
    max_episode_steps = 200,
)