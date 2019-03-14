import gym
from .download import available_scenes

for key, name in available_scenes():
    gym.register(
        id = 'Graph' + key + '-v0',
        entry_point = 'environments.gym_graph.graph:OrientedGraphEnv',
        max_episode_steps = 100,
        kwargs = dict(
            graph_name = name
        )
    )