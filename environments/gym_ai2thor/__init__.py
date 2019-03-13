import gym

NUMBER_OF_SCENES = 120

for i in range(1, NUMBER_OF_SCENES + 1):
    gym.register(
        id = 'ContinuousThor%s-v0' % i,
        entry_point = 'environments.gym_ai2thor.envs.continuous:ContinuousEnv',
        max_episode_steps = 300,
        kwargs = dict(scene_id = i)
    )

    gym.register(
        id = 'DiscreteThor%s-v0' % i,
        entry_point = 'environments.gym_ai2thor.envs.discrete:DiscreteEnv',
        max_episode_steps = 150,
        kwargs = dict(scene_id = i)
    )