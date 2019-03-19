import gym

gym.register(
    id = 'House-v0',
    entry_point = 'environments.gym_house.env:GymHouseEnvOriginal',
    max_episode_steps = 900
)
