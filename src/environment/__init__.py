from gym.envs.registration import register

register(
    id = 'Mushroom-v0',
    entry_point = 'environment.mushroom:MushroomEnv',
    max_episode_steps = 200,
)