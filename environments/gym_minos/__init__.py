import gym

gym.register(
    id = 'ContinuousGoalThor-v0',
    entry_point = 'environments.gym_minos.envs.env:MinosEnv',
    max_episode_steps = 900,
)