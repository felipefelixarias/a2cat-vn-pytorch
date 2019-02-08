
import gym

class EnvironmentWrap:
    def __init__(self, env, action_space_size, use_goal_input):
        self.env = env
        self.action_space_size = action_space_size
        self.use_goal_input = use_goal_input

    def reset(self):
        self.env.reset()
        return self.env.last_state

    @property
    def observation_space(self):
        if self.use_goal_input:
            return gym.spaces.Dict({
                "goal": gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3)),
                "image": gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3))
            })
        else:
            return gym.spaces.Dict({
                "image": gym.spaces.Box(0.0, 1.0, shape=(84, 84, 3))
            })

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.action_space_size)

    def step(self, action):
        new_obs, rew, done, _ = self.env.process(action)
        return (new_obs, rew, done, None)

    def stop(self):
        self.env.stop()

def wrap_environment(env, action_space_size, use_goal_input = True):
    return EnvironmentWrap(env, action_space_size, use_goal_input)