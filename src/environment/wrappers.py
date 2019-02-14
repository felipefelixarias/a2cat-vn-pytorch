from gym.core import ObservationWrapper
import gym
import numpy as np

class ColorObservationWrapper(ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super(ColorObservationWrapper, self).__init__(*args, **kwargs)

        

        self._original_space = self.observation_space
        self.observation_space = self._transform_space(self._original_space)

    def observation(self, observation):
        return self._transform_observation(observation, self._original_space)

    def _transform_observation(self, observation, space_ref):
        if type(space_ref) == gym.spaces.Dict:
            return { key: self._transform_observation(observation[key], s_ref) for key, s_ref in space_ref.spaces.items()}
        else:
            return (observation.astype(np.float32) - float(self._original_space.low)) / float(self._original_space.high - self._original_space.low)

    def _transform_space(self, space):
        if type(space) == gym.spaces.Dict:
            return gym.spaces.Dict({key: self._transform_space(space) for key, space in space.spaces.items()})
        elif type(space) == gym.spaces.Box:
            return gym.spaces.Box(0.0, 1.0, space.shape, dtype = np.float32)
        else:
            raise Exception('Observation space not supported')

class UnrealObservationWrapper(ColorObservationWrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        deprecated_warn_once("%s doesn't implement 'observation' method. Maybe it implements deprecated '_observation' method." % type(self))
        return self._observation(observation)