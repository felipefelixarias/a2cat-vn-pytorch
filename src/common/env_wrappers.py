from gym.core import ObservationWrapper, Wrapper
import gym
from collections import deque
from gym import spaces
import numpy as np

class ColorObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super(ColorObservationWrapper, self).__init__(env)

        self._original_space = self.observation_space
        self.observation_space = self._transform_space(self._original_space)

    def observation(self, observation):
        return self._transform_observation(observation, self._original_space)

    def _transform_observation(self, observation, space_ref):
        if type(space_ref) == gym.spaces.Dict:
            return { key: self._transform_observation(observation[key], s_ref) for key, s_ref in space_ref.spaces.items()}
        else:
            return np.divide(observation.astype(np.float32) - space_ref.low, space_ref.high - space_ref.low)

    def _transform_space(self, space):
        if type(space) == gym.spaces.Dict:
            return gym.spaces.Dict({key: self._transform_space(space) for key, space in space.spaces.items()})
        elif type(space) == gym.spaces.Box:
            return gym.spaces.Box(0.0, 1.0, space.shape, dtype = np.float32)
        else:
            raise Exception('Observation space not supported')

class UnrealObservationWrapper(Wrapper):
    def __init__(self, env):
        transformed_env = ColorObservationWrapper(env)
        super(UnrealObservationWrapper, self).__init__(transformed_env)

        reward_range = self.reward_range or (-1.0, 1.0)

        self._is_dict = True
        if type(self.observation_space) != gym.spaces.Dict:
            self.observation_space = gym.spaces.Dict(dict(
                observation = self.observation_space
            ))

            self._is_dict = False

        observation_shape = self.observation_space.spaces['observation'].shape
        self._pixel_change_shape = (observation_shape[0] // 4, observation_shape[1] // 4,)
        self.observation_space = gym.spaces.Dict(dict(
            last_action_reward = gym.spaces.Box(
                min(reward_range[0], 0.0), 
                max(reward_range[1], 1.0), 
                shape = (self.action_space.n + 1,),
                dtype = np.float32
            ),
            pixel_change = gym.spaces.Box(0.0, 1.0, 
                shape = self._pixel_change_shape, 
                dtype = np.float32),
            **self.observation_space.spaces
        ))

    def _get_action_reward(self, action, reward):
        frame = np.zeros((self.action_space.n + 1,), dtype = np.float32)
        frame[action] = 1.0
        frame[-1] = np.clip(float(reward), -1, 1)
        return frame

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self.observation(observation)
        
        self._last_action_reward = self._get_action_reward(action, reward)
        self._last_observation = observation['observation'] 
        return observation, reward, done, info

    def _subsample(self, a, average_width):
        s = a.shape
        sh = s[0]//average_width, average_width, s[1]//average_width, average_width
        return a.reshape(sh).mean(-1).mean(1)  

    def _calc_pixel_change(self, state, last_state):
        d = np.absolute(state[2:-2,2:-2,:] - last_state[2:-2,2:-2,:])
        # (80,80,3)
        m = np.mean(d, 2)
        c = self._subsample(m, 4)
        return c

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        
        self._last_action_reward = np.zeros(shape = (self.action_space.n + 1,), dtype = np.float32)
        if self._is_dict:
            self._last_observation = observation['observation']
        else:
            self._last_observation = observation
        return self.observation(observation)

    def observation(self, observation):
        if not self._is_dict:
            observation = dict(observation = observation)
        else:
            observation = dict(**observation) # clone the original observation

        image = observation['observation']
        observation.update(dict(
            pixel_change = self._calc_pixel_change(image, self._last_observation),
            last_action_reward = self._last_action_reward
        ))

        return observation



class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

