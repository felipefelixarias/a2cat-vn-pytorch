import gym
import numpy as np

class Catch(gym.Env):
    def __init__(self, grid_size=7, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size * 12,)*2
        state = self.state[0]
        canvas = np.zeros(im_size + (3,))
        canvas[state[0] * 12:(state[0] + 1) * 12, state[1] * 12:(state[1] + 1) * 12,:] = 255
        canvas[-13:-1, (state[2]-1) * 12:(state[2] + 2) * 12,:] = 255
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas

    def step(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over, dict()

    def reset(self):
        n = np.random.randint(0, self.grid_size-1)
        m = np.random.randint(1, self.grid_size-2)
        self.state = np.asarray([0, n, m])[np.newaxis]
        return self.observe()

gym.register(
    id = 'Catch-v0',
    entry_point='deepq.catch_experiment:Catch',
)