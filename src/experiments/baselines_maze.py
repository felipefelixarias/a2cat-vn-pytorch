if __name__ == '__main__':
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)


import gym
import gym.wrappers
import gym_maze

from baselines import deepq
from common.abstraction import AbstractAgent

class BaselinesDqnAgent(AbstractAgent):
    def __init__(self, path):
        super().__init__(self, 'dqn')
        self.actor = deepq.deepq.ActWrapper.load_act(path)

    def act(self, state):
        return self.actor.step(state)


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 0.99
    return is_solved


def main():
    env = gym.wrappers.TimeLimit(gym.make("Maze-v0"), max_episode_steps=70)
    act = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.90,
    )
    print("Saving model to cartpole_model.pkl")
    act.save("maze_model.pkl")


if __name__ == '__main__':
    main()
    #env = gym.make("Maze-v0")